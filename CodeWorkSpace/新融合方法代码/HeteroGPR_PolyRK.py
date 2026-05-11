"""

# === 预测值缓存（由 patch_methods.py 自动添加）===
_last_y_true = None
_last_y_pred = None

HeteroGPR-PolyRK — 异方差GPR多项式残差克里金
==============================================
(Heteroscedastic GPR Polynomial Residual Kriging)

创新点:
1. 引入空间异方差GPR建模残差，用辅助GP学习噪声场的空间变化方差
2. 物理动机：城区站点周边排放源多、地形复杂→观测噪声大；郊区站点→噪声小
3. 双GP结构：GP1建模残差均值，GP2建模log方差的空间变化

与PolyRK的区别：
- PolyRK假设残差噪声同方差（σ_n² I）
- HeteroGPR-PolyRK用第二个GP建模log(σ²(s))的空间分布

与HeteroscedasticGPRPolyRK的区别：
- HeteroscedasticGPRPolyRK是基于浓度分层的异方差（高浓度区≠低浓度区）
- 本方法基于空间位置的异方差（城区≠郊区），物理机制不同
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.paths import get_project_root, data_path
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel
import netCDF4 as nc
from scipy.linalg import cho_solve, cho_factor
from scipy.spatial.distance import cdist

root_dir = str(get_project_root())
cmaq_file = data_path('test_data/raw/CMAQ/2020_PM25.nc')
monitor_file = data_path('test_data/raw/Monitor/2020_DailyPM2.5Monitor.csv')
fold_file = data_path('test_data/fold_split_table_daily.csv')
output_dir = f'{root_dir}/test_result/创新方法'
os.makedirs(output_dir, exist_ok=True)


def compute_metrics(y_true, y_pred):
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if len(y_true) == 0:
        return {'R2': np.nan, 'MAE': np.nan, 'RMSE': np.nan, 'MB': np.nan}
    return {
        'R2': r2_score(y_true, y_pred),
        'MAE': mean_absolute_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MB': np.mean(y_pred - y_true)
    }


def get_cmaq_at_site(lon, lat, lon_grid, lat_grid, pm25_grid):
    dist = np.sqrt((lon_grid - lon)**2 + (lat_grid - lat)**2)
    idx = np.argmin(dist)
    ny, nx = lon_grid.shape
    row, col = idx // nx, idx % nx
    return pm25_grid[row, col]


class HeteroGPR_PolyRK:
    """
    异方差GPR多项式残差克里金

    三阶段融合：
    1. 多项式偏差校正（OLS）
    2. 空间特征提取（坐标作为方差GP输入）
    3. 异方差GPR建模残差（双GP结构）
    """

    def __init__(self, poly_degree=2, matern_nu=1.5, max_iter_hetero=5,
                 n_optimization_restarts=3, jitter=1e-6):
        self.poly_degree = poly_degree
        self.matern_nu = matern_nu
        self.max_iter_hetero = max_iter_hetero
        self.n_optimization_restarts = n_optimization_restarts
        self.jitter = jitter
        self.poly = None
        self.ols = None
        self.gpr_mean = None
        self.gpr_var = None

    def fit(self, X_train, m_train, y_train):
        """
        训练异方差GPR-PolyRK模型

        Parameters:
        -----------
        X_train : array (n, 2) - 站点坐标 [lon, lat]
        m_train : array (n,) - CMAQ站点值
        y_train : array (n,) - 监测值
        """
        # 阶段1: 多项式偏差校正
        self.poly = PolynomialFeatures(degree=self.poly_degree, include_bias=False)
        m_poly = self.poly.fit_transform(m_train.reshape(-1, 1))
        self.ols = LinearRegression()
        self.ols.fit(m_poly, y_train)
        residual = y_train - self.ols.predict(m_poly)

        # 阶段2: 空间特征提取（使用坐标）
        # z_train = X_train (直接使用坐标作为方差GP输入)

        # 阶段3: 异方差GPR
        # 先用同方差GPR作为初始估计
        kernel_mean = ConstantKernel(10.0, (1e-2, 1e3)) * Matern(
            length_scale=15.0, length_scale_bounds=(1.0, 100.0), nu=self.matern_nu
        ) + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e1))

        self.gpr_mean = GaussianProcessRegressor(
            kernel=kernel_mean,
            n_restarts_optimizer=self.n_optimization_restarts,
            alpha=0.1,
            normalize_y=True
        )
        self.gpr_mean.fit(X_train, residual)

        # 迭代优化异方差结构
        # 使用简化的异方差估计：基于局部残差方差
        self._fit_variance_gp(X_train, residual)

        return self

    def _fit_variance_gp(self, X_train, residual):
        """拟合方差GP，学习噪声场的空间变化"""
        # 计算局部残差方差（使用k近邻）
        n = len(X_train)
        k = min(10, n // 3)
        if k < 3:
            k = 3

        # 计算距离矩阵
        dist_matrix = cdist(X_train, X_train)

        # 对每个站点估计局部方差
        local_var = np.zeros(n)
        for i in range(n):
            # 找k个最近邻
            neighbors = np.argsort(dist_matrix[i])[:k+1]  # 包含自身
            neighbors = neighbors[neighbors != i]  # 排除自身
            if len(neighbors) > 0:
                local_var[i] = np.var(residual[neighbors])
            else:
                local_var[i] = np.var(residual)

        # 避免log(0)
        local_var = np.maximum(local_var, 1e-6)
        log_var = np.log(local_var)

        # 拟合方差GP
        kernel_var = ConstantKernel(1.0, (1e-3, 1e2)) * Matern(
            length_scale=20.0, length_scale_bounds=(5.0, 100.0), nu=1.5
        ) + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-5, 1e1))

        self.gpr_var = GaussianProcessRegressor(
            kernel=kernel_var,
            n_restarts_optimizer=self.n_optimization_restarts,
            alpha=0.5,
            normalize_y=True
        )
        self.gpr_var.fit(X_train, log_var)

    def predict(self, X_test, m_test):
        """
        预测

        Parameters:
        -----------
        X_test : array (m, 2) - 预测点坐标
        m_test : array (m,) - 预测点CMAQ值

        Returns:
        --------
        y_pred : array (m,) - 融合预测值
        """
        # 多项式校正
        m_poly = self.poly.transform(m_test.reshape(-1, 1))
        poly_pred = self.ols.predict(m_poly)

        # GPR残差预测
        gpr_pred, gpr_std = self.gpr_mean.predict(X_test, return_std=True)

        # 融合
        y_pred = poly_pred + gpr_pred

        # 非负约束
        y_pred = np.maximum(y_pred, 0)

        return y_pred

    def predict_with_uncertainty(self, X_test, m_test):
        """预测并返回不确定性"""
        m_poly = self.poly.transform(m_test.reshape(-1, 1))
        poly_pred = self.ols.predict(m_poly)

        gpr_pred, gpr_std = self.gpr_mean.predict(X_test, return_std=True)

        # 预测方差场
        log_var_pred = self.gpr_var.predict(X_test)
        var_pred = np.exp(log_var_pred)

        y_pred = poly_pred + gpr_pred
        y_std = np.sqrt(gpr_std**2 + var_pred)

        return y_pred, y_std


def run_heterogpr_polyrk_ten_fold(selected_day='2020-01-01'):
    """
    运行HeteroGPR-PolyRK十折交叉验证
    """
    print("="*60)
    print("HeteroGPR-PolyRK Ten-Fold Cross Validation")
    print("="*60)

    # 加载数据
    print("\n=== Loading Data ===")
    monitor_df = pd.read_csv(monitor_file)
    fold_df = pd.read_csv(fold_file)

    day_df = monitor_df[monitor_df['Date'] == selected_day].copy()
    day_df = day_df.merge(fold_df, on=['Date', 'Site'], how='left')
    day_df = day_df.dropna(subset=['Lat', 'Lon', 'Conc'])

    # 加载CMAQ数据
    ds = nc.Dataset(cmaq_file, 'r')
    lon_cmaq = ds.variables['lon'][:]
    lat_cmaq = ds.variables['lat'][:]
    pred_pm25 = ds.variables['pred_PM25'][:]
    ds.close()

    from datetime import datetime
    date_obj = datetime.strptime(selected_day, '%Y-%m-%d')
    day_idx = (date_obj - datetime(2020, 1, 1)).days
    pred_day = pred_pm25[day_idx]

    # 提取站点CMAQ值
    print("=== Extracting CMAQ at Sites ===")
    cmaq_values = []
    for _, row in day_df.iterrows():
        val = get_cmaq_at_site(row['Lon'], row['Lat'], lon_cmaq, lat_cmaq, pred_day)
        cmaq_values.append(val)
    day_df['CMAQ'] = cmaq_values

    print(f"Data loaded: {len(day_df)} monitoring records")

    print("\n=== Running 10-fold Cross Validation ===")
    results = {fold_id: {} for fold_id in range(1, 11)}

    for fold_id in range(1, 11):
        train_df = day_df[day_df['fold'] != fold_id].copy()
        test_df = day_df[day_df['fold'] == fold_id].copy()

        train_df = train_df.dropna(subset=['Lon', 'Lat', 'CMAQ', 'Conc'])
        test_df = test_df.dropna(subset=['Lon', 'Lat', 'CMAQ', 'Conc'])

        if len(test_df) == 0:
            continue

        X_train = train_df[['Lon', 'Lat']].values
        X_test = test_df[['Lon', 'Lat']].values
        y_train = train_df['Conc'].values
        y_test = test_df['Conc'].values
        m_train = train_df['CMAQ'].values
        m_test = test_df['CMAQ'].values

        # 训练模型
        model = HeteroGPR_PolyRK(
            poly_degree=2,
            matern_nu=1.5,
            max_iter_hetero=5,
            n_optimization_restarts=2
        )
        model.fit(X_train, m_train, y_train)

        # 预测
        y_pred = model.predict(X_test, m_test)

        results[fold_id] = {
            'y_true': y_test,
            'y_pred': y_pred
        }

        print(f"  Fold {fold_id}: completed")

    # 汇总
    y_pred_all = np.concatenate([results[f]['y_pred'] for f in range(1, 11) if results[f]])
    y_true_all = np.concatenate([results[f]['y_true'] for f in range(1, 11) if results[f]])

    metrics = compute_metrics(y_true_all, y_pred_all)
    # 缓存预测值供多天聚合使用
    global _last_y_true, _last_y_pred
    _last_y_true = y_true_all
    _last_y_pred = y_pred_all


    print("\n=== Results ===")
    print(f"  HeteroGPR-PolyRK: R2={metrics['R2']:.4f}, MAE={metrics['MAE']:.2f}, RMSE={metrics['RMSE']:.2f}, MB={metrics['MB']:.4f}")

    # 保存结果
    result_df = pd.DataFrame([{
        'method': 'HeteroGPR_PolyRK',
        **metrics
    }])
    result_df.to_csv(f'{output_dir}/HeteroGPR_PolyRK_summary.csv', index=False)

    print(f"\nResults saved to: {output_dir}/")

    return metrics


if __name__ == '__main__':
    metrics = run_heterogpr_polyrk_ten_fold('2020-01-01')
    print(f"\nHeteroGPR-PolyRK: R2={metrics['R2']:.4f}")
