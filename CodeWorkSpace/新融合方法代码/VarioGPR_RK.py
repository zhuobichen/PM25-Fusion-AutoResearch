"""

# === 预测值缓存（由 patch_methods.py 自动添加）===
_last_y_true = None
_last_y_pred = None

VarioGPR-RK — 变异函数引导的GPR残差克里金
==========================================
(Variogram-Guided GPR Residual Kriging)

创新点:
1. 用经验变异函数的空间相关结构直接引导GPR核函数设计
2. 从变异函数自动提取range→GPR length_scale，nugget→noise_level
3. 物理意义：变异函数描述了大气污染物的空间相关结构

与PolyRK的区别：
- PolyRK使用固定RBF核（length_scale需手动调参）
- VarioGPR-RK从变异函数自动提取参数

与SLOOCV_AK的区别：
- SLOOCV_AK使用逐站点留一交叉验证选择带宽（计算量大）
- VarioGPR-RK用变异函数一次拟合获取全局空间结构参数（计算效率高）
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
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
import netCDF4 as nc
from scipy.spatial.distance import cdist
from scipy.optimize import minimize

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


def compute_empirical_variogram(coords, values, n_lags=15, max_dist=None):
    """
    计算经验变异函数

    Parameters:
    -----------
    coords : array (n, 2) - 坐标
    values : array (n,) - 残差值
    n_lags : int - 滞后距离数
    max_dist : float - 最大距离（度）

    Returns:
    --------
    lag_distances : array - 滞后距离
    gamma_values : array - 半方差值
    """
    n = len(coords)
    dist_matrix = cdist(coords, coords)

    if max_dist is None:
        max_dist = np.max(dist_matrix) / 3

    # 定义滞后区间
    lag_edges = np.linspace(0, max_dist, n_lags + 1)
    lag_distances = []
    gamma_values = []

    for i in range(n_lags):
        lag_min = lag_edges[i]
        lag_max = lag_edges[i + 1]
        lag_mid = (lag_min + lag_max) / 2

        # 找到该滞后区间内的点对
        mask = (dist_matrix >= lag_min) & (dist_matrix < lag_max)
        mask = np.triu(mask, k=1)  # 只取上三角

        if np.sum(mask) < 5:
            continue

        # 计算半方差
        pairs_i, pairs_j = np.where(mask)
        squared_diffs = (values[pairs_i] - values[pairs_j]) ** 2
        gamma = np.mean(squared_diffs) / 2

        lag_distances.append(lag_mid)
        gamma_values.append(gamma)

    return np.array(lag_distances), np.array(gamma_values)


def fit_variogram_model(lag_distances, gamma_values):
    """
    拟合指数变异函数模型

    γ(h) = c0 + c1 * [1 - exp(-h/a)]

    Returns:
    --------
    nugget : float - 块金效应 c0
    sill : float - 基台值 c0 + c1
    range : float - 变程 a
    """
    if len(lag_distances) < 3:
        # 数据不足，返回默认值
        return 0.1, 1.0, 15.0

    # 初始猜测
    c0_init = gamma_values[0] if len(gamma_values) > 0 else 0.1
    c1_init = gamma_values[-1] - c0_init if len(gamma_values) > 1 else 0.9
    a_init = lag_distances[len(lag_distances)//2] if len(lag_distances) > 2 else 15.0

    # 指数变异函数模型
    def exponential_model(params, h):
        c0, c1, a = params
        return c0 + c1 * (1 - np.exp(-h / max(a, 1e-6)))

    # 损失函数（加权最小二乘）
    def loss(params):
        c0, c1, a = params
        if c0 < 0 or c1 < 0 or a < 0:
            return 1e10
        predicted = exponential_model(params, lag_distances)
        weights = 1.0 / (lag_distances + 1e-6)  # 近距离权重更大
        return np.sum(weights * (gamma_values - predicted) ** 2)

    # 优化
    result = minimize(loss, [c0_init, c1_init, a_init],
                     bounds=[(1e-6, 10), (1e-6, 10), (1, 200)],
                     method='L-BFGS-B')

    c0, c1, a = result.x
    nugget = c0
    sill = c0 + c1
    range_param = a

    return nugget, sill, range_param


class VarioGPR_RK:
    """
    变异函数引导的GPR残差克里金

    三阶段融合：
    1. 多项式偏差校正（OLS）
    2. 变异函数分析提取空间结构
    3. 以变异函数参数引导GPR建模残差
    """

    def __init__(self, poly_degree=2, matern_nu=1.5,
                 variogram_n_lags=15, optimize_kernel=True):
        self.poly_degree = poly_degree
        self.matern_nu = matern_nu
        self.variogram_n_lags = variogram_n_lags
        self.optimize_kernel = optimize_kernel
        self.poly = None
        self.ols = None
        self.gpr = None
        self.variogram_params = None

    def fit(self, X_train, m_train, y_train):
        """
        训练VarioGPR-RK模型

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

        # 阶段2: 变异函数分析
        lag_distances, gamma_values = compute_empirical_variogram(
            X_train, residual, n_lags=self.variogram_n_lags
        )

        if len(lag_distances) >= 3:
            nugget, sill, range_param = fit_variogram_model(lag_distances, gamma_values)
        else:
            # 默认参数
            nugget = 0.1
            sill = 1.0
            range_param = 15.0

        self.variogram_params = {
            'nugget': nugget,
            'sill': sill,
            'range': range_param
        }

        # 阶段3: 变异函数引导的GPR
        # 将变异函数参数映射到GPR核参数
        length_scale = range_param  # range → length_scale
        signal_var = sill  # sill → signal variance
        noise_var = nugget  # nugget → noise variance

        # 构建Matérn核
        kernel = ConstantKernel(signal_var, (1e-3, 1e3)) * Matern(
            length_scale=length_scale,
            length_scale_bounds=(max(length_scale * 0.1, 1), min(length_scale * 10, 200)),
            nu=self.matern_nu
        ) + WhiteKernel(noise_level=noise_var, noise_level_bounds=(1e-6, 10))

        self.gpr = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=3 if self.optimize_kernel else 0,
            alpha=0.1,
            normalize_y=True
        )
        self.gpr.fit(X_train, residual)

        return self

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
        gpr_pred = self.gpr.predict(X_test)

        # 融合
        y_pred = poly_pred + gpr_pred

        # 非负约束
        y_pred = np.maximum(y_pred, 0)

        return y_pred

    def predict_with_uncertainty(self, X_test, m_test):
        """预测并返回不确定性"""
        m_poly = self.poly.transform(m_test.reshape(-1, 1))
        poly_pred = self.ols.predict(m_poly)

        gpr_pred, gpr_std = self.gpr.predict(X_test, return_std=True)

        y_pred = poly_pred + gpr_pred
        y_std = gpr_std

        return y_pred, y_std


def run_variogpr_rk_ten_fold(selected_day='2020-01-01'):
    """
    运行VarioGPR-RK十折交叉验证
    """
    print("="*60)
    print("VarioGPR-RK Ten-Fold Cross Validation")
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
        model = VarioGPR_RK(
            poly_degree=2,
            matern_nu=1.5,
            variogram_n_lags=15,
            optimize_kernel=True
        )
        model.fit(X_train, m_train, y_train)

        # 预测
        y_pred = model.predict(X_test, m_test)

        results[fold_id] = {
            'y_true': y_test,
            'y_pred': y_pred
        }

        # 打印变异函数参数
        vp = model.variogram_params
        print(f"  Fold {fold_id}: nugget={vp['nugget']:.3f}, sill={vp['sill']:.3f}, range={vp['range']:.1f}")

    # 汇总
    y_pred_all = np.concatenate([results[f]['y_pred'] for f in range(1, 11) if results[f]])
    y_true_all = np.concatenate([results[f]['y_true'] for f in range(1, 11) if results[f]])

    metrics = compute_metrics(y_true_all, y_pred_all)
    # 缓存预测值供多天聚合使用
    global _last_y_true, _last_y_pred
    _last_y_true = y_true_all
    _last_y_pred = y_pred_all


    print("\n=== Results ===")
    print(f"  VarioGPR-RK: R2={metrics['R2']:.4f}, MAE={metrics['MAE']:.2f}, RMSE={metrics['RMSE']:.2f}, MB={metrics['MB']:.4f}")

    # 保存结果
    result_df = pd.DataFrame([{
        'method': 'VarioGPR_RK',
        **metrics
    }])
    result_df.to_csv(f'{output_dir}/VarioGPR_RK_summary.csv', index=False)

    print(f"\nResults saved to: {output_dir}/")

    return metrics


if __name__ == '__main__':
    metrics = run_variogpr_rk_ten_fold('2020-01-01')
    print(f"\nVarioGPR-RK: R2={metrics['R2']:.4f}")
