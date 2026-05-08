"""
MKGPR-RK - Multi-Kernel GPR PolyRK
===================================
GPR 使用多核（RBF短程 + RBF中程 + WhiteKernel）

核心创新：
1. 全局多项式 OLS 校正
2. GPR 使用多核：
   - ConstantKernel(10.0) * RBF(length_scale=10.0)  # 短程 ~10km
   - ConstantKernel(10.0) * RBF(length_scale=40.0)  # 中程 ~40km
   - WhiteKernel(noise_level=1.0)

多核配置：
kernel = (
    ConstantKernel(10.0) * RBF(length_scale=10.0) +   # 短程 ~10km
    ConstantKernel(10.0) * RBF(length_scale=40.0) +   # 中程 ~40km
    WhiteKernel(noise_level=1.0)
)
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
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
import netCDF4 as nc

root_dir = str(get_project_root())
cmaq_file = data_path('test_data/raw/CMAQ/2020_PM25.nc')
monitor_file = data_path('test_data/raw/Monitor/2020_DailyPM2.5Monitor.csv')
fold_file = data_path('test_data/fold_split_table_daily.csv')
output_dir = f'{root_dir}/test_result/创新方法'
os.makedirs(output_dir, exist_ok=True)


def compute_metrics(y_true, y_pred):
    """计算评估指标"""
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
    """获取站点位置的 CMAQ 值"""
    dist = np.sqrt((lon_grid - lon)**2 + (lat_grid - lat)**2)
    idx = np.argmin(dist)
    ny, nx = lon_grid.shape
    row, col = idx // nx, idx % nx
    return pm25_grid[row, col]


def create_multi_kernel():
    """
    创建多核 GPR 核函数

    Returns:
        kernel: 多核组合核函数
    """
    kernel = (
        ConstantKernel(10.0, (1e-2, 1e3)) * RBF(length_scale=10.0, length_scale_bounds=(1e-2, 1e3)) +   # 短程 ~10km
        ConstantKernel(10.0, (1e-2, 1e3)) * RBF(length_scale=40.0, length_scale_bounds=(1e-2, 1e3)) +   # 中程 ~40km
        WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e1))
    )
    return kernel


def create_single_kernel():
    """
    创建单核 GPR 核函数（用于对比）

    Returns:
        kernel: 单核核函数
    """
    kernel = ConstantKernel(10.0, (1e-2, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e1))
    return kernel


class MKGPRK:
    """
    MKGPR-RK: Multi-Kernel GPR PolyRK

    多核高斯过程残差克里金
    """

    def __init__(self, poly_degree=2, use_multi_kernel=True):
        self.poly_degree = poly_degree
        self.poly = PolynomialFeatures(degree=poly_degree, include_bias=False)
        self.ols = None
        self.gpr = None
        self.X_train = None
        self.residual_mean = None
        self.residual_std = None
        self.use_multi_kernel = use_multi_kernel

    def fit(self, X, y, m):
        """
        训练 MKGPR-RK 模型

        参数：
        X: 位置坐标 (n, 2) - [Lon, Lat]
        y: 真实浓度值 (n,)
        m: CMAQ 模型值 (n,)
        """
        X = np.asarray(X)
        y = np.asarray(y)
        m = np.asarray(m)

        # 多项式 OLS 拟合
        m_poly = self.poly.fit_transform(m.reshape(-1, 1))
        self.ols = LinearRegression()
        self.ols.fit(m_poly, y)

        # 预测并计算残差
        residual = y - self.ols.predict(m_poly)

        # 残差标准化
        self.residual_mean = np.mean(residual)
        self.residual_std = np.std(residual) + 1e-8
        residual_normalized = (residual - self.residual_mean) / self.residual_std

        # 选择核函数
        if self.use_multi_kernel:
            kernel = create_multi_kernel()
        else:
            kernel = create_single_kernel()

        # GPR 拟合
        self.gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, alpha=0.1, normalize_y=True)
        self.gpr.fit(X, residual_normalized)
        self.X_train = X

    def predict(self, X, m):
        """
        预测

        参数：
        X: 位置坐标 (n, 2) - [Lon, Lat]
        m: CMAQ 模型值 (n,)

        返回：
        预测浓度值 (n,)
        """
        X = np.asarray(X)
        m = np.asarray(m)

        # OLS 预测
        m_poly = self.poly.transform(m.reshape(-1, 1))
        pred_ols = self.ols.predict(m_poly)

        # GPR 克里金校正
        if self.gpr is not None:
            residual_pred_normalized, _ = self.gpr.predict(X, return_std=True)
            residual_pred = residual_pred_normalized * self.residual_std + self.residual_mean
            pred = pred_ols + residual_pred
        else:
            pred = pred_ols

        return pred


def run_mkgprk_ten_fold(selected_day='2020-01-01'):
    """运行 MKGPR-RK 十折交叉验证"""
    print("=" * 60)
    print("MKGPR-RK Ten-Fold Cross Validation")
    print("=" * 60)

    # 加载数据
    print("\n=== Loading Data ===")
    monitor_df = pd.read_csv(monitor_file)
    fold_df = pd.read_csv(fold_file)

    day_df = monitor_df[monitor_df['Date'] == selected_day].copy()
    day_df = day_df.merge(fold_df, on='Site', how='left')
    day_df = day_df.dropna(subset=['Lat', 'Lon', 'Conc'])

    # 加载 CMAQ 数据
    ds = nc.Dataset(cmaq_file, 'r')
    lon_cmaq = ds.variables['lon'][:]
    lat_cmaq = ds.variables['lat'][:]
    pred_pm25 = ds.variables['pred_PM25'][:]
    ds.close()

    from datetime import datetime
    date_obj = datetime.strptime(selected_day, '%Y-%m-%d')
    day_idx = (date_obj - datetime(2020, 1, 1)).days
    pred_day = pred_pm25[day_idx]

    # 提取站点 CMAQ 值
    print("=== Extracting CMAQ at Sites ===")
    cmaq_values = []
    for _, row in day_df.iterrows():
        val = get_cmaq_at_site(row['Lon'], row['Lat'], lon_cmaq, lat_cmaq, pred_day)
        cmaq_values.append(val)
    day_df['CMAQ'] = cmaq_values

    ny, nx = lon_cmaq.shape
    X_grid_full = np.column_stack([lon_cmaq.ravel(), lat_cmaq.ravel()])
    y_grid_model_full = pred_day.ravel()

    print(f"Data loaded: {len(day_df)} monitoring records")

    # 定义单核 GPR 核函数（用于对比）
    single_kernel = create_single_kernel()

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

        # === 二次多项式 OLS 校正 ===
        poly = PolynomialFeatures(degree=2, include_bias=False)
        m_train_poly = poly.fit_transform(m_train.reshape(-1, 1))
        m_test_poly = poly.transform(m_test.reshape(-1, 1))

        ols_poly = LinearRegression()
        ols_poly.fit(m_train_poly, y_train)
        pred_poly = ols_poly.predict(m_test_poly)
        residual_poly = y_train - ols_poly.predict(m_train_poly)

        # === 单核 GPR (对比基准) ===
        residual_mean = np.mean(residual_poly)
        residual_std = np.std(residual_poly) + 1e-8
        residual_norm = (residual_poly - residual_mean) / residual_std

        gpr_single = GaussianProcessRegressor(kernel=single_kernel, n_restarts_optimizer=2, alpha=0.1, normalize_y=True)
        gpr_single.fit(X_train, residual_norm)
        gpr_single_pred, _ = gpr_single.predict(X_test, return_std=True)
        gpr_single_pred = gpr_single_pred * residual_std + residual_mean

        rk_single_pred = pred_poly + gpr_single_pred

        # === MKGPR-RK (多核) ===
        mkgprk = MKGPRK(poly_degree=2, use_multi_kernel=True)
        mkgprk.fit(X_train, y_train, m_train)
        mkgprk_pred = mkgprk.predict(X_test, m_test)

        results[fold_id] = {
            'y_true': y_test,
            'rk_single': rk_single_pred,
            'mkgprk': mkgprk_pred
        }

        print(f"  Fold {fold_id}: completed")

    # 汇总
    rk_single_all = np.concatenate([results[f]['rk_single'] for f in range(1, 11) if results[f]])
    mkgprk_all = np.concatenate([results[f]['mkgprk'] for f in range(1, 11) if results[f]])
    true_all = np.concatenate([results[f]['y_true'] for f in range(1, 11) if results[f]])

    # 计算 R2
    print("\n=== Results ===")
    rk_single_metrics = compute_metrics(true_all, rk_single_all)
    mkgprk_metrics = compute_metrics(true_all, mkgprk_all)

    print(f"  RK-Single: R2={rk_single_metrics['R2']:.4f}, MAE={rk_single_metrics['MAE']:.2f}, RMSE={rk_single_metrics['RMSE']:.2f}")
    print(f"  MKGPR-RK:  R2={mkgprk_metrics['R2']:.4f}, MAE={mkgprk_metrics['MAE']:.2f}, RMSE={mkgprk_metrics['RMSE']:.2f}")

    # 保存结果
    result_df = pd.DataFrame([{
        'method': 'RK_Single',
        **rk_single_metrics
    }, {
        'method': 'MKGPR_RK',
        **mkgprk_metrics
    }])
    result_df.to_csv(f'{output_dir}/MKGPRK_summary.csv', index=False)

    print(f"\nResults saved to: {output_dir}/MKGPRK_summary.csv")

    return rk_single_metrics, mkgprk_metrics


if __name__ == '__main__':
    rk_single_metrics, mkgprk_metrics = run_mkgprk_ten_fold('2020-01-01')
    print(f"\nRK-Single: R2={rk_single_metrics['R2']:.4f}")
    print(f"MKGPR-RK:  R2={mkgprk_metrics['R2']:.4f}")