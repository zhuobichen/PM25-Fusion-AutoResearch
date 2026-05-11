"""

# === 预测值缓存（由 patch_methods.py 自动添加）===
_last_y_true = None
_last_y_pred = None

MultiKernelGPRPolyRK - 多核高斯过程残差多项式克里金
=====================================================
在PolyRK的二次多项式OLS全局校正 + 局部GPR克里金混合架构基础上，
使用多核(multi-kernel)GPR替代单一RBF核GPR。

创新点:
1. 短程核 (RBF, ell~10km): 捕捉城市尺度局地变异
2. 中程核 (RBF, ell~40km): 捕捉区域梯度
3. WhiteKernel: 建模监测噪声
4. 多核联合优化，同时覆盖多个空间相关尺度
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
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true, y_pred = y_true[mask], y_pred[mask]
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


def create_multi_kernel():
    """短程(~10km) + 中程(~40km) + 噪声"""
    kernel = (
        ConstantKernel(10.0, (1e-2, 1e3)) * RBF(length_scale=10.0, length_scale_bounds=(1.0, 20.0)) +
        ConstantKernel(10.0, (1e-2, 1e3)) * RBF(length_scale=40.0, length_scale_bounds=(20.0, 100.0)) +
        WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e1))
    )
    return kernel


def create_single_kernel():
    """单核对比基准"""
    kernel = ConstantKernel(10.0, (1e-2, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e1))
    return kernel


def run_MultiKernelGPRPolyRK_ten_fold(selected_day='2020-01-01'):
    print("=" * 60)
    print("MultiKernelGPRPolyRK Ten-Fold Cross Validation")
    print("=" * 60)

    print("\n=== Loading Data ===")
    monitor_df = pd.read_csv(monitor_file)
    fold_df = pd.read_csv(fold_file)

    day_df = monitor_df[monitor_df['Date'] == selected_day].copy()
    day_df = day_df.merge(fold_df, on=['Date', 'Site'], how='left')
    day_df = day_df.dropna(subset=['Lat', 'Lon', 'Conc'])

    ds = nc.Dataset(cmaq_file, 'r')
    lon_cmaq = ds.variables['lon'][:]
    lat_cmaq = ds.variables['lat'][:]
    pred_pm25 = ds.variables['pred_PM25'][:]
    ds.close()

    from datetime import datetime
    date_obj = datetime.strptime(selected_day, '%Y-%m-%d')
    day_idx = (date_obj - datetime(2020, 1, 1)).days
    pred_day = pred_pm25[day_idx]

    print("=== Extracting CMAQ at Sites ===")
    cmaq_values = []
    for _, row in day_df.iterrows():
        val = get_cmaq_at_site(row['Lon'], row['Lat'], lon_cmaq, lat_cmaq, pred_day)
        cmaq_values.append(val)
    day_df['CMAQ'] = cmaq_values

    print(f"Data loaded: {len(day_df)} monitoring records")

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

        # Step 1: 二次多项式OLS校正
        poly = PolynomialFeatures(degree=2, include_bias=False)
        m_train_poly = poly.fit_transform(m_train.reshape(-1, 1))
        m_test_poly = poly.transform(m_test.reshape(-1, 1))

        ols = LinearRegression()
        ols.fit(m_train_poly, y_train)
        pred_ols = ols.predict(m_test_poly)
        residual = y_train - ols.predict(m_train_poly)

        # Step 2: 残差标准化
        residual_mean = np.mean(residual)
        residual_std = np.std(residual) + 1e-8
        residual_norm = (residual - residual_mean) / residual_std

        # Step 3: 多核GPR拟合
        multi_kernel = create_multi_kernel()
        gpr_multi = GaussianProcessRegressor(kernel=multi_kernel, n_restarts_optimizer=3, alpha=0.1, normalize_y=True)
        gpr_multi.fit(X_train, residual_norm)
        gpr_multi_pred = gpr_multi.predict(X_test)
        gpr_multi_pred = gpr_multi_pred * residual_std + residual_mean

        # Step 4: 融合
        mkgprk_pred = pred_ols + gpr_multi_pred

        results[fold_id] = {
            'y_true': y_test,
            'mkgprk': mkgprk_pred
        }
        print(f"  Fold {fold_id}: completed")

    # 汇总
    mkgprk_all = np.concatenate([results[f]['mkgprk'] for f in range(1, 11) if results[f]])
    true_all = np.concatenate([results[f]['y_true'] for f in range(1, 11) if results[f]])

    print("\n=== Results ===")
    mkgprk_metrics = compute_metrics(true_all, mkgprk_all)
    # 缓存预测值供多天聚合使用
    global _last_y_true, _last_y_pred
    _last_y_true = true_all
    _last_y_pred = mkgprk_all

    print(f"  MKGPR-RK: R2={mkgprk_metrics['R2']:.4f}, MAE={mkgprk_metrics['MAE']:.2f}, RMSE={mkgprk_metrics['RMSE']:.2f}")

    result_df = pd.DataFrame([{'method': 'MultiKernelGPRPolyRK', **mkgprk_metrics}])
    result_df.to_csv(f'{output_dir}/MultiKernelGPRPolyRK_summary.csv', index=False)
    print(f"\nResults saved to: {output_dir}/MultiKernelGPRPolyRK_summary.csv")

    return mkgprk_metrics


if __name__ == '__main__':
    metrics = run_MultiKernelGPRPolyRK_ten_fold('2020-01-01')
    print(f"\nFinal: R2={metrics['R2']:.4f}")
