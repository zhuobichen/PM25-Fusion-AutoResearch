"""

# === 预测值缓存（由 patch_methods.py 自动添加）===
_last_y_true = None
_last_y_pred = None

HeteroscedasticGPRPolyRK - 异方差高斯过程残差多项式克里金
=========================================================
在PolyRK的二次多项式OLS全局校正 + 局部GPR克里金混合架构基础上，
放宽GPR残差建模的同方差假设，改用异方差GPR建模残差的空间分布。

创新点:
1. 分层计算残差方差: 低/中/高浓度区各有不同方差
2. 构建异方差权重: w_i = sigma2_min / sigma2_layer_i
3. 用 sample_weight 传入 GPR 拟合，高浓度大方差区域自动降权

阈值: T1=35 ug/m3, T2=75 ug/m3
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

# 浓度分层阈值
T1 = 35.0
T2 = 75.0


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


def get_concentration_layer(m_value):
    """0=低层, 1=中层, 2=高层"""
    if m_value < T1:
        return 0
    elif m_value < T2:
        return 1
    else:
        return 2


def compute_heteroscedastic_weights(m_values, residual_values):
    """计算异方差权重: w_i = sigma2_min / sigma2_layer_i"""
    layers = np.array([get_concentration_layer(v) for v in m_values])
    layer_variances = {}
    for layer_id in [0, 1, 2]:
        mask = layers == layer_id
        if np.sum(mask) > 0:
            layer_variances[layer_id] = np.var(residual_values[mask])
        else:
            layer_variances[layer_id] = np.var(residual_values)
    sigma_min_sq = min(layer_variances.values())
    weights = np.array([sigma_min_sq / (layer_variances[l] + 1e-8) for l in layers])
    return weights


def run_HeteroscedasticGPRPolyRK_ten_fold(selected_day='2020-01-01'):
    print("=" * 60)
    print("HeteroscedasticGPRPolyRK Ten-Fold Cross Validation")
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

    kernel = ConstantKernel(10.0, (1e-2, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e1))

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

        # Step 2: 计算异方差权重
        weights = compute_heteroscedastic_weights(m_train, residual)

        # Step 3: 异方差加权GPR拟合
        gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, alpha=0.1, normalize_y=True)
        gpr.fit(X_train, residual, sample_weight=weights)
        gpr_pred = gpr.predict(X_test)

        # Step 4: 融合
        hgprk_pred = pred_ols + gpr_pred

        results[fold_id] = {
            'y_true': y_test,
            'hgprk': hgprk_pred
        }
        print(f"  Fold {fold_id}: completed")

    # 汇总
    hgprk_all = np.concatenate([results[f]['hgprk'] for f in range(1, 11) if results[f]])
    true_all = np.concatenate([results[f]['y_true'] for f in range(1, 11) if results[f]])

    print("\n=== Results ===")
    hgprk_metrics = compute_metrics(true_all, hgprk_all)
    # 缓存预测值供多天聚合使用
    global _last_y_true, _last_y_pred
    _last_y_true = true_all
    _last_y_pred = hgprk_all

    print(f"  HGP-RK: R2={hgprk_metrics['R2']:.4f}, MAE={hgprk_metrics['MAE']:.2f}, RMSE={hgprk_metrics['RMSE']:.2f}")

    result_df = pd.DataFrame([{'method': 'HeteroscedasticGPRPolyRK', **hgprk_metrics}])
    result_df.to_csv(f'{output_dir}/HeteroscedasticGPRPolyRK_summary.csv', index=False)
    print(f"\nResults saved to: {output_dir}/HeteroscedasticGPRPolyRK_summary.csv")

    return hgprk_metrics


if __name__ == '__main__':
    metrics = run_HeteroscedasticGPRPolyRK_ten_fold('2020-01-01')
    print(f"\nFinal: R2={metrics['R2']:.4f}")
