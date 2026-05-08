# -*- coding: utf-8 -*-
"""
CGARK 十折交叉验证 - 标准模式
============================
CMAQ Gradient Anisotropic Residual Kriging

按照十折交叉验证架构文档：
- 训练：9折监测站的CMAQ网格坐标
- 预测：对1折站点所在的CMAQ网格坐标预测

创新点:
1. 多项式 OLS 偏差校正: O = a + b*M + c*M² + ε
2. CMAQ 梯度引导的各向异性 GPR (IDW-like):
   - 沿梯度方向：相关长度 = 25km
   - 垂直梯度方向：相关长度 = 10km

参数 (固定):
- lambda_along: 25.0 (km)
- lambda_across: 10.0 (km)
- poly_degree: 2
- n_neighbor: 12
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from shared.paths import get_project_root, data_path
import json
import numpy as np
import pandas as pd
import netCDF4 as nc
from datetime import datetime, timedelta
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from joblib import Parallel, delayed

ROOT_DIR = str(get_project_root())
CMAQ_FILE = data_path('test_data/raw/CMAQ/2020_PM25.nc')
MONITOR_FILE = data_path('test_data/raw/Monitor/2020_DailyPM2.5Monitor.csv')
FOLD_FILE = data_path('test_data/fold_split_table_daily.csv')
OUTPUT_FILE = f'{ROOT_DIR}/Innovation/success/CGARK/CGARK_all_stages.json'

# VNA baseline
BASELINE = {
    'pre_exp': {'R2': 0.8907, 'RMSE': 16.68, 'MB': 0.70},
    'stage1':  {'R2': 0.9034, 'RMSE': 16.48, 'MB': 0.50},
    'stage2':  {'R2': 0.8408, 'RMSE': 5.05, 'MB': 0.05},
    'stage3':  {'R2': 0.9031, 'RMSE': 12.20, 'MB': 0.42},
}

STAGES = {
    'pre_exp': ('2020-01-01', '2020-01-05'),
    'stage1':  ('2020-01-01', '2020-01-31'),
    'stage2':  ('2020-07-01', '2020-07-31'),
    'stage3':  ('2020-12-01', '2020-12-31'),
}

# CGARK parameters (fixed)
LAMBDA_ALONG = 25.0  # km
LAMBDA_ACROSS = 10.0  # km
POLY_DEGREE = 2
N_NEIGHBOR = 12


def compute_metrics(y_true, y_pred):
    mask = ~(np.isnan(y_true) | np.isnan(y_pred) | np.isinf(y_true) | np.isinf(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if len(y_true) == 0:
        return {'R2': np.nan, 'MAE': np.nan, 'RMSE': np.nan, 'MB': np.nan}
    return {
        'R2': float(r2_score(y_true, y_pred)),
        'MAE': float(mean_absolute_error(y_true, y_pred)),
        'RMSE': float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'MB': float(np.mean(y_pred - y_true))
    }


def get_cmaq_at_site(lon, lat, lon_grid, lat_grid, pm25_grid):
    dist = np.sqrt((lon_grid - lon)**2 + (lat_grid - lat)**2)
    idx = np.argmin(dist)
    ny, nx = lon_grid.shape
    row, col = idx // nx, idx % nx
    return pm25_grid[row, col]


def compute_gradient_direction(lon, lat, lon_grid, lat_grid, pm25_grid):
    """计算CMAQ梯度方向"""
    dist = np.sqrt((lon_grid - lon)**2 + (lat_grid - lat)**2)
    idx = np.argmin(dist)
    ny, nx = lon_grid.shape
    row, col = idx // nx, idx % nx

    row = np.clip(row, 1, ny - 2)
    col = np.clip(col, 1, nx - 2)

    pm_xp = pm25_grid[row, min(col + 1, nx - 1)]
    pm_xm = pm25_grid[row, max(col - 1, 0)]
    pm_yp = pm25_grid[min(row + 1, ny - 1), col]
    pm_ym = pm25_grid[max(row - 1, 0), col]

    lon_step = np.mean(np.diff(lon_grid, axis=1))
    lat_step = np.mean(np.diff(lat_grid, axis=0))

    dpm_dlon = (pm_xp - pm_xm) / (2 * lon_step) if lon_step != 0 else 0
    dpm_dlat = (pm_yp - pm_ym) / (2 * lat_step) if lat_step != 0 else 0

    gradient_magnitude = np.sqrt(dpm_dlon**2 + dpm_dlat**2)
    gradient_direction = np.arctan2(dpm_dlon, dpm_dlat)

    return gradient_magnitude, gradient_direction


def anisotropic_distance_weighted_predict(x_obs, residual_obs, x_pred, direction_pred,
                                         lambda_along, lambda_across, n_neighbor=12):
    """各向异性距离加权预测"""
    n_pred = x_pred.shape[0]
    n_obs = x_obs.shape[0]

    # 广播计算所有距离
    diff = x_pred[:, np.newaxis, :] - x_obs[np.newaxis, :, :]  # (n_pred, n_obs, 2)

    cos_d = np.cos(direction_pred)[:, np.newaxis]
    sin_d = np.sin(direction_pred)[:, np.newaxis]

    d_along = diff[:, :, 0] * cos_d + diff[:, :, 1] * sin_d
    d_perp = -diff[:, :, 0] * sin_d + diff[:, :, 1] * cos_d

    dist_aniso = np.sqrt((d_along / lambda_along)**2 + (d_perp / lambda_across)**2)

    # 选择最近邻
    if n_neighbor < n_obs:
        idx = np.argpartition(dist_aniso, n_neighbor, axis=1)[:, :n_neighbor]
        row_idx = np.arange(n_pred)[:, np.newaxis]
        dists_k = dist_aniso[row_idx, idx]
        values_k = residual_obs[idx]
    else:
        dists_k = dist_aniso
        values_k = np.tile(residual_obs, (n_pred, 1))

    dists_k = np.maximum(dists_k, 1e-10)

    sigma = lambda_across / 3.0
    weights = np.exp(-0.5 * (dists_k / sigma)**2)
    weights = weights / weights.sum(axis=1, keepdims=True)

    pred_values = np.sum(weights * values_k, axis=1)

    return pred_values


def ten_fold_cgark(selected_day):
    """CGARK 十折验证"""
    monitor_df = pd.read_csv(MONITOR_FILE)
    fold_df = pd.read_csv(FOLD_FILE)

    day_df = monitor_df[monitor_df['Date'] == selected_day].copy()
    day_df = day_df.merge(fold_df, on=['Date', 'Site'], how='left')
    day_df = day_df.dropna(subset=['Lat', 'Lon', 'Conc'])

    if len(day_df) < 100:
        return np.array([]), np.array([])

    ds = nc.Dataset(CMAQ_FILE, 'r')
    lon_cmaq = ds.variables['lon'][:]
    lat_cmaq = ds.variables['lat'][:]
    pred_pm25 = ds.variables['pred_PM25'][:]
    ds.close()

    date_obj = datetime.strptime(selected_day, '%Y-%m-%d')
    day_idx = (date_obj - datetime(2020, 1, 1)).days
    if day_idx >= pred_pm25.shape[0]:
        return np.array([]), np.array([])
    cmaq_day = pred_pm25[day_idx]

    # 获取CMAQ值和梯度方向
    cmaq_values = []
    grad_dirs = []
    for _, row in day_df.iterrows():
        val = get_cmaq_at_site(row['Lon'], row['Lat'], lon_cmaq, lat_cmaq, cmaq_day)
        _, direction = compute_gradient_direction(row['Lon'], row['Lat'], lon_cmaq, lat_cmaq, cmaq_day)
        cmaq_values.append(val)
        grad_dirs.append(direction)
    day_df['CMAQ'] = cmaq_values
    day_df['gradient_direction'] = grad_dirs

    all_y_true = []
    all_y_pred = []

    for fold_id in range(1, 11):
        train_df = day_df[day_df['fold'] != fold_id].copy()
        test_df = day_df[day_df['fold'] == fold_id].copy()

        train_df = train_df.dropna(subset=['Lon', 'Lat', 'CMAQ', 'Conc', 'gradient_direction'])
        test_df = test_df.dropna(subset=['Lon', 'Lat', 'CMAQ', 'Conc', 'gradient_direction'])

        if len(test_df) == 0 or len(train_df) == 0:
            continue

        y_train = train_df['Conc'].values
        y_test = test_df['Conc'].values
        m_train = train_df['CMAQ'].values
        m_test = test_df['CMAQ'].values

        # 多项式特征
        poly = PolynomialFeatures(degree=POLY_DEGREE, include_bias=False)
        m_train_poly = poly.fit_transform(m_train.reshape(-1, 1))
        m_test_poly = poly.transform(m_test.reshape(-1, 1))

        # OLS 多项式校正
        ols = LinearRegression()
        ols.fit(m_train_poly, y_train)
        pred_ols = ols.predict(m_test_poly)
        residual_train = y_train - ols.predict(m_train_poly)

        X_train = train_df[['Lon', 'Lat']].values
        X_test = test_df[['Lon', 'Lat']].values
        direction_train = train_df['gradient_direction'].values
        direction_test = test_df['gradient_direction'].values

        # 各向异性预测
        cgark_residual_pred = anisotropic_distance_weighted_predict(
            X_train, residual_train, X_test, direction_test,
            LAMBDA_ALONG, LAMBDA_ACROSS, N_NEIGHBOR
        )

        # 融合预测
        cgark_pred = pred_ols + cgark_residual_pred

        all_y_true.extend(y_test)
        all_y_pred.extend(cgark_pred)

    return np.array(all_y_true), np.array(all_y_pred)


def run_stage_validation(stage_name, start_date, end_date):
    sep = "=" * 70
    print(sep)
    print(f"CGARK Stage: {stage_name} ({start_date} ~ {end_date})")
    print(f"Parameters: lambda_along={LAMBDA_ALONG}, lambda_across={LAMBDA_ACROSS}, poly_degree={POLY_DEGREE}")
    print(sep)

    base = BASELINE[stage_name]
    threshold_r2 = base['R2']
    print(f"VNA Baseline: R2={base['R2']:.4f}, RMSE={base['RMSE']:.2f}, MB={base['MB']:.2f}")

    date_list = []
    current_date = datetime.strptime(start_date, '%Y-%m-%d')
    end_date_obj = datetime.strptime(end_date, '%Y-%m-%d')
    while current_date <= end_date_obj:
        date_list.append(current_date.strftime('%Y-%m-%d'))
        current_date += timedelta(days=1)

    print(f"Days: {len(date_list)}")

    n_jobs = min(8, len(date_list))
    results = Parallel(n_jobs=n_jobs)(
        delayed(ten_fold_cgark)(date_str)
        for date_str in date_list
    )

    all_y_true = []
    all_y_pred = []
    day_count = 0
    for y_true, y_pred in results:
        if len(y_true) > 0:
            all_y_true.extend(y_true)
            all_y_pred.extend(y_pred)
            day_count += 1

    print(f"Processed: {day_count} days, {len(all_y_true)} predictions")

    if len(all_y_true) == 0:
        return {'R2': np.nan, 'MAE': np.nan, 'RMSE': np.nan, 'MB': np.nan}, False

    metrics = compute_metrics(np.array(all_y_true), np.array(all_y_pred))

    r2_pass = metrics['R2'] > threshold_r2
    rmse_pass = metrics['RMSE'] <= base['RMSE']
    mb_pass = abs(metrics['MB']) <= abs(base['MB'])
    innovation_pass = r2_pass and rmse_pass and mb_pass

    print(f"Result: R2={metrics['R2']:.4f}, RMSE={metrics['RMSE']:.2f}, MAE={metrics['MAE']:.2f}, MB={metrics['MB']:.2f}")
    r2_str = "PASS" if r2_pass else "FAIL"
    rmse_str = "PASS" if rmse_pass else "FAIL"
    mb_str = "PASS" if mb_pass else "FAIL"
    innov_str = "VERIFIED" if innovation_pass else "NOT VERIFIED"
    print(f"Check: R2>{threshold_r2:.4f}? {r2_str} | RMSE<={base['RMSE']}? {rmse_str} | |MB|<={abs(base['MB'])}? {mb_str}")
    print(f"Innovation: {innov_str}")

    return metrics, innovation_pass


def main():
    sep = "=" * 70
    print(sep)
    print("CGARK All Stages - CMAQ Gradient Anisotropic Residual Kriging")
    print(sep)

    results = {}
    all_pass = True

    for stage_name, (start, end) in STAGES.items():
        metrics, innovation_pass = run_stage_validation(stage_name, start, end)
        results[stage_name] = {'metrics': metrics, '判定': {'innovation_verified': innovation_pass}}
        if not innovation_pass:
            all_pass = False

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(sep)
    print("SUMMARY")
    print(sep)
    for stage, data in results.items():
        m = data['metrics']
        status = 'VERIFIED' if data['判定']['innovation_verified'] else 'NOT VERIFIED'
        print(f"{stage}: R2={m['R2']:.4f}, RMSE={m['RMSE']:.2f}, MB={m['MB']:.2f} -> {status}")

    print(f"\nAll stages passed: {all_pass}")
    print(f"Results saved: {OUTPUT_FILE}")

    return results


if __name__ == '__main__':
    main()
