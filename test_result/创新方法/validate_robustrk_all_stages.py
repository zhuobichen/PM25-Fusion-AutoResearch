# -*- coding: utf-8 -*-
"""
RobustRK All Stages Validation
==============================
验证稳健多项式残差克里金方法的创新性
使用Huber回归进行稳健的多项式残差克里金
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
from sklearn.linear_model import HuberRegressor, LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from joblib import Parallel, delayed

ROOT_DIR = str(get_project_root())
CMAQ_FILE = data_path('test_data/raw/CMAQ/2020_PM25.nc')
MONITOR_FILE = data_path('test_data/raw/Monitor/2020_DailyPM2.5Monitor.csv')
FOLD_FILE = data_path('test_data/fold_split_table_daily.csv')
OUTPUT_FILE = f'{ROOT_DIR}/Innovation/success/RobustRK/RobustRK_all_stages.json'

# VNA baseline (from benchmark_multistage.json, updated 2026-04-16)
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


def get_cmaq_grid_coord(lon, lat, lon_grid, lat_grid):
    dist = np.sqrt((lon_grid - lon)**2 + (lat_grid - lat)**2)
    idx = np.argmin(dist)
    ny, nx = lon_grid.shape
    row, col = idx // nx, idx % nx
    return lon_grid[row, col], lat_grid[row, col]


def ten_fold_for_day_robustrk(selected_day, poly_degree=2):
    monitor_df = pd.read_csv(MONITOR_FILE)
    fold_df = pd.read_csv(FOLD_FILE)

    day_df = monitor_df[monitor_df['Date'] == selected_day].copy()
    day_df = day_df.merge(fold_df, on='Site', how='left')
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

    cmaq_values = []
    for _, row in day_df.iterrows():
        val = get_cmaq_at_site(row['Lon'], row['Lat'], lon_cmaq, lat_cmaq, cmaq_day)
        cmaq_values.append(val)
    day_df['CMAQ'] = cmaq_values

    kernel = (ConstantKernel(10.0, (1e-2, 1e3)) *
              RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) +
              WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e1)))

    all_y_true = []
    all_y_pred = []

    for fold_id in range(1, 11):
        train_df = day_df[day_df['fold'] != fold_id].copy()
        test_df = day_df[day_df['fold'] == fold_id].copy()

        train_df = train_df.dropna(subset=['Lon', 'Lat', 'CMAQ', 'Conc'])
        test_df = test_df.dropna(subset=['Lon', 'Lat', 'CMAQ', 'Conc'])

        if len(test_df) == 0 or len(train_df) == 0:
            continue

        y_train = train_df['Conc'].values
        m_train = train_df['CMAQ'].values
        m_test = test_df['CMAQ'].values

        # CMAQ grid coords
        train_cmaq_coords = []
        for _, row in train_df.iterrows():
            cmaq_lon, cmaq_lat = get_cmaq_grid_coord(row['Lon'], row['Lat'], lon_cmaq, lat_cmaq)
            train_cmaq_coords.append([cmaq_lon, cmaq_lat])
        X_train = np.array(train_cmaq_coords)

        test_cmaq_coords = []
        for _, row in test_df.iterrows():
            cmaq_lon, cmaq_lat = get_cmaq_grid_coord(row['Lon'], row['Lat'], lon_cmaq, lat_cmaq)
            test_cmaq_coords.append([cmaq_lon, cmaq_lat])
        X_test = np.array(test_cmaq_coords)

        y_test = test_df['Conc'].values

        # Polynomial features
        poly = PolynomialFeatures(degree=poly_degree, include_bias=False)
        m_train_poly = poly.fit_transform(m_train.reshape(-1, 1))
        m_test_poly = poly.transform(m_test.reshape(-1, 1))

        # Huber Robust OLS
        huber_poly = HuberRegressor(epsilon=1.35, max_iter=1000)
        huber_poly.fit(m_train_poly, y_train)
        pred_huber = huber_poly.predict(m_test_poly)
        residual_huber = y_train - huber_poly.predict(m_train_poly)

        # 过滤 NaN/Inf 残差
        valid_mask = ~(np.isnan(residual_huber) | np.isinf(residual_huber))
        if np.sum(valid_mask) < 50:
            # 数据不足，使用简单预测
            robustrk_pred = pred_huber
        else:
            X_train_clean = X_train[valid_mask]
            residual_clean = residual_huber[valid_mask]

            # GPR kriging on residuals
            gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, alpha=0.1, normalize_y=True)
            try:
                gpr.fit(X_train_clean, residual_clean)
                gpr_pred = gpr.predict(X_test)
            except Exception:
                # GPR失败，使用零残差
                gpr_pred = np.zeros(len(X_test))
            robustrk_pred = pred_huber + gpr_pred

        all_y_true.extend(y_test)
        all_y_pred.extend(robustrk_pred)

    return np.array(all_y_true), np.array(all_y_pred)


def run_stage_validation(stage_name, start_date, end_date):
    sep = "=" * 70
    print(sep)
    print(f"Stage: {stage_name} ({start_date} ~ {end_date})")
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

    # 串行处理避免GPR内存问题
    results = []
    for date_str in date_list:
        print(f"  Processing {date_str}...")
        result = ten_fold_for_day_robustrk(date_str)
        results.append(result)

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
    print("RobustRK All Stages Validation (Huber Polynomial RK)")
    print(sep)

    results = {}

    for stage_name, (start, end) in STAGES.items():
        metrics, innovation_pass = run_stage_validation(stage_name, start, end)
        results[stage_name] = {
            'metrics': metrics,
            '判定': {'innovation_verified': innovation_pass}
        }

    # Save
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(sep)
    print("SUMMARY")
    print(sep)
    passed = 0
    for stage, data in results.items():
        m = data['metrics']
        status = 'VERIFIED' if data['判定']['innovation_verified'] else 'NOT VERIFIED'
        print(f"{stage}: R2={m['R2']:.4f}, RMSE={m['RMSE']:.2f}, MB={m['MB']:.2f} -> {status}")
        if data['判定']['innovation_verified']:
            passed += 1

    print(f"\nTotal: {passed}/4 stages passed")
    print(f"Results saved: {OUTPUT_FILE}")

    return results


if __name__ == '__main__':
    main()