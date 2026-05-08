# -*- coding: utf-8 -*-
"""
ARK_OLS 十折交叉验证 - 标准模式（方式2b）
==========================================
结合OLS线性校正和多尺度残差插值
训练：CMAQ网格坐标（不聚合）
预测：1折站点所在的CMAQ网格坐标
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
from joblib import Parallel, delayed

ROOT_DIR = str(get_project_root())
CMAQ_FILE = data_path('test_data/raw/CMAQ/2020_PM25.nc')
MONITOR_FILE = data_path('test_data/raw/Monitor/2020_DailyPM2.5Monitor.csv')
FOLD_FILE = data_path('test_data/fold_split_table_daily.csv')
OUTPUT_DIR = f'{ROOT_DIR}/Innovation/success/ARK_OLS'
os.makedirs(OUTPUT_DIR, exist_ok=True)

BASELINE = {
    'pre_exp': {'R2': 0.8941, 'RMSE': 16.42, 'MB': 0.76},
    'stage1':  {'R2': 0.9057, 'RMSE': 16.28, 'MB': 0.50},
    'stage2':  {'R2': 0.8458, 'RMSE': 4.97, 'MB': 0.04},
    'stage3':  {'R2': 0.9078, 'RMSE': 11.90, 'MB': 0.36},
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


def idw_predict(x_obs, values, x_pred, power=-2, k=None):
    """IDW插值"""
    if len(x_obs) == 0:
        return np.full(len(x_pred), np.nan)

    dist = np.sqrt((x_obs[:, 0][:, np.newaxis] - x_pred[:, 0])**2 +
                    (x_obs[:, 1][:, np.newaxis] - x_pred[:, 1])**2)

    if k is not None:
        nearest_k = np.argsort(dist, axis=0)[:k]  # shape: (k, n_pred)
        dist = dist[nearest_k, np.arange(dist.shape[1])]  # shape: (k, n_pred)
        weights = np.where(dist > 0, 1.0 / (dist ** power), 1.0)
        weights = weights / weights.sum(axis=0)  # shape: (k, n_pred)
        values_k = values[nearest_k]  # shape: (k, n_pred)
        return np.sum(weights * values_k, axis=0)  # shape: (n_pred,)
    else:
        dist = np.where(dist == 0, 1e-10, dist)
        weights = 1.0 / (dist ** power)
        weights = weights / weights.sum(axis=0)
        return np.sum(weights * values, axis=0)


def ten_fold_ark_ols(selected_day):
    """标准模式十折验证 - 方式2b"""
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

    cmaq_values = []
    for _, row in day_df.iterrows():
        val = get_cmaq_at_site(row['Lon'], row['Lat'], lon_cmaq, lat_cmaq, cmaq_day)
        cmaq_values.append(val)
    day_df['CMAQ'] = cmaq_values

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

        # OLS线性校正
        ols = LinearRegression()
        ols.fit(m_train.reshape(-1, 1), y_train)
        residual = y_train - ols.predict(m_train.reshape(-1, 1))

        # 方式2b：训练用CMAQ网格坐标（不聚合）
        train_coords = []
        for _, row in train_df.iterrows():
            cmaq_lon, cmaq_lat = get_cmaq_grid_coord(row['Lon'], row['Lat'], lon_cmaq, lat_cmaq)
            train_coords.append([cmaq_lon, cmaq_lat])
        X_train = np.array(train_coords)

        # 测试坐标
        test_coords = []
        for _, row in test_df.iterrows():
            cmaq_lon, cmaq_lat = get_cmaq_grid_coord(row['Lon'], row['Lat'], lon_cmaq, lat_cmaq)
            test_coords.append([cmaq_lon, cmaq_lat])
        X_test = np.array(test_coords)

        y_test = test_df['Conc'].values
        m_test = test_df['CMAQ'].values

        # 多尺度IDW插值残差
        k_values = [5, 10, 20]
        residual_preds = []
        for k in k_values:
            residual_pred = idw_predict(X_train, residual, X_test, power=-2, k=k)
            residual_preds.append(residual_pred)
        residual_interp = np.mean(residual_preds, axis=0)

        # 融合预测
        pred_ols_test = ols.predict(m_test.reshape(-1, 1))
        ark_pred = pred_ols_test + residual_interp

        all_y_true.extend(y_test)
        all_y_pred.extend(ark_pred)

    return np.array(all_y_true), np.array(all_y_pred)


def run_stage_validation(stage_name, start_date, end_date):
    sep = "=" * 70
    print(sep)
    print(f"ARK_OLS Stage: {stage_name} ({start_date} ~ {end_date})")
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
        delayed(ten_fold_ark_ols)(date_str)
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

    print(f"Result: R2={metrics['R2']:.4f}, RMSE={metrics['RMSE']:.2f}, MB={metrics['MB']:.2f}")
    print(f"Check: R2>{threshold_r2:.4f}? {'PASS' if r2_pass else 'FAIL'} | RMSE<={base['RMSE']}? {'PASS' if rmse_pass else 'FAIL'} | |MB|<={abs(base['MB'])}? {'PASS' if mb_pass else 'FAIL'}")
    print(f"Innovation: {'VERIFIED' if innovation_pass else 'NOT VERIFIED'}")

    return metrics, innovation_pass


def main():
    sep = "=" * 70
    print(sep)
    print("ARK_OLS All Stages - 标准模式（方式2b）")
    print(sep)

    stages = {
        'pre_exp':  ('2020-01-01', '2020-01-05'),
        'stage1':   ('2020-01-01', '2020-01-31'),
        'stage2':   ('2020-07-01', '2020-07-31'),
        'stage3':   ('2020-12-01', '2020-12-31'),
    }

    results = {}
    all_pass = True

    for stage_name, (start, end) in stages.items():
        metrics, innovation_pass = run_stage_validation(stage_name, start, end)
        results[stage_name] = {'metrics': metrics, '判定': {'innovation_verified': innovation_pass}}
        if not innovation_pass:
            all_pass = False

    output_file = f'{OUTPUT_DIR}/ARK_OLS_all_stages.json'
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(sep)
    print("SUMMARY")
    print(sep)
    for stage, data in results.items():
        m = data['metrics']
        status = 'VERIFIED' if data['判定']['innovation_verified'] else 'NOT VERIFIED'
        print(f"{stage}: R2={m['R2']:.4f}, RMSE={m['RMSE']:.2f}, MB={m['MB']:.2f} -> {status}")

    print(f"\nAll stages passed: {all_pass}")
    print(f"Results saved: {output_file}")

    return results


if __name__ == '__main__':
    main()