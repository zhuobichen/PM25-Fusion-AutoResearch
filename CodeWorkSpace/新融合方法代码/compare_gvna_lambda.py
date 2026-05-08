import os
from shared.paths import get_project_root, data_path
# -*- coding: utf-8 -*-
"""
gVNA Auto std vs Fixed 15.0 多天对比测试
"""

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

sys.path.insert(0, 'E:/CodeProject/ClaudeRoom/Data_Fusion_AutoResearch/CodeWorkSpace/新融合方法代码')

import numpy as np
import pandas as pd
import netCDF4 as nc
from datetime import datetime, timedelta
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from CodeWorkSpace.新融合方法代码.gVNA import gVNA

ROOT_DIR = str(get_project_root())
CMAQ_FILE = data_path('test_data/raw/CMAQ/2020_PM25.nc')
MONITOR_FILE = data_path('test_data/raw/Monitor/2020_DailyPM2.5Monitor.csv')
FOLD_FILE = data_path('test_data/fold_split_table_daily.csv')


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


def ten_fold_gvna_single_day(selected_day, lambda_bg, lambda_method='median'):
    monitor_df = pd.read_csv(MONITOR_FILE)
    fold_df = pd.read_csv(FOLD_FILE)

    day_df = monitor_df[monitor_df['Date'] == selected_day].copy()
    day_df = day_df.merge(fold_df, on=['Date', 'Site'], how='left')
    day_df = day_df.dropna(subset=['Lat', 'Lon', 'Conc'])

    if len(day_df) < 100:
        return None, None

    ds = nc.Dataset(CMAQ_FILE, 'r')
    lon_cmaq = ds.variables['lon'][:]
    lat_cmaq = ds.variables['lat'][:]
    pred_pm25 = ds.variables['pred_PM25'][:]
    ds.close()

    date_obj = datetime.strptime(selected_day, '%Y-%m-%d')
    day_idx = (date_obj - datetime(2020, 1, 1)).days
    if day_idx >= pred_pm25.shape[0]:
        return None, None
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

        model = gVNA(k=30, p=2, lambda_bg=lambda_bg, lambda_method=lambda_method)
        model.fit(
            train_df['Lon'].values,
            train_df['Lat'].values,
            train_df['Conc'].values,
            train_df['CMAQ'].values
        )

        X_test = test_df[['Lon', 'Lat']].values
        mod_test = test_df['CMAQ'].values
        y_pred = model.predict(X_test, mod=mod_test)

        all_y_true.extend(test_df['Conc'].values)
        all_y_pred.extend(y_pred)

    if len(all_y_true) == 0:
        return None, None

    return np.array(all_y_true), np.array(all_y_pred)


def run_comparison(start_date, end_date, n_days=None):
    print("=" * 70)
    print("gVNA Auto std vs Fixed 15.0 对比测试")
    print("=" * 70)

    date_list = []
    current_date = datetime.strptime(start_date, '%Y-%m-%d')
    end_date_obj = datetime.strptime(end_date, '%Y-%m-%d')
    while current_date <= end_date_obj:
        date_list.append(current_date.strftime('%Y-%m-%d'))
        current_date += timedelta(days=1)

    if n_days is not None:
        date_list = date_list[:n_days]

    print(f"Test days: {len(date_list)} ({date_list[0]} ~ {date_list[-1]})")

    results_fixed = {'R2': [], 'RMSE': [], 'MB': [], 'lambda': []}
    results_auto = {'R2': [], 'RMSE': [], 'MB': [], 'lambda': []}

    for i, date_str in enumerate(date_list):
        if (i+1) % 5 == 0 or i == 0:
            print(f"  Day {i+1}/{len(date_list)}: {date_str}")

        # Fixed 15.0
        y_true, y_pred = ten_fold_gvna_single_day(date_str, lambda_bg=15.0)
        if y_true is not None:
            m = compute_metrics(y_true, y_pred)
            results_fixed['R2'].append(m['R2'])
            results_fixed['RMSE'].append(m['RMSE'])
            results_fixed['MB'].append(abs(m['MB']))
            results_fixed['lambda'].append(15.0)

        # Auto std
        y_true, y_pred = ten_fold_gvna_single_day(date_str, lambda_bg=None, lambda_method='std')
        if y_true is not None:
            m = compute_metrics(y_true, y_pred)
            results_auto['R2'].append(m['R2'])
            results_auto['RMSE'].append(m['RMSE'])
            results_auto['MB'].append(abs(m['MB']))

    # summary
    print("\n" + "=" * 70)
    print("Summary:")
    print("=" * 70)

    for name, res in [('Fixed 15.0', results_fixed), ('Auto std', results_auto)]:
        if res['R2']:
            print(f"\n{name}:")
            print(f"  lambda = {np.mean(res['lambda']):.2f}")
            print(f"  R2    = {np.mean(res['R2']):.4f} +/- {np.std(res['R2']):.4f}")
            print(f"  RMSE  = {np.mean(res['RMSE']):.2f} +/- {np.std(res['RMSE']):.2f}")
            print(f"  |MB|  = {np.mean(res['MB']):.2f} +/- {np.std(res['MB']):.2f}")

    # 对比
    if results_fixed['R2'] and results_auto['R2']:
        r2_diff = np.mean(results_fixed['R2']) - np.mean(results_auto['R2'])
        rmse_diff = np.mean(results_fixed['RMSE']) - np.mean(results_auto['RMSE'])
        print(f"\nDifference (Fixed - Auto std):")
        print(f"  delta_R2   = {r2_diff:+.4f}")
        print(f"  delta_RMSE = {rmse_diff:+.2f}")


if __name__ == '__main__':
    # 测试 stage1 (1月全月)
    run_comparison('2020-01-01', '2020-01-31')
