# -*- coding: utf-8 -*-
"""
gVNA 自适应λ 四阶段验证
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

sys.path.insert(0, 'E:/CodeProject/ClaudeRoom/Data_Fusion_AutoResearch/CodeWorkSpace/新融合方法代码')

from shared.paths import get_project_root, data_path
import json
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
OUTPUT_DIR = ROOT_DIR + '/Innovation/success/gVNA'
os.makedirs(OUTPUT_DIR, exist_ok=True)

BASELINE = {
    'pre_exp': {'R2': 0.8907, 'RMSE': 16.68, 'MB': 0.70},
    'stage1':  {'R2': 0.9034, 'RMSE': 16.48, 'MB': 0.50},
    'stage2':  {'R2': 0.8408, 'RMSE': 5.05, 'MB': 0.05},
    'stage3':  {'R2': 0.9031, 'RMSE': 12.20, 'MB': 0.42},
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


def ten_fold_gvna_adaptive_single_day(selected_day):
    """gVNA with adaptive lambda"""
    monitor_df = pd.read_csv(MONITOR_FILE)
    fold_df = pd.read_csv(FOLD_FILE)

    day_df = monitor_df[monitor_df['Date'] == selected_day].copy()
    day_df = day_df.merge(fold_df, on=['Date', 'Site'], how='left')
    day_df = day_df.dropna(subset=['Lat', 'Lon', 'Conc'])

    if len(day_df) < 100:
        return None, None, None

    ds = nc.Dataset(CMAQ_FILE, 'r')
    lon_cmaq = ds.variables['lon'][:]
    lat_cmaq = ds.variables['lat'][:]
    pred_pm25 = ds.variables['pred_PM25'][:]
    ds.close()

    date_obj = datetime.strptime(selected_day, '%Y-%m-%d')
    day_idx = (date_obj - datetime(2020, 1, 1)).days
    if day_idx >= pred_pm25.shape[0]:
        return None, None, None
    cmaq_day = pred_pm25[day_idx]

    cmaq_values = []
    for _, row in day_df.iterrows():
        val = get_cmaq_at_site(row['Lon'], row['Lat'], lon_cmaq, lat_cmaq, cmaq_day)
        cmaq_values.append(val)
    day_df['CMAQ'] = cmaq_values

    # Compute CMAQ R2 on training fold (for adaptive lambda)
    # We compute it on each fold's training set
    all_y_true = []
    all_y_pred = []
    all_lam_used = []

    for fold_id in range(1, 11):
        train_df = day_df[day_df['fold'] != fold_id].copy()
        test_df = day_df[day_df['fold'] == fold_id].copy()

        train_df = train_df.dropna(subset=['Lon', 'Lat', 'CMAQ', 'Conc'])
        test_df = test_df.dropna(subset=['Lon', 'Lat', 'CMAQ', 'Conc'])

        if len(test_df) == 0 or len(train_df) == 0:
            continue

        # Compute CMAQ R2 on training set for adaptive lambda
        valid = ~np.isnan(train_df['CMAQ']) & ~np.isnan(train_df['Conc'])
        if valid.sum() > 10:
            cmaq_r2 = r2_score(train_df.loc[valid, 'Conc'], train_df.loc[valid, 'CMAQ'])
        else:
            cmaq_r2 = np.nan

        # Use adaptive lambda (default)
        model = gVNA(k=30, p=2, adaptive=True)
        model.fit(
            train_df['Lon'].values,
            train_df['Lat'].values,
            train_df['Conc'].values,
            train_df['CMAQ'].values,
            cmaq_r2=cmaq_r2
        )

        X_test = test_df[['Lon', 'Lat']].values
        mod_test = test_df['CMAQ'].values
        y_pred = model.predict(X_test, mod=mod_test)

        all_y_true.extend(test_df['Conc'].values)
        all_y_pred.extend(y_pred)
        all_lam_used.append(model.lambda_bg_estimated)

    if len(all_y_true) == 0:
        return None, None, None

    return np.array(all_y_true), np.array(all_y_pred), all_lam_used


def run_stage(stage_name, start_date, end_date):
    sep = "=" * 60
    print(sep)
    print("gVNA Adaptive Stage: " + stage_name + " (" + start_date + " ~ " + end_date + ")")
    print(sep)

    base = BASELINE[stage_name]
    print("VNA Baseline: R2=" + str(base['R2']) + ", RMSE=" + str(base['RMSE']))

    date_list = []
    current_date = datetime.strptime(start_date, '%Y-%m-%d')
    end_date_obj = datetime.strptime(end_date, '%Y-%m-%d')
    while current_date <= end_date_obj:
        date_list.append(current_date.strftime('%Y-%m-%d'))
        current_date += timedelta(days=1)

    print("Days: " + str(len(date_list)))

    all_y_true = []
    all_y_pred = []
    all_lam_by_day = []
    day_count = 0

    for i, date_str in enumerate(date_list):
        if (i+1) % 10 == 0:
            print("  Processing day " + str(i+1) + "/" + str(len(date_list)) + "...")
        result = ten_fold_gvna_adaptive_single_day(date_str)
        if result[0] is not None:
            y_true, y_pred, lam_used = result
            all_y_true.extend(y_true)
            all_y_pred.extend(y_pred)
            all_lam_by_day.extend(lam_used)
            day_count += 1

    print("Processed: " + str(day_count) + " days")

    if len(all_y_true) == 0:
        return {'R2': np.nan, 'MAE': np.nan, 'RMSE': np.nan, 'MB': np.nan}

    metrics = compute_metrics(np.array(all_y_true), np.array(all_y_pred))

    print("gVNA Adaptive Result: R2=" + str(metrics['R2']) + ", RMSE=" + str(metrics['RMSE']) + ", MB=" + str(metrics['MB']))
    print("Lambda distribution: mean=" + str(np.mean(all_lam_by_day)) +
          ", median=" + str(np.median(all_lam_by_day)) +
          ", used values: " + str(sorted(set(all_lam_by_day))))

    r2_pass = metrics['R2'] >= base['R2'] + 0.01
    rmse_pass = metrics['RMSE'] <= base['RMSE']
    mb_pass = abs(metrics['MB']) <= base['MB']

    r2_str = "OK" if r2_pass else "FAIL"
    rmse_str = "OK" if rmse_pass else "FAIL"
    mb_str = "OK" if mb_pass else "FAIL"

    print("Innovation: R2>=" + str(base['R2']+0.01) + "? " + r2_str +
          "  RMSE<=" + str(base['RMSE']) + "? " + rmse_str +
          "  |MB|<=" + str(base['MB']) + "? " + mb_str)

    return metrics


def main():
    print("=" * 70)
    print("gVNA Adaptive Lambda Multi-Stage Validation")
    print("=" * 70)

    stages = {
        'pre_exp':  ('2020-01-01', '2020-01-05'),
        'stage1':   ('2020-01-01', '2020-01-31'),
        'stage2':   ('2020-07-01', '2020-07-31'),
        'stage3':   ('2020-12-01', '2020-12-31'),
    }

    results = {}
    for stage_name, (start, end) in stages.items():
        metrics = run_stage(stage_name, start, end)
        results[stage_name] = metrics

    output_file = OUTPUT_DIR + '/gVNA_adaptive_all_stages.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("=" * 70)
    print("Summary:")
    print("=" * 70)
    for stage, m in results.items():
        print("  " + stage + ": R2=" + str(m['R2']) + ", RMSE=" + str(m['RMSE']) + ", MB=" + str(m['MB']))


if __name__ == '__main__':
    main()
