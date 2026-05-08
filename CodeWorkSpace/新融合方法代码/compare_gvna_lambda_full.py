import os
from shared.paths import get_project_root, data_path
# -*- coding: utf-8 -*-
"""
gVNA lambda sensitivity test across wide range
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


def run_sensitivity_test(start_date, end_date, lambda_values, stage_name=""):
    print("=" * 70)
    print("gVNA Lambda Sensitivity Test" + (" - " + stage_name if stage_name else ""))
    print("=" * 70)

    date_list = []
    current_date = datetime.strptime(start_date, '%Y-%m-%d')
    end_date_obj = datetime.strptime(end_date, '%Y-%m-%d')
    while current_date <= end_date_obj:
        date_list.append(current_date.strftime('%Y-%m-%d'))
        current_date += timedelta(days=1)

    print(f"Test days: {len(date_list)} ({date_list[0]} ~ {date_list[-1]})")
    print(f"Lambda values: {lambda_values}")
    print()

    results = {lam: {'R2': [], 'RMSE': [], 'MB': [], 'lambda_used': []} for lam in lambda_values}

    for i, date_str in enumerate(date_list):
        if (i+1) % 5 == 0 or i == 0:
            print(f"  Day {i+1}/{len(date_list)}: {date_str}")

        for lam in lambda_values:
            y_true, y_pred = ten_fold_gvna_single_day(date_str, lambda_bg=lam)
            if y_true is not None:
                m = compute_metrics(y_true, y_pred)
                results[lam]['R2'].append(m['R2'])
                results[lam]['RMSE'].append(m['RMSE'])
                results[lam]['MB'].append(abs(m['MB']))
                results[lam]['lambda_used'].append(lam)

    # Summary
    print("\n" + "=" * 70)
    print("Summary (mean +/- std over days):")
    print("=" * 70)
    print(f"{'lambda':>8}  {'R2':>8}  {'RMSE':>8}  {'|MB|':>8}  {'R2_std':>8}")
    print("-" * 50)

    summary = []
    for lam in lambda_values:
        r2_list = results[lam]['R2']
        if r2_list:
            r2_mean = np.mean(r2_list)
            r2_std = np.std(r2_list)
            rmse_mean = np.mean(results[lam]['RMSE'])
            mb_mean = np.mean(results[lam]['MB'])
            print(f"{lam:>8.1f}  {r2_mean:>8.4f}  {rmse_mean:>8.2f}  {mb_mean:>8.2f}  {r2_std:>8.4f}")
            summary.append((lam, r2_mean, rmse_mean, mb_mean, r2_std))

    # Best
    if summary:
        best = max(summary, key=lambda x: x[1])
        print(f"\nBest lambda = {best[0]:.1f} with R2 = {best[1]:.4f}")

        # Physical interpretation: show distribution of |M-M_i| in training data
        print("\n" + "=" * 70)
        print("Physical Interpretation (CMAQ diff distribution on day 1)" + (" - " + stage_name if stage_name else "") + ":")
        print("=" * 70)
        day_df = pd.read_csv(MONITOR_FILE)
        day_df = day_df[day_df['Date'] == date_list[0]]
        day_df = day_df.dropna(subset=['Lat', 'Lon', 'Conc'])
        if len(day_df) >= 100:
            ds = nc.Dataset(CMAQ_FILE, 'r')
            lon_cmaq = ds.variables['lon'][:]
            lat_cmaq = ds.variables['lat'][:]
            pred_pm25 = ds.variables['pred_PM25'][:]
            ds.close()
            date_obj = datetime.strptime(date_list[0], '%Y-%m-%d')
            day_idx = (date_obj - datetime(2020, 1, 1)).days
            cmaq_day = pred_pm25[day_idx]
            cmaq_vals = []
            for _, row in day_df.iterrows():
                val = get_cmaq_at_site(row['Lon'], row['Lat'], lon_cmaq, lat_cmaq, cmaq_day)
                cmaq_vals.append(val)
            cmaq_vals = np.array(cmaq_vals)

            # Pairwise differences (sampled)
            diffs = []
            for i in range(min(50, len(cmaq_vals))):
                for j in range(i+1, min(i+20, len(cmaq_vals))):
                    diffs.append(abs(cmaq_vals[i] - cmaq_vals[j]))
            diffs = np.array(diffs)

            print(f"  Min diff:  {np.min(diffs):.1f} ug/m3")
            print(f"  Median:    {np.median(diffs):.1f} ug/m3")
            print(f"  Mean:      {np.mean(diffs):.1f} ug/m3")
            print(f"  Std:       {np.std(diffs):.1f} ug/m3")
            print(f"  P25:       {np.percentile(diffs, 25):.1f} ug/m3")
            print(f"  P75:       {np.percentile(diffs, 75):.1f} ug/m3")
            print(f"  P90:       {np.percentile(diffs, 90):.1f} ug/m3")
            print(f"  Max diff:  {np.max(diffs):.1f} ug/m3")
            print()
            print(f"  When |M-M_i| = lambda:")
            for lam in [10, 12, 15, 18, 20, 25]:
                weight_at_lambda = np.exp(-1.0)  # at lambda, weight = e^-1
                weight_at_half = np.exp(-0.5)
                weight_at_quarter = np.exp(-0.25)
                print(f"    lambda={lam}: exp(-1) = {weight_at_lambda:.3f} (37% of max weight)")
                if lam == 15:
                    print(f"             75% of diffs < {np.percentile(diffs, 75):.1f} -> weight > 0.37")


if __name__ == '__main__':
    lambda_values = [5, 8, 10, 12, 14, 15, 16, 18, 20, 25, 30, 40, 50]

    stages = [
        ('stage1 (Jan)', '2020-01-01', '2020-01-10'),
        ('stage2 (Jul)', '2020-07-01', '2020-07-10'),
        ('stage3 (Dec)', '2020-12-01', '2020-12-10'),
    ]

    for name, start, end in stages:
        print()
        run_sensitivity_test(start, end, lambda_values, stage_name=name)
        print()

    # Overall summary across all stages
    print("=" * 70)
    print("Cross-Stage Comparison: Best lambda per stage")
    print("=" * 70)
