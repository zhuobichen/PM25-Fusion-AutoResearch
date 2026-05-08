import os
from shared.paths import get_project_root, data_path
# -*- coding: utf-8 -*-
"""
Analyze: how close is optimal lambda to P25 of |M-M_i| distribution
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


def get_diff_percentiles(selected_day):
    """Get percentiles of |M-M_i| from training data"""
    monitor_df = pd.read_csv(MONITOR_FILE)
    day_df = monitor_df[monitor_df['Date'] == selected_day].copy()
    day_df = day_df.dropna(subset=['Lat', 'Lon', 'Conc'])

    if len(day_df) < 100:
        return None

    ds = nc.Dataset(CMAQ_FILE, 'r')
    lon_cmaq = ds.variables['lon'][:]
    lat_cmaq = ds.variables['lat'][:]
    pred_pm25 = ds.variables['pred_PM25'][:]
    ds.close()

    date_obj = datetime.strptime(selected_day, '%Y-%m-%d')
    day_idx = (date_obj - datetime(2020, 1, 1)).days
    cmaq_day = pred_pm25[day_idx]

    cmaq_vals = []
    for _, row in day_df.iterrows():
        val = get_cmaq_at_site(row['Lon'], row['Lat'], lon_cmaq, lat_cmaq, cmaq_day)
        cmaq_vals.append(val)
    cmaq_vals = np.array(cmaq_vals)

    # Pairwise differences (sampled to avoid O(n^2))
    diffs = []
    for i in range(min(80, len(cmaq_vals))):
        for j in range(i+1, min(i+30, len(cmaq_vals))):
            diffs.append(abs(cmaq_vals[i] - cmaq_vals[j]))
    diffs = np.array(diffs)

    percentiles = {}
    for p in [10, 15, 20, 25, 30, 35, 40, 50]:
        percentiles[p] = np.percentile(diffs, p)

    return percentiles, diffs


def ten_fold_gvna_single_day(selected_day, lambda_bg):
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

        model = gVNA(k=30, p=2, lambda_bg=lambda_bg)
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


def percentile_of_value(diffs, value):
    """Find what percentile of diffs is below a given value"""
    return np.mean(diffs < value) * 100


def main():
    print("=" * 80)
    print("Lambda vs P25 Analysis: How close is optimal lambda to P25?")
    print("=" * 80)

    stages = [
        ('stage1 (Jan)', '2020-01-01', '2020-01-10', [5, 8, 10, 12, 14, 15, 16]),
        ('stage2 (Jul)', '2020-07-01', '2020-07-10', [3, 4, 5, 6, 8, 10, 12]),
        ('stage3 (Dec)', '2020-12-01', '2020-12-10', [5, 8, 10, 12, 14, 15, 16, 18]),
    ]

    lambda_values_full = [3, 4, 5, 6, 8, 10, 12, 14, 15, 16, 18, 20, 25]

    for stage_name, start, end, zoom_lambdas in stages:
        print(f"\n{'='*80}")
        print(f"Stage: {stage_name}")
        print(f"{'='*80}")

        # Get date list
        date_list = []
        current_date = datetime.strptime(start, '%Y-%m-%d')
        end_date_obj = datetime.strptime(end, '%Y-%m-%d')
        while current_date <= end_date_obj:
            date_list.append(current_date.strftime('%Y-%m-%d'))
            current_date += timedelta(days=1)

        # Compute P values (average across days)
        all_percentiles = {p: [] for p in [10, 15, 20, 25, 30, 35, 40, 50]}
        for date_str in date_list[:5]:  # use first 5 days for percentiles
            result = get_diff_percentiles(date_str)
            if result:
                pcts, _ = result
                for p, v in pcts.items():
                    all_percentiles[p].append(v)

        avg_percentiles = {p: np.mean(v) for p, v in all_percentiles.items() if v}

        # Compute R2 for all lambda values
        r2_by_lambda = {lam: [] for lam in lambda_values_full}
        for date_str in date_list:
            for lam in lambda_values_full:
                y_true, y_pred = ten_fold_gvna_single_day(date_str, lam)
                if y_true is not None:
                    m = compute_metrics(y_true, y_pred)
                    r2_by_lambda[lam].append(m['R2'])

        avg_r2 = {lam: np.mean(v) for lam, v in r2_by_lambda.items() if v}

        # Find best lambda
        best_lam = max(avg_r2, key=lambda x: avg_r2[x])
        best_r2 = avg_r2[best_lam]

        # Show P percentiles
        print(f"\n  Percentiles of |M-M_i| (average over 5 days):")
        for p in [10, 15, 20, 25, 30, 35, 40, 50]:
            print(f"    P{p:>2} = {avg_percentiles.get(p, 0):>6.1f}  ug/m3")

        print(f"\n  R2 by lambda (average over {len(date_list)} days):")
        for lam in sorted(avg_r2.keys()):
            marker = " <-- BEST" if lam == best_lam else ""
            print(f"    lambda={lam:>4.0f}: R2={avg_r2[lam]:>7.4f}{marker}")

        # What percentile does best_lambda correspond to?
        best_pct = None
        for p, val in sorted(avg_percentiles.items()):
            if abs(val - best_lam) < abs((avg_percentiles.get(best_pct, 9999) if best_pct else 9999) - best_lam):
                best_pct = p

        print(f"\n  --> Optimal lambda={best_lam:.0f} corresponds to P{best_pct:.0f}={avg_percentiles.get(best_pct, 0):.1f} ug/m3")
        print(f"  --> Best lambda / P25 ratio: {best_lam / avg_percentiles.get(25, 1):.2f}")

        # Also check P15, P20, P25 ratios
        print(f"\n  Ratio (lambda / Pn):")
        for p in [15, 20, 25, 30, 35]:
            if p in avg_percentiles:
                ratio = best_lam / avg_percentiles[p]
                pct_retained = np.exp(-1) * 100 if p == 25 else None
                print(f"    lambda/P{p}: {best_lam:.0f}/{avg_percentiles[p]:.1f} = {ratio:.2f}")

        # What percentile does each tested lambda fall at?
        print(f"\n  Percentile rank of each lambda in |M-M_i| distribution:")
        pct_rank = {}
        for date_str in date_list[:3]:
            result = get_diff_percentiles(date_str)
            if result:
                _, diffs = result
                for lam in sorted(r2_by_lambda.keys()):
                    pct = percentile_of_value(diffs, lam)
                    if lam not in pct_rank:
                        pct_rank[lam] = []
                    pct_rank[lam].append(pct)

        for lam in sorted(pct_rank.keys()):
            avg_pct = np.mean(pct_rank[lam])
            marker = " <-- BEST" if lam == best_lam else ""
            print(f"    lambda={lam:>4.0f}: P = {avg_pct:>5.1f}%{marker}")


if __name__ == '__main__':
    main()
