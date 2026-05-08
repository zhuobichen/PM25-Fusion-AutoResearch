import os
from shared.paths import get_project_root, data_path
# -*- coding: utf-8 -*-
"""
Explore different methods to determine optimal lambda for gVNA
"""

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

sys.path.insert(0, 'E:/CodeProject/ClaudeRoom/Data_Fusion_AutoResearch/CodeWorkSpace/新融合方法代码')

import numpy as np
import pandas as pd
import netCDF4 as nc
from datetime import datetime, timedelta
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
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


def ten_fold_gvna_single_day(selected_day, lambda_bg):
    monitor_df = pd.read_csv(MONITOR_FILE)
    fold_df = pd.read_csv(FOLD_FILE)

    day_df = monitor_df[monitor_df['Date'] == selected_day].copy()
    day_df = day_df.merge(fold_df, on=['Date', 'Site'], how='left')
    day_df = day_df.dropna(subset=['Lat', 'Lon', 'Conc'])

    if len(day_df) < 100:
        return None, None, None, None

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
        return None, None, None, None

    # Also compute CMAQ-only R2 on training fold (for Method 5)
    train_biases = train_df['Conc'].values - train_df['CMAQ'].values
    cmaq_r2_train = r2_score(train_df['Conc'].values, train_df['CMAQ'].values) if len(train_df) > 0 else np.nan

    return np.array(all_y_true), np.array(all_y_pred), cmaq_r2_train, np.mean(np.abs(train_biases))


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

    diffs = []
    for i in range(min(80, len(cmaq_vals))):
        for j in range(i+1, min(i+30, len(cmaq_vals))):
            diffs.append(abs(cmaq_vals[i] - cmaq_vals[j]))
    diffs = np.array(diffs)

    return {p: np.percentile(diffs, p) for p in [15, 20, 25, 30]}


def run_all_stages_all_methods():
    """Run all lambda determination methods across all stages"""
    print("=" * 80)
    print("gVNA Lambda Determination Methods Comparison")
    print("=" * 80)

    stages = [
        ('stage1 (Jan)', '2020-01-01', '2020-01-10', [3, 5, 6, 8, 10, 12, 15]),
        ('stage2 (Jul)', '2020-07-01', '2020-07-10', [3, 4, 5, 6, 8, 10, 12]),
        ('stage3 (Dec)', '2020-12-01', '2020-12-10', [5, 6, 8, 10, 12, 15, 18]),
    ]

    lambda_values_full = [3, 4, 5, 6, 8, 10, 12, 14, 15, 16, 18, 20, 25]

    for stage_name, start, end, zoom_lambdas in stages:
        print(f"\n{'='*80}")
        print(f"Stage: {stage_name}")
        print(f"{'='*80}")

        date_list = []
        current_date = datetime.strptime(start, '%Y-%m-%d')
        end_date_obj = datetime.strptime(end, '%Y-%m-%d')
        while current_date <= end_date_obj:
            date_list.append(current_date.strftime('%Y-%m-%d'))
            current_date += timedelta(days=1)

        # Collect per-day results
        day_results = {}
        for date_str in date_list:
            day_results[date_str] = {}
            for lam in lambda_values_full:
                y_true, y_pred, cmaq_r2, bias_mean = ten_fold_gvna_single_day(date_str, lam)
                if y_true is not None:
                    m = compute_metrics(y_true, y_pred)
                    day_results[date_str][lam] = m['R2']

            # Also get percentiles
            pcts = get_diff_percentiles(date_str)
            if pcts:
                day_results[date_str]['P15'] = pcts[15]
                day_results[date_str]['P20'] = pcts[20]
                day_results[date_str]['P25'] = pcts[25]
                day_results[date_str]['P30'] = pcts[30]

        # Compute per-day best lambda
        per_day_best = {}
        for date_str, results in day_results.items():
            lams = {lam: r for lam, r in results.items() if isinstance(lam, (int, float)) and lam in lambda_values_full}
            if lams:
                best_lam = max(lams, key=lambda x: lams[x])
                per_day_best[date_str] = {'best_lam': best_lam, 'best_r2': lams[best_lam]}

        print(f"  Processed {len(per_day_best)} valid days")

        # =========================================================================
        # METHOD 1: Global best (grid search on all data pooled)
        # =========================================================================
        pooled = {lam: [] for lam in lambda_values_full}
        for date_str, results in day_results.items():
            for lam in lambda_values_full:
                if lam in results:
                    pooled[lam].append(results[lam])
        pooled_avg = {lam: np.mean(v) for lam, v in pooled.items() if v}
        method1_best = max(pooled_avg, key=lambda x: pooled_avg[x])
        print(f"\n  Method 1 - Global Best:")
        print(f"    Optimal lambda = {method1_best:.0f}, R2 = {pooled_avg[method1_best]:.4f}")

        # =========================================================================
        # METHOD 2: Per-day optimal, then average
        # =========================================================================
        best_lams = [v['best_lam'] for v in per_day_best.values()]
        method2_median = np.median(best_lams)
        method2_mean = np.mean(best_lams)
        print(f"\n  Method 2 - Per-day optimal lambda:")
        print(f"    Mean   of per-day best = {method2_mean:.1f}")
        print(f"    Median of per-day best = {method2_median:.1f}")
        print(f"    Distribution: min={min(best_lams)}, max={max(best_lams)}, std={np.std(best_lams):.1f}")

        # =========================================================================
        # METHOD 3: Leave-some-days-out cross-validation for lambda
        # =========================================================================
        # Use first 5 days to find best lambda, test on remaining 5
        n_train = 5
        train_dates = date_list[:n_train]
        test_dates = date_list[n_train:]

        train_pooled = {lam: [] for lam in lambda_values_full}
        for date_str in train_dates:
            for lam in lambda_values_full:
                if lam in day_results.get(date_str, {}):
                    train_pooled[lam].append(day_results[date_str][lam])
        train_avg = {lam: np.mean(v) for lam, v in train_pooled.items() if v}
        method3_lam = max(train_avg, key=lambda x: train_avg[x])

        # Test on held-out days
        test_pooled = {lam: [] for lam in lambda_values_full}
        for date_str in test_dates:
            for lam in lambda_values_full:
                if lam in day_results.get(date_str, {}):
                    test_pooled[lam].append(day_results[date_str][lam])
        test_avg = {lam: np.mean(v) for lam, v in test_pooled.items() if v}

        print(f"\n  Method 3 - Leave-some-days-out CV:")
        print(f"    Train on {n_train} days: optimal lambda = {method3_lam:.0f}")
        if method3_lam in test_avg:
            print(f"    Test  on {len(test_dates)} days: R2 = {test_avg[method3_lam]:.4f}")
            # Compare with global best on test
            if method1_best in test_avg:
                print(f"    (Global best lambda={method1_best:.0f} on test: R2 = {test_avg[method1_best]:.4f})")

        # =========================================================================
        # METHOD 4: Percentile-based lambda
        # =========================================================================
        p15_vals = [day_results[d].get('P15') for d in date_list if day_results[d].get('P15')]
        p20_vals = [day_results[d].get('P20') for d in date_list if day_results[d].get('P20')]
        p25_vals = [day_results[d].get('P25') for d in date_list if day_results[d].get('P25')]

        method4_p15 = np.mean(p15_vals) if p15_vals else 0
        method4_p20 = np.mean(p20_vals) if p20_vals else 0
        method4_p25 = np.mean(p25_vals) if p25_vals else 0

        print(f"\n  Method 4 - Percentile-based lambda:")
        print(f"    lambda = P15 of |M-M_i|: {method4_p15:.1f}")
        print(f"    lambda = P20 of |M-M_i|: {method4_p20:.1f}")
        print(f"    lambda = P25 of |M-M_i|: {method4_p25:.1f}")

        # Evaluate these on pooled data
        for pct_lam, lam_val in [('P15', method4_p15), ('P20', method4_p20), ('P25', method4_p25)]:
            closest = min(lambda_values_full, key=lambda x: abs(x - lam_val))
            if closest in pooled_avg:
                print(f"    lambda={closest:.0f} ({pct_lam}): R2 = {pooled_avg[closest]:.4f}")

        # =========================================================================
        # SUMMARY: Compare all fixed lambda candidates
        # =========================================================================
        print(f"\n  Fixed lambda candidates (pooled R2):")
        print(f"    {'lambda':>6}  {'R2':>8}  {'vs_best':>8}")
        for lam in sorted(pooled_avg.keys()):
            diff = pooled_avg[lam] - pooled_avg[method1_best]
            marker = " <-- BEST" if lam == method1_best else ""
            print(f"    {lam:>6.0f}  {pooled_avg[lam]:>8.4f}  {diff:>+8.4f}{marker}")


if __name__ == '__main__':
    run_all_stages_all_methods()
