import os
from shared.paths import get_project_root, data_path
# -*- coding: utf-8 -*-
"""
Efficient gVNA lambda analysis - data loaded once per day
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

LAMBDA_VALUES = [3, 4, 5, 6, 8, 10, 12, 14, 15, 16, 18, 20, 25]


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


class DayDataCache:
    """Pre-load all data for a day, then run all lambda tests on it"""
    def __init__(self, selected_day):
        self.day = selected_day
        monitor_df = pd.read_csv(MONITOR_FILE)
        fold_df = pd.read_csv(FOLD_FILE)

        self.day_df = monitor_df[monitor_df['Date'] == selected_day].copy()
        self.day_df = self.day_df.merge(fold_df, on=['Date', 'Site'], how='left')
        self.day_df = self.day_df.dropna(subset=['Lat', 'Lon', 'Conc'])

        if len(self.day_df) < 100:
            self.valid = False
            return
        self.valid = True

        ds = nc.Dataset(CMAQ_FILE, 'r')
        lon_cmaq = ds.variables['lon'][:]
        lat_cmaq = ds.variables['lat'][:]
        pred_pm25 = ds.variables['pred_PM25'][:]
        ds.close()

        date_obj = datetime.strptime(selected_day, '%Y-%m-%d')
        day_idx = (date_obj - datetime(2020, 1, 1)).days
        cmaq_day = pred_pm25[day_idx]

        cmaq_values = []
        for _, row in self.day_df.iterrows():
            val = get_cmaq_at_site(row['Lon'], row['Lat'], lon_cmaq, lat_cmaq, cmaq_day)
            cmaq_values.append(val)
        self.day_df = self.day_df.copy()
        self.day_df['CMAQ'] = cmaq_values

        # Compute |M_i - M_j| percentiles from ALL sites (not per fold)
        self.cmaq_vals = self.day_df['CMAQ'].dropna().values
        diffs = []
        for i in range(min(100, len(self.cmaq_vals))):
            for j in range(i+1, min(i+30, len(self.cmaq_vals))):
                diffs.append(abs(self.cmaq_vals[i] - self.cmaq_vals[j]))
        diffs = np.array(diffs)
        self.pcts = {p: np.percentile(diffs, p) for p in [10, 15, 20, 25, 30]}

        # CMAQ-only R2 on all data
        valid_mask = ~np.isnan(self.day_df['CMAQ'].values) & ~np.isnan(self.day_df['Conc'].values)
        if valid_mask.sum() > 10:
            self.cmaq_r2 = r2_score(self.day_df.loc[valid_mask, 'Conc'].values,
                                    self.day_df.loc[valid_mask, 'CMAQ'].values)
        else:
            self.cmaq_r2 = np.nan

    def test_all_lambdas(self):
        """Run 10-fold CV for ALL lambda values on pre-loaded data"""
        results = {}

        for fold_id in range(1, 11):
            train_df = self.day_df[self.day_df['fold'] != fold_id].copy()
            test_df = self.day_df[self.day_df['fold'] == fold_id].copy()

            train_df = train_df.dropna(subset=['Lon', 'Lat', 'CMAQ', 'Conc'])
            test_df = test_df.dropna(subset=['Lon', 'Lat', 'CMAQ', 'Conc'])

            if len(test_df) == 0 or len(train_df) == 0:
                continue

            X_test = test_df[['Lon', 'Lat']].values
            mod_test = test_df['CMAQ'].values
            y_true = test_df['Conc'].values

            for lam in LAMBDA_VALUES:
                if lam not in results:
                    results[lam] = {'y_true': [], 'y_pred': []}

                model = gVNA(k=30, p=2, lambda_bg=lam)
                model.fit(
                    train_df['Lon'].values,
                    train_df['Lat'].values,
                    train_df['Conc'].values,
                    train_df['CMAQ'].values
                )
                y_pred = model.predict(X_test, mod=mod_test)
                results[lam]['y_true'].extend(y_true)
                results[lam]['y_pred'].extend(y_pred)

        metrics = {}
        for lam in LAMBDA_VALUES:
            if results[lam]['y_true']:
                y_t = np.array(results[lam]['y_true'])
                y_p = np.array(results[lam]['y_pred'])
                metrics[lam] = compute_metrics(y_t, y_p)

        return metrics


def run_stage(stage_name, start, end, n_days=10):
    print(f"\n{'='*80}")
    print(f"Stage: {stage_name}")
    print(f"{'='*80}")

    date_list = []
    current_date = datetime.strptime(start, '%Y-%m-%d')
    end_date_obj = datetime.strptime(end, '%Y-%m-%d')
    while current_date <= end_date_obj:
        date_list.append(current_date.strftime('%Y-%m-%d'))
        current_date += timedelta(days=1)
    date_list = date_list[:n_days]
    print(f"Days: {len(date_list)}")

    # Load all days once
    caches = []
    for date_str in date_list:
        cache = DayDataCache(date_str)
        if cache.valid:
            caches.append(cache)
            print(f"  Loaded: {date_str}")
    print(f"Valid days: {len(caches)}")

    # Test all lambdas on all days
    all_lam_metrics = {lam: {'R2': [], 'RMSE': [], 'MB': []} for lam in LAMBDA_VALUES}
    per_day_best = {}
    per_day_pcts = []
    per_day_cmaq_r2 = []

    for i, cache in enumerate(caches):
        if (i+1) % 5 == 0 or i == 0:
            print(f"  Processing day {i+1}/{len(caches)}: {cache.day}")
        metrics = cache.test_all_lambdas()

        day_best_lam = None
        day_best_r2 = -999
        for lam, m in metrics.items():
            all_lam_metrics[lam]['R2'].append(m['R2'])
            all_lam_metrics[lam]['RMSE'].append(m['RMSE'])
            all_lam_metrics[lam]['MB'].append(abs(m['MB']))
            if m['R2'] > day_best_r2:
                day_best_r2 = m['R2']
                day_best_lam = lam

        per_day_best[cache.day] = day_best_lam
        per_day_pcts.append(cache.pcts)
        per_day_cmaq_r2.append(cache.cmaq_r2)

    # =========================================================================
    # RESULTS
    # =========================================================================
    pooled = {lam: np.mean(v['R2']) for lam, v in all_lam_metrics.items() if v['R2']}
    best_global = max(pooled, key=lambda x: pooled[x])

    print(f"\n  --- Pooled R2 by lambda ---")
    print(f"  {'lambda':>6}  {'R2':>8}  {'RMSE':>8}  {'|MB|':>8}  {'vs_best':>8}")
    for lam in sorted(pooled.keys()):
        diff = pooled[lam] - pooled[best_global]
        marker = " <-- BEST" if lam == best_global else ""
        print(f"  {lam:>6.0f}  {pooled[lam]:>8.4f}  {np.mean(all_lam_metrics[lam]['RMSE']):>8.2f}  "
              f"{np.mean(all_lam_metrics[lam]['MB']):>8.2f}  {diff:>+8.4f}{marker}")

    # Per-day best lambda distribution
    best_lams = list(per_day_best.values())
    print(f"\n  --- Per-day best lambda ---")
    print(f"  Mean={np.mean(best_lams):.1f}, Median={np.median(best_lams):.1f}, "
          f"Std={np.std(best_lams):.1f}, Min={min(best_lams)}, Max={max(best_lams)}")
    from collections import Counter
    cnt = Counter(best_lams)
    print(f"  Distribution: {dict(sorted(cnt.items()))}")

    # =========================================================================
    # METHOD COMPARISONS
    # =========================================================================
    print(f"\n  --- Method Comparisons ---")

    # Method 1: Global best
    print(f"  M1 Global best:       lambda={best_global:.0f}, R2={pooled[best_global]:.4f}")

    # Method 2: Per-day median
    method2_lam = np.median(best_lams)
    closest2 = min(LAMBDA_VALUES, key=lambda x: abs(x - method2_lam))
    print(f"  M2 Per-day median:    lambda={method2_lam:.1f} -> {closest2:.0f}, R2={pooled.get(closest2, pooled[best_global]):.4f}")

    # Method 3: P15-based
    avg_p15 = np.mean([p[15] for p in per_day_pcts])
    closest3 = min(LAMBDA_VALUES, key=lambda x: abs(x - avg_p15))
    print(f"  M3 lambda=P15:        P15={avg_p15:.1f} -> {closest3:.0f}, R2={pooled.get(closest3, pooled[best_global]):.4f}")

    # Method 4: P20-based
    avg_p20 = np.mean([p[20] for p in per_day_pcts])
    closest4 = min(LAMBDA_VALUES, key=lambda x: abs(x - avg_p20))
    print(f"  M4 lambda=P20:         P20={avg_p20:.1f} -> {closest4:.0f}, R2={pooled.get(closest4, pooled[best_global]):.4f}")

    # Method 5: P25-based
    avg_p25 = np.mean([p[25] for p in per_day_pcts])
    closest5 = min(LAMBDA_VALUES, key=lambda x: abs(x - avg_p25))
    print(f"  M5 lambda=P25:         P25={avg_p25:.1f} -> {closest5:.0f}, R2={pooled.get(closest5, pooled[best_global]):.4f}")

    # Method 6: Leave-some-days-out (first 5 train, last 5 test)
    n_train = 5
    train_pooled = {lam: [] for lam in LAMBDA_VALUES}
    for cache in caches[:n_train]:
        metrics = cache.test_all_lambdas()
        for lam, m in metrics.items():
            train_pooled[lam].append(m['R2'])
    train_avg = {lam: np.mean(v) for lam, v in train_pooled.items() if v}
    method6_lam = max(train_avg, key=lambda x: train_avg[x])

    test_pooled = {lam: [] for lam in LAMBDA_VALUES}
    for cache in caches[n_train:]:
        metrics = cache.test_all_lambdas()
        for lam, m in metrics.items():
            test_pooled[lam].append(m['R2'])
    test_avg = {lam: np.mean(v) for lam, v in test_pooled.items() if v}
    test_r2_method6 = test_avg.get(method6_lam, pooled[best_global])
    test_r2_global = test_avg.get(best_global, pooled[best_global])
    print(f"  M6 Leave-5-out CV:    lambda={method6_lam:.0f}, test_R2={test_r2_method6:.4f}")
    print(f"    (vs Global best on test: lambda={best_global:.0f}, test_R2={test_r2_global:.4f})")

    # Method 7: Median of P15-P25
    avg_p15_25 = np.mean([np.mean([p[15], p[20], p[25]]) for p in per_day_pcts])
    closest7 = min(LAMBDA_VALUES, key=lambda x: abs(x - avg_p15_25))
    print(f"  M7 lambda=P15-25_avg: avg={avg_p15_25:.1f} -> {closest7:.0f}, R2={pooled.get(closest7, pooled[best_global]):.4f}")

    # Method 8: Adaptive based on CMAQ R2
    avg_cmaq_r2 = np.mean([x for x in per_day_cmaq_r2 if not np.isnan(x)])
    print(f"  M8 CMAQ R2:           avg={avg_cmaq_r2:.3f}")

    # Correlation: best_lambda vs CMAQ R2
    valid_cmaq = [(per_day_cmaq_r2[i], best_lams[i]) for i in range(len(best_lams))
                  if not np.isnan(per_day_cmaq_r2[i])]
    if len(valid_cmaq) >= 5:
        corr = np.corrcoef([x[0] for x in valid_cmaq], [x[1] for x in valid_cmaq])[0, 1]
        print(f"    Corr(best_lambda, CMAQ_R2): {corr:.3f}")
        if abs(corr) > 0.3:
            print(f"    -> {'Negative' if corr < 0 else 'Positive'} correlation (|r| > 0.3)")

    return pooled, best_global, per_day_best


def main():
    print("=" * 80)
    print("gVNA Lambda Determination Methods - Efficiency Optimized")
    print("=" * 80)

    stages = [
        ('stage1 (Jan)', '2020-01-01', '2020-01-10'),
        ('stage2 (Jul)', '2020-07-01', '2020-07-10'),
        ('stage3 (Dec)', '2020-12-01', '2020-12-10'),
    ]

    all_results = {}
    for stage_name, start, end in stages:
        pooled, best_global, per_day_best = run_stage(stage_name, start, end, n_days=3)
        all_results[stage_name] = {
            'pooled': pooled,
            'best_global': best_global,
            'per_day_best': per_day_best
        }

    # Cross-stage summary
    print(f"\n{'='*80}")
    print("CROSS-STAGE SUMMARY")
    print(f"{'='*80}")
    print(f"  {'Stage':<20}  {'Best_lam':>8}  {'R2_best':>8}  {'R2(lam=10)':>10}  {'R2(lam=15)':>10}")
    for stage, res in all_results.items():
        p = res['pooled']
        bg = res['best_global']
        print(f"  {stage:<20}  {bg:>8.0f}  {p[bg]:>8.4f}  {p.get(10, np.nan):>10.4f}  {p.get(15, np.nan):>10.4f}")

    best_global_overall = max(all_results.items(), key=lambda x: x[1]['pooled'][x[1]['best_global']])
    print(f"\n  Overall best: {best_global_overall[0]}, lambda={best_global_overall[1]['best_global']:.0f}")


if __name__ == '__main__':
    main()
