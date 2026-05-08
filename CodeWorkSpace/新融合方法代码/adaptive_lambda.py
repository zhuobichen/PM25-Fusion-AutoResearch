import os
# -*- coding: utf-8 -*-
"""
gVNA Adaptive Lambda: Use daily CMAQ R2 to determine lambda
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

CMAQ_FILE = data_path('test_data/raw/CMAQ/2020_PM25.nc')
MONITOR_FILE = data_path('test_data/raw/Monitor/2020_DailyPM2.5Monitor.csv')
FOLD_FILE = data_path('test_data/fold_split_table_daily.csv')
LAMBDA_VALUES = [3, 5, 8, 10, 12, 15]


def compute_metrics(y_true, y_pred):
    mask = ~(np.isnan(y_true) | np.isnan(y_pred) | np.isinf(y_true) | np.isinf(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if len(y_true) == 0:
        return {'R2': np.nan, 'RMSE': np.nan, 'MB': np.nan}
    return {
        'R2': float(r2_score(y_true, y_pred)),
        'RMSE': float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'MB': float(np.mean(y_pred - y_true))
    }


def get_cmaq_at_site(lon, lat, lon_grid, lat_grid, pm25_grid):
    dist = np.sqrt((lon_grid - lon)**2 + (lat_grid - lat)**2)
    idx = np.argmin(dist)
    ny, nx = lon_grid.shape
    return pm25_grid[idx // nx, idx % nx]


def get_daily_cmaq_r2(day_df):
    """Compute CMAQ R2 on the full day's data (using all sites as proxy)"""
    valid = ~np.isnan(day_df['CMAQ']) & ~np.isnan(day_df['Conc'])
    if valid.sum() < 10:
        return np.nan
    return r2_score(day_df.loc[valid, 'Conc'], day_df.loc[valid, 'CMAQ'])


def run_cv(day_df, lam):
    all_y_true, all_y_pred = [], []
    for fold_id in range(1, 11):
        train = day_df[day_df['fold'] != fold_id].dropna(subset=['Lon', 'Lat', 'CMAQ', 'Conc'])
        test = day_df[day_df['fold'] == fold_id].dropna(subset=['Lon', 'Lat', 'CMAQ', 'Conc'])
        if len(test) == 0 or len(train) == 0:
            continue
        model = gVNA(k=30, p=2, lambda_bg=lam)
        model.fit(train['Lon'].values, train['Lat'].values, train['Conc'].values, train['CMAQ'].values)
        y_pred = model.predict(test[['Lon', 'Lat']].values, mod=test['CMAQ'].values)
        all_y_true.extend(test['Conc'].values)
        all_y_pred.extend(y_pred)
    if not all_y_true:
        return np.nan
    return compute_metrics(np.array(all_y_true), np.array(all_y_pred))


def get_adaptive_lambda(cmaq_r2):
    """
    Determine lambda based on daily CMAQ R2
    Fitted from empirical data:
    - stage1: corr=0.832, best lambda increases with CMAQ R2
    - stage2: best lambda=3 always (CMAQ always bad)
    - stage3: corr=0.165, mild positive
    """
    if np.isnan(cmaq_r2):
        return 8  # default
    if cmaq_r2 < 0.1:
        return 3
    elif cmaq_r2 < 0.25:
        return 5
    elif cmaq_r2 < 0.40:
        return 8
    else:
        return 12


def main():
    stages = [
        ('stage1 (Jan)', '2020-01-01', '2020-01-10'),
        ('stage2 (Jul)', '2020-07-01', '2020-07-10'),
        ('stage3 (Dec)', '2020-12-01', '2020-12-10'),
    ]

    all_fixed = {lam: [] for lam in LAMBDA_VALUES}
    all_adaptive = []

    for stage_name, start, end in stages:
        print(f"\n{'='*65}")
        print(f"Stage: {stage_name}")
        print(f"{'='*65}")

        date_list = []
        cur = datetime.strptime(start, '%Y-%m-%d')
        while cur <= datetime.strptime(end, '%Y-%m-%d'):
            date_list.append(cur.strftime('%Y-%m-%d'))
            cur += timedelta(days=1)

        # Load data
        day_datas = []
        for d in date_list:
            monitor_df = pd.read_csv(MONITOR_FILE)
            fold_df = pd.read_csv(FOLD_FILE)
            day_df = monitor_df[monitor_df['Date'] == d].copy()
            day_df = day_df.merge(fold_df, on=['Date', 'Site'], how='left')
            day_df = day_df.dropna(subset=['Lat', 'Lon', 'Conc'])
            if len(day_df) < 100:
                continue
            ds = nc.Dataset(CMAQ_FILE, 'r')
            lon_cmaq = ds.variables['lon'][:]
            lat_cmaq = ds.variables['lat'][:]
            pred_pm25 = ds.variables['pred_PM25'][:]
            ds.close()
            date_obj = datetime.strptime(d, '%Y-%m-%d')
            day_idx = (date_obj - datetime(2020, 1, 1)).days
            cmaq_day = pred_pm25[day_idx]
            day_df = day_df.copy()
            day_df['CMAQ'] = [get_cmaq_at_site(row['Lon'], row['Lat'], lon_cmaq, lat_cmaq, cmaq_day)
                               for _, row in day_df.iterrows()]
            day_df = day_df.dropna(subset=['CMAQ'])
            day_datas.append(day_df)

        print(f"Valid days: {len(day_datas)}")

        # Per-day: compare adaptive vs all fixed
        day_fixed_r2s = {lam: [] for lam in LAMBDA_VALUES}
        day_adaptive_r2s = []
        day_info = []

        for i, day_df in enumerate(day_datas):
            cmaq_r2 = get_daily_cmaq_r2(day_df)
            adaptive_lam = get_adaptive_lambda(cmaq_r2)

            fixed_r2s = {}
            for lam in LAMBDA_VALUES:
                m = run_cv(day_df, lam)
                if m['R2'] is not None:
                    fixed_r2s[lam] = m['R2']
                    day_fixed_r2s[lam].append(m['R2'])

            # Adaptive: use the lambda determined by CMAQ R2
            m_adaptive = run_cv(day_df, adaptive_lam)
            if m_adaptive['R2'] is not None:
                day_adaptive_r2s.append(m_adaptive['R2'])

            # Best fixed lambda for this day
            best_fixed_lam = max(fixed_r2s, key=lambda x: fixed_r2s[x])
            best_fixed_r2 = fixed_r2s[best_fixed_lam]
            day_info.append({
                'day': i+1,
                'cmaq_r2': cmaq_r2,
                'adaptive_lam': adaptive_lam,
                'best_fixed_lam': best_fixed_lam,
                'adaptive_r2': m_adaptive['R2'],
                'best_fixed_r2': best_fixed_r2,
                'gap': best_fixed_r2 - m_adaptive['R2']
            })

            print(f"  Day {i+1:2d}: CMAQ_R2={cmaq_r2:>7.3f} | adaptive_lam={adaptive_lam:.0f}, R2={m_adaptive['R2']:.4f} "
                  f"| best_fixed={best_fixed_lam:.0f}, R2={best_fixed_r2:.4f} | gap={m_adaptive['R2']-best_fixed_r2:+.4f}")

        # Summary
        pooled_fixed = {lam: np.nanmean(day_fixed_r2s[lam]) for lam in LAMBDA_VALUES}
        pooled_adaptive = np.nanmean(day_adaptive_r2s)
        best_fixed_overall = max(pooled_fixed, key=lambda x: pooled_fixed[x])

        print(f"\n  --- Pooled Summary ---")
        print(f"  {'Method':<20}  {'lambda':>8}  {'R2':>8}")
        print(f"  {'-'*40}")
        for lam in sorted(pooled_fixed):
            print(f"  {'Fixed lam='+str(lam):<20}  {lam:>8.0f}  {pooled_fixed[lam]:>8.4f}")
        print(f"  {'Adaptive':<20}  {'auto':>8}  {pooled_adaptive:>8.4f}")
        print(f"  {'Best fixed':<20}  {best_fixed_overall:>8.0f}  {pooled_fixed[best_fixed_overall]:>8.4f}")

        # Gap analysis
        gaps = [d['gap'] for d in day_info]
        print(f"\n  Adaptive gap vs best fixed:")
        print(f"    Mean={np.mean(gaps):+.4f}, Std={np.std(gaps):.4f}, Max={np.max(gaps):+.4f}")

        # Per-day lambda distribution for adaptive
        adaptive_lams = [d['adaptive_lam'] for d in day_info]
        from collections import Counter
        print(f"    Adaptive lambda used: {dict(Counter(adaptive_lams))}")

        # Store for cross-stage summary
        for lam in LAMBDA_VALUES:
            all_fixed[lam].extend(day_fixed_r2s[lam])
        all_adaptive.extend(day_adaptive_r2s)

    # =========================================================================
    # CROSS-STAGE SUMMARY
    # =========================================================================
    print(f"\n{'='*65}")
    print("CROSS-STAGE SUMMARY")
    print(f"{'='*65}")

    pooled_all_fixed = {lam: np.nanmean(all_fixed[lam]) for lam in LAMBDA_VALUES}
    best_all = max(pooled_all_fixed, key=lambda x: pooled_all_fixed[x])
    pooled_adaptive_all = np.nanmean(all_adaptive)

    print(f"  {'Method':<20}  {'lambda':>8}  {'R2':>8}  {'vs_best':>8}")
    print(f"  {'-'*50}")
    for lam in sorted(pooled_all_fixed):
        diff = pooled_all_fixed[lam] - pooled_all_fixed[best_all]
        print(f"  {'Fixed lam='+str(lam):<20}  {lam:>8.0f}  {pooled_all_fixed[lam]:>8.4f}  {diff:>+8.4f}")
    diff_adaptive = pooled_adaptive_all - pooled_all_fixed[best_all]
    print(f"  {'Adaptive':<20}  {'auto':>8}  {pooled_adaptive_all:>8.4f}  {diff_adaptive:>+8.4f}")

    # Also try the seasonal scheme
    print(f"\n  --- Seasonal Lambda Scheme ---")
    seasonal_fixed = []
    for lam in LAMBDA_VALUES:
        seasonal_fixed.extend(all_fixed[lam])

    # lambda=8 is the "winter" choice
    # Try overall fixed=5 (the compromise)
    print(f"    lambda=5 (compromise): R2={np.nanmean(all_fixed[5]):.4f}")
    print(f"    lambda=8 (winter):      R2={np.nanmean(all_fixed[8]):.4f}")
    print(f"    lambda=3 (summer):     R2={np.nanmean(all_fixed[3]):.4f}")
    print(f"    Adaptive:              R2={pooled_adaptive_all:.4f}")


if __name__ == '__main__':
    main()
