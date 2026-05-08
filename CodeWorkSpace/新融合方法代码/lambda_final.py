import os
# -*- coding: utf-8 -*-
"""Quick lambda analysis - 3 days, 7 lambdas per stage"""
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
LAMBDA_VALUES = [3, 5, 8, 10, 12, 15, 20]

def compute_metrics(y_true, y_pred):
    mask = ~(np.isnan(y_true) | np.isnan(y_pred) | np.isinf(y_true) | np.isinf(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if len(y_true) == 0:
        return {'R2': np.nan}
    return {'R2': float(r2_score(y_true, y_pred))}

def get_cmaq_at_site(lon, lat, lon_grid, lat_grid, pm25_grid):
    dist = np.sqrt((lon_grid - lon)**2 + (lat_grid - lat)**2)
    idx = np.argmin(dist)
    ny, nx = lon_grid.shape
    return pm25_grid[idx // nx, idx % nx]

def load_day(selected_day):
    monitor_df = pd.read_csv(MONITOR_FILE)
    fold_df = pd.read_csv(FOLD_FILE)
    day_df = monitor_df[monitor_df['Date'] == selected_day].copy()
    day_df = day_df.merge(fold_df, on=['Date', 'Site'], how='left')
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
    cmaq_values = [get_cmaq_at_site(row['Lon'], row['Lat'], lon_cmaq, lat_cmaq, cmaq_day)
                   for _, row in day_df.iterrows()]
    day_df = day_df.copy()
    day_df['CMAQ'] = cmaq_values
    # percentiles
    cmaq_vals = day_df['CMAQ'].dropna().values
    diffs = []
    for i in range(min(80, len(cmaq_vals))):
        for j in range(i+1, min(i+30, len(cmaq_vals))):
            diffs.append(abs(cmaq_vals[i] - cmaq_vals[j]))
    pcts = {p: np.percentile(diffs, p) for p in [15, 20, 25]}
    # CMAQ R2
    valid = ~np.isnan(day_df['CMAQ']) & ~np.isnan(day_df['Conc'])
    cmaq_r2 = r2_score(day_df.loc[valid, 'Conc'], day_df.loc[valid, 'CMAQ']) if valid.sum() > 10 else np.nan
    return day_df, pcts, cmaq_r2

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
    return compute_metrics(np.array(all_y_true), np.array(all_y_pred))['R2']

def main():
    stages = [
        ('stage1 (Jan)', '2020-01-01', '2020-01-10'),
        ('stage2 (Jul)', '2020-07-01', '2020-07-10'),
        ('stage3 (Dec)', '2020-12-01', '2020-12-10'),
    ]

    for stage_name, start, end in stages:
        date_list = []
        cur = datetime.strptime(start, '%Y-%m-%d')
        while cur <= datetime.strptime(end, '%Y-%m-%d'):
            date_list.append(cur.strftime('%Y-%m-%d'))
            cur += timedelta(days=1)

        print(f"\n{'='*60}")
        print(f"Stage: {stage_name}")
        print(f"{'='*60}")

        day_datas = []
        for d in date_list:
            data = load_day(d)
            if data:
                day_datas.append(data)

        print(f"Valid days: {len(day_datas)}")

        # Per-day results
        results = {lam: [] for lam in LAMBDA_VALUES}
        per_day_best = []
        per_day_pcts = [d[1] for d in day_datas]
        per_day_cmaq = [d[2] for d in day_datas]

        for i, (day_df, pcts, cmaq_r2) in enumerate(day_datas):
            day_r2s = {}
            for lam in LAMBDA_VALUES:
                r2 = run_cv(day_df, lam)
                results[lam].append(r2)
                day_r2s[lam] = r2
            best_lam = max(day_r2s, key=lambda x: day_r2s[x])
            per_day_best.append(best_lam)
            print(f"  Day {i+1}: best_lambda={best_lam:.0f}, CMAQ_R2={cmaq_r2:.3f}, P25={pcts[25]:.1f}")

        # Pooled
        pooled = {lam: np.nanmean(results[lam]) for lam in LAMBDA_VALUES}
        best_global = max(pooled, key=lambda x: pooled[x])

        print(f"\n  Pooled R2:")
        print(f"  {'lam':>5}  {'R2':>8}  {'vs_best':>8}")
        for lam in sorted(pooled):
            diff = pooled[lam] - pooled[best_global]
            m = " <-- BEST" if lam == best_global else ""
            print(f"  {lam:>5.0f}  {pooled[lam]:>8.4f}  {diff:>+8.4f}{m}")

        print(f"\n  Per-day best lambda: {sorted({l: per_day_best.count(l) for l in per_day_best}.items())}")
        print(f"  Mean={np.mean(per_day_best):.1f}, Median={np.median(per_day_best):.1f}")

        avg_p15 = np.mean([p[15] for p in per_day_pcts])
        avg_p20 = np.mean([p[20] for p in per_day_pcts])
        avg_p25 = np.mean([p[25] for p in per_day_pcts])
        print(f"  P15={avg_p15:.1f}, P20={avg_p20:.1f}, P25={avg_p25:.1f}")

        valid = [(per_day_cmaq[i], per_day_best[i]) for i in range(len(per_day_best)) if not np.isnan(per_day_cmaq[i])]
        if len(valid) >= 3:
            corr = np.corrcoef([x[0] for x in valid], [x[1] for x in valid])[0, 1]
            print(f"  Corr(best_lambda, CMAQ_R2): {corr:.3f}")

if __name__ == '__main__':
    main()
