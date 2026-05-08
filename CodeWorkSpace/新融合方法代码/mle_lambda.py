import os
# -*- coding: utf-8 -*-
"""Fast MLE lambda: compute lambda ONCE per day, then 10-fold CV"""
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

sys.path.insert(0, 'E:/CodeProject/ClaudeRoom/Data_Fusion_AutoResearch/CodeWorkSpace/新融合方法代码')
import numpy as np
import pandas as pd
import netCDF4 as nc
from datetime import datetime, timedelta
from sklearn.metrics import r2_score, mean_squared_error
from CodeWorkSpace.新融合方法代码.gVNA import gVNA, adaptive_lambda

CMAQ_FILE = data_path('test_data/raw/CMAQ/2020_PM25.nc')
MONITOR_FILE = data_path('test_data/raw/Monitor/2020_DailyPM2.5Monitor.csv')
FOLD_FILE = data_path('test_data/fold_split_table_daily.csv')


def compute_metrics(y_true, y_pred):
    mask = ~(np.isnan(y_true) | np.isnan(y_pred) | np.isinf(y_true) | np.isinf(y_pred))
    y_true, y_pred = y_true[mask], y_pred[mask]
    return {'R2': float(r2_score(y_true, y_pred)),
            'RMSE': float(np.sqrt(mean_squared_error(y_true, y_pred)))}


def get_cmaq_at_site(lon, lat, lon_grid, lat_grid, pm25_grid):
    idx = np.argmin(np.sqrt((lon_grid - lon)**2 + (lat_grid - lat)**2))
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
    lon_cmaq, lat_cmaq = ds.variables['lon'][:], ds.variables['lat'][:]
    pred_pm25 = ds.variables['pred_PM25'][:]
    ds.close()
    day_idx = (datetime.strptime(selected_day, '%Y-%m-%d') - datetime(2020, 1, 1)).days
    day_df = day_df.copy()
    day_df['CMAQ'] = [get_cmaq_at_site(r['Lon'], r['Lat'], lon_cmaq, lat_cmaq, pred_pm25[day_idx])
                       for _, r in day_df.iterrows()]
    return day_df.dropna(subset=['CMAQ'])


# --- MLE lambda estimators (O(n), no pairwise) ---
def est_sigma(df):
    b = (df['Conc'] - df['CMAQ']).dropna()
    return np.std(b) if len(b) > 0 else 8.0


def est_mad(df):
    b = (df['Conc'] - df['CMAQ']).dropna()
    return np.median(np.abs(b - np.median(b))) * 1.4826 if len(b) > 5 else 8.0


def est_iqr(df):
    b = (df['Conc'] - df['CMAQ']).dropna()
    if len(b) < 5:
        return 8.0
    q75, q25 = np.percentile(b, [75, 25])
    return (q75 - q25) / 1.35


def est_adaptive(df):
    valid = ~np.isnan(df['CMAQ']) & ~np.isnan(df['Conc'])
    r2 = r2_score(df.loc[valid, 'Conc'], df.loc[valid, 'CMAQ']) if valid.sum() > 10 else np.nan
    return adaptive_lambda(r2)


# --- Run 10-fold CV for a given lambda ---
def run_cv(day_df, lam):
    all_t, all_p = [], []
    for fid in range(1, 11):
        tr = day_df[day_df['fold'] != fid].dropna(subset=['Lon', 'Lat', 'CMAQ', 'Conc'])
        te = day_df[day_df['fold'] == fid].dropna(subset=['Lon', 'Lat', 'CMAQ', 'Conc'])
        if len(te) == 0 or len(tr) == 0:
            continue
        m = gVNA(k=30, p=2, adaptive=False, lambda_bg=lam)
        m.fit(tr['Lon'].values, tr['Lat'].values, tr['Conc'].values, tr['CMAQ'].values)
        pred = m.predict(te[['Lon', 'Lat']].values, mod=te['CMAQ'].values)
        all_t.extend(te['Conc'].values)
        all_p.extend(pred)
    return compute_metrics(np.array(all_t), np.array(all_p)) if all_t else None


def main():
    stages = [('Jan', '2020-01-01', '2020-01-05'), ('Jul', '2020-07-01', '2020-07-05'),
              ('Dec', '2020-12-01', '2020-12-05')]

    estimators = [
        ('Fixed lam=8',    lambda df: 8.0),
        ('Fixed lam=15',   lambda df: 15.0),
        ('Adaptive CMAQ',  est_adaptive),
        ('MLE sigma',      est_sigma),
        ('MLE MAD',        est_mad),
        ('MLE IQR',        est_iqr),
    ]

    all_r2s = {n: [] for n, _ in estimators}
    all_lams = {n: [] for n, _ in estimators}

    for name, start, end in stages:
        print(f"\n=== {name} ===")
        dates = [(datetime.strptime(start, '%Y-%m-%d') + timedelta(days=i)).strftime('%Y-%m-%d')
                 for i in range((datetime.strptime(end, '%Y-%m-%d') - datetime.strptime(start, '%Y-%m-%d')).days + 1)]

        for di, d in enumerate(dates):
            df = load_day(d)
            if df is None:
                continue
            print(f"  Day {di+1}/{len(dates)}: {d}", end='')

            for ename, est in estimators:
                lam = float(np.clip(est(df), 1.0, 50.0))
                res = run_cv(df, lam)
                if res:
                    all_r2s[ename].append(res['R2'])
                    all_lams[ename].append(lam)
                    if di == 0:
                        print(f"  | lam={lam:.1f}", end='')
            print()

    # Summary
    print("\n" + "=" * 55)
    print(f"{'Method':<18}  {'R2':>8}  {'lambda':>8}")
    print("-" * 55)
    pooled = {n: np.mean(r2s) for n, r2s in all_r2s.items() if r2s}
    for n in sorted(pooled, key=pooled.get, reverse=True):
        diff = pooled[n] - max(pooled.values())
        star = " ***" if pooled[n] == max(pooled.values()) else ""
        print(f"  {n:<18}  {pooled[n]:>8.4f}  {np.mean(all_lams[n]):>8.1f}{star}")

    print("\n--- MLE formulas ---")
    print("  sigma:  lambda = std(biases)")
    print("  MAD:    lambda = MAD(biases) * 1.4826")
    print("  IQR:    lambda = IQR(biases) / 1.35")
    print("  (all derived from max likelihood under normal assumption)")


if __name__ == '__main__':
    main()
