import os
# -*- coding: utf-8 -*-
"""Quick compare: Adaptive CMAQ vs Fixed lam=8 (31 days per stage)"""
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

sys.path.insert(0, 'E:/CodeProject/ClaudeRoom/Data_Fusion_AutoResearch/CodeWorkSpace/新融合方法代码')
import numpy as np
import pandas as pd
import netCDF4 as nc
from datetime import datetime, timedelta
from sklearn.metrics import r2_score
from CodeWorkSpace.新融合方法代码.gVNA import gVNA, adaptive_lambda

CMAQ_FILE = data_path('test_data/raw/CMAQ/2020_PM25.nc')
MONITOR_FILE = data_path('test_data/raw/Monitor/2020_DailyPM2.5Monitor.csv')
FOLD_FILE = data_path('test_data/fold_split_table_daily.csv')


def load_day(d):
    monitor_df = pd.read_csv(MONITOR_FILE)
    fold_df = pd.read_csv(FOLD_FILE)
    day_df = monitor_df[monitor_df['Date'] == d].copy()
    day_df = day_df.merge(fold_df, on=['Date', 'Site'], how='left')
    day_df = day_df.dropna(subset=['Lat', 'Lon', 'Conc'])
    if len(day_df) < 100:
        return None
    ds = nc.Dataset(CMAQ_FILE, 'r')
    lon_cmaq, lat_cmaq = ds.variables['lon'][:], ds.variables['lat'][:]
    pred_pm25 = ds.variables['pred_PM25'][:]
    ds.close()
    day_idx = (datetime.strptime(d, '%Y-%m-%d') - datetime(2020, 1, 1)).days
    if day_idx >= pred_pm25.shape[0]:
        return None
    day_df = day_df.copy()
    day_df['CMAQ'] = [np.nan] * len(day_df)
    for i, (_, r) in enumerate(day_df.iterrows()):
        idx = np.argmin(np.sqrt((lon_cmaq - r['Lon'])**2 + (lat_cmaq - r['Lat'])**2))
        ny, nx = lon_cmaq.shape
        day_df.iloc[i, day_df.columns.get_loc('CMAQ')] = pred_pm25[day_idx, idx // nx, idx % nx]
    return day_df.dropna(subset=['CMAQ'])


def run_cv(day_df, lam, cmaq_r2=None):
    all_t, all_p = [], []
    for fid in range(1, 11):
        tr = day_df[day_df['fold'] != fid].dropna(subset=['Lon', 'Lat', 'CMAQ', 'Conc'])
        te = day_df[day_df['fold'] == fid].dropna(subset=['Lon', 'Lat', 'CMAQ', 'Conc'])
        if len(te) == 0 or len(tr) == 0:
            continue
        m = gVNA(k=30, p=2, adaptive=False, lambda_bg=lam)
        m.fit(tr['Lon'].values, tr['Lat'].values, tr['Conc'].values, tr['CMAQ'].values, cmaq_r2=cmaq_r2)
        pred = m.predict(te[['Lon', 'Lat']].values, mod=te['CMAQ'].values)
        all_t.extend(te['Conc'].values)
        all_p.extend(pred)
    if not all_t:
        return np.nan
    mask = ~(np.isnan(all_t) | np.isnan(all_p))
    return r2_score(np.array(all_t)[mask], np.array(all_p)[mask])


print("=" * 60, flush=True)
print("Adaptive CMAQ vs Fixed lam=8 (31 days x 3 stages)", flush=True)
print("=" * 60, flush=True)

stages = [('Jan', '2020-01-01', '2020-01-31'),
          ('Jul', '2020-07-01', '2020-07-31'),
          ('Dec', '2020-12-01', '2020-12-31')]

for name, start, end in stages:
    dates = [(datetime.strptime(start, '%Y-%m-%d') + timedelta(days=i)).strftime('%Y-%m-%d')
             for i in range((datetime.strptime(end, '%Y-%m-%d') - datetime.strptime(start, '%Y-%m-%d')).days + 1)]
    r2_fixed, r2_adapt = [], []
    for di, d in enumerate(dates):
        df = load_day(d)
        if df is None:
            continue
        if (di+1) % 10 == 0:
            print(f"  {name} day {di+1}/{len(dates)}", flush=True)
        # Fixed lam=8
        r2_f = run_cv(df, 8.0)
        r2_fixed.append(r2_f)
        # Adaptive
        valid = ~np.isnan(df['CMAQ']) & ~np.isnan(df['Conc'])
        r2_cmaq = r2_score(df.loc[valid, 'Conc'], df.loc[valid, 'CMAQ']) if valid.sum() > 10 else np.nan
        lam_a = adaptive_lambda(r2_cmaq)
        r2_a = run_cv(df, lam_a)
        r2_adapt.append(r2_a)
    print(f"\n{name}: Fixed lam=8 R2={np.mean(r2_fixed):.4f} | Adaptive R2={np.mean(r2_adapt):.4f} | "
          f"Δ={np.mean(r2_adapt)-np.mean(r2_fixed):+.4f}", flush=True)

# Overall
all_f, all_a = [], []
for name, start, end in stages:
    dates = [(datetime.strptime(start, '%Y-%m-%d') + timedelta(days=i)).strftime('%Y-%m-%d')
             for i in range((datetime.strptime(end, '%Y-%m-%d') - datetime.strptime(start, '%Y-%m-%d')).days + 1)]
    for di, d in enumerate(dates):
        df = load_day(d)
        if df is None:
            continue
        r2_f = run_cv(df, 8.0)
        all_f.append(r2_f)
        valid = ~np.isnan(df['CMAQ']) & ~np.isnan(df['Conc'])
        r2_cmaq = r2_score(df.loc[valid, 'Conc'], df.loc[valid, 'CMAQ']) if valid.sum() > 10 else np.nan
        all_a.append(run_cv(df, adaptive_lambda(r2_cmaq)))
print(f"\nOverall: Fixed lam=8 R2={np.nanmean(all_f):.4f} | Adaptive R2={np.nanmean(all_a):.4f} | "
      f"Δ={np.nanmean(all_a)-np.nanmean(all_f):+.4f}", flush=True)
