# -*- coding: utf-8 -*-
"""
SVCD vs SVCD-RAMP 对比测试 — pre_exp 阶段 (2020-01-01 ~ 2020-01-05)
"""
import sys, os, json
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _PROJECT_ROOT)

from shared.paths import get_project_root, data_path
import numpy as np
import pandas as pd
import netCDF4 as nc
from datetime import datetime, timedelta
from joblib import Parallel, delayed
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

ROOT_DIR = str(get_project_root())
CMAQ_FILE = data_path('test_data/raw/CMAQ/2020_PM25.nc')
MONITOR_FILE = data_path('test_data/raw/Monitor/2020_DailyPM2.5Monitor.csv')
FOLD_FILE = data_path('test_data/fold_split_table_daily.csv')

_code_path = os.path.join(ROOT_DIR, 'CodeWorkSpace', '新融合方法代码')
sys.path.insert(0, _code_path)
from SVCD import SVCD
from SVCD_RAMP import SVCD_RAMP


def compute_metrics(y_true, y_pred):
    mask = ~(np.isnan(y_true) | np.isnan(y_pred) | np.isinf(y_true) | np.isinf(y_pred))
    yt, yp = y_true[mask], y_pred[mask]
    if len(yt) == 0:
        return {'R2': np.nan, 'MAE': np.nan, 'RMSE': np.nan, 'MB': np.nan}
    return {
        'R2': float(r2_score(yt, yp)),
        'MAE': float(mean_absolute_error(yt, yp)),
        'RMSE': float(np.sqrt(mean_squared_error(yt, yp))),
        'MB': float(np.mean(yp - yt))
    }


def get_cmaq_at_site(lon, lat, lon_grid, lat_grid, pm25_grid):
    dist = np.sqrt((lon_grid - lon)**2 + (lat_grid - lat)**2)
    idx = np.argmin(dist)
    ny, nx = lon_grid.shape
    row, col = idx // nx, idx % nx
    return pm25_grid[row, col]


def get_cmaq_grid_coord(lon, lat, lon_grid, lat_grid):
    dist = np.sqrt((lon_grid - lon)**2 + (lat_grid - lat)**2)
    idx = np.argmin(dist)
    ny, nx = lon_grid.shape
    row, col = idx // nx, idx % nx
    return lon_grid[row, col], lat_grid[row, col]


def ten_fold_compare(selected_day):
    """单日十折: 同时跑 SVCD 和 SVCD-RAMP, 收集所有预测值"""
    monitor_df = pd.read_csv(MONITOR_FILE)
    fold_df = pd.read_csv(FOLD_FILE)

    day_df = monitor_df[monitor_df['Date'] == selected_day].copy()
    day_df = day_df.merge(fold_df, on=['Date', 'Site'], how='left')
    day_df = day_df.dropna(subset=['Lat', 'Lon', 'Conc'])
    if len(day_df) < 50:
        return None

    ds = nc.Dataset(CMAQ_FILE, 'r')
    lon_cmaq = ds.variables['lon'][:]
    lat_cmaq = ds.variables['lat'][:]
    pred_pm25 = ds.variables['pred_PM25'][:]
    ds.close()

    date_obj = datetime.strptime(selected_day, '%Y-%m-%d')
    day_idx = (date_obj - datetime(2020, 1, 1)).days
    if day_idx >= pred_pm25.shape[0]:
        return None
    cmaq_day = pred_pm25[day_idx]

    cmaq_values, cmaq_coords = [], []
    for _, row in day_df.iterrows():
        val = get_cmaq_at_site(row['Lon'], row['Lat'], lon_cmaq, lat_cmaq, cmaq_day)
        clon, clat = get_cmaq_grid_coord(row['Lon'], row['Lat'], lon_cmaq, lat_cmaq)
        cmaq_values.append(val)
        cmaq_coords.append([clon, clat])
    day_df['CMAQ'] = cmaq_values
    day_df['CMAQ_Lon'] = [c[0] for c in cmaq_coords]
    day_df['CMAQ_Lat'] = [c[1] for c in cmaq_coords]

    svcd_true, svcd_pred, svcd_resid = [], [], []
    ramp_true, ramp_pred, ramp_resid = [], [], []

    for fold_id in range(1, 11):
        train = day_df[day_df['fold'] != fold_id].dropna(subset=['Lon', 'Lat', 'CMAQ', 'Conc'])
        test = day_df[day_df['fold'] == fold_id].dropna(subset=['Lon', 'Lat', 'CMAQ', 'Conc'])
        if len(test) == 0 or len(train) < 20:
            continue

        try:
            Xt = train[['CMAQ_Lon', 'CMAQ_Lat']].values
            yt = train['Conc'].values
            mt = train['CMAQ'].values
            Xe = test[['CMAQ_Lon', 'CMAQ_Lat']].values
            ye = test['Conc'].values
            me = test['CMAQ'].values

            # SVCD 原生
            m_svcd = SVCD()
            m_svcd.fit(Xt, yt, mt)
            p_svcd = m_svcd.predict(Xe, me)
            # 训练集残差（诊断用）
            p_svcd_train = m_svcd.predict(Xt, mt)
            r_svcd_train = yt - p_svcd_train

            svcd_true.extend(ye)
            svcd_pred.extend(p_svcd)
            svcd_resid.extend(ye - p_svcd)

            # SVCD-RAMP
            m_ramp = SVCD_RAMP(ramp_kwargs={'n_bins': 10, 'n_spatial': 8, 'alpha': 0.5})
            m_ramp.fit(Xt, yt, mt)
            p_ramp = m_ramp.predict(Xe, me)

            ramp_true.extend(ye)
            ramp_pred.extend(p_ramp)
            ramp_resid.extend(ye - p_ramp)

        except Exception as e:
            continue

    svcd_true = np.array(svcd_true)
    svcd_pred = np.array(svcd_pred)
    ramp_true = np.array(ramp_true)
    ramp_pred = np.array(ramp_pred)

    return {
        'day': selected_day,
        'n_sites': len(day_df),
        'SVCD': {
            'metrics': compute_metrics(svcd_true, svcd_pred),
            'true': svcd_true, 'pred': svcd_pred,
        },
        'SVCD_RAMP': {
            'metrics': compute_metrics(ramp_true, ramp_pred),
            'true': ramp_true, 'pred': ramp_pred,
        },
    }


if __name__ == '__main__':
    print("=" * 70)
    print("SVCD vs SVCD-RAMP 十折交叉验证 (pre_exp: 2020-01-01 ~ 05)")
    print("=" * 70)

    date_list = ['2020-01-01', '2020-01-02', '2020-01-03', '2020-01-04', '2020-01-05']
    results = Parallel(n_jobs=4, verbose=1)(delayed(ten_fold_compare)(d) for d in date_list)
    results = [r for r in results if r is not None]

    # 每日对比
    print(f"\n{'Day':<12} {'SVCD R²':>8} {'SVCD RMSE':>10} {'SVCD MB':>8}  |  {'RAMP R²':>8} {'RAMP RMSE':>10} {'RAMP MB':>8}  |  ΔR²")
    print("-" * 75)

    for r in results:
        s = r['SVCD']['metrics']
        m = r['SVCD_RAMP']['metrics']
        dr2 = m['R2'] - s['R2']
        sign = '+' if dr2 >= 0 else ''
        print(f"{r['day']:<12} {s['R2']:>8.4f} {s['RMSE']:>10.2f} {s['MB']:>8.2f}  |  "
              f"{m['R2']:>8.4f} {m['RMSE']:>10.2f} {m['MB']:>8.2f}  |  {sign}{dr2:.4f}")

    # 汇总
    all_svcd_true = np.concatenate([r['SVCD']['true'] for r in results])
    all_svcd_pred = np.concatenate([r['SVCD']['pred'] for r in results])
    all_ramp_true = np.concatenate([r['SVCD_RAMP']['true'] for r in results])
    all_ramp_pred = np.concatenate([r['SVCD_RAMP']['pred'] for r in results])

    svcd_total = compute_metrics(all_svcd_true, all_svcd_pred)
    ramp_total = compute_metrics(all_ramp_true, all_ramp_pred)

    print("\n" + "=" * 70)
    print("5天汇总")
    print(f"SVCD:      R²={svcd_total['R2']:.4f}, RMSE={svcd_total['RMSE']:.2f}, "
          f"MAE={svcd_total['MAE']:.2f}, MB={svcd_total['MB']:.2f}")
    print(f"SVCD-RAMP: R²={ramp_total['R2']:.4f}, RMSE={ramp_total['RMSE']:.2f}, "
          f"MAE={ramp_total['MAE']:.2f}, MB={ramp_total['MB']:.2f}")
    dr2 = ramp_total['R2'] - svcd_total['R2']
    print(f"ΔR² = {dr2:+.4f}  {'✅ 改善' if dr2 > 0 else '❌ 无改善或退化'}")
    print("=" * 70)
