# -*- coding: utf-8 -*-
"""
GD-IDW 十折交叉验证
==================
验证 Gradient-Direction IDW 方法的效果
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
from joblib import Parallel, delayed

from CodeWorkSpace.新融合方法代码.GDIDW import GDIDW

ROOT_DIR = str(get_project_root())
CMAQ_FILE = data_path('test_data/raw/CMAQ/2020_PM25.nc')
MONITOR_FILE = data_path('test_data/raw/Monitor/2020_DailyPM2.5Monitor.csv')
FOLD_FILE = data_path('test_data/fold_split_table_daily.csv')
OUTPUT_DIR = f'{ROOT_DIR}/Innovation/success/GDIDW'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# VNA基准值（用于创新判定）
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


def ten_fold_gdidw(selected_day, beta=0.5):
    """GD-IDW十折交叉验证"""
    monitor_df = pd.read_csv(MONITOR_FILE)
    fold_df = pd.read_csv(FOLD_FILE)

    day_df = monitor_df[monitor_df['Date'] == selected_day].copy()
    day_df = day_df.merge(fold_df, on=['Date', 'Site'], how='left')
    day_df = day_df.dropna(subset=['Lat', 'Lon', 'Conc'])

    if len(day_df) < 100:
        return np.array([]), np.array([])

    ds = nc.Dataset(CMAQ_FILE, 'r')
    lon_cmaq = ds.variables['lon'][:]
    lat_cmaq = ds.variables['lat'][:]
    pred_pm25 = ds.variables['pred_PM25'][:]
    ds.close()

    date_obj = datetime.strptime(selected_day, '%Y-%m-%d')
    day_idx = (date_obj - datetime(2020, 1, 1)).days
    if day_idx >= pred_pm25.shape[0]:
        return np.array([]), np.array([])
    cmaq_day = pred_pm25[day_idx]

    # 获取站点CMAQ值
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

        # ===== GD-IDW =====
        model = GDIDW(k=30, power=-2, beta=beta, use_gradient=True)
        model.fit(
            train_df['Lon'].values,
            train_df['Lat'].values,
            train_df['Conc'].values,
            train_df['CMAQ'].values,
            lon_grid=lon_cmaq,
            lat_grid=lat_cmaq,
            pm25_grid=cmaq_day
        )

        # 预测 (使用GD方法: CMAQ + bias插值)
        X_test = test_df[['Lon', 'Lat']].values
        bias_pred = model.predict(X_test, method='gd')
        y_pred = test_df['CMAQ'].values + bias_pred

        all_y_true.extend(test_df['Conc'].values)
        all_y_pred.extend(y_pred)

    return np.array(all_y_true), np.array(all_y_pred)


def run_stage_validation(stage_name, start_date, end_date, beta=0.5):
    print(f"\n{'='*60}")
    print(f"GD-IDW Stage: {stage_name} ({start_date} ~ {end_date}), beta={beta}")
    print(f"{'='*60}")

    base = BASELINE[stage_name]
    print(f"VNA Baseline: R2={base['R2']:.4f}, RMSE={base['RMSE']:.2f}")

    date_list = []
    current_date = datetime.strptime(start_date, '%Y-%m-%d')
    end_date_obj = datetime.strptime(end_date, '%Y-%m-%d')
    while current_date <= end_date_obj:
        date_list.append(current_date.strftime('%Y-%m-%d'))
        current_date += timedelta(days=1)

    print(f"Days: {len(date_list)}")

    n_jobs = min(8, len(date_list))
    results = Parallel(n_jobs=n_jobs)(
        delayed(ten_fold_gdidw)(date_str, beta)
        for date_str in date_list
    )

    all_y_true = []
    all_y_pred = []
    day_count = 0
    for result in results:
        y_true, y_pred = result
        if len(y_true) > 0:
            all_y_true.extend(y_true)
            all_y_pred.extend(y_pred)
            day_count += 1

    print(f"Processed: {day_count} days")

    if len(all_y_true) == 0:
        return {'R2': np.nan, 'MAE': np.nan, 'RMSE': np.nan, 'MB': np.nan}

    metrics = compute_metrics(np.array(all_y_true), np.array(all_y_pred))

    print(f"GD-IDW Result: R2={metrics['R2']:.4f}, RMSE={metrics['RMSE']:.2f}, MB={metrics['MB']:.2f}")

    # 创新判定
    r2_pass = metrics['R2'] >= base['R2'] + 0.01
    rmse_pass = metrics['RMSE'] <= base['RMSE']
    mb_pass = abs(metrics['MB']) <= base['MB']

    print(f"Innovation: R2>={base['R2']+0.01:.4f}? {'OK' if r2_pass else 'FAIL'}  "
          f"RMSE<={base['RMSE']:.2f}? {'OK' if rmse_pass else 'FAIL'}  "
          f"|MB|<={base['MB']:.2f}? {'OK' if mb_pass else 'FAIL'}")

    return metrics


def main():
    print("="*70)
    print("GD-IDW Multi-Stage Validation")
    print("="*70)

    stages = {
        'pre_exp':  ('2020-01-01', '2020-01-05'),
        'stage1':   ('2020-01-01', '2020-01-31'),
        'stage2':   ('2020-07-01', '2020-07-31'),
        'stage3':   ('2020-12-01', '2020-12-31'),
    }

    # 测试不同beta值
    beta_values = [0.0, 0.3, 0.5, 1.0]

    for beta in beta_values:
        results = {}
        for stage_name, (start, end) in stages.items():
            metrics = run_stage_validation(stage_name, start, end, beta)
            results[stage_name] = metrics

        # 保存结果
        output_file = f'{OUTPUT_DIR}/GDIDW_beta{beta}_all_stages.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"\nBeta={beta} Results:")
        for stage, m in results.items():
            print(f"  {stage}: R2={m['R2']:.4f}, RMSE={m['RMSE']:.2f}")


if __name__ == '__main__':
    main()
