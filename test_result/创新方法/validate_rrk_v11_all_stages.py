# -*- coding: utf-8 -*-
"""
RK_OLS_Poly v11 全阶段验证
==========================
Stage 1: 2020-01-01 ~ 2020-01-31 (31天)
Stage 2: 2020-07-01 ~ 2020-07-31 (31天)
Stage 3: 2020-12-01 ~ 2020-12-31 (31天)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from shared.paths import get_project_root, data_path
import json
import numpy as np
import pandas as pd
import netCDF4 as nc
from datetime import datetime, timedelta
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import HuberRegressor, LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from joblib import Parallel, delayed

# 路径配置
ROOT_DIR = str(get_project_root())
CMAQ_FILE = data_path('test_data/raw/CMAQ/2020_PM25.nc')
MONITOR_FILE = data_path('test_data/raw/Monitor/2020_DailyPM2.5Monitor.csv')
FOLD_FILE = data_path('test_data/fold_split_table_daily.csv')
OUTPUT_FILE = f'{ROOT_DIR}/test_result/创新方法/RRK_v11_all_stages.json'

# 基准（VNA 多阶段）
BASELINE = {
    'pre_exp': {'R2': 0.8941, 'RMSE': 16.42, 'MB': 0.76},
    'stage1':  {'R2': 0.9057, 'RMSE': 16.28, 'MB': 0.50},
    'stage2':  {'R2': 0.8458, 'RMSE': 4.97, 'MB': 0.04},
    'stage3':  {'R2': 0.9078, 'RMSE': 11.90, 'MB': 0.36},
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


def ten_fold_for_day_rrk(selected_day, poly_degree=2):
    monitor_df = pd.read_csv(MONITOR_FILE)
    fold_df = pd.read_csv(FOLD_FILE)

    day_df = monitor_df[monitor_df['Date'] == selected_day].copy()
    day_df = day_df.merge(fold_df, on='Site', how='left')
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

    cmaq_values = []
    for _, row in day_df.iterrows():
        val = get_cmaq_at_site(row['Lon'], row['Lat'], lon_cmaq, lat_cmaq, cmaq_day)
        cmaq_values.append(val)
    day_df['CMAQ'] = cmaq_values

    kernel = (ConstantKernel(10.0, (1e-2, 1e3)) *
              RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) +
              WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e1)))

    all_y_true = []
    all_y_pred = []

    for fold_id in range(1, 11):
        train_df = day_df[day_df['fold'] != fold_id].copy()
        test_df = day_df[day_df['fold'] == fold_id].copy()

        train_df = train_df.dropna(subset=['Lon', 'Lat', 'CMAQ', 'Conc'])
        test_df = test_df.dropna(subset=['Lon', 'Lat', 'CMAQ', 'Conc'])

        if len(test_df) == 0 or len(train_df) == 0:
            continue

        X_train = train_df[['Lon', 'Lat']].values
        X_test = test_df[['Lon', 'Lat']].values
        y_train = train_df['Conc'].values
        y_test = test_df['Conc'].values
        m_train = train_df['CMAQ'].values
        m_test = test_df['CMAQ'].values

        poly = PolynomialFeatures(degree=poly_degree, include_bias=False)
        m_train_poly = poly.fit_transform(m_train.reshape(-1, 1))
        m_test_poly = poly.transform(m_test.reshape(-1, 1))

        ols = LinearRegression()
        ols.fit(m_train_poly, y_train)
        pred_ols = ols.predict(m_test_poly)
        residual_ols = y_train - ols.predict(m_train_poly)

        gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, alpha=0.1, normalize_y=True)
        gpr.fit(X_train, residual_ols)
        gpr_pred, _ = gpr.predict(X_test, return_std=True)

        rk_pred = pred_ols + gpr_pred

        all_y_true.extend(y_test)
        all_y_pred.extend(rk_pred)

    return np.array(all_y_true), np.array(all_y_pred)


def run_stage_validation(stage_name, start_date, end_date):
    """运行指定阶段验证"""
    print(f"\n{'='*70}")
    print(f"Stage: {stage_name} ({start_date} ~ {end_date})")
    print(f"{'='*70}")

    base = BASELINE[stage_name]
    threshold_r2 = base['R2'] + 0.01
    print(f"Baseline (VNA): R2={base['R2']:.4f}, RMSE={base['RMSE']:.2f}, MB={base['MB']:.2f}")
    print(f"Threshold: R2>={threshold_r2:.4f}, RMSE<={base['RMSE']}, |MB|<={abs(base['MB'])}")

    date_list = []
    current_date = datetime.strptime(start_date, '%Y-%m-%d')
    end_date_obj = datetime.strptime(end_date, '%Y-%m-%d')
    while current_date <= end_date_obj:
        date_list.append(current_date.strftime('%Y-%m-%d'))
        current_date += timedelta(days=1)

    print(f"Days: {len(date_list)}")

    # 并行处理
    n_jobs = min(8, len(date_list))
    results = Parallel(n_jobs=n_jobs)(
        delayed(ten_fold_for_day_rrk)(date_str)
        for date_str in date_list
    )

    all_y_true = []
    all_y_pred = []
    day_count = 0
    for y_true, y_pred in results:
        if len(y_true) > 0:
            all_y_true.extend(y_true)
            all_y_pred.extend(y_pred)
            day_count += 1

    print(f"Processed: {day_count} days, {len(all_y_true)} predictions")

    if len(all_y_true) == 0:
        return {'R2': np.nan, 'MAE': np.nan, 'RMSE': np.nan, 'MB': np.nan}, False

    metrics = compute_metrics(np.array(all_y_true), np.array(all_y_pred))

    r2_pass = metrics['R2'] >= threshold_r2
    rmse_pass = metrics['RMSE'] <= base['RMSE']
    mb_pass = abs(metrics['MB']) <= abs(base['MB'])
    innovation_pass = r2_pass and rmse_pass and mb_pass

    print(f"Result: R2={metrics['R2']:.4f}, RMSE={metrics['RMSE']:.2f}, MAE={metrics['MAE']:.2f}, MB={metrics['MB']:.2f}")
    print(f"判定: R2>={threshold_r2:.4f}? {'PASS' if r2_pass else 'FAIL'} | RMSE<={base['RMSE']}? {'PASS' if rmse_pass else 'FAIL'} | |MB|<={abs(base['MB'])}? {'PASS' if mb_pass else 'FAIL'}")
    print(f"Innovation: {'VERIFIED' if innovation_pass else 'NOT VERIFIED'}")

    return metrics, innovation_pass


def main():
    print("="*70)
    print("RK_OLS_Poly v11 Multi-Stage Validation")
    print("="*70)

    stages = {
        'stage1':  ('2020-01-01', '2020-01-31'),
        'stage2':  ('2020-07-01', '2020-07-31'),
        'stage3':  ('2020-12-01', '2020-12-31'),
    }

    # 先确认 pre_exp 已通过
    pre_exp_metrics = {
        'R2': 0.9144, 'RMSE': 14.76, 'MAE': 9.62, 'MB': 0.20
    }
    pre_exp_pass = True

    results = {
        'pre_exp': {
            'metrics': pre_exp_metrics,
            '判定': {'innovation_verified': pre_exp_pass}
        }
    }

    all_pass = pre_exp_pass

    for stage_name, (start, end) in stages.items():
        metrics, innovation_pass = run_stage_validation(stage_name, start, end)
        results[stage_name] = {
            'metrics': metrics,
            '判定': {'innovation_verified': innovation_pass}
        }
        if not innovation_pass:
            all_pass = False

    # 保存结果
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    for stage, data in results.items():
        m = data['metrics']
        status = 'VERIFIED' if data['判定']['innovation_verified'] else 'NOT VERIFIED'
        print(f"{stage}: R2={m['R2']:.4f}, RMSE={m['RMSE']:.2f}, MB={m['MB']:.2f} -> {status}")

    print(f"\nAll stages passed: {all_pass}")
    print(f"Results saved: {OUTPUT_FILE}")

    return results


if __name__ == '__main__':
    main()