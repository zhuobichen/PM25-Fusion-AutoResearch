# -*- coding: utf-8 -*-
"""
RK_OLS_Poly v11 aligned validation
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
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from joblib import Parallel, delayed

ROOT_DIR = str(get_project_root())
CMAQ_FILE = data_path('test_data/raw/CMAQ/2020_PM25.nc')
MONITOR_FILE = data_path('test_data/raw/Monitor/2020_DailyPM2.5Monitor.csv')
FOLD_FILE = data_path('test_data/fold_split_table_daily.csv')
OUTPUT_FILE = f'{ROOT_DIR}/test_result/创新方法/RRK_v11_preexp_aligned.json'

BASELINE = {'VNA': {'R2': 0.8941, 'RMSE': 16.42, 'MB': 0.76}}
THRESHOLD_R2 = BASELINE['VNA']['R2'] + 0.01


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


def get_cmaq_grid_coord(lon, lat, lon_grid, lat_grid):
    dist = np.sqrt((lon_grid - lon)**2 + (lat_grid - lat)**2)
    idx = np.argmin(dist)
    ny, nx = lon_grid.shape
    row, col = idx // nx, idx % nx
    return lon_grid[row, col], lat_grid[row, col]


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

        y_train = train_df['Conc'].values
        m_train = train_df['CMAQ'].values

        poly = PolynomialFeatures(degree=poly_degree, include_bias=False)
        m_train_poly = poly.fit_transform(m_train.reshape(-1, 1))

        ols = LinearRegression()
        ols.fit(m_train_poly, y_train)
        pred_ols_train = ols.predict(m_train_poly)
        residual_ols = y_train - pred_ols_train

        # CMAQ grid coords for training (aligned with VNA)
        train_cmaq_coords = []
        for _, row in train_df.iterrows():
            cmaq_lon, cmaq_lat = get_cmaq_grid_coord(row['Lon'], row['Lat'], lon_cmaq, lat_cmaq)
            train_cmaq_coords.append([cmaq_lon, cmaq_lat])
        X_train = np.array(train_cmaq_coords)

        # CMAQ grid coords for testing (aligned with VNA)
        test_cmaq_coords = []
        for _, row in test_df.iterrows():
            cmaq_lon, cmaq_lat = get_cmaq_grid_coord(row['Lon'], row['Lat'], lon_cmaq, lat_cmaq)
            test_cmaq_coords.append([cmaq_lon, cmaq_lat])
        X_test = np.array(test_cmaq_coords)

        y_test = test_df['Conc'].values
        m_test = test_df['CMAQ'].values

        # GPR kriging on CMAQ grid coords
        gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, alpha=0.1, normalize_y=True)
        gpr.fit(X_train, residual_ols)
        gpr_pred = gpr.predict(X_test)

        # Final prediction
        m_test_poly = poly.transform(m_test.reshape(-1, 1))
        pred_ols_test = ols.predict(m_test_poly)
        rk_pred = pred_ols_test + gpr_pred

        all_y_true.extend(y_test)
        all_y_pred.extend(rk_pred)

    return np.array(all_y_true), np.array(all_y_pred)


def run_pre_exp_validation():
    sep = "=" * 70
    print(sep)
    print("RK_OLS_Poly v11 pre_exp Validation (Aligned)")
    print(sep)
    print(f"Baseline (VNA pre_exp): R2={BASELINE['VNA']['R2']:.4f}, RMSE={BASELINE['VNA']['RMSE']:.2f}, MB={BASELINE['VNA']['MB']:.2f}")
    print(f"Threshold: R2>={THRESHOLD_R2:.4f}, RMSE<={BASELINE['VNA']['RMSE']}, |MB|<={abs(BASELINE['VNA']['MB'])}")

    date_list = []
    current_date = datetime.strptime('2020-01-01', '%Y-%m-%d')
    end_date = datetime.strptime('2020-01-05', '%Y-%m-%d')
    while current_date <= end_date:
        date_list.append(current_date.strftime('%Y-%m-%d'))
        current_date += timedelta(days=1)

    print(f"Date range: 2020-01-01 ~ 2020-01-05 ({len(date_list)} days)")

    results = Parallel(n_jobs=4)(
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

    r2_pass = metrics['R2'] >= THRESHOLD_R2
    rmse_pass = metrics['RMSE'] <= BASELINE['VNA']['RMSE']
    mb_pass = abs(metrics['MB']) <= abs(BASELINE['VNA']['MB'])
    innovation_pass = r2_pass and rmse_pass and mb_pass

    print(sep)
    print(f"Result: R2={metrics['R2']:.4f}, RMSE={metrics['RMSE']:.2f}, MAE={metrics['MAE']:.2f}, MB={metrics['MB']:.2f}")
    r2_str = "PASS" if r2_pass else "FAIL"
    rmse_str = "PASS" if rmse_pass else "FAIL"
    mb_str = "PASS" if mb_pass else "FAIL"
    innov_str = "VERIFIED" if innovation_pass else "NOT VERIFIED"
    print(f"Check: R2>={THRESHOLD_R2:.4f}? {r2_str} | RMSE<={BASELINE['VNA']['RMSE']}? {rmse_str} | |MB|<={abs(BASELINE['VNA']['MB'])}? {mb_str}")
    print(f"Innovation: {innov_str}")
    print(sep)

    return metrics, innovation_pass


def main():
    metrics, innovation_pass = run_pre_exp_validation()

    result = {
        'method': 'RK_OLS_Poly',
        'stage': 'pre_exp',
        'note': 'GPR on CMAQ grid coords (aligned with VNA)',
        'baseline': BASELINE,
        'threshold': {'R2': THRESHOLD_R2, 'RMSE': BASELINE['VNA']['RMSE'], 'MB': abs(BASELINE['VNA']['MB'])},
        'metrics': metrics,
        '判定': {
            'R2_pass': metrics['R2'] >= THRESHOLD_R2,
            'RMSE_pass': metrics['RMSE'] <= BASELINE['VNA']['RMSE'],
            'MB_pass': abs(metrics['MB']) <= abs(BASELINE['VNA']['MB']),
            'innovation_verified': innovation_pass
        }
    }

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"Results saved: {OUTPUT_FILE}")
    return result


if __name__ == '__main__':
    main()