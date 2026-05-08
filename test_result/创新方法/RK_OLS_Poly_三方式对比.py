import os
from shared.paths import get_project_root, data_path
# -*- coding: utf-8 -*-
"""
RK_OLS_Poly 三种训练方式对比测试（多日）
======================================
对比：
- 方式1: 训练监测站坐标 / 预测CMAQ网格坐标
- 方式2a: 训练CMAQ网格坐标(聚合) / 预测CMAQ网格坐标
- 方式2b: 训练CMAQ网格坐标(不聚合) / 预测CMAQ网格坐标
"""

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import numpy as np
import pandas as pd
import netCDF4 as nc
from datetime import datetime
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel

ROOT_DIR = str(get_project_root())
CMAQ_FILE = data_path('test_data/raw/CMAQ/2020_PM25.nc')
MONITOR_FILE = data_path('test_data/raw/Monitor/2020_DailyPM2.5Monitor.csv')
FOLD_FILE = data_path('test_data/fold_split_table_daily.csv')

SELECTED_DAYS = ['2020-01-01', '2020-01-02', '2020-01-03', '2020-01-04']


def compute_metrics(y_true, y_pred):
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
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


def run_method_for_day(selected_day, lon_cmaq, lat_cmaq, cmaq_day, method):
    """对单日运行十折验证"""
    monitor_df = pd.read_csv(MONITOR_FILE)
    fold_df = pd.read_csv(FOLD_FILE)

    day_df = monitor_df[monitor_df['Date'] == selected_day].copy()
    day_df = day_df.merge(fold_df, on='Site', how='left')
    day_df = day_df.dropna(subset=['Lat', 'Lon', 'Conc'])

    if len(day_df) < 100:
        return np.array([]), np.array([])

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

        poly = PolynomialFeatures(degree=2, include_bias=False)
        m_train_poly = poly.fit_transform(m_train.reshape(-1, 1))
        ols = LinearRegression()
        ols.fit(m_train_poly, y_train)
        residual_ols = y_train - ols.predict(m_train_poly)

        # ========== 训练坐标选择 ==========
        if method == '方式1':
            X_train = train_df[['Lon', 'Lat']].values
        elif method == '方式2a':
            train_coords = []
            for _, row in train_df.iterrows():
                cmaq_lon, cmaq_lat = get_cmaq_grid_coord(row['Lon'], row['Lat'], lon_cmaq, lat_cmaq)
                train_coords.append([cmaq_lon, cmaq_lat])
            X_train = np.array(train_coords)

            coord_to_idx = {}
            for i, coord in enumerate(X_train):
                key = tuple(coord)
                if key not in coord_to_idx:
                    coord_to_idx[key] = []
                coord_to_idx[key].append(i)

            aggregated_coords = []
            aggregated_residual = []
            for key, indices in coord_to_idx.items():
                aggregated_coords.append(key)
                aggregated_residual.append(np.mean(residual_ols[indices]))
            X_train = np.array(aggregated_coords)
            residual_ols = np.array(aggregated_residual)
        elif method == '方式2b':
            train_coords = []
            for _, row in train_df.iterrows():
                cmaq_lon, cmaq_lat = get_cmaq_grid_coord(row['Lon'], row['Lat'], lon_cmaq, lat_cmaq)
                train_coords.append([cmaq_lon, cmaq_lat])
            X_train = np.array(train_coords)
        # ==================================

        gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, alpha=0.1, normalize_y=True)
        gpr.fit(X_train, residual_ols)

        test_cmaq_coords = []
        for _, row in test_df.iterrows():
            cmaq_lon, cmaq_lat = get_cmaq_grid_coord(row['Lon'], row['Lat'], lon_cmaq, lat_cmaq)
            test_cmaq_coords.append([cmaq_lon, cmaq_lat])
        X_test = np.array(test_cmaq_coords)

        y_test = test_df['Conc'].values
        m_test = test_df['CMAQ'].values

        gpr_pred = gpr.predict(X_test)
        m_test_poly = poly.transform(m_test.reshape(-1, 1))
        pred_ols_test = ols.predict(m_test_poly)
        rk_pred = pred_ols_test + gpr_pred

        all_y_true.extend(y_test)
        all_y_pred.extend(rk_pred)

    return np.array(all_y_true), np.array(all_y_pred)


def main():
    print("=" * 70)
    print(f"RK_OLS_Poly 三种方式对比 - {SELECTED_DAYS}")
    print("=" * 70)

    ds = nc.Dataset(CMAQ_FILE, 'r')
    lon_cmaq = ds.variables['lon'][:]
    lat_cmaq = ds.variables['lat'][:]
    pred_pm25 = ds.variables['pred_PM25'][:]
    ds.close()

    days_data = {}
    for day_str in SELECTED_DAYS:
        date_obj = datetime.strptime(day_str, '%Y-%m-%d')
        day_idx = (date_obj - datetime(2020, 1, 1)).days
        days_data[day_str] = pred_pm25[day_idx]

    methods = ['方式1', '方式2a', '方式2b']

    # 每日结果
    daily_results = {m: {} for m in methods}

    for method in methods:
        print(f"\n{method}:")
        for day_str in SELECTED_DAYS:
            y_true, y_pred = run_method_for_day(day_str, lon_cmaq, lat_cmaq, days_data[day_str], method)
            if len(y_true) > 0:
                m = compute_metrics(y_true, y_pred)
                daily_results[method][day_str] = m
                print(f"  {day_str}: R2={m['R2']:.4f} RMSE={m['RMSE']:.2f} MB={m['MB']:.2f}")
            else:
                daily_results[method][day_str] = {'R2': np.nan, 'RMSE': np.nan, 'MB': np.nan}
                print(f"  {day_str}: 无数据")

    # 每日明细表
    print("\n" + "=" * 70)
    print("每日明细对比")
    print("=" * 70)
    print(f"{'日期':<12} {'方式1 R2':>10} {'方式2a R2':>10} {'方式2b R2':>10} | {'方式1 RMSE':>10} {'方式2a RMSE':>10} {'方式2b RMSE':>10}")
    print("-" * 75)
    for day_str in SELECTED_DAYS:
        r1 = daily_results['方式1'][day_str]['R2']
        r2a = daily_results['方式2a'][day_str]['R2']
        r2b = daily_results['方式2b'][day_str]['R2']
        rm1 = daily_results['方式1'][day_str]['RMSE']
        rm2a = daily_results['方式2a'][day_str]['RMSE']
        rm2b = daily_results['方式2b'][day_str]['RMSE']
        r1_str = f"{r1:.4f}" if not np.isnan(r1) else "  N/A  "
        r2a_str = f"{r2a:.4f}" if not np.isnan(r2a) else "  N/A  "
        r2b_str = f"{r2b:.4f}" if not np.isnan(r2b) else "  N/A  "
        print(f"{day_str:<12} {r1_str:>10} {r2a_str:>10} {r2b_str:>10} | {rm1:>10.2f} {rm2a:>10.2f} {rm2b:>10.2f}")

    # 汇总
    print("\n" + "=" * 70)
    print("四天汇总")
    print("=" * 70)
    print(f"{'方式':<8} {'平均R2':>10} {'平均RMSE':>10} {'平均|MB|':>10}")
    print("-" * 40)
    for method in methods:
        r2_vals = [daily_results[method][d]['R2'] for d in SELECTED_DAYS]
        rmse_vals = [daily_results[method][d]['RMSE'] for d in SELECTED_DAYS]
        mb_vals = [daily_results[method][d]['MB'] for d in SELECTED_DAYS]
        r2_avg = np.nanmean(r2_vals)
        rmse_avg = np.nanmean(rmse_vals)
        mb_avg = np.nanmean([abs(m) for m in mb_vals])
        print(f"{method:<8} {r2_avg:>10.4f} {rmse_avg:>10.2f} {mb_avg:>10.2f}")

    print("\n" + "=" * 70)


if __name__ == '__main__':
    main()
