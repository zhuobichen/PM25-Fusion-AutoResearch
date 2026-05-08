import os
from shared.paths import get_project_root, data_path
# -*- coding: utf-8 -*-
"""
对比RK_OLS_Poly的两种训练方式
方式1: 直接用所有监测站点（有重复CMAQ坐标）
方式2: 对同一CMAQ网格点的监测站做平均聚合
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


def ten_fold_method1(selected_day):
    """方式1: 直接用所有监测站点（有重复CMAQ坐标）"""
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

        # 方式1: CMAQ网格坐标（可能有重复）
        train_cmaq_coords = []
        for _, row in train_df.iterrows():
            cmaq_lon, cmaq_lat = get_cmaq_grid_coord(row['Lon'], row['Lat'], lon_cmaq, lat_cmaq)
            train_cmaq_coords.append([cmaq_lon, cmaq_lat])
        X_train = np.array(train_cmaq_coords)

        test_cmaq_coords = []
        for _, row in test_df.iterrows():
            cmaq_lon, cmaq_lat = get_cmaq_grid_coord(row['Lon'], row['Lat'], lon_cmaq, lat_cmaq)
            test_cmaq_coords.append([cmaq_lon, cmaq_lat])
        X_test = np.array(test_cmaq_coords)

        y_train = train_df['Conc'].values
        y_test = test_df['Conc'].values
        m_train = train_df['CMAQ'].values
        m_test = test_df['CMAQ'].values

        # 多项式校正
        poly = PolynomialFeatures(degree=2, include_bias=False)
        m_train_poly = poly.fit_transform(m_train.reshape(-1, 1))
        m_test_poly = poly.transform(m_test.reshape(-1, 1))

        ols = LinearRegression()
        ols.fit(m_train_poly, y_train)
        residual_ols = y_train - ols.predict(m_train_poly)

        # GPR
        gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, alpha=0.1, normalize_y=True)
        gpr.fit(X_train, residual_ols)
        gpr_pred = gpr.predict(X_test)

        # 融合
        pred_ols_test = ols.predict(m_test_poly)
        rk_pred = pred_ols_test + gpr_pred

        all_y_true.extend(y_test)
        all_y_pred.extend(rk_pred)

    return np.array(all_y_true), np.array(all_y_pred)


def ten_fold_method2(selected_day):
    """方式2: 对同一CMAQ网格点的监测站做平均聚合"""
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
    cmaq_coords = []
    for _, row in day_df.iterrows():
        val = get_cmaq_at_site(row['Lon'], row['Lat'], lon_cmaq, lat_cmaq, cmaq_day)
        cmaq_values.append(val)
        cmaq_lon, cmaq_lat = get_cmaq_grid_coord(row['Lon'], row['Lat'], lon_cmaq, lat_cmaq)
        cmaq_coords.append((cmaq_lon, cmaq_lat))

    day_df['CMAQ'] = cmaq_values
    day_df['cmaq_lon'] = [c[0] for c in cmaq_coords]
    day_df['cmaq_lat'] = [c[1] for c in cmaq_coords]
    day_df['cmaq_coord'] = list(zip(day_df['cmaq_lon'], day_df['cmaq_lat']))

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

        # 方式2: 对同一CMAQ网格点的监测站做平均聚合
        # 按cmaq_coord分组，取平均
        train_grouped = train_df.groupby('cmaq_coord').agg({
            'Conc': 'mean',
            'CMAQ': 'mean',
            'cmaq_lon': 'first',
            'cmaq_lat': 'first'
        }).reset_index()

        X_train = train_grouped[['cmaq_lon', 'cmaq_lat']].values
        y_train = train_grouped['Conc'].values
        m_train = train_grouped['CMAQ'].values

        # 测试集保持原样（因为测试时是在真实监测站位置评估）
        test_cmaq_coords = test_df[['cmaq_lon', 'cmaq_lat']].values
        X_test = test_cmaq_coords
        y_test = test_df['Conc'].values
        m_test = test_df['CMAQ'].values

        # 多项式校正
        poly = PolynomialFeatures(degree=2, include_bias=False)
        m_train_poly = poly.fit_transform(m_train.reshape(-1, 1))
        m_test_poly = poly.transform(m_test.reshape(-1, 1))

        ols = LinearRegression()
        ols.fit(m_train_poly, y_train)
        residual_ols = y_train - ols.predict(m_train_poly)

        # GPR
        gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, alpha=0.1, normalize_y=True)
        gpr.fit(X_train, residual_ols)
        gpr_pred = gpr.predict(X_test)

        # 融合
        pred_ols_test = ols.predict(m_test_poly)
        rk_pred = pred_ols_test + gpr_pred

        all_y_true.extend(y_test)
        all_y_pred.extend(rk_pred)

    return np.array(all_y_true), np.array(all_y_pred)


def main():
    print("=" * 70)
    print("RK_OLS_Poly 两种训练方式对比 - 预实验 (5天)")
    print("=" * 70)

    selected_days = ['2020-01-01', '2020-01-02', '2020-01-03', '2020-01-04', '2020-01-05']

    results = {
        'method1': {'y_true': [], 'y_pred': []},
        'method2': {'y_true': [], 'y_pred': []}
    }

    for day in selected_days:
        print(f"\n日期: {day}")

        y_true1, y_pred1 = ten_fold_method1(day)
        y_true2, y_pred2 = ten_fold_method2(day)

        if len(y_true1) > 0:
            m1 = compute_metrics(y_true1, y_pred1)
            print(f"  方式1 (直接站点): R2={m1['R2']:.4f}, RMSE={m1['RMSE']:.2f}, MB={m1['MB']:.2f}, n={len(y_true1)}")
            results['method1']['y_true'].extend(y_true1)
            results['method1']['y_pred'].extend(y_pred1)

        if len(y_true2) > 0:
            m2 = compute_metrics(y_true2, y_pred2)
            print(f"  方式2 (聚合网格): R2={m2['R2']:.4f}, RMSE={m2['RMSE']:.2f}, MB={m2['MB']:.2f}, n={len(y_true2)}")

            # 检查聚合后训练样本数
            ds = nc.Dataset(CMAQ_FILE, 'r')
            lon_cmaq = ds.variables['lon'][:]
            lat_cmaq = ds.variables['lat'][:]
            ds.close()

            monitor_df = pd.read_csv(MONITOR_FILE)
            fold_df = pd.read_csv(FOLD_FILE)
            day_df = monitor_df[monitor_df['Date'] == day].copy()
            day_df = day_df.merge(fold_df, on='Site', how='left')
            day_df = day_df.dropna(subset=['Lat', 'Lon', 'Conc'])

            day_df['cmaq_coord'] = day_df.apply(
                lambda row: get_cmaq_grid_coord(row['Lon'], row['Lat'], lon_cmaq, lat_cmaq),
                axis=1
            )

            train_df = day_df[day_df['fold'] != 1].copy()
            train_grouped = train_df.groupby('cmaq_coord').size()
            print(f"  方式2训练样本: 聚合前={len(train_df)}, 聚合后={len(train_grouped)}")

            results['method2']['y_true'].extend(y_true2)
            results['method2']['y_pred'].extend(y_pred2)

    print("\n" + "=" * 70)
    print("汇总结果 (5天合并)")
    print("=" * 70)

    y_true1 = np.array(results['method1']['y_true'])
    y_pred1 = np.array(results['method1']['y_pred'])
    y_true2 = np.array(results['method2']['y_true'])
    y_pred2 = np.array(results['method2']['y_pred'])

    m1 = compute_metrics(y_true1, y_pred1)
    m2 = compute_metrics(y_true2, y_pred2)

    print(f"\n方式1 (直接站点):")
    print(f"  R²   = {m1['R2']:.4f}")
    print(f"  RMSE = {m1['RMSE']:.2f}")
    print(f"  MAE  = {m1['MAE']:.2f}")
    print(f"  MB   = {m1['MB']:.2f}")
    print(f"  样本数 = {len(y_true1)}")

    print(f"\n方式2 (CMAQ网格聚合):")
    print(f"  R²   = {m2['R2']:.4f}")
    print(f"  RMSE = {m2['RMSE']:.2f}")
    print(f"  MAE  = {m2['MAE']:.2f}")
    print(f"  MB   = {m2['MB']:.2f}")
    print(f"  样本数 = {len(y_true2)}")

    print(f"\n差异 (方式1 - 方式2):")
    print(f"  ΔR²   = {m1['R2'] - m2['R2']:+.4f}")
    print(f"  ΔRMSE = {m1['RMSE'] - m2['RMSE']:+.2f}")
    print(f"  ΔMB   = {m1['MB'] - m2['MB']:+.2f}")


if __name__ == '__main__':
    main()