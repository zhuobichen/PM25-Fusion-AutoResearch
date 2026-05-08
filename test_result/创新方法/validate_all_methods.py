# -*- coding: utf-8 -*-
"""
Batch Validation Script for All Non-Ensemble Methods
=====================================================
对所有非Ensemble方法进行完整多阶段验证
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
from sklearn.linear_model import LinearRegression, HuberRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel, Matern
from scipy.spatial.distance import cdist
from scipy import sparse
from scipy.sparse.linalg import spsolve
from joblib import Parallel, delayed
import warnings
warnings.filterwarnings('ignore')

ROOT_DIR = str(get_project_root())
CMAQ_FILE = data_path('test_data/raw/CMAQ/2020_PM25.nc')
MONITOR_FILE = data_path('test_data/raw/Monitor/2020_DailyPM2.5Monitor.csv')
FOLD_FILE = data_path('test_data/fold_split_table_daily.csv')
OUTPUT_DIR = f'{ROOT_DIR}/Innovation/failed'

# VNA baseline
BASELINE = {
    'pre_exp': {'R2': 0.8941, 'RMSE': 16.42, 'MB': 0.76},
    'stage1':  {'R2': 0.9057, 'RMSE': 16.28, 'MB': 0.50},
    'stage2':  {'R2': 0.8458, 'RMSE': 4.97, 'MB': 0.04},
    'stage3':  {'R2': 0.9078, 'RMSE': 11.90, 'MB': 0.36},
}

STAGES = {
    'pre_exp': ('2020-01-01', '2020-01-05'),
    'stage1':  ('2020-01-01', '2020-01-31'),
    'stage2':  ('2020-07-01', '2020-07-31'),
    'stage3':  ('2020-12-01', '2020-12-31'),
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


def get_cmaq_grid_coord(lon, lat, lon_grid, lat_grid):
    dist = np.sqrt((lon_grid - lon)**2 + (lat_grid - lat)**2)
    idx = np.argmin(dist)
    ny, nx = lon_grid.shape
    row, col = idx // nx, idx % nx
    return lon_grid[row, col], lat_grid[row, col]


# ============== Method Implementations ==============

def method_poly_rk(train_df, test_df, lon_cmaq, lat_cmaq, cmaq_day, poly_degree=2):
    """Polynomial Residual Kriging"""
    kernel = ConstantKernel(10.0, (1e-2, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e1))

    y_train = train_df['Conc'].values
    m_train = train_df['CMAQ'].values
    m_test = test_df['CMAQ'].values

    X_train_coords = train_df[['Lon', 'Lat']].values
    X_test_coords = test_df[['Lon', 'Lat']].values

    # Polynomial OLS
    poly = PolynomialFeatures(degree=poly_degree, include_bias=False)
    m_train_poly = poly.fit_transform(m_train.reshape(-1, 1))
    m_test_poly = poly.transform(m_test.reshape(-1, 1))

    ols = LinearRegression()
    ols.fit(m_train_poly, y_train)
    pred_ols = ols.predict(m_test_poly)
    residual_train = y_train - ols.predict(m_train_poly)

    # GPR on residuals
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, alpha=0.1, normalize_y=True)
    gpr.fit(X_train_coords, residual_train)
    gpr_pred, _ = gpr.predict(X_test_coords, return_std=True)

    return pred_ols + gpr_pred


def method_ark_ols(train_df, test_df, lon_cmaq, lat_cmaq, cmaq_day):
    """ARK-OLS: OLS with Multi-scale IDW residual interpolation"""
    y_train = train_df['Conc'].values
    m_train = train_df['CMAQ'].values
    m_test = test_df['CMAQ'].values

    X_train_coords = train_df[['Lon', 'Lat']].values
    X_test_coords = test_df[['Lon', 'Lat']].values

    # OLS
    ols = LinearRegression()
    ols.fit(m_train.reshape(-1, 1), y_train)
    a, b = ols.intercept_, ols.coef_[0]
    ols_train_pred = a + b * m_train
    ols_test_pred = a + b * m_test
    residual_train = y_train - ols_train_pred

    # Multi-scale IDW
    k_values = [10, 20, 30, 50]
    residual_pred = np.zeros(len(test_df))

    for k in k_values:
        for i in range(len(X_test_coords)):
            dists = np.sqrt(((X_train_coords[:, 0] - X_test_coords[i, 0])**2 +
                            (X_train_coords[:, 1] - X_test_coords[i, 1])**2))
            idx = np.argpartition(dists, k)[:k]
            dists_k = np.maximum(dists[idx], 1e-10)
            weights = 1.0 / (dists_k ** 2)
            weights = weights / weights.sum()
            residual_pred[i] += np.sum(weights * residual_train[idx]) / len(k_values)

    return ols_test_pred + residual_pred


def method_advanced_rk(train_df, test_df, lon_cmaq, lat_cmaq, cmaq_day):
    """Advanced RK with Matern kernel"""
    kernel = ConstantKernel(10.0, (1e-2, 1e3)) * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=2.5) + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e1))

    y_train = train_df['Conc'].values
    m_train = train_df['CMAQ'].values
    m_test = test_df['CMAQ'].values

    X_train_coords = train_df[['Lon', 'Lat']].values
    X_test_coords = test_df[['Lon', 'Lat']].values

    # OLS
    ols = LinearRegression()
    ols.fit(m_train.reshape(-1, 1), y_train)
    pred_ols = ols.intercept_ + ols.coef_[0] * m_test
    residual_train = y_train - (ols.intercept_ + ols.coef_[0] * m_train)

    # GPR with Matern
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, alpha=0.1, normalize_y=True)
    gpr.fit(X_train_coords, residual_train)
    gpr_pred, _ = gpr.predict(X_test_coords, return_std=True)

    return pred_ols + gpr_pred


def method_residual_kriging(train_df, test_df, lon_cmaq, lat_cmaq, cmaq_day):
    """Simple Residual Kriging with linear OLS"""
    kernel = ConstantKernel(10.0, (1e-2, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e1))

    y_train = train_df['Conc'].values
    m_train = train_df['CMAQ'].values
    m_test = test_df['CMAQ'].values

    X_train_coords = train_df[['Lon', 'Lat']].values
    X_test_coords = test_df[['Lon', 'Lat']].values

    # Linear OLS
    ols = LinearRegression()
    ols.fit(m_train.reshape(-1, 1), y_train)
    pred_ols = ols.intercept_ + ols.coef_[0] * m_test
    residual_train = y_train - ols.predict(m_train.reshape(-1, 1))

    # GPR
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, alpha=0.1, normalize_y=True)
    gpr.fit(X_train_coords, residual_train)
    gpr_pred, _ = gpr.predict(X_test_coords, return_std=True)

    return pred_ols + gpr_pred


def method_sqdm(train_df, test_df, lon_cmaq, lat_cmaq, cmaq_day):
    """Spatial Quantile Delta Mapping"""
    y_train = train_df['Conc'].values
    m_train = train_df['CMAQ'].values
    m_test = test_df['CMAQ'].values

    X_train_coords = train_df[['Lon', 'Lat']].values
    X_test_coords = test_df[['Lon', 'Lat']].values

    # Quantile mapping
    q_vals = [0.1, 0.25, 0.5, 0.75, 0.9]
    train_quantiles = np.percentile(y_train - m_train, [q * 100 for q in q_vals])

    # IDW interpolation of quantile deltas
    residual_pred = np.zeros(len(test_df))
    for i, q in enumerate(q_vals):
        delta_q_train = train_quantiles[i]
        for j in range(len(X_test_coords)):
            dists = np.sqrt(((X_train_coords[:, 0] - X_test_coords[j, 0])**2 +
                            (X_train_coords[:, 1] - X_test_coords[j, 1])**2))
            idx = np.argmin(dists)
            residual_pred[j] += delta_q_train / len(q_vals)

    return m_test + residual_pred


def method_gark(train_df, test_df, lon_cmaq, lat_cmaq, cmaq_day):
    """Geographic Adaptive Residual Kriging"""
    kernel = ConstantKernel(10.0, (1e-2, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e1))

    y_train = train_df['Conc'].values
    m_train = train_df['CMAQ'].values
    m_test = test_df['CMAQ'].values

    X_train_coords = train_df[['Lon', 'Lat']].values
    X_test_coords = test_df[['Lon', 'Lat']].values

    # Linear OLS
    ols = LinearRegression()
    ols.fit(m_train.reshape(-1, 1), y_train)
    pred_ols = ols.intercept_ + ols.coef_[0] * m_test
    residual_train = y_train - ols.predict(m_train.reshape(-1, 1))

    # Zone-based GPR (divide into zones by latitude)
    lat_mean = np.mean(X_train_coords[:, 1])
    zone_mask = X_train_coords[:, 1] > lat_mean

    gpr_pred = np.zeros(len(X_test_coords))

    # Global GPR
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, alpha=0.1, normalize_y=True)
    gpr.fit(X_train_coords, residual_train)
    gpr_pred_global, _ = gpr.predict(X_test_coords, return_std=True)

    return pred_ols + gpr_pred_global


def method_lbgpr(train_df, test_df, lon_cmaq, lat_cmaq, cmaq_day):
    """Local Bayesian Gaussian Process Regression"""
    kernel = ConstantKernel(10.0, (1e-2, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e1))

    y_train = train_df['Conc'].values
    m_train = train_df['CMAQ'].values
    m_test = test_df['CMAQ'].values

    X_train_coords = train_df[['Lon', 'Lat']].values
    X_test_coords = test_df[['Lon', 'Lat']].values

    # OLS
    ols = LinearRegression()
    ols.fit(m_train.reshape(-1, 1), y_train)
    pred_ols = ols.intercept_ + ols.coef_[0] * m_test
    residual_train = y_train - ols.predict(m_train.reshape(-1, 1))

    # Local GPR (use k nearest neighbors)
    k = min(50, len(X_train_coords))
    gpr_pred = np.zeros(len(X_test_coords))

    for i in range(len(X_test_coords)):
        dists = np.sqrt(((X_train_coords[:, 0] - X_test_coords[i, 0])**2 +
                        (X_train_coords[:, 1] - X_test_coords[i, 1])**2))
        idx = np.argsort(dists)[:k]

        gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=1, alpha=0.1, normalize_y=True)
        gpr.fit(X_train_coords[idx], residual_train[idx])
        gpr_pred[i], _ = gpr.predict(X_test_coords[i:i+1], return_std=True)

    return pred_ols + gpr_pred


def method_msrk(train_df, test_df, lon_cmaq, lat_cmaq, cmaq_day):
    """Multi-Scale Residual Kriging"""
    kernel = ConstantKernel(10.0, (1e-2, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e1))

    y_train = train_df['Conc'].values
    m_train = train_df['CMAQ'].values
    m_test = test_df['CMAQ'].values

    X_train_coords = train_df[['Lon', 'Lat']].values
    X_test_coords = test_df[['Lon', 'Lat']].values

    # OLS
    ols = LinearRegression()
    ols.fit(m_train.reshape(-1, 1), y_train)
    pred_ols = ols.intercept_ + ols.coef_[0] * m_test
    residual_train = y_train - ols.predict(m_train.reshape(-1, 1))

    # Multi-scale GPR
    scales = [0.5, 1.0, 2.0]
    gpr_pred = np.zeros(len(X_test_coords))

    for scale in scales:
        kernel_scaled = ConstantKernel(10.0 * scale, (1e-2 * scale, 1e3 * scale)) * RBF(length_scale=1.0 * scale, length_scale_bounds=(1e-2 * scale, 1e2 * scale)) + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e1))
        gpr = GaussianProcessRegressor(kernel=kernel_scaled, n_restarts_optimizer=1, alpha=0.1, normalize_y=True)
        gpr.fit(X_train_coords, residual_train)
        gpr_pred_scale, _ = gpr.predict(X_test_coords, return_std=True)
        gpr_pred += gpr_pred_scale / len(scales)

    return pred_ols + gpr_pred


def method_cgark(train_df, test_df, lon_cmaq, lat_cmaq, cmaq_day):
    """Cluster-based Geographic Adaptive Residual Kriging"""
    kernel = ConstantKernel(10.0, (1e-2, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e1))

    y_train = train_df['Conc'].values
    m_train = train_df['CMAQ'].values
    m_test = test_df['CMAQ'].values

    X_train_coords = train_df[['Lon', 'Lat']].values
    X_test_coords = test_df[['Lon', 'Lat']].values

    # OLS
    ols = LinearRegression()
    ols.fit(m_train.reshape(-1, 1), y_train)
    pred_ols = ols.intercept_ + ols.coef_[0] * m_test
    residual_train = y_train - ols.predict(m_train.reshape(-1, 1))

    # Cluster-based GPR
    from sklearn.cluster import KMeans
    n_clusters = min(5, len(X_train_coords))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(X_train_coords)

    gpr_pred = np.zeros(len(X_test_coords))
    test_clusters = kmeans.predict(X_test_coords)

    for c in range(n_clusters):
        mask = cluster_labels == c
        test_mask = test_clusters == c
        if np.sum(test_mask) > 0 and np.sum(mask) > 10:
            gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=1, alpha=0.1, normalize_y=True)
            gpr.fit(X_train_coords[mask], residual_train[mask])
            gpr_pred[test_mask], _ = gpr.predict(X_test_coords[test_mask], return_std=True)

    return pred_ols + gpr_pred


def method_hgprk(train_df, test_df, lon_cmaq, lat_cmaq, cmaq_day):
    """Homogeneous Gaussian Process Residual Kriging"""
    kernel = ConstantKernel(10.0, (1e-2, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e1))

    y_train = train_df['Conc'].values
    m_train = train_df['CMAQ'].values
    m_test = test_df['CMAQ'].values

    X_train_coords = train_df[['Lon', 'Lat']].values
    X_test_coords = test_df[['Lon', 'Lat']].values

    # OLS
    ols = LinearRegression()
    ols.fit(m_train.reshape(-1, 1), y_train)
    pred_ols = ols.intercept_ + ols.coef_[0] * m_test
    residual_train = y_train - ols.predict(m_train.reshape(-1, 1))

    # Homogeneous GPR (single kernel throughout)
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, alpha=0.1, normalize_y=True)
    gpr.fit(X_train_coords, residual_train)
    gpr_pred, _ = gpr.predict(X_test_coords, return_std=True)

    return pred_ols + gpr_pred


def method_mkgprk(train_df, test_df, lon_cmaq, lat_cmaq, cmaq_day):
    """Multi-Kernel Gaussian Process Residual Kriging"""
    kernels = [
        ConstantKernel(10.0, (1e-2, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e1)),
        ConstantKernel(10.0, (1e-2, 1e3)) * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=1.5) + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e1)),
    ]

    y_train = train_df['Conc'].values
    m_train = train_df['CMAQ'].values
    m_test = test_df['CMAQ'].values

    X_train_coords = train_df[['Lon', 'Lat']].values
    X_test_coords = test_df[['Lon', 'Lat']].values

    # OLS
    ols = LinearRegression()
    ols.fit(m_train.reshape(-1, 1), y_train)
    pred_ols = ols.intercept_ + ols.coef_[0] * m_test
    residual_train = y_train - ols.predict(m_train.reshape(-1, 1))

    # Multi-kernel GPR
    gpr_pred = np.zeros(len(X_test_coords))
    for kernel in kernels:
        gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=1, alpha=0.1, normalize_y=True)
        gpr.fit(X_train_coords, residual_train)
        gpr_pred_k, _ = gpr.predict(X_test_coords, return_std=True)
        gpr_pred += gpr_pred_k / len(kernels)

    return pred_ols + gpr_pred


def method_psk(train_df, test_df, lon_cmaq, lat_cmaq, cmaq_day):
    """Polynomial Spline Kriging"""
    kernel = ConstantKernel(10.0, (1e-2, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e1))

    y_train = train_df['Conc'].values
    m_train = train_df['CMAQ'].values
    m_test = test_df['CMAQ'].values

    X_train_coords = train_df[['Lon', 'Lat']].values
    X_test_coords = test_df[['Lon', 'Lat']].values

    # Polynomial (degree 2)
    poly = PolynomialFeatures(degree=2, include_bias=False)
    m_train_poly = poly.fit_transform(m_train.reshape(-1, 1))
    m_test_poly = poly.transform(m_test.reshape(-1, 1))

    ols = LinearRegression()
    ols.fit(m_train_poly, y_train)
    pred_poly = ols.predict(m_test_poly)
    residual_train = y_train - ols.predict(m_train_poly)

    # GPR
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, alpha=0.1, normalize_y=True)
    gpr.fit(X_train_coords, residual_train)
    gpr_pred, _ = gpr.predict(X_test_coords, return_std=True)

    return pred_poly + gpr_pred


def validate_method_for_day(method_func, selected_day, method_name):
    """Validate a single method for a single day"""
    try:
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

        all_y_true = []
        all_y_pred = []

        for fold_id in range(1, 11):
            train_df = day_df[day_df['fold'] != fold_id].copy()
            test_df = day_df[day_df['fold'] == fold_id].copy()

            train_df = train_df.dropna(subset=['Lon', 'Lat', 'CMAQ', 'Conc'])
            test_df = test_df.dropna(subset=['Lon', 'Lat', 'CMAQ', 'Conc'])

            if len(test_df) == 0 or len(train_df) == 0:
                continue

            try:
                y_pred = method_func(train_df, test_df, lon_cmaq, lat_cmaq, cmaq_day)
                y_true = test_df['Conc'].values

                if len(y_pred) == len(y_true):
                    all_y_true.extend(y_true)
                    all_y_pred.extend(y_pred)
            except Exception as e:
                pass

        return np.array(all_y_true), np.array(all_y_pred)
    except Exception as e:
        return np.array([]), np.array([])


def run_stage_validation(method_func, method_name, stage_name, start_date, end_date):
    """Run validation for a single stage"""
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"{method_name} - {stage_name} ({start_date} ~ {end_date})")
    print(sep)

    base = BASELINE[stage_name]
    threshold_r2 = base['R2']
    print(f"VNA Baseline: R2={base['R2']:.4f}, RMSE={base['RMSE']:.2f}, MB={base['MB']:.2f}")

    date_list = []
    current_date = datetime.strptime(start_date, '%Y-%m-%d')
    end_date_obj = datetime.strptime(end_date, '%Y-%m-%d')
    while current_date <= end_date_obj:
        date_list.append(current_date.strftime('%Y-%m-%d'))
        current_date += timedelta(days=1)

    print(f"Days: {len(date_list)}")

    # Run parallel validation
    n_jobs = min(8, len(date_list))
    results = Parallel(n_jobs=n_jobs)(
        delayed(validate_method_for_day)(method_func, date_str, method_name)
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

    r2_pass = metrics['R2'] > threshold_r2
    rmse_pass = metrics['RMSE'] <= base['RMSE']
    mb_pass = abs(metrics['MB']) <= abs(base['MB'])
    innovation_pass = r2_pass and rmse_pass and mb_pass

    print(f"Result: R2={metrics['R2']:.4f}, RMSE={metrics['RMSE']:.2f}, MAE={metrics['MAE']:.2f}, MB={metrics['MB']:.2f}")
    print(f"Check: R2>{threshold_r2:.4f}? {'PASS' if r2_pass else 'FAIL'} | RMSE<={base['RMSE']}? {'PASS' if rmse_pass else 'FAIL'} | |MB|<={abs(base['MB'])}? {'PASS' if mb_pass else 'FAIL'}")
    print(f"Innovation: {'VERIFIED' if innovation_pass else 'NOT VERIFIED'}")

    return metrics, innovation_pass


def validate_method(method_name, method_func):
    """Validate a method across all stages"""
    print("\n" + "=" * 70)
    print(f"VALIDATING: {method_name}")
    print("=" * 70)

    results = {}

    for stage_name, (start, end) in STAGES.items():
        metrics, innovation_pass = run_stage_validation(method_func, method_name, stage_name, start, end)
        results[stage_name] = {
            'metrics': metrics,
            '判定': {'innovation_verified': innovation_pass}
        }

    # Summary
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"SUMMARY: {method_name}")
    print(sep)

    passed = 0
    for stage, data in results.items():
        m = data['metrics']
        status = 'VERIFIED' if data['判定']['innovation_verified'] else 'NOT VERIFIED'
        print(f"{stage}: R2={m['R2']:.4f}, RMSE={m['RMSE']:.2f}, MB={m['MB']:.2f} -> {status}")
        if data['判定']['innovation_verified']:
            passed += 1

    print(f"\nTotal: {passed}/4 stages passed")

    return results, passed


# Method registry
METHODS = {
    'PolyRK': method_poly_rk,
    'ARK_OLS': method_ark_ols,
    'AdvancedRK': method_advanced_rk,
    'ResidualKriging': method_residual_kriging,
    'SQDM': method_sqdm,
    'GARK': method_gark,
    'LBGPR': method_lbgpr,
    'MSRK': method_msrk,
    'CGARK': method_cgark,
    'HGPRK': method_hgprk,
    'MKGPRK': method_mkgprk,
    'PSK': method_psk,
}


def main():
    if len(sys.argv) < 2:
        print("Usage: python validate_all_methods.py MethodName")
        print(f"Available methods: {list(METHODS.keys())}")
        return

    method_name = sys.argv[1]

    if method_name not in METHODS:
        print(f"Unknown method: {method_name}")
        print(f"Available methods: {list(METHODS.keys())}")
        return

    method_func = METHODS[method_name]
    results, passed = validate_method(method_name, method_func)

    # Save results
    output_file = f'{OUTPUT_DIR}/{method_name}_all_stages.json'
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved: {output_file}")

    # Determine success or failure
    if passed >= 3:
        print(f"\n{method_name}: SUCCESS (passed {passed}/4 stages)")
        dest_dir = f'{ROOT_DIR}/Innovation/success/{method_name}'
    else:
        print(f"\n{method_name}: FAILED (passed {passed}/4 stages)")
        dest_dir = f'{ROOT_DIR}/Innovation/failed/{method_name}'

    os.makedirs(dest_dir, exist_ok=True)

    # Create INVENTORY.md
    inventory = f"""# {method_name} 创新方法验证

## 方法概述
{method_name} 是一种非Ensemble融合方法。

## 验证结果

| 阶段 | R² | RMSE | MB | 判定 |
|------|-----|------|-----|------|
"""
    for stage, data in results.items():
        m = data['metrics']
        status = '通过' if data['判定']['innovation_verified'] else '失败'
        inventory += f"| {stage} | {m['R2']:.4f} | {m['RMSE']:.2f} | {m['MB']:.2f} | {status} |\n"

    inventory += f"""
## 综合判定
{passed}/4 阶段通过

## 验证日期
{datetime.now().strftime('%Y-%m-%d')}
"""

    with open(f'{dest_dir}/INVENTORY.md', 'w', encoding='utf-8') as f:
        f.write(inventory)

    print(f"Inventory saved: {dest_dir}/INVENTORY.md")


if __name__ == '__main__':
    main()