# -*- coding: utf-8 -*-
"""
BayesianVariationalFusion All Stages Validation
================================================
验证贝叶斯变分融合方法的创新性
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
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.spatial.distance import cdist
from joblib import Parallel, delayed

ROOT_DIR = str(get_project_root())
CMAQ_FILE = data_path('test_data/raw/CMAQ/2020_PM25.nc')
MONITOR_FILE = data_path('test_data/raw/Monitor/2020_DailyPM2.5Monitor.csv')
FOLD_FILE = data_path('test_data/fold_split_table_daily.csv')
OUTPUT_FILE = f'{ROOT_DIR}/Innovation/failed/BayesianVariationalFusion_all_stages.json'

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


def build_laplacian_matrix(lon, lat, n_neighbors=10):
    """构建拉普拉斯平滑矩阵"""
    n = len(lon)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        dist_matrix[i, :] = np.sqrt((lon - lon[i])**2 + (lat - lat[i])**2)

    row_indices = []
    col_indices = []
    data = []

    for i in range(n):
        nearest_idx = np.argsort(dist_matrix[i, :])[1:n_neighbors+1]
        for j in nearest_idx:
            if i != j:
                d = dist_matrix[i, j]
                weight = np.exp(-d**2 / 2.0)
                row_indices.append(i)
                col_indices.append(j)
                data.append(weight)

    W = sparse.csr_matrix((data, (row_indices, col_indices)), shape=(n, n))
    D = sparse.diags(W.sum(axis=1).A1) - W
    return D


def bayesian_variational_fusion(y_obs, m_cmaq, lon, lat, omega=1.0, epsilon=1e-2, delta=1e-2):
    """贝叶斯变分融合"""
    n = len(y_obs)
    D = build_laplacian_matrix(lon, lat, n_neighbors=10)
    DTD = D.T @ D
    system_matrix = omega * sparse.eye(n) + epsilon * DTD + delta * sparse.eye(n)
    rhs = omega * (y_obs - m_cmaq)
    bias = spsolve(system_matrix.tocsc(), rhs)
    return bias


def ten_fold_for_day_bayesian(selected_day, omega=1.0, epsilon=0.01, delta=0.01):
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

        y_train = train_df['Conc'].values
        m_train = train_df['CMAQ'].values
        lon_train = train_df['Lon'].values
        lat_train = train_df['Lat'].values

        y_test = test_df['Conc'].values
        m_test = test_df['CMAQ'].values

        # 训练偏差
        bias_train = bayesian_variational_fusion(y_train, m_train, lon_train, lat_train,
                                                  omega=omega, epsilon=epsilon, delta=delta)

        # IDW插值偏差到测试站点
        test_locations = test_df[['Lon', 'Lat']].values
        train_locations = train_df[['Lon', 'Lat']].values

        distances = cdist(test_locations, train_locations)
        weights = 1.0 / (distances + 1e-6)
        weights = weights / weights.sum(axis=1, keepdims=True)
        bias_test = weights @ bias_train

        # 融合预测
        pred_test = m_test + bias_test

        all_y_true.extend(y_test)
        all_y_pred.extend(pred_test)

    return np.array(all_y_true), np.array(all_y_pred)


def run_stage_validation(stage_name, start_date, end_date, best_params):
    sep = "=" * 70
    print(sep)
    print(f"Stage: {stage_name} ({start_date} ~ {end_date})")
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

    n_jobs = min(8, len(date_list))
    results = Parallel(n_jobs=n_jobs)(
        delayed(ten_fold_for_day_bayesian)(date_str, **best_params)
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
    r2_str = "PASS" if r2_pass else "FAIL"
    rmse_str = "PASS" if rmse_pass else "FAIL"
    mb_str = "PASS" if mb_pass else "FAIL"
    innov_str = "VERIFIED" if innovation_pass else "NOT VERIFIED"
    print(f"Check: R2>{threshold_r2:.4f}? {r2_str} | RMSE<={base['RMSE']}? {rmse_str} | |MB|<={abs(base['MB'])}? {mb_str}")
    print(f"Innovation: {innov_str}")

    return metrics, innovation_pass


def find_best_params():
    """使用单日数据寻找最佳参数"""
    print("\n=== Finding Best Parameters ===")
    selected_day = '2020-01-01'

    monitor_df = pd.read_csv(MONITOR_FILE)
    fold_df = pd.read_csv(FOLD_FILE)

    day_df = monitor_df[monitor_df['Date'] == selected_day].copy()
    day_df = day_df.merge(fold_df, on='Site', how='left')
    day_df = day_df.dropna(subset=['Lat', 'Lon', 'Conc'])

    ds = nc.Dataset(CMAQ_FILE, 'r')
    lon_cmaq = ds.variables['lon'][:]
    lat_cmaq = ds.variables['lat'][:]
    pred_pm25 = ds.variables['pred_PM25'][:]
    ds.close()

    date_obj = datetime.strptime(selected_day, '%Y-%m-%d')
    day_idx = (date_obj - datetime(2020, 1, 1)).days
    pred_day = pred_pm25[day_idx]

    cmaq_values = []
    for _, row in day_df.iterrows():
        val = get_cmaq_at_site(row['Lon'], row['Lat'], lon_cmaq, lat_cmaq, pred_day)
        cmaq_values.append(val)
    day_df['CMAQ'] = cmaq_values

    best_r2 = -np.inf
    best_params = None

    omega_values = [0.5, 1.0, 2.0, 5.0, 10.0]
    epsilon_values = [1e-3, 1e-2, 1e-1, 0.5]
    delta_values = [1e-3, 1e-2, 1e-1, 0.5]

    for omega in omega_values:
        for epsilon in epsilon_values:
            for delta in delta_values:
                all_true = []
                all_pred = []

                for fold_id in range(1, 11):
                    train_df = day_df[day_df['fold'] != fold_id].copy()
                    test_df = day_df[day_df['fold'] == fold_id].copy()

                    train_df = train_df.dropna(subset=['Lon', 'Lat', 'CMAQ', 'Conc'])
                    test_df = test_df.dropna(subset=['Lon', 'Lat', 'CMAQ', 'Conc'])

                    if len(test_df) == 0:
                        continue

                    y_train = train_df['Conc'].values
                    m_train = train_df['CMAQ'].values
                    lon_train = train_df['Lon'].values
                    lat_train = train_df['Lat'].values

                    y_test = test_df['Conc'].values
                    m_test = test_df['CMAQ'].values

                    bias_train = bayesian_variational_fusion(y_train, m_train, lon_train, lat_train,
                                                            omega=omega, epsilon=epsilon, delta=delta)

                    test_locations = test_df[['Lon', 'Lat']].values
                    train_locations = train_df[['Lon', 'Lat']].values

                    distances = cdist(test_locations, train_locations)
                    weights = 1.0 / (distances + 1e-6)
                    weights = weights / weights.sum(axis=1, keepdims=True)
                    bias_test = weights @ bias_train

                    pred_test = m_test + bias_test

                    all_true.extend(y_test)
                    all_pred.extend(pred_test)

                if len(all_true) > 0:
                    metrics = compute_metrics(np.array(all_true), np.array(all_pred))
                    if metrics['R2'] > best_r2:
                        best_r2 = metrics['R2']
                        best_params = {'omega': omega, 'epsilon': epsilon, 'delta': delta}
                        print(f"  New best: omega={omega}, epsilon={epsilon}, delta={delta}, R2={best_r2:.4f}")

    print(f"\nBest params: omega={best_params['omega']}, epsilon={best_params['epsilon']}, delta={best_params['delta']}")
    print(f"Best R2: {best_r2:.4f}")
    return best_params


def main():
    sep = "=" * 70
    print(sep)
    print("BayesianVariationalFusion All Stages Validation")
    print(sep)

    # 找最佳参数
    best_params = find_best_params()

    results = {}

    for stage_name, (start, end) in STAGES.items():
        metrics, innovation_pass = run_stage_validation(stage_name, start, end, best_params)
        results[stage_name] = {
            'metrics': metrics,
            '判定': {'innovation_verified': innovation_pass}
        }

    # Save
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(sep)
    print("SUMMARY")
    print(sep)
    passed = 0
    for stage, data in results.items():
        m = data['metrics']
        status = 'VERIFIED' if data['判定']['innovation_verified'] else 'NOT VERIFIED'
        print(f"{stage}: R2={m['R2']:.4f}, RMSE={m['RMSE']:.2f}, MB={m['MB']:.2f} -> {status}")
        if data['判定']['innovation_verified']:
            passed += 1

    print(f"\nTotal: {passed}/4 stages passed")
    print(f"Results saved: {OUTPUT_FILE}")

    return results


if __name__ == '__main__':
    main()