# -*- coding: utf-8 -*-
"""
RobustRK 十折交叉验证 - 标准模式（方式2b）
==========================================
训练：CMAQ网格坐标（不聚合）
预测：1折站点所在的CMAQ网格坐标
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

ROOT_DIR = str(get_project_root())
CMAQ_FILE = data_path('test_data/raw/CMAQ/2020_PM25.nc')
MONITOR_FILE = data_path('test_data/raw/Monitor/2020_DailyPM2.5Monitor.csv')
FOLD_FILE = data_path('test_data/fold_split_table_daily.csv')
OUTPUT_DIR = f'{ROOT_DIR}/Innovation/success/RobustRK'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# VNA baseline
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


def get_cmaq_grid_coord(lon, lat, lon_grid, lat_grid):
    """获取站点对应的CMAQ网格坐标"""
    dist = np.sqrt((lon_grid - lon)**2 + (lat_grid - lat)**2)
    idx = np.argmin(dist)
    ny, nx = lon_grid.shape
    row, col = idx // nx, idx % nx
    return lon_grid[row, col], lat_grid[row, col]


def ten_fold_robust_rk(selected_day, poly_degree=2):
    """标准模式十折验证 - 方式2b：训练用CMAQ网格坐标（不聚合）"""
    monitor_df = pd.read_csv(MONITOR_FILE)
    fold_df = pd.read_csv(FOLD_FILE)

    day_df = monitor_df[monitor_df['Date'] == selected_day].copy()
    day_df = day_df.merge(fold_df, on=['Date', 'Site'], how='left')
    day_df = day_df.dropna(subset=['Lat', 'Lon', 'Conc'])

    if len(day_df) < 100:
        return np.array([]), np.array([]), np.array([])

    ds = nc.Dataset(CMAQ_FILE, 'r')
    lon_cmaq = ds.variables['lon'][:]
    lat_cmaq = ds.variables['lat'][:]
    pred_pm25 = ds.variables['pred_PM25'][:]
    ds.close()

    date_obj = datetime.strptime(selected_day, '%Y-%m-%d')
    day_idx = (date_obj - datetime(2020, 1, 1)).days
    if day_idx >= pred_pm25.shape[0]:
        return np.array([]), np.array([]), np.array([])
    cmaq_day = pred_pm25[day_idx]

    # 获取每个站点的CMAQ值
    cmaq_values = []
    for _, row in day_df.iterrows():
        val = get_cmaq_at_site(row['Lon'], row['Lat'], lon_cmaq, lat_cmaq, cmaq_day)
        cmaq_values.append(val)
    day_df['CMAQ'] = cmaq_values

    # GPR核函数
    kernel = (ConstantKernel(10.0, (1e-2, 1e3)) *
              RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) +
              WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e1)))

    all_y_true = []
    all_y_pred_ols = []
    all_y_pred_huber = []

    for fold_id in range(1, 11):
        train_df = day_df[day_df['fold'] != fold_id].copy()
        test_df = day_df[day_df['fold'] == fold_id].copy()

        train_df = train_df.dropna(subset=['Lon', 'Lat', 'CMAQ', 'Conc'])
        test_df = test_df.dropna(subset=['Lon', 'Lat', 'CMAQ', 'Conc'])

        if len(test_df) == 0 or len(train_df) == 0:
            continue

        y_train = train_df['Conc'].values
        m_train = train_df['CMAQ'].values

        # 多项式校正
        poly = PolynomialFeatures(degree=poly_degree, include_bias=False)
        m_train_poly = poly.fit_transform(m_train.reshape(-1, 1))

        # OLS多项式
        ols = LinearRegression()
        ols.fit(m_train_poly, y_train)
        residual_ols = y_train - ols.predict(m_train_poly)

        # Huber稳健多项式
        huber = HuberRegressor(epsilon=1.35, max_iter=1000)
        huber.fit(m_train_poly, y_train)
        residual_huber = y_train - huber.predict(m_train_poly)

        # ========== GPR训练：方式2b（不聚合）==========
        train_cmaq_coords = []
        for _, row in train_df.iterrows():
            cmaq_lon, cmaq_lat = get_cmaq_grid_coord(row['Lon'], row['Lat'], lon_cmaq, lat_cmaq)
            train_cmaq_coords.append([cmaq_lon, cmaq_lat])
        X_train = np.array(train_cmaq_coords)
        # =============================================

        gpr_ols = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, alpha=0.1, normalize_y=True)
        gpr_ols.fit(X_train, residual_ols)

        gpr_huber = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, alpha=0.1, normalize_y=True)
        gpr_huber.fit(X_train, residual_huber)

        # 测试：获取验证站点所在的CMAQ网格坐标
        test_cmaq_coords = []
        for _, row in test_df.iterrows():
            cmaq_lon, cmaq_lat = get_cmaq_grid_coord(row['Lon'], row['Lat'], lon_cmaq, lat_cmaq)
            test_cmaq_coords.append([cmaq_lon, cmaq_lat])
        X_test = np.array(test_cmaq_coords)

        y_test = test_df['Conc'].values
        m_test = test_df['CMAQ'].values

        # GPR预测
        gpr_ols_pred = gpr_ols.predict(X_test)
        gpr_huber_pred = gpr_huber.predict(X_test)

        # 融合预测
        m_test_poly = poly.transform(m_test.reshape(-1, 1))
        pred_ols_test = ols.predict(m_test_poly)
        pred_huber_test = huber.predict(m_test_poly)

        rk_ols_pred = pred_ols_test + gpr_ols_pred
        rk_huber_pred = pred_huber_test + gpr_huber_pred

        all_y_true.extend(y_test)
        all_y_pred_ols.extend(rk_ols_pred)
        all_y_pred_huber.extend(rk_huber_pred)

    return np.array(all_y_true), np.array(all_y_pred_ols), np.array(all_y_pred_huber)


def run_stage_validation(stage_name, start_date, end_date, poly_degree=2):
    sep = "=" * 70
    print(sep)
    print(f"RobustRK Stage: {stage_name} ({start_date} ~ {end_date})")
    print(sep)

    base = BASELINE[stage_name]
    threshold_r2 = base['R2']
    print(f"VNA Baseline: R2={base['R2']:.4f}, RMSE={base['RMSE']:.2f}, MB={base['MB']:.2f}")
    print(f"Threshold: R2>{threshold_r2:.4f}, RMSE<={base['RMSE']}, |MB|<={abs(base['MB'])}")

    date_list = []
    current_date = datetime.strptime(start_date, '%Y-%m-%d')
    end_date_obj = datetime.strptime(end_date, '%Y-%m-%d')
    while current_date <= end_date_obj:
        date_list.append(current_date.strftime('%Y-%m-%d'))
        current_date += timedelta(days=1)

    print(f"Days: {len(date_list)}")

    n_jobs = 4  # 限制并行数避免内存崩溃
    results = Parallel(n_jobs=n_jobs)(
        delayed(ten_fold_robust_rk)(date_str, poly_degree)
        for date_str in date_list
    )

    all_y_true = []
    all_y_pred_ols = []
    all_y_pred_huber = []
    day_count = 0
    for y_true, y_pred_ols, y_pred_huber in results:
        if len(y_true) > 0:
            all_y_true.extend(y_true)
            all_y_pred_ols.extend(y_pred_ols)
            all_y_pred_huber.extend(y_pred_huber)
            day_count += 1

    print(f"Processed: {day_count} days, {len(all_y_true)} predictions")

    if len(all_y_true) == 0:
        return {'R2_ols': np.nan, 'R2_huber': np.nan}, {'ols': False, 'huber': False}

    metrics_ols = compute_metrics(np.array(all_y_true), np.array(all_y_pred_ols))
    metrics_huber = compute_metrics(np.array(all_y_true), np.array(all_y_pred_huber))

    r2_pass_ols = metrics_ols['R2'] > threshold_r2
    rmse_pass_ols = metrics_ols['RMSE'] <= base['RMSE']
    mb_pass_ols = abs(metrics_ols['MB']) <= abs(base['MB'])
    innovation_pass_ols = r2_pass_ols and rmse_pass_ols and mb_pass_ols

    r2_pass_huber = metrics_huber['R2'] > threshold_r2
    rmse_pass_huber = metrics_huber['RMSE'] <= base['RMSE']
    mb_pass_huber = abs(metrics_huber['MB']) <= abs(base['MB'])
    innovation_pass_huber = r2_pass_huber and rmse_pass_huber and mb_pass_huber

    print(f"\nRK_OLS_Poly: R2={metrics_ols['R2']:.4f}, RMSE={metrics_ols['RMSE']:.2f}, MB={metrics_ols['MB']:.2f}")
    print(f"  Check: R2>{threshold_r2:.4f}? {'PASS' if r2_pass_ols else 'FAIL'} | RMSE<={base['RMSE']}? {'PASS' if rmse_pass_ols else 'FAIL'} | |MB|<={abs(base['MB'])}? {'PASS' if mb_pass_ols else 'FAIL'}")
    print(f"  Innovation: {'VERIFIED' if innovation_pass_ols else 'NOT VERIFIED'}")

    print(f"\nRK_Huber_Poly: R2={metrics_huber['R2']:.4f}, RMSE={metrics_huber['RMSE']:.2f}, MB={metrics_huber['MB']:.2f}")
    print(f"  Check: R2>{threshold_r2:.4f}? {'PASS' if r2_pass_huber else 'FAIL'} | RMSE<={base['RMSE']}? {'PASS' if rmse_pass_huber else 'FAIL'} | |MB|<={abs(base['MB'])}? {'PASS' if mb_pass_huber else 'FAIL'}")
    print(f"  Innovation: {'VERIFIED' if innovation_pass_huber else 'NOT VERIFIED'}")

    return {
        'ols': metrics_ols,
        'huber': metrics_huber
    }, {
        'ols': innovation_pass_ols,
        'huber': innovation_pass_huber
    }


def main():
    sep = "=" * 70
    print(sep)
    print("RobustRK All Stages - 标准模式（方式2b）")
    print("GPR训练: CMAQ网格坐标(不聚合) | 预测: CMAQ网格坐标")
    print(sep)

    stages = {
        'pre_exp':  ('2020-01-01', '2020-01-05'),
        'stage1':   ('2020-01-01', '2020-01-31'),
        'stage2':   ('2020-07-01', '2020-07-31'),
        'stage3':   ('2020-12-01', '2020-12-31'),
    }

    results = {}
    all_pass_ols = True
    all_pass_huber = True

    for stage_name, (start, end) in stages.items():
        metrics, pass_info = run_stage_validation(stage_name, start, end)
        results[stage_name] = {
            'metrics': metrics,
            '判定': {
                'ols_innovation_verified': pass_info['ols'],
                'huber_innovation_verified': pass_info['huber']
            }
        }
        if not pass_info['ols']:
            all_pass_ols = False
        if not pass_info['huber']:
            all_pass_huber = False

    # 保存结果
    output_file = f'{OUTPUT_DIR}/RobustRK_all_stages.json'
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(sep)
    print("SUMMARY")
    print(sep)
    for stage, data in results.items():
        m_ols = data['metrics']['ols']
        m_huber = data['metrics']['huber']
        status_ols = 'VERIFIED' if data['判定']['ols_innovation_verified'] else 'NOT VERIFIED'
        status_huber = 'VERIFIED' if data['判定']['huber_innovation_verified'] else 'NOT VERIFIED'
        print(f"{stage}:")
        print(f"  OLS:   R2={m_ols['R2']:.4f}, RMSE={m_ols['RMSE']:.2f}, MB={m_ols['MB']:.2f} -> {status_ols}")
        print(f"  Huber: R2={m_huber['R2']:.4f}, RMSE={m_huber['RMSE']:.2f}, MB={m_huber['MB']:.2f} -> {status_huber}")

    print(f"\nOLS all stages passed: {all_pass_ols}")
    print(f"Huber all stages passed: {all_pass_huber}")
    print(f"Results saved: {output_file}")

    return results


if __name__ == '__main__':
    main()