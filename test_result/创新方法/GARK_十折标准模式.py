# -*- coding: utf-8 -*-
"""
GARK 十折交叉验证 - 标准模式
============================
Gradient Anisotropic Residual Kriging

按照十折交叉验证架构文档：
- 训练：9折监测站的CMAQ网格坐标 + 梯度方向
- 预测：对1折站点所在的CMAQ网格坐标预测

创新点:
1. CMAQ 梯度引导的各向异性克里金
2. 沿梯度方向和垂直梯度方向使用不同相关长度
3. 固定参数 (a_min=8.0km, a_max=20.0km, alpha=2.0)

参数 (固定，不学习):
- a_min: 8.0 (km) - 垂直梯度方向相关长度
- a_max: 20.0 (km) - 沿梯度方向相关长度
- alpha: 2.0 - 各向异性指数
- n_neighbor: 12
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
from joblib import Parallel, delayed

ROOT_DIR = str(get_project_root())
CMAQ_FILE = data_path('test_data/raw/CMAQ/2020_PM25.nc')
MONITOR_FILE = data_path('test_data/raw/Monitor/2020_DailyPM2.5Monitor.csv')
FOLD_FILE = data_path('test_data/fold_split_table_daily.csv')
OUTPUT_FILE = f'{ROOT_DIR}/Innovation/success/GARK/GARK_all_stages.json'

# VNA baseline
BASELINE = {
    'pre_exp': {'R2': 0.8907, 'RMSE': 16.68, 'MB': 0.70},
    'stage1':  {'R2': 0.9034, 'RMSE': 16.48, 'MB': 0.50},
    'stage2':  {'R2': 0.8408, 'RMSE': 5.05, 'MB': 0.05},
    'stage3':  {'R2': 0.9031, 'RMSE': 12.20, 'MB': 0.42},
}

STAGES = {
    'pre_exp': ('2020-01-01', '2020-01-05'),
    'stage1':  ('2020-01-01', '2020-01-31'),
    'stage2':  ('2020-07-01', '2020-07-31'),
    'stage3':  ('2020-12-01', '2020-12-31'),
}

# GARK parameters (fixed)
A_MIN = 8.0  # km, perpendicular to gradient
A_MAX = 20.0  # km, along gradient
ALPHA = 2.0
N_NEIGHBOR = 12


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


def compute_gradient_direction(lon, lat, lon_grid, lat_grid, pm25_grid):
    """计算CMAQ梯度方向"""
    dist = np.sqrt((lon_grid - lon)**2 + (lat_grid - lat)**2)
    idx = np.argmin(dist)
    ny, nx = lon_grid.shape
    row, col = idx // nx, idx % nx

    row = np.clip(row, 1, ny - 2)
    col = np.clip(col, 1, nx - 2)

    pm_xp = pm25_grid[row, min(col + 1, nx - 1)]
    pm_xm = pm25_grid[row, max(col - 1, 0)]
    pm_yp = pm25_grid[min(row + 1, ny - 1), col]
    pm_ym = pm25_grid[max(row - 1, 0), col]

    lon_step = np.mean(np.diff(lon_grid, axis=1))
    lat_step = np.mean(np.diff(lat_grid, axis=0))

    dpm_dlon = (pm_xp - pm_xm) / (2 * lon_step) if lon_step != 0 else 0
    dpm_dlat = (pm_yp - pm_ym) / (2 * lat_step) if lat_step != 0 else 0

    gradient_magnitude = np.sqrt(dpm_dlon**2 + dpm_dlat**2)
    gradient_direction = np.arctan2(dpm_dlon, dpm_dlat)

    return gradient_magnitude, gradient_direction


def anisotropic_distance(x1, x2, direction, a_min, a_max, alpha=2.0):
    """计算各向异性距离"""
    dx = x1[0] - x2[0]
    dy = x1[1] - x2[1]

    cos_d = np.cos(direction)
    sin_d = np.sin(direction)

    d_along = dx * cos_d + dy * sin_d
    d_perp = -dx * sin_d + dy * cos_d

    dist = np.sqrt((d_along / a_max)**2 + (d_perp / a_min)**2)

    return dist


def gark_predict_vectorized(x_obs, residual_obs, x_pred, direction_pred, a_min=8.0, a_max=20.0, alpha=2.0, n_neighbor=12):
    """GARK预测 - 向量化版本"""
    n_pred = x_pred.shape[0]
    n_obs = x_obs.shape[0]

    # 广播：x_pred[:, np.newaxis, :] - x_obs[np.newaxis, :, :]
    # 结果形状 (n_pred, n_obs, 2)
    diff = x_pred[:, np.newaxis, :] - x_obs[np.newaxis, :, :]

    # 测试点梯度方向
    cos_d = np.cos(direction_pred)[:, np.newaxis]  # (n_pred, 1)
    sin_d = np.sin(direction_pred)[:, np.newaxis]  # (n_pred, 1)

    # 旋转到梯度主轴坐标系
    d_along = diff[:, :, 0] * cos_d + diff[:, :, 1] * sin_d  # (n_pred, n_obs)
    d_perp = -diff[:, :, 0] * sin_d + diff[:, :, 1] * cos_d  # (n_pred, n_obs)

    # 各向异性距离
    dist_aniso = np.sqrt((d_along / a_max)**2 + (d_perp / a_min)**2)  # (n_pred, n_obs)

    # 对每个预测点，选择最近的邻居
    if n_neighbor < n_obs:
        # argpartition 更高效地选择前 n_neighbor 个
        idx = np.argpartition(dist_aniso, n_neighbor, axis=1)[:, :n_neighbor]  # (n_pred, n_neighbor)
        # 获取这些索引对应的距离和残差值
        row_idx = np.arange(n_pred)[:, np.newaxis]  # (n_pred, 1)
        dists_k = dist_aniso[row_idx, idx]  # (n_pred, n_neighbor)
        values_k = residual_obs[idx]  # (n_pred, n_neighbor)
    else:
        dists_k = dist_aniso
        values_k = np.tile(residual_obs, (n_pred, 1))

    # 避免除零
    dists_k = np.maximum(dists_k, 1e-10)

    # 高斯核权重
    sigma = a_min / 3.0
    weights = np.exp(-0.5 * (dists_k / sigma)**2)  # (n_pred, n_neighbor)
    weights = weights / weights.sum(axis=1, keepdims=True)  # 归一化

    # 加权平均
    pred_values = np.sum(weights * values_k, axis=1)  # (n_pred,)

    return pred_values


def gark_predict(x_obs, residual_obs, x_pred, direction_pred, a_min=8.0, a_max=20.0, alpha=2.0, n_neighbor=12):
    """GARK预测 - 调用向量化版本"""
    return gark_predict_vectorized(x_obs, residual_obs, x_pred, direction_pred, a_min, a_max, alpha, n_neighbor)


def ten_fold_gark(selected_day):
    """GARK 十折验证 - 标准模式"""
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

    # 获取CMAQ值和梯度方向
    cmaq_values = []
    grad_dirs = []
    for _, row in day_df.iterrows():
        val = get_cmaq_at_site(row['Lon'], row['Lat'], lon_cmaq, lat_cmaq, cmaq_day)
        _, direction = compute_gradient_direction(row['Lon'], row['Lat'], lon_cmaq, lat_cmaq, cmaq_day)
        cmaq_values.append(val)
        grad_dirs.append(direction)
    day_df['CMAQ'] = cmaq_values
    day_df['gradient_direction'] = grad_dirs

    all_y_true = []
    all_y_pred = []

    for fold_id in range(1, 11):
        train_df = day_df[day_df['fold'] != fold_id].copy()
        test_df = day_df[day_df['fold'] == fold_id].copy()

        train_df = train_df.dropna(subset=['Lon', 'Lat', 'CMAQ', 'Conc', 'gradient_direction'])
        test_df = test_df.dropna(subset=['Lon', 'Lat', 'CMAQ', 'Conc', 'gradient_direction'])

        if len(test_df) == 0 or len(train_df) == 0:
            continue

        X_train = train_df[['Lon', 'Lat']].values
        X_test = test_df[['Lon', 'Lat']].values
        residual_train = train_df['Conc'].values - train_df['CMAQ'].values
        direction_train = train_df['gradient_direction'].values
        direction_test = test_df['gradient_direction'].values
        y_test = test_df['Conc'].values
        cmaq_test = test_df['CMAQ'].values

        # GARK预测
        gark_residual_pred = gark_predict(
            X_train, residual_train, X_test, direction_test,
            A_MIN, A_MAX, ALPHA, N_NEIGHBOR
        )

        # 融合预测
        gark_pred = cmaq_test + gark_residual_pred

        all_y_true.extend(y_test)
        all_y_pred.extend(gark_pred)

    return np.array(all_y_true), np.array(all_y_pred)


def run_stage_validation(stage_name, start_date, end_date):
    sep = "=" * 70
    print(sep)
    print(f"GARK Stage: {stage_name} ({start_date} ~ {end_date})")
    print(f"Parameters: a_min={A_MIN}, a_max={A_MAX}, alpha={ALPHA}, n_neighbor={N_NEIGHBOR}")
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

    # 并行处理
    n_jobs = min(8, len(date_list))
    results = Parallel(n_jobs=n_jobs)(
        delayed(ten_fold_gark)(date_str)
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


def main():
    sep = "=" * 70
    print(sep)
    print("GARK All Stages - Gradient Anisotropic Residual Kriging")
    print(sep)

    results = {}
    all_pass = True

    for stage_name, (start, end) in STAGES.items():
        metrics, innovation_pass = run_stage_validation(stage_name, start, end)
        results[stage_name] = {'metrics': metrics, '判定': {'innovation_verified': innovation_pass}}
        if not innovation_pass:
            all_pass = False

    # 保存结果
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(sep)
    print("SUMMARY")
    print(sep)
    for stage, data in results.items():
        m = data['metrics']
        status = 'VERIFIED' if data['判定']['innovation_verified'] else 'NOT VERIFIED'
        print(f"{stage}: R2={m['R2']:.4f}, RMSE={m['RMSE']:.2f}, MB={m['MB']:.2f} -> {status}")

    print(f"\nAll stages passed: {all_pass}")
    print(f"Results saved: {OUTPUT_FILE}")

    return results


if __name__ == '__main__':
    main()
