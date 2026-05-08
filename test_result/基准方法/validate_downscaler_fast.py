# -*- coding: utf-8 -*-
"""
Downscaler 多阶段验证脚本 (加速版)
=======================
减少迭代次数以加快验证速度
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

from Code.Downscaler.pm25_downscaler import PM25Downscaler
from Code.Downscaler.common_setting import CommonSetting

# 路径配置
ROOT_DIR = str(get_project_root())
CMAQ_FILE = data_path('test_data/raw/CMAQ/2020_PM25.nc')
MONITOR_FILE = data_path('test_data/raw/Monitor/2020_DailyPM2.5Monitor.csv')
FOLD_FILE = data_path('test_data/fold_split_table_daily.csv')
OUTPUT_FILE = f'{ROOT_DIR}/test_result/基准方法/downscaler_multistage.json'


def compute_metrics(y_true, y_pred):
    """计算评估指标"""
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


def ten_fold_for_day_downscaler(selected_day, setting):
    """
    Downscaler 的十折交叉验证
    """
    # 加载数据
    monitor_df = pd.read_csv(MONITOR_FILE)
    fold_df = pd.read_csv(FOLD_FILE)

    # 筛选日期
    day_df = monitor_df[monitor_df['Date'] == selected_day].copy()
    day_df = day_df.merge(fold_df, on='Site', how='left')
    day_df = day_df.dropna(subset=['Lat', 'Lon', 'Conc'])

    # 如果站点数太少（少于100个），跳过这一天
    if len(day_df) < 100:
        return np.array([]), np.array([])

    # 加载CMAQ全网格数据
    ds = nc.Dataset(CMAQ_FILE, 'r')
    lon_cmaq = ds.variables['lon'][:]
    lat_cmaq = ds.variables['lat'][:]
    pred_pm25 = ds.variables['pred_PM25'][:]
    ds.close()

    # 获取当天的CMAQ数据
    date_obj = datetime.strptime(selected_day, '%Y-%m-%d')
    day_idx = (date_obj - datetime(2020, 1, 1)).days
    if day_idx >= pred_pm25.shape[0]:
        return np.array([]), np.array([])
    cmaq_day = pred_pm25[day_idx]

    # 构建全网格坐标和值
    lon_flat = lon_cmaq.flatten()
    lat_flat = lat_cmaq.flatten()
    cmaq_flat = cmaq_day.flatten()

    # 有效的网格点（非NaN）
    valid_mask = ~np.isnan(cmaq_flat)

    # 空间滤波：只保留监测站周围 buffer 度范围内的网格点
    buffer = 15.0
    lon_min, lon_max = day_df['Lon'].min() - buffer, day_df['Lon'].max() + buffer
    lat_min, lat_max = day_df['Lat'].min() - buffer, day_df['Lat'].max() + buffer

    spatial_mask = valid_mask & (
        (lon_flat >= lon_min) & (lon_flat <= lon_max) &
        (lat_flat >= lat_min) & (lat_flat <= lat_max)
    )

    matrix_latlon_model = np.column_stack([lon_flat[spatial_mask], lat_flat[spatial_mask]])
    matrix_model = cmaq_flat[spatial_mask].reshape(-1, 1)

    all_y_true = []
    all_y_pred = []

    for fold_id in range(1, 11):
        print(f"    Fold {fold_id}...", end=" ", flush=True)
        train_df = day_df[day_df['fold'] != fold_id].copy()
        test_df = day_df[day_df['fold'] == fold_id].copy()

        train_df = train_df.dropna(subset=['Lon', 'Lat', 'Conc'])
        test_df = test_df.dropna(subset=['Lon', 'Lat', 'Conc'])

        if len(test_df) == 0 or len(train_df) == 0:
            print("SKIP")
            continue

        # 准备监测站点数据
        matrix_latlon_monitor = train_df[['Lon', 'Lat']].values
        matrix_monitor = train_df['Conc'].values.reshape(-1, 1)

        # 运行原始 Downscaler
        downscaler = PM25Downscaler(setting)
        result = downscaler.run(
            matrix_latlon_model.astype(float),
            matrix_latlon_monitor.astype(float),
            matrix_model.astype(float),
            matrix_monitor.astype(float),
            seed=42
        )

        if result is None:
            print(f"FAILED - {downscaler.error_msg[:50]}")
            continue

        pred_grid, _ = result

        # 提取验证站点位置的预测值
        test_lon = test_df['Lon'].values
        test_lat = test_df['Lat'].values
        y_true = test_df['Conc'].values

        y_pred = []
        for lon, lat in zip(test_lon, test_lat):
            dist = np.sqrt((matrix_latlon_model[:, 0] - lon)**2 +
                          (matrix_latlon_model[:, 1] - lat)**2)
            nearest_idx = np.argmin(dist)
            y_pred.append(pred_grid[nearest_idx])

        y_pred = np.array(y_pred)

        all_y_true.extend(y_true)
        all_y_pred.extend(y_pred)
        print("OK")

    return np.array(all_y_true), np.array(all_y_pred)


def run_stage_validation(start_date, end_date, setting):
    """运行阶段验证"""
    date_list = []
    current_date = datetime.strptime(start_date, '%Y-%m-%d')
    end_date_obj = datetime.strptime(end_date, '%Y-%m-%d')
    while current_date <= end_date_obj:
        date_list.append(current_date.strftime('%Y-%m-%d'))
        current_date += timedelta(days=1)

    # 并行处理多天
    results = Parallel(n_jobs=2)(
        delayed(ten_fold_for_day_downscaler)(date_str, setting)
        for date_str in date_list
    )

    # 合并所有天的结果
    all_y_true = []
    all_y_pred = []
    day_count = 0
    for y_true, y_pred in results:
        if len(y_true) > 0:
            all_y_true.extend(y_true)
            all_y_pred.extend(y_pred)
            day_count += 1

    print(f"    成功处理 {day_count} 天")

    if len(all_y_true) == 0:
        return {'R2': np.nan, 'MAE': np.nan, 'RMSE': np.nan, 'MB': np.nan}

    metrics = compute_metrics(np.array(all_y_true), np.array(all_y_pred))
    print(f"    R2={metrics['R2']:.4f}, RMSE={metrics['RMSE']:.2f}, MB={metrics['MB']:.2f}")

    return metrics


def main():
    print("="*70)
    print("Original Downscaler Multi-Stage Validation (Accelerated)")
    print("="*70)

    # 创建加速设置：减少迭代次数
    setting = CommonSetting()
    setting.numit = 200   # 原始2500 -> 200
    setting.burn = 100    # 原始500 -> 100
    setting.thin = 1

    stages = {
        'pre_exp': ('2020-01-01', '2020-01-05'),
    }

    results = {}

    for stage_name, (start, end) in stages.items():
        print(f"\n  [{stage_name}] {start} ~ {end}")
        metrics = run_stage_validation(start, end, setting)
        results[stage_name] = metrics

    # 保存结果
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {OUTPUT_FILE}")

    return results


if __name__ == '__main__':
    main()