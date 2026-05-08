# -*- coding: utf-8 -*-
"""
SparseKrig 多阶段验证脚本
=======================
对 SparseKrig 进行多阶段验证

每天十折验证，合并所有天预测计算整体指标
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

from Code.Downscaler.pm25_sparse_krig import SparseKrigCalculator
from Code.Downscaler.common_setting import CommonSetting

# 路径配置
ROOT_DIR = str(get_project_root())
CMAQ_FILE = data_path('test_data/raw/CMAQ/2020_PM25.nc')
MONITOR_FILE = data_path('test_data/raw/Monitor/2020_DailyPM2.5Monitor.csv')
FOLD_FILE = data_path('test_data/fold_split_table_daily.csv')
OUTPUT_FILE = f'{ROOT_DIR}/test_result/基准方法/sparsekrig_multistage.json'


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


def ten_fold_for_day_sparsekrig(selected_day, setting):
    """SparseKrig 的十折交叉验证"""
    monitor_df = pd.read_csv(MONITOR_FILE)
    fold_df = pd.read_csv(FOLD_FILE)

    day_df = monitor_df[monitor_df['Date'] == selected_day].copy()
    day_df = day_df.merge(fold_df, on=['Date', 'Site'], how='left')
    day_df = day_df.dropna(subset=['Lat', 'Lon', 'Conc'])

    # 加载CMAQ
    ds = nc.Dataset(CMAQ_FILE, 'r')
    lon_cmaq = ds.variables['lon'][:]
    lat_cmaq = ds.variables['lat'][:]
    pred_pm25 = ds.variables['pred_PM25'][:]
    ds.close()

    date_obj = datetime.strptime(selected_day, '%Y-%m-%d')
    day_idx = (date_obj - datetime(2020, 1, 1)).days
    cmaq_day = pred_pm25[day_idx]

    lon_flat = lon_cmaq.flatten()
    lat_flat = lat_cmaq.flatten()
    cmaq_flat = cmaq_day.flatten()

    all_y_true = []
    all_y_pred = []

    for fold_id in range(1, 11):
        train_df = day_df[day_df['fold'] != fold_id].copy()
        test_df = day_df[day_df['fold'] == fold_id].copy()

        train_df = train_df.dropna(subset=['Lon', 'Lat', 'Conc'])
        test_df = test_df.dropna(subset=['Lon', 'Lat', 'Conc'])

        if len(test_df) == 0 or len(train_df) == 0:
            continue

        # 空间滤波：只保留监测站周围 buffer 度范围内的网格点
        buffer = 0.5
        lon_min = train_df['Lon'].min() - buffer
        lon_max = train_df['Lon'].max() + buffer
        lat_min = train_df['Lat'].min() - buffer
        lat_max = train_df['Lat'].max() + buffer

        spatial_mask = (
            (lon_flat >= lon_min) & (lon_flat <= lon_max) &
            (lat_flat >= lat_min) & (lat_flat <= lat_max) &
            ~np.isnan(cmaq_flat)
        )

        matrix_latlon_model = np.column_stack([lon_flat[spatial_mask], lat_flat[spatial_mask]])
        matrix_model = cmaq_flat[spatial_mask].reshape(-1, 1)

        matrix_latlon_monitor = train_df[['Lon', 'Lat']].values
        matrix_monitor = train_df[['Conc']].values.reshape(-1, 1)

        # 运行 SparseKrig
        result = SparseKrigCalculator.run(
            matrix_latlon_model.astype(float),
            matrix_latlon_monitor.astype(float),
            matrix_model.astype(float),
            matrix_monitor.astype(float),
            setting,
            seed=42 + fold_id,  # 不同fold用不同seed
            kernel_threshold=0.01
        )

        if result is None:
            print(f"    Fold {fold_id}: FAILED")
            continue

        pred_grid = result[0]

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

    return np.array(all_y_true), np.array(all_y_pred)


def run_stage_validation(start_date, end_date, setting):
    """运行阶段验证（时间维度并行）"""
    # 生成日期列表
    date_list = []
    current_date = datetime.strptime(start_date, '%Y-%m-%d')
    end_date_obj = datetime.strptime(end_date, '%Y-%m-%d')
    while current_date <= end_date_obj:
        date_list.append(current_date.strftime('%Y-%m-%d'))
        current_date += timedelta(days=1)

    # 并行处理多天
    results = Parallel(n_jobs=4)(
        delayed(ten_fold_for_day_sparsekrig)(date_str, setting)
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
    print("SparseKrig Multi-Stage Validation")
    print("="*70)

    # 设置参数
    setting = CommonSetting()
    setting.numit = 100  # 加速测试

    # 完整阶段测试
    stages = {
        'pre_exp': ('2020-01-01', '2020-01-05'),
        'stage1':  ('2020-01-01', '2020-01-31'),
        'stage2':  ('2020-07-01', '2020-07-31'),
        'stage3':  ('2020-12-01', '2020-12-31'),
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