# -*- coding: utf-8 -*-
"""
基准方法多阶段验证脚本
=====================
验证 VNA, eVNA, aVNA, Downscaler 在各阶段的真实表现

验证方式：
- VNA/eVNA/aVNA（标准模式）：直接对验证站点的CMAQ网格坐标预测
- Downscaler（特例模式）：必须预测全网格再提取

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

from Code.VNAeVNAaVNA.nna_methods import NNA
from common_setting import CommonSetting
from pm25_downscaler import PM25Downscaler

# 路径配置
ROOT_DIR = str(get_project_root())
CMAQ_FILE = data_path('test_data/raw/CMAQ/2020_PM25.nc')
MONITOR_FILE = data_path('test_data/raw/Monitor/2020_DailyPM2.5Monitor.csv')
FOLD_FILE = data_path('test_data/fold_split_table_daily.csv')
OUTPUT_FILE = f'{ROOT_DIR}/test_result/基准方法/benchmark_multistage.json'


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


def get_cmaq_at_site(lon, lat, lon_grid, lat_grid, pm25_grid):
    """获取站点位置的CMAQ值（最近邻）"""
    dist = np.sqrt((lon_grid - lon)**2 + (lat_grid - lat)**2)
    idx = np.argmin(dist)
    ny, nx = lon_grid.shape
    row, col = idx // nx, idx % nx
    return pm25_grid[row, col]


def get_cmaq_grid_coord(lon, lat, lon_grid, lat_grid):
    """获取站点对应的CMAQ网格坐标（最近邻）"""
    dist = np.sqrt((lon_grid - lon)**2 + (lat_grid - lat)**2)
    idx = np.argmin(dist)
    ny, nx = lon_grid.shape
    row, col = idx // nx, idx % nx
    return lon_grid[row, col], lat_grid[row, col]


def ten_fold_for_day_vna(method_name, selected_day):
    """
    VNA/eVNA/aVNA 的十折交叉验证
    标准模式：直接对验证站点的CMAQ网格坐标预测
    """
    # 加载数据
    monitor_df = pd.read_csv(MONITOR_FILE)
    fold_df = pd.read_csv(FOLD_FILE)

    # 筛选日期
    day_df = monitor_df[monitor_df['Date'] == selected_day].copy()
    day_df = day_df.merge(fold_df, on=['Date', 'Site'], how='left')
    day_df = day_df.dropna(subset=['Lat', 'Lon', 'Conc'])

    # 如果站点数太少（少于100个），跳过这一天
    if len(day_df) < 100:
        return np.array([]), np.array([])

    # 加载CMAQ
    ds = nc.Dataset(CMAQ_FILE, 'r')
    lon_cmaq = ds.variables['lon'][:]
    lat_cmaq = ds.variables['lat'][:]
    pred_pm25 = ds.variables['pred_PM25'][:]
    ds.close()

    # 获取当天的CMAQ数据
    date_obj = datetime.strptime(selected_day, '%Y-%m-%d')
    day_idx = (date_obj - datetime(2020, 1, 1)).days
    # 检查 day_idx 是否越界（CMAQ数据只有365天，2020是闰年12月31日会越界）
    if day_idx >= pred_pm25.shape[0]:
        return np.array([]), np.array([])
    cmaq_day = pred_pm25[day_idx]

    # 提取站点CMAQ值
    cmaq_values = []
    for _, row in day_df.iterrows():
        val = get_cmaq_at_site(row['Lon'], row['Lat'], lon_cmaq, lat_cmaq, cmaq_day)
        cmaq_values.append(val)
    day_df['CMAQ'] = cmaq_values

    # 十折验证
    all_y_true = []
    all_y_pred = []

    for fold_id in range(1, 11):
        train_df = day_df[day_df['fold'] != fold_id].copy()
        test_df = day_df[day_df['fold'] == fold_id].copy()

        train_df = train_df.dropna(subset=['Lon', 'Lat', 'CMAQ', 'Conc'])
        test_df = test_df.dropna(subset=['Lon', 'Lat', 'CMAQ', 'Conc'])

        if len(test_df) == 0 or len(train_df) == 0:
            continue

        # 准备训练数据
        train_df['x'] = train_df['Lon']
        train_df['y'] = train_df['Lat']
        train_df['mod'] = train_df['CMAQ']
        train_df['bias'] = train_df['Conc'] - train_df['CMAQ']
        train_df['r_n'] = train_df['Conc'] / train_df['CMAQ']

        if method_name == 'CMAQ':
            # CMAQ直接用模型值
            y_pred = test_df['CMAQ'].values
        else:
            nn = NNA(method='voronoi', k=30, power=-2)
            nn.fit(
                train_df[['x', 'y']],
                train_df[['Conc', 'mod', 'bias', 'r_n']]
            )

            # 【关键区别】标准模式：直接对验证站点的CMAQ网格坐标预测
            # 先获取测试站点对应的CMAQ网格坐标，然后用该坐标进行预测
            cmaq_coords = []
            for _, row in test_df.iterrows():
                cmaq_lon, cmaq_lat = get_cmaq_grid_coord(
                    row['Lon'], row['Lat'], lon_cmaq, lat_cmaq
                )
                cmaq_coords.append([cmaq_lon, cmaq_lat])
            X_test = np.array(cmaq_coords)

            # NNA.predict 返回的是在给定CMAQ网格坐标上的插值结果（不使用空间并行，数据量小）
            zdf = nn.predict(X_test)
            vna_pred = zdf[:, 0]  # VNA预测值
            vna_bias_pred = zdf[:, 2]  # bias预测值
            vna_rn_pred = zdf[:, 3]  # rn预测值

            # 不裁剪，让残差自由预测
            if method_name == 'VNA':
                y_pred = vna_pred
            elif method_name == 'aVNA':
                y_pred = test_df['CMAQ'].values + vna_bias_pred
            elif method_name == 'eVNA':
                y_pred = test_df['CMAQ'].values * vna_rn_pred

        all_y_true.extend(test_df['Conc'].values)
        all_y_pred.extend(y_pred)

    return np.array(all_y_true), np.array(all_y_pred)


def ten_fold_for_day_downscaler(selected_day):
    """
    Downscaler 的十折交叉验证
    特例模式：必须预测全网格，然后提取验证站点位置的值
    """
    # 加载数据
    monitor_df = pd.read_csv(MONITOR_FILE)
    fold_df = pd.read_csv(FOLD_FILE)

    # 筛选日期
    day_df = monitor_df[monitor_df['Date'] == selected_day].copy()
    day_df = day_df.merge(fold_df, on=['Date', 'Site'], how='left')
    day_df = day_df.dropna(subset=['Lat', 'Lon', 'Conc'])

    # 加载CMAQ全网格数据
    ds = nc.Dataset(CMAQ_FILE, 'r')
    lon_cmaq = ds.variables['lon'][:]
    lat_cmaq = ds.variables['lat'][:]
    pred_pm25 = ds.variables['pred_PM25'][:]
    ds.close()

    # 获取当天的CMAQ数据
    date_obj = datetime.strptime(selected_day, '%Y-%m-%d')
    day_idx = (date_obj - datetime(2020, 1, 1)).days
    cmaq_day = pred_pm25[day_idx]

    # 构建全网格坐标和值
    ny, nx = lon_cmaq.shape
    lon_flat = lon_cmaq.flatten()
    lat_flat = lat_cmaq.flatten()
    cmaq_flat = cmaq_day.flatten()

    # 有效的网格点（非NaN）
    valid_mask = ~np.isnan(cmaq_flat)

    # 空间滤波：只保留监测站周围 buffer 度范围内的网格点
    # 这不会显著影响结果，因为超出相关范围的残差插值贡献趋近于0
    buffer = 15.0  # 度，大约覆盖有效的空间相关性范围
    # 使用当天所有站点的边界
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
        train_df = day_df[day_df['fold'] != fold_id].copy()
        test_df = day_df[day_df['fold'] == fold_id].copy()

        train_df = train_df.dropna(subset=['Lon', 'Lat', 'Conc'])
        test_df = test_df.dropna(subset=['Lon', 'Lat', 'Conc'])

        if len(test_df) == 0 or len(train_df) == 0:
            continue

        # 准备监测站点数据
        matrix_latlon_monitor = train_df[['Lon', 'Lat']].values
        matrix_monitor = train_df['Conc'].values.reshape(-1, 1)

        # 运行 Downscaler
        setting = CommonSetting()
        downscaler = PM25Downscaler(setting)
        result = downscaler.run(
            matrix_latlon_model.astype(float),
            matrix_latlon_monitor.astype(float),
            matrix_model.astype(float),
            matrix_monitor.astype(float),
            seed=42
        )

        if result is None:
            print(f"    Downscaler failed: {downscaler.error_msg}")
            continue

        pred_grid, _ = result  # pred_grid 是全网格预测，与 matrix_latlon_model 对应

        # 提取验证站点位置的预测值
        test_lon = test_df['Lon'].values
        test_lat = test_df['Lat'].values
        y_true = test_df['Conc'].values

        # 找到每个验证站点在 matrix_latlon_model 中最近的索引
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


def ten_fold_for_day(method_name, selected_day):
    """
    根据方法类型调用对应的十折验证
    """
    if method_name == 'Downscaler':
        return ten_fold_for_day_downscaler(selected_day)
    else:
        return ten_fold_for_day_vna(method_name, selected_day)


def run_stage_validation(method_name, start_date, end_date, n_jobs=4):
    """
    运行阶段验证：每天十折，合并所有天预测计算整体指标
    时间维度并行：同时处理多天
    """
    # 生成日期列表
    date_list = []
    current_date = datetime.strptime(start_date, '%Y-%m-%d')
    end_date_obj = datetime.strptime(end_date, '%Y-%m-%d')
    while current_date <= end_date_obj:
        date_list.append(current_date.strftime('%Y-%m-%d'))
        current_date += timedelta(days=1)

    # 并行处理多天
    results = Parallel(n_jobs=n_jobs)(
        delayed(ten_fold_for_day)(method_name, date_str)
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

    print(f"    {method_name}: 成功处理 {day_count} 天")

    if len(all_y_true) == 0:
        return {'R2': np.nan, 'MAE': np.nan, 'RMSE': np.nan, 'MB': np.nan}

    metrics = compute_metrics(np.array(all_y_true), np.array(all_y_pred))
    print(f"    {method_name}: R2={metrics['R2']:.4f}, RMSE={metrics['RMSE']:.2f}, MB={metrics['MB']:.2f}")

    return metrics


def main():
    print("="*70)
    print("Baseline Methods Multi-Stage Validation")
    print("="*70)

    # 定义阶段
    stages = {
        'pre_exp': ('2020-01-01', '2020-01-05'),
        'stage1':  ('2020-01-01', '2020-01-31'),
        'stage2':  ('2020-07-01', '2020-07-31'),
        'stage3':  ('2020-12-01', '2020-12-31'),
    }

    # 基准方法 (Downscaler单独脚本)
    methods = ['CMAQ', 'VNA', 'aVNA', 'eVNA']

    results = {}

    for method in methods:
        print(f"\n{'='*50}")
        print(f"Method: {method}")
        print(f"{'='*50}")

        results[method] = {}

        for stage_name, (start, end) in stages.items():
            print(f"\n  [{stage_name}] {start} ~ {end}")
            metrics = run_stage_validation(method, start, end)
            results[method][stage_name] = metrics

    # 保存结果
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*70}")
    print(f"Results saved to: {OUTPUT_FILE}")
    print(f"{'='*70}")

    # 打印汇总表
    print("\nSummary:")
    print("-" * 80)
    header = f"{'Method':<12} {'Stage':<10} {'R2':<10} {'RMSE':<10} {'MB':<10}"
    print(header)
    print("-" * 80)

    for method in methods:
        for stage_name in ['pre_exp', 'stage1', 'stage2', 'stage3']:
            m = results[method][stage_name]
            r2 = f"{m['R2']:.4f}" if not np.isnan(m['R2']) else "N/A"
            rmse = f"{m['RMSE']:.2f}" if not np.isnan(m['RMSE']) else "N/A"
            mb = f"{m['MB']:.2f}" if not np.isnan(m['MB']) else "N/A"
            print(f"{method:<12} {stage_name:<10} {r2:<10} {rmse:<10} {mb:<10}")

    # 找出每个阶段的最优基准
    print("\nOptimal Baseline per Stage (excluding CMAQ):")
    print("-" * 80)
    for stage_name in ['pre_exp', 'stage1', 'stage2', 'stage3']:
        best_method = None
        best_r2 = -np.inf
        for method in ['VNA', 'aVNA', 'eVNA']:  # 排除CMAQ和Downscaler
            r2 = results[method][stage_name]['R2']
            if not np.isnan(r2) and r2 > best_r2:
                best_r2 = r2
                best_method = method
        if best_method:
            print(f"{stage_name}: {best_method} (R2={best_r2:.4f})")
        else:
            print(f"{stage_name}: N/A")

    return results


if __name__ == '__main__':
    main()
