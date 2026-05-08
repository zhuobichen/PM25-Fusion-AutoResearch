# -*- coding: utf-8 -*-
"""
通用多阶段验证脚本
对非Ensemble方法进行完整多阶段验证
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from shared.paths import get_project_root, data_path
import json
import importlib
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import netCDF4 as nc
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from joblib import Parallel, delayed

ROOT_DIR = str(get_project_root())
CMAQ_FILE = data_path('test_data/raw/CMAQ/2020_PM25.nc')
MONITOR_FILE = data_path('test_data/raw/Monitor/2020_DailyPM2.5Monitor.csv')
FOLD_FILE = data_path('test_data/fold_split_table_daily.csv')
OUTPUT_DIR = f'{ROOT_DIR}/Innovation/failed'  # 默认存失败

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

# 需要跳过的Ensemble方法
ENSEMBLE_KEYWORDS = ['Ensemble', 'Stacking', 'Super', 'Extreme', 'Ultimate',
                     'Adaptive', 'LogRatio', 'MultiLevel', 'Feature', 'Triple']


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


def is_ensemble_method(method_name):
    """检查是否是Ensemble方法（应跳过）"""
    for kw in ENSEMBLE_KEYWORDS:
        if kw in method_name:
            return True
    return False


def validate_method_all_stages(method_name, method_module):
    """
    对单个方法进行全阶段验证

    Args:
        method_name: 方法名（用于显示）
        method_module: 包含predict方法的模块

    Returns:
        dict: 各阶段验证结果
    """
    if is_ensemble_method(method_name):
        print(f"[跳过] {method_name} 是Ensemble方法")
        return None

    print(f"\n{'='*60}")
    print(f"验证方法: {method_name}")
    print(f"{'='*60}")

    results = {}

    # 尝试获取predict方法
    if not hasattr(method_module, 'predict'):
        print(f"[错误] {method_name} 没有predict方法")
        return None

    predict_func = method_module.predict

    # 加载数据
    monitor_df = pd.read_csv(MONITOR_FILE)
    fold_df = pd.read_csv(FOLD_FILE)

    ds = nc.Dataset(CMAQ_FILE, 'r')
    lon_cmaq = ds.variables['lon'][:]
    lat_cmaq = ds.variables['lat'][:]
    pred_pm25 = ds.variables['pred_PM25'][:]
    ds.close()

    for stage_name, (start_date, end_date) in STAGES.items():
        print(f"\n--- {stage_name} ({start_date} ~ {end_date}) ---")

        base = BASELINE[stage_name]
        threshold_r2 = base['R2']  # > VNA to pass

        date_list = []
        current_date = datetime.strptime(start_date, '%Y-%m-%d')
        end_date_obj = datetime.strptime(end_date, '%Y-%m-%d')
        while current_date <= end_date_obj:
            date_list.append(current_date.strftime('%Y-%m-%d'))
            current_date += timedelta(days=1)

        print(f"天数: {len(date_list)}")

        all_y_true = []
        all_y_pred = []
        day_count = 0

        for date_str in date_list:
            try:
                date_obj = datetime.strptime(date_str, '%Y-%m-%d')
                day_idx = (date_obj - datetime(2020, 1, 1)).days
                if day_idx >= pred_pm25.shape[0] or day_idx < 0:
                    continue

                day_df = monitor_df[monitor_df['Date'] == date_str].copy()
                day_df = day_df.merge(fold_df, on='Site', how='left')
                day_df = day_df.dropna(subset=['Lat', 'Lon', 'Conc'])

                if len(day_df) < 100:
                    continue

                cmaq_day = pred_pm25[day_idx]
                cmaq_values = []
                for _, row in day_df.iterrows():
                    val = get_cmaq_at_site(row['Lon'], row['Lat'], lon_cmaq, lat_cmaq, cmaq_day)
                    cmaq_values.append(val)
                day_df['CMAQ'] = cmaq_values

                # 十折验证
                for fold_id in range(1, 11):
                    train_df = day_df[day_df['fold'] != fold_id].copy()
                    test_df = day_df[day_df['fold'] == fold_id].copy()

                    train_df = train_df.dropna(subset=['Lon', 'Lat', 'CMAQ', 'Conc'])
                    test_df = test_df.dropna(subset=['Lon', 'Lat', 'CMAQ', 'Conc'])

                    if len(test_df) == 0 or len(train_df) == 0:
                        continue

                    # 调用方法的predict函数
                    try:
                        y_pred = predict_func(train_df, test_df, lon_cmaq, lat_cmaq, cmaq_day)
                        y_true = test_df['Conc'].values

                        if len(y_pred) == len(y_true):
                            all_y_true.extend(y_true)
                            all_y_pred.extend(y_pred)
                    except Exception as e:
                        pass

                day_count += 1
            except Exception as e:
                pass

        if len(all_y_true) == 0:
            print(f"[警告] 无有效数据")
            continue

        metrics = compute_metrics(np.array(all_y_true), np.array(all_y_pred))
        r2_pass = metrics['R2'] > threshold_r2
        rmse_pass = metrics['RMSE'] <= base['RMSE']
        mb_pass = abs(metrics['MB']) <= abs(base['MB'])
        innovation_pass = r2_pass and rmse_pass and mb_pass

        results[stage_name] = {
            'metrics': metrics,
            '判定': {
                'innovation_verified': innovation_pass,
                'r2_pass': r2_pass,
                'rmse_pass': rmse_pass,
                'mb_pass': mb_pass
            }
        }

        status = "✅ 通过" if innovation_pass else "❌ 失败"
        print(f"结果: R2={metrics['R2']:.4f}, RMSE={metrics['RMSE']:.2f}, MB={metrics['MB']:.2f} {status}")

    return results


def main():
    if len(sys.argv) < 2:
        print("用法: python batch_validate.py MethodName")
        print("示例: python batch_validate.py PolyRK")
        return

    method_name = sys.argv[1]

    # 尝试导入方法模块
    try:
        # 先尝试从CodeWorkSpace导入
        module_path = f'CodeWorkSpace.新融合方法代码.{method_name}'
        method_module = importlib.import_module(module_path.replace('/', '.').replace('\\', '.'))
    except:
        try:
            # 从test_result导入
            module_path = f'test_result.创新方法.{method_name}'
            method_module = importlib.import_module(module_path.replace('/', '.').replace('\\', '.'))
        except Exception as e:
            print(f"[错误] 无法导入 {method_name}: {e}")
            return

    results = validate_method_all_stages(method_name, method_module)

    if results:
        # 保存结果
        output_file = f'{ROOT_DIR}/Innovation/failed/{method_name}_all_stages.json'
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        # 统计通过阶段数
        passed = sum(1 for r in results.values() if r['判定']['innovation_verified'])
        print(f"\n{'='*60}")
        print(f"{method_name} 综合结果: {passed}/4 阶段通过")
        print(f"结果已保存: {output_file}")


if __name__ == '__main__':
    main()
