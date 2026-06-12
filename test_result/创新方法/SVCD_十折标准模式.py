# -*- coding: utf-8 -*-
"""
SVCD 十折交叉验证 - 标准模式
==============================
Spatially Varying Coefficient Downscaler
训练：使用CMAQ网格坐标（标准模式）
预测：1折站点所在的CMAQ网格坐标

支持参数:
  --pre-only    仅运行 pre_exp 阶段（快速筛选）
  无参数        运行全部4阶段验证
"""

import sys
import os
# 项目根目录: test_result/创新方法/ -> test_result/ -> 项目根
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PROJECT_ROOT)

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
OUTPUT_DIR = os.path.join(ROOT_DIR, 'test_result', '创新方法')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# VNA 基准阈值（主级创新要求 R² >= baseline + 0.01）
BASELINE = {
    'pre_exp': {'R2': 0.8941, 'RMSE': 16.42, 'MB': 0.76},
    'stage1':  {'R2': 0.9057, 'RMSE': 16.28, 'MB': 0.50},
    'stage2':  {'R2': 0.8458, 'RMSE': 4.97, 'MB': 0.04},
    'stage3':  {'R2': 0.9078, 'RMSE': 11.90, 'MB': 0.36},
}


def compute_metrics(y_true, y_pred):
    """计算评估指标"""
    mask = ~(np.isnan(y_true) | np.isnan(y_pred) |
             np.isinf(y_true) | np.isinf(y_pred))
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
    """获取站点对应的 CMAQ 网格值"""
    dist = np.sqrt((lon_grid - lon)**2 + (lat_grid - lat)**2)
    idx = np.argmin(dist)
    ny, nx = lon_grid.shape
    row, col = idx // nx, idx % nx
    return pm25_grid[row, col]


def get_cmaq_grid_coord(lon, lat, lon_grid, lat_grid):
    """获取站点对应的 CMAQ 网格坐标"""
    dist = np.sqrt((lon_grid - lon)**2 + (lat_grid - lat)**2)
    idx = np.argmin(dist)
    ny, nx = lon_grid.shape
    row, col = idx // nx, idx % nx
    return lon_grid[row, col], lat_grid[row, col]


def ten_fold_svcd(selected_day, c=1.0, nu0=1.5, nu1=1.5):
    """
    单日十折交叉验证 - SVCD方法
    
    返回: (all_y_true, all_y_pred)
    """
    # 延迟导入 SVCD（避免路径问题）
    svcd_path = os.path.join(ROOT_DIR, 'CodeWorkSpace', '新融合方法代码')
    if svcd_path not in sys.path:
        sys.path.insert(0, svcd_path)
    from SVCD import SVCD
    
    monitor_df = pd.read_csv(MONITOR_FILE)
    fold_df = pd.read_csv(FOLD_FILE)
    
    day_df = monitor_df[monitor_df['Date'] == selected_day].copy()
    day_df = day_df.merge(fold_df, on=['Date', 'Site'], how='left')
    day_df = day_df.dropna(subset=['Lat', 'Lon', 'Conc'])
    
    if len(day_df) < 50:
        return np.array([]), np.array([])
    
    # 加载 CMAQ 数据
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
    
    # 获取每个站点对应的 CMAQ 值和网格坐标
    cmaq_values = []
    cmaq_coords = []
    for _, row in day_df.iterrows():
        val = get_cmaq_at_site(row['Lon'], row['Lat'], lon_cmaq, lat_cmaq, cmaq_day)
        cmaq_lon, cmaq_lat = get_cmaq_grid_coord(row['Lon'], row['Lat'], lon_cmaq, lat_cmaq)
        cmaq_values.append(val)
        cmaq_coords.append([cmaq_lon, cmaq_lat])
    day_df['CMAQ'] = cmaq_values
    day_df['CMAQ_Lon'] = [c[0] for c in cmaq_coords]
    day_df['CMAQ_Lat'] = [c[1] for c in cmaq_coords]
    
    all_y_true = []
    all_y_pred = []
    
    for fold_id in range(1, 11):
        train_df = day_df[day_df['fold'] != fold_id].copy()
        test_df = day_df[day_df['fold'] == fold_id].copy()
        
        train_df = train_df.dropna(subset=['Lon', 'Lat', 'CMAQ', 'Conc'])
        test_df = test_df.dropna(subset=['Lon', 'Lat', 'CMAQ', 'Conc'])
        
        if len(test_df) == 0 or len(train_df) < 20:
            continue
        
        try:
            # 使用 CMAQ 网格坐标（标准模式）
            X_train = train_df[['CMAQ_Lon', 'CMAQ_Lat']].values
            y_train = train_df['Conc'].values
            m_train = train_df['CMAQ'].values
            
            X_test = test_df[['CMAQ_Lon', 'CMAQ_Lat']].values
            y_test = test_df['Conc'].values
            m_test = test_df['CMAQ'].values
            
            # 训练 SVCD
            model = SVCD(c=c, nu0=nu0, nu1=nu1)
            model.fit(X_train, y_train, m_train)
            
            # 预测
            y_pred = model.predict(X_test, m_test)
            
            all_y_true.extend(y_test)
            all_y_pred.extend(y_pred)
            
        except Exception as e:
            # 单折失败不影响整体
            continue
    
    return np.array(all_y_true), np.array(all_y_pred)


def run_stage_validation(stage_name, start_date, end_date):
    """运行单阶段验证"""
    sep = "=" * 70
    print(sep)
    print(f"SVCD Stage: {stage_name} ({start_date} ~ {end_date})")
    print(sep)
    
    base = BASELINE[stage_name]
    threshold_r2 = base['R2']
    print(f"VNA Baseline: R2={base['R2']:.4f}, RMSE={base['RMSE']:.2f}, MB={base['MB']:.2f}")
    
    # 生成日期列表
    date_list = []
    current_date = datetime.strptime(start_date, '%Y-%m-%d')
    end_date_obj = datetime.strptime(end_date, '%Y-%m-%d')
    while current_date <= end_date_obj:
        date_list.append(current_date.strftime('%Y-%m-%d'))
        current_date += timedelta(days=1)
    
    print(f"Days: {len(date_list)}")
    
    # 并行执行（SVCD 计算量较大，适当减少并行数）
    n_jobs = min(4, len(date_list))
    results = Parallel(n_jobs=n_jobs, verbose=1)(
        delayed(ten_fold_svcd)(date_str)
        for date_str in date_list
    )
    
    # 汇总
    all_y_true = []
    all_y_pred = []
    day_count = 0
    for y_true, y_pred in results:
        if len(y_true) > 0:
            all_y_true.extend(y_true)
            all_y_pred.extend(y_pred)
            day_count += 1
    
    print(f"Processed: {day_count}/{len(date_list)} days, {len(all_y_true)} predictions")
    
    if len(all_y_true) == 0:
        return {'R2': np.nan, 'MAE': np.nan, 'RMSE': np.nan, 'MB': np.nan}, False
    
    metrics = compute_metrics(np.array(all_y_true), np.array(all_y_pred))
    
    # 创新判定
    r2_pass = metrics['R2'] > threshold_r2
    rmse_pass = metrics['RMSE'] <= base['RMSE']
    mb_pass = abs(metrics['MB']) <= abs(base['MB'])
    innovation_pass = r2_pass and rmse_pass and mb_pass
    
    print(f"Result: R2={metrics['R2']:.4f}, RMSE={metrics['RMSE']:.2f}, MB={metrics['MB']:.2f}")
    print(f"Check: R2>{threshold_r2:.4f}? {'PASS' if r2_pass else 'FAIL'} | "
          f"RMSE<={base['RMSE']}? {'PASS' if rmse_pass else 'FAIL'} | "
          f"|MB|<={abs(base['MB'])}? {'PASS' if mb_pass else 'FAIL'}")
    print(f"Innovation: {'VERIFIED' if innovation_pass else 'NOT VERIFIED'}")
    
    return metrics, innovation_pass


def run_pre_exp_only():
    """仅运行 pre_exp 预验证"""
    metrics, passed = run_stage_validation('pre_exp', '2020-01-01', '2020-01-05')
    
    # 保存预验证结果
    result = {
        'method': 'SVCD',
        'stage': 'pre_exp',
        'metrics': metrics,
        'passed': passed,
        'timestamp': datetime.now().isoformat()
    }
    
    output_file = os.path.join(OUTPUT_DIR, 'SVCD_pre_exp.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"\nPre-exp result saved: {output_file}")
    return passed


def main():
    """完整四阶段验证"""
    sep = "=" * 70
    print(sep)
    print("SVCD (Spatially Varying Coefficient Downscaler)")
    print("All Stages - 标准模式")
    print(sep)
    
    # 检查是否仅运行预验证
    if '--pre-only' in sys.argv:
        passed = run_pre_exp_only()
        sys.exit(0 if passed else 1)
    
    stages = {
        'pre_exp': ('2020-01-01', '2020-01-05'),
        'stage1':  ('2020-01-01', '2020-01-31'),
        'stage2':  ('2020-07-01', '2020-07-31'),
        'stage3':  ('2020-12-01', '2020-12-31'),
    }
    
    results = {}
    all_pass = True
    
    for stage_name, (start, end) in stages.items():
        metrics, innovation_pass = run_stage_validation(stage_name, start, end)
        results[stage_name] = {
            'metrics': metrics,
            '判定': {'innovation_verified': innovation_pass}
        }
        if not innovation_pass:
            all_pass = False
    
    # 保存完整结果
    output_file = os.path.join(OUTPUT_DIR, 'SVCD_all_stages.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # 保存 summary CSV
    summary_file = os.path.join(OUTPUT_DIR, 'SVCD_summary.csv')
    rows = []
    for stage, data in results.items():
        m = data['metrics']
        rows.append({
            'stage': stage,
            'R2': m['R2'],
            'MAE': m['MAE'],
            'RMSE': m['RMSE'],
            'MB': m['MB'],
            'innovation_verified': data['判定']['innovation_verified']
        })
    pd.DataFrame(rows).to_csv(summary_file, index=False)
    
    # 打印总结
    print("\n" + sep)
    print("SUMMARY")
    print(sep)
    for stage, data in results.items():
        m = data['metrics']
        status = 'VERIFIED' if data['判定']['innovation_verified'] else 'NOT VERIFIED'
        print(f"{stage}: R2={m['R2']:.4f}, RMSE={m['RMSE']:.2f}, MB={m['MB']:.2f} -> {status}")
    
    print(f"\nAll stages passed: {all_pass}")
    print(f"Results saved: {output_file}")
    print(f"Summary saved: {summary_file}")
    
    return results


if __name__ == '__main__':
    main()
