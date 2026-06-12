# -*- coding: utf-8 -*-
"""
SVCD 十折交叉验证示例
====================
演示如何使用 SVCD 模型进行单日/多日十折交叉验证。

依赖:
- numpy, scipy, scikit-learn, pandas, netCDF4

用法:
    python example_cv.py                    # 运行5天pre_exp验证
    python example_cv.py --days 1           # 仅运行1天
    python example_cv.py --start 2020-07-01 --days 31  # 自定义日期范围
"""

import sys
import os
import time
import argparse
import numpy as np
import pandas as pd
import netCDF4 as nc
from datetime import datetime, timedelta

# 将当前目录加入路径以导入 SVCD
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from svcd import SVCD, compute_metrics


def get_cmaq_at_sites(lons, lats, lon_grid, lat_grid, cmaq_day):
    """批量获取站点对应的 CMAQ 值和网格坐标"""
    ny, nx = lon_grid.shape
    lon_flat = lon_grid.ravel()
    lat_flat = lat_grid.ravel()
    
    n = len(lons)
    cmaq_vals = np.empty(n)
    cmaq_lons = np.empty(n)
    cmaq_lats = np.empty(n)
    
    for i in range(n):
        dist = (lon_flat - lons[i])**2 + (lat_flat - lats[i])**2
        idx = np.argmin(dist)
        r, c = idx // nx, idx % nx
        cmaq_vals[i] = cmaq_day[r, c]
        cmaq_lons[i] = lon_grid[r, c]
        cmaq_lats[i] = lat_grid[r, c]
    
    return cmaq_vals, cmaq_lons, cmaq_lats


def run_cv_one_day(day_df, warm_theta=None):
    """单日十折交叉验证，返回 (y_true_list, y_pred_list, last_theta)"""
    n_folds = int(day_df['fold'].max())
    all_true, all_pred = [], []
    prev_theta = warm_theta
    
    for fold_id in range(1, n_folds + 1):
        train_df = day_df[day_df['fold'] != fold_id].dropna(subset=['CMAQ', 'Conc'])
        test_df = day_df[day_df['fold'] == fold_id].dropna(subset=['CMAQ', 'Conc'])
        if len(test_df) == 0 or len(train_df) < 20:
            continue
        
        X_tr = train_df[['CMAQ_Lon', 'CMAQ_Lat']].values
        y_tr = train_df['Conc'].values
        m_tr = train_df['CMAQ'].values
        X_te = test_df[['CMAQ_Lon', 'CMAQ_Lat']].values
        y_te = test_df['Conc'].values
        m_te = test_df['CMAQ'].values
        
        try:
            model = SVCD(c=1.0)
            model.fit(X_tr, y_tr, m_tr, theta_init=prev_theta,
                      maxiter=80, ftol=1e-5, gtol=1e-4)
            y_pred = model.predict(X_te, m_te)
            prev_theta = model.theta  # warm start for next fold
            
            all_true.extend(y_te)
            all_pred.extend(y_pred)
        except Exception as e:
            print(f"    Fold {fold_id} failed: {e}")
            continue
    
    return all_true, all_pred, prev_theta


def main():
    parser = argparse.ArgumentParser(description='SVCD Cross-Validation')
    parser.add_argument('--cmaq', required=True, help='CMAQ netCDF file path')
    parser.add_argument('--monitor', required=True, help='Monitor CSV file path')
    parser.add_argument('--folds', required=True, help='Fold split CSV file path')
    parser.add_argument('--start', default='2020-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--days', type=int, default=5, help='Number of days')
    args = parser.parse_args()
    
    print("=" * 60)
    print("SVCD Cross-Validation")
    print("=" * 60)
    t_total = time.time()
    
    # 加载数据
    monitor_df = pd.read_csv(args.monitor)
    fold_df = pd.read_csv(args.folds)
    
    ds = nc.Dataset(args.cmaq, 'r')
    lon_cmaq = ds.variables['lon'][:]
    lat_cmaq = ds.variables['lat'][:]
    pred_pm25 = ds.variables['pred_PM25'][:]
    ds.close()
    
    # 日期列表
    start_dt = datetime.strptime(args.start, '%Y-%m-%d')
    dates = [(start_dt + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(args.days)]
    print(f"日期范围: {dates[0]} ~ {dates[-1]} ({len(dates)}天)")
    print("-" * 60)
    
    # 合并所有天的预测
    ALL_TRUE, ALL_PRED = [], []
    warm_theta = None
    
    for day_str in dates:
        t_day = time.time()
        
        day_df = monitor_df[monitor_df['Date'] == day_str].copy()
        day_df = day_df.merge(fold_df, on=['Date', 'Site'], how='left')
        day_df = day_df.dropna(subset=['Lat', 'Lon', 'Conc'])
        
        if len(day_df) < 50:
            print(f"  {day_str}: 站点不足({len(day_df)}), 跳过")
            continue
        
        # 提取 CMAQ
        date_obj = datetime.strptime(day_str, '%Y-%m-%d')
        day_idx = (date_obj - datetime(2020, 1, 1)).days
        if day_idx < 0 or day_idx >= pred_pm25.shape[0]:
            print(f"  {day_str}: 超出CMAQ时间范围, 跳过")
            continue
        cmaq_day = pred_pm25[day_idx]
        
        cmaq_vals, cmaq_lons, cmaq_lats = get_cmaq_at_sites(
            day_df['Lon'].values, day_df['Lat'].values, lon_cmaq, lat_cmaq, cmaq_day
        )
        day_df['CMAQ'] = cmaq_vals
        day_df['CMAQ_Lon'] = cmaq_lons
        day_df['CMAQ_Lat'] = cmaq_lats
        
        # 十折
        day_true, day_pred, warm_theta = run_cv_one_day(day_df, warm_theta)
        
        if len(day_true) > 0:
            day_metrics = compute_metrics(np.array(day_true), np.array(day_pred))
            elapsed = time.time() - t_day
            print(f"  {day_str}: n={len(day_true)}, R2={day_metrics['R2']:.4f}, "
                  f"RMSE={day_metrics['RMSE']:.2f} [{elapsed:.0f}s]")
            ALL_TRUE.extend(day_true)
            ALL_PRED.extend(day_pred)
        
        sys.stdout.flush()
    
    # 汇总
    if len(ALL_TRUE) == 0:
        print("无有效预测结果！")
        return
    
    overall = compute_metrics(np.array(ALL_TRUE), np.array(ALL_PRED))
    total_time = time.time() - t_total
    
    print("=" * 60)
    print(f"SVCD 合并指标 (n={len(ALL_TRUE)}):")
    print(f"  R2   = {overall['R2']:.4f}")
    print(f"  RMSE = {overall['RMSE']:.2f}")
    print(f"  MAE  = {overall['MAE']:.2f}")
    print(f"  MB   = {overall['MB']:.2f}")
    print(f"  总耗时: {total_time:.0f}s ({total_time/60:.1f}min)")
    print("=" * 60)


if __name__ == '__main__':
    main()
