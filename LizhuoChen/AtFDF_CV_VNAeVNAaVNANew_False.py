# -*- coding: utf-8 -*-
"""
基准方法多阶段验证脚本
=====================
验证 VNA, eVNA, aVNA, AdvancedRK 在各阶段的真实表现

验证方式：标准模式 - 直接对验证站点的CMAQ网格坐标预测

依据：十折交叉验证架构文档 - 方式2b（CMAQ网格坐标，不聚合）

每天十折验证，合并所有天预测计算整体指标
"""

import sys
import os

# 路径配置（与 AtFDF_CV_New_Geng.py 保持一致）
MONITOR_DIR = "/data/workspace/DataFusion_China_CleanAir/Data/Monitor_Data"
MODEL_DIR = "/data/workspace/DataFusion_China_CleanAir/Data/Model_Data/Geng_QH_20251031_AnnualMeanCMAQ"
OUTPUT_DIR = "/data/workspace/DataFusion_China_CleanAir/Data/AtFOutput_CV_Geng"

try:
    import pyrsig
except ImportError:
    pyrsig = None
import pyproj
import numpy as np
import pandas as pd
import multiprocessing
from datetime import datetime, timedelta
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from joblib import Parallel, delayed

import nna_methods_Define

# 导入 AdvancedRK 方法
try:
    from AdvancedRK import AdvancedRK
except ImportError:
    print("警告: AdvancedRK 模块导入失败，将不支持该方法")
    AdvancedRK = None


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


def _map_obs_to_model_indices(df_obs_xy, model_2d, full_grid_df):
    """将监测站点坐标映射到模型网格索引"""
    COL_vals = full_grid_df['COL'].values.reshape(model_2d.shape)
    ROW_vals = full_grid_df['ROW'].values.reshape(model_2d.shape)
    col_centers_1d = COL_vals[0, :]
    row_centers_1d = ROW_vals[:, 0]
    x_obs = df_obs_xy['x'].values
    y_obs = df_obs_xy['y'].values
    col_idx = np.clip(np.searchsorted(col_centers_1d, x_obs) - 1, 0, col_centers_1d.size - 1)
    row_idx = np.clip(np.searchsorted(row_centers_1d, y_obs) - 1, 0, row_centers_1d.size - 1)
    return row_idx, col_idx, COL_vals, ROW_vals


def get_cmaq_grid_coord_from_row_col(row_idx, col_idx, lon_grid_2d, lat_grid_2d):
    """根据行列索引获取CMAQ网格的经纬度坐标"""
    return lon_grid_2d[row_idx, col_idx], lat_grid_2d[row_idx, col_idx]


def ten_fold_for_day_vna(method_name, selected_day):
    """
    VNA/eVNA/aVNA 的十折交叉验证
    标准模式：直接对验证站点的CMAQ网格坐标预测
    
    数据输入与 AtFDF_CV_New_Geng.py 保持一致
    """
    # 解析日期获取年份
    date_obj = datetime.strptime(selected_day, '%Y-%m-%d')
    year = date_obj.year
    
    # 监测数据文件
    monitor_file = os.path.join(MONITOR_DIR, f"{year}_DailyPM2.5Monitor.csv")
    if not os.path.exists(monitor_file):
        print(f"警告: 监测数据文件不存在: {monitor_file}")
        return np.array([]), np.array([])
    
    # 模型数据文件（年均）
    annual_model_file = os.path.join(MODEL_DIR, f"Annualmean.BASE{year}.nc")
    if not os.path.exists(annual_model_file):
        print(f"警告: 模型数据文件不存在: {annual_model_file}")
        return np.array([]), np.array([])
    
    # 读取监测数据
    df_obs = pd.read_csv(monitor_file)
    if not pd.api.types.is_numeric_dtype(df_obs.get("Lat", pd.Series(dtype=float))):
        df_obs["Lat"] = pd.to_numeric(df_obs["Lat"], errors="coerce")
    if not pd.api.types.is_numeric_dtype(df_obs.get("Lon", pd.Series(dtype=float))):
        df_obs["Lon"] = pd.to_numeric(df_obs["Lon"], errors="coerce")
    df_obs["Date"] = pd.to_datetime(df_obs["Date"])
    
    # 筛选当天的数据
    day_df = df_obs[df_obs["Date"] == selected_day].copy()
    day_df = day_df.dropna(subset=['Lat', 'Lon', 'Conc'])
    
    # 如果站点数太少，跳过
    if len(day_df) < 10:
        return np.array([]), np.array([])
    
    # 读取模型数据（使用 pyrsig 读取 IOAPI 格式）
    if pyrsig is None:
        print("警告: pyrsig 未安装")
        return np.array([]), np.array([])
    
    ds_model = pyrsig.open_ioapi(annual_model_file)
    proj = pyproj.Proj(ds_model.crs_proj4)
    model_2d = ds_model['PM25'].values.squeeze()
    
    # 替换0为0.1
    if np.sum(model_2d == 0) > 0:
        model_2d[model_2d == 0] = 0.1
    
    full_grid_df = ds_model[["ROW", "COL"]].to_dataframe().reset_index()[["ROW", "COL"]].copy()
    
    # 站点投影与网格匹配
    day_df["x"], day_df["y"] = proj(day_df["Lon"], day_df["Lat"])
    row_idx, col_idx, COL_vals, ROW_vals = _map_obs_to_model_indices(day_df, model_2d, full_grid_df)
    
    # 获取站点对应的CMAQ值
    matched_model = model_2d[row_idx, col_idx]
    valid_mask = np.isfinite(matched_model)
    day_df = day_df.loc[valid_mask].copy().reset_index(drop=True)
    row_idx = row_idx[valid_mask]
    col_idx = col_idx[valid_mask]
    day_df['matched_model'] = matched_model[valid_mask]
    day_df['row_idx'] = row_idx
    day_df['col_idx'] = col_idx
    
    # 获取全网格经纬度坐标
    nrows, ncols = model_2d.shape
    col_centers_1d = COL_vals[0, :]
    row_centers_1d = ROW_vals[:, 0]
    lon_1d, lat_1d = proj(col_centers_1d, row_centers_1d, inverse=True)
    lon_grid = np.tile(lon_1d, nrows)
    lat_grid = np.repeat(lat_1d, ncols)
    lon_grid_2d = lon_grid.reshape(nrows, ncols)
    lat_grid_2d = lat_grid.reshape(nrows, ncols)
    
    # 为每个站点添加CMAQ网格坐标
    day_df['cmaq_lon'] = [lon_grid_2d[int(r), int(c)] for r, c in zip(day_df['row_idx'], day_df['col_idx'])]
    day_df['cmaq_lat'] = [lat_grid_2d[int(r), int(c)] for r, c in zip(day_df['row_idx'], day_df['col_idx'])]
    
    ds_model.close()
    
    # 十折验证
    all_y_true = []
    all_y_pred = []
    
    for fold_id in range(1, 11):
        train_df = day_df[day_df['row_idx'] % 10 != fold_id - 1].copy()
        test_df = day_df[day_df['row_idx'] % 10 == fold_id - 1].copy()
        
        train_df = train_df.dropna(subset=['Lon', 'Lat', 'matched_model', 'Conc'])
        test_df = test_df.dropna(subset=['Lon', 'Lat', 'matched_model', 'Conc'])
        
        if len(test_df) == 0 or len(train_df) == 0:
            continue
        
        # NNA 训练
        nn = nna_methods_Define.NNA(k=30, power=-2, method='nearest', GAT=1.0)
        nn.fit_GAT(
            X=train_df[["x", "y"]].values,
            y_model=train_df["matched_model"].values,
            y_obs=train_df['Conc'].values
        )
        
        # 测试集预测
        test_rr = test_df["row_idx"].values
        test_cc = test_df["col_idx"].values
        X_test = np.column_stack([COL_vals[test_rr, test_cc], ROW_vals[test_rr, test_cc]])
        x_model_test = model_2d[test_rr, test_cc].ravel()
        
        fusion_results = nn.predict_GAT_Parrel(
            X=X_test,
            x_model=x_model_test,
            njobs=min(20, max(1, multiprocessing.cpu_count()-1)),
            adjustment_method=['VNA', 'aVNA', 'eVNA']
        )
        
        if method_name == 'CMAQ':
            y_pred = test_df['matched_model'].values
        elif method_name == 'VNA':
            y_pred = np.asarray(fusion_results.get('VNA', [])).ravel()
        elif method_name == 'aVNA':
            y_pred = np.asarray(fusion_results.get('aVNA', [])).ravel()
        elif method_name == 'eVNA':
            y_pred = np.asarray(fusion_results.get('eVNA', [])).ravel()
        else:
            y_pred = test_df['Conc'].values
        
        all_y_true.extend(test_df['Conc'].values)
        all_y_pred.extend(y_pred)
    
    return np.array(all_y_true), np.array(all_y_pred)


def ten_fold_for_day_advanced_rk(selected_day):
    """
    AdvancedRK 的十折交叉验证
    标准模式：直接对验证站点的CMAQ网格坐标预测
    
    依据：十折交叉验证架构文档 - 方式2b（CMAQ网格坐标，不聚合）
    数据输入与 AtFDF_CV_New_Geng.py 保持一致
    """
    if AdvancedRK is None:
        return np.array([]), np.array([])
    
    # 解析日期获取年份
    date_obj = datetime.strptime(selected_day, '%Y-%m-%d')
    year = date_obj.year
    
    # 监测数据文件
    monitor_file = os.path.join(MONITOR_DIR, f"{year}_DailyPM2.5Monitor.csv")
    if not os.path.exists(monitor_file):
        print(f"警告: 监测数据文件不存在: {monitor_file}")
        return np.array([]), np.array([])
    
    # 模型数据文件（年均）
    annual_model_file = os.path.join(MODEL_DIR, f"Annualmean.BASE{year}.nc")
    if not os.path.exists(annual_model_file):
        print(f"警告: 模型数据文件不存在: {annual_model_file}")
        return np.array([]), np.array([])
    
    # 读取监测数据
    df_obs = pd.read_csv(monitor_file)
    if not pd.api.types.is_numeric_dtype(df_obs.get("Lat", pd.Series(dtype=float))):
        df_obs["Lat"] = pd.to_numeric(df_obs["Lat"], errors="coerce")
    if not pd.api.types.is_numeric_dtype(df_obs.get("Lon", pd.Series(dtype=float))):
        df_obs["Lon"] = pd.to_numeric(df_obs["Lon"], errors="coerce")
    df_obs["Date"] = pd.to_datetime(df_obs["Date"])
    
    # 筛选当天的数据
    day_df = df_obs[df_obs["Date"] == selected_day].copy()
    day_df = day_df.dropna(subset=['Lat', 'Lon', 'Conc'])
    
    # 如果站点数太少，跳过
    if len(day_df) < 10:
        return np.array([]), np.array([])
    
    # 读取模型数据（使用 pyrsig 读取 IOAPI 格式）
    if pyrsig is None:
        print("警告: pyrsig 未安装")
        return np.array([]), np.array([])
    
    ds_model = pyrsig.open_ioapi(annual_model_file)
    proj = pyproj.Proj(ds_model.crs_proj4)
    model_2d = ds_model['PM25'].values.squeeze()
    
    # 替换0为0.1
    if np.sum(model_2d == 0) > 0:
        model_2d[model_2d == 0] = 0.1
    
    full_grid_df = ds_model[["ROW", "COL"]].to_dataframe().reset_index()[["ROW", "COL"]].copy()
    
    # 站点投影与网格匹配
    day_df["x"], day_df["y"] = proj(day_df["Lon"], day_df["Lat"])
    row_idx, col_idx, COL_vals, ROW_vals = _map_obs_to_model_indices(day_df, model_2d, full_grid_df)
    
    # 获取站点对应的CMAQ值
    matched_model = model_2d[row_idx, col_idx]
    valid_mask = np.isfinite(matched_model)
    day_df = day_df.loc[valid_mask].copy().reset_index(drop=True)
    row_idx = row_idx[valid_mask]
    col_idx = col_idx[valid_mask]
    day_df['matched_model'] = matched_model[valid_mask]
    day_df['row_idx'] = row_idx
    day_df['col_idx'] = col_idx
    
    # 获取全网格经纬度坐标
    nrows, ncols = model_2d.shape
    col_centers_1d = COL_vals[0, :]
    row_centers_1d = ROW_vals[:, 0]
    lon_1d, lat_1d = proj(col_centers_1d, row_centers_1d, inverse=True)
    lon_grid = np.tile(lon_1d, nrows)
    lat_grid = np.repeat(lat_1d, ncols)
    lon_grid_2d = lon_grid.reshape(nrows, ncols)
    lat_grid_2d = lat_grid.reshape(nrows, ncols)
    
    # 为每个站点添加CMAQ网格坐标（方式2b：不聚合）
    day_df['cmaq_lon'] = [lon_grid_2d[int(r), int(c)] for r, c in zip(day_df['row_idx'], day_df['col_idx'])]
    day_df['cmaq_lat'] = [lat_grid_2d[int(r), int(c)] for r, c in zip(day_df['row_idx'], day_df['col_idx'])]
    
    ds_model.close()
    
    # 十折验证
    all_y_true = []
    all_y_pred = []
    
    for fold_id in range(1, 11):
        train_df = day_df[day_df['row_idx'] % 10 != fold_id - 1].copy()
        test_df = day_df[day_df['row_idx'] % 10 == fold_id - 1].copy()
        
        train_df = train_df.dropna(subset=['cmaq_lon', 'cmaq_lat', 'matched_model', 'Conc'])
        test_df = test_df.dropna(subset=['cmaq_lon', 'cmaq_lat', 'matched_model', 'Conc'])
        
        if len(test_df) == 0 or len(train_df) == 0:
            continue
        
        
        # ========== 训练数据：使用 CMAQ 网格坐标（方式2b：不聚合）==========
        X_train = train_df[['cmaq_lon', 'cmaq_lat']].values  # CMAQ网格经纬度
        y_train = train_df['Conc'].values  # 监测值
        CMAQ_train = train_df['matched_model'].values  # CMAQ网格点的模型预测值
        
        # ========== 预测：直接对测试站点所在的CMAQ网格坐标预测（标准模式）==========
        X_test = test_df[['cmaq_lon', 'cmaq_lat']].values  # 测试站点所在网格的经纬度
        CMAQ_test = test_df['matched_model'].values  # 测试站点所在网格的CMAQ值
        
        # 训练 AdvancedRK 模型
        rk_model = AdvancedRK(poly_degree=2, nu=1.5, kernel_optimize=True)
        rk_model.fit(X_train, y_train, CMAQ_train)
        
        # 直接预测测试站点所在网格坐标（标准模式）
        y_pred = rk_model.predict(X_test, CMAQ_test)
        
        all_y_true.extend(test_df['Conc'].values)
        all_y_pred.extend(y_pred)
    
    return np.array(all_y_true), np.array(all_y_pred)


def ten_fold_for_day(method_name, selected_day):
    """
    根据方法类型调用对应的十折验证
    """
    if method_name == 'AdvancedRK':
        return ten_fold_for_day_advanced_rk(selected_day)
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

    # 基准方法
    methods = ['CMAQ', 'VNA', 'aVNA', 'eVNA', 'AdvancedRK']

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
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    import json
    output_file = os.path.join(OUTPUT_DIR, "benchmark_multistage.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*70}")
    print(f"Results saved to: {output_file}")
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
        for method in ['VNA', 'aVNA', 'eVNA', 'AdvancedRK']:  # 排除CMAQ
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
