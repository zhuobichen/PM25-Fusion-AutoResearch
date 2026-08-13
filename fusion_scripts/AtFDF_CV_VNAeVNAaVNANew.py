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

# 新增：十折划分表路径
FOLD_FILE = "/data/workspace/DataFusion_China_CleanAir/Data/fold_split_table_daily.csv"

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

# 原项目NNA路径
sys.path.insert(0, 'E:/CodeProject/ClaudeRoom/Data_Fusion_AutoResearch/Code')
from Code.VNAeVNAaVNA.nna_methods import NNA

# AdvancedRK路径
sys.path.insert(0, 'E:/CodeProject/ClaudeRoom/Data_Fusion_AutoResearch/Innovation/success/AdvancedRK')
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


def _load_or_create_fold_table():
    """
    加载十折划分表
    格式：Date,Site,fold（每年每天独立划分）
    """
    global FOLD_FILE

    if os.path.exists(FOLD_FILE):
        print(f"  加载十折划分表: {FOLD_FILE}")
        return pd.read_csv(FOLD_FILE)

    raise FileNotFoundError(f"划分表不存在，请先运行 generate_fold_table.py 生成: {FOLD_FILE}")


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
    
    # ========== 正确的全网格经纬度坐标获取（批量投影转换）==========
    nrows, ncols = model_2d.shape
    
    # 创建全网格索引
    row_idx_all = np.repeat(np.arange(nrows), ncols)
    col_idx_all = np.tile(np.arange(ncols), nrows)
    
    # 获取全网格的投影坐标
    x_all = COL_vals[row_idx_all, col_idx_all]
    y_all = ROW_vals[row_idx_all, col_idx_all]
    
    # 批量投影转换为经纬度
    lon_all, lat_all = proj(x_all, y_all, inverse=True)
    
    lon_grid_2d = lon_all.reshape(nrows, ncols)
    lat_grid_2d = lat_all.reshape(nrows, ncols)
    
    # 为每个站点添加CMAQ网格坐标（向量化操作）
    day_df['cmaq_lon'] = lon_grid_2d[day_df['row_idx'].values, day_df['col_idx'].values]
    day_df['cmaq_lat'] = lat_grid_2d[day_df['row_idx'].values, day_df['col_idx'].values]
    
    ds_model.close()
    
    # ========== 加载十折划分表（每年每天独立划分）==========
    fold_df = _load_or_create_fold_table()
    
    # 将Date转换为字符串以便合并
    day_df['Date'] = day_df['Date'].astype(str)
    fold_df['Date'] = fold_df['Date'].astype(str)
    
    # 将fold信息合并到day_df（按Date+Site）
    day_df = day_df.merge(fold_df, on=['Date', 'Site'], how='left')
    
    # 如果某些站点没有fold编号，随机分配
    if day_df['fold'].isna().any():
        print(f"  警告: {day_df['fold'].isna().sum()} 个站点无fold编号，将随机分配")
        unassigned = day_df[day_df['fold'].isna()].index
        day_df.loc[unassigned, 'fold'] = np.random.randint(1, 11, size=len(unassigned))
    
    # 确保fold是整数类型
    day_df['fold'] = day_df['fold'].astype(int)
    
    # 十折验证
    all_y_true = []
    all_y_pred = []
    
    for fold_id in range(1, 11):
        train_df = day_df[day_df['fold'] != fold_id].copy()
        test_df = day_df[day_df['fold'] == fold_id].copy()
        
        train_df = train_df.dropna(subset=['Lon', 'Lat', 'matched_model', 'Conc'])
        test_df = test_df.dropna(subset=['Lon', 'Lat', 'matched_model', 'Conc'])
        
        if len(test_df) == 0 or len(train_df) == 0:
            continue
        
        if method_name == 'CMAQ':
            y_pred = test_df['matched_model'].values
        else:
            # ========== 使用原项目NNA接口 ==========
            # 准备训练数据：X=经纬度坐标, y=[Conc, mod, bias, r_n]
            train_df['mod'] = train_df['matched_model']
            train_df['bias'] = train_df['Conc'] - train_df['mod']
            train_df['r_n'] = train_df['Conc'] / train_df['mod']

            # 训练NNA（原项目接口）
            nn = NNA(method='voronoi', k=30, power=-2)
            nn.fit(
                train_df[['Lon', 'Lat']],
                train_df[['Conc', 'mod', 'bias', 'r_n']]
            )

            # 测试集预测：使用CMAQ网格经纬度坐标
            X_test = test_df[['cmaq_lon', 'cmaq_lat']].values

            # NNA.predict 返回 array[n, 4]，列对应 [Conc, mod, bias, r_n]
            zdf = nn.predict(X_test)
            vna_pred = zdf[:, 0]       # VNA预测值
            vna_bias_pred = zdf[:, 2]  # bias预测值
            vna_rn_pred = zdf[:, 3]    # rn预测值

            # 三种方法
            if method_name == 'VNA':
                y_pred = vna_pred
            elif method_name == 'aVNA':
                y_pred = test_df['matched_model'].values + vna_bias_pred
            elif method_name == 'eVNA':
                y_pred = test_df['matched_model'].values * vna_rn_pred
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
    
    # ========== 正确的全网格经纬度坐标获取（批量投影转换）==========
    nrows, ncols = model_2d.shape
    
    # 创建全网格索引
    row_idx_all = np.repeat(np.arange(nrows), ncols)
    col_idx_all = np.tile(np.arange(ncols), nrows)
    
    # 获取全网格的投影坐标
    x_all = COL_vals[row_idx_all, col_idx_all]
    y_all = ROW_vals[row_idx_all, col_idx_all]
    
    # 批量投影转换为经纬度
    lon_all, lat_all = proj(x_all, y_all, inverse=True)
    
    lon_grid_2d = lon_all.reshape(nrows, ncols)
    lat_grid_2d = lat_all.reshape(nrows, ncols)
    
    # 为每个站点添加CMAQ网格坐标（向量化操作）
    day_df['cmaq_lon'] = lon_grid_2d[day_df['row_idx'].values, day_df['col_idx'].values]
    day_df['cmaq_lat'] = lat_grid_2d[day_df['row_idx'].values, day_df['col_idx'].values]
    
    ds_model.close()
    
    # ========== 加载十折划分表（每年每天独立划分）==========
    fold_df = _load_or_create_fold_table()
    
    # 将Date转换为字符串以便合并
    day_df['Date'] = day_df['Date'].astype(str)
    fold_df['Date'] = fold_df['Date'].astype(str)
    
    # 将fold信息合并到day_df（按Date+Site）
    day_df = day_df.merge(fold_df, on=['Date', 'Site'], how='left')
    
    # 如果某些站点没有fold编号，随机分配
    if day_df['fold'].isna().any():
        print(f"  警告: {day_df['fold'].isna().sum()} 个站点无fold编号，将随机分配")
        unassigned = day_df[day_df['fold'].isna()].index
        day_df.loc[unassigned, 'fold'] = np.random.randint(1, 11, size=len(unassigned))
    
    # 确保fold是整数类型
    day_df['fold'] = day_df['fold'].astype(int)
    
    # 十折验证
    all_y_true = []
    all_y_pred = []
    
    for fold_id in range(1, 11):
        train_df = day_df[day_df['fold'] != fold_id].copy()
        test_df = day_df[day_df['fold'] == fold_id].copy()
        
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

    # ============================================================
    # 用户配置：指定处理日期范围，按天进行十折交叉验证
    # ============================================================
    CONFIG = {
        'date_range': ('2014-05-13', '2020-05-13'),  # 日期范围 (起始日期, 结束日期)
        'n_jobs': 4,                                   # 并行进程数
    }
    
    start_date = CONFIG['date_range'][0]
    end_date = CONFIG['date_range'][1]
    print(f"\n配置:")
    print(f"  日期范围: {start_date} ~ {end_date}")
    print(f"  并行进程: {CONFIG['n_jobs']}")
    
    # 基准方法
    methods = ['CMAQ', 'VNA', 'aVNA', 'eVNA', 'AdvancedRK']

    results = {}

    for method in methods:
        print(f"\n{'='*50}")
        print(f"Method: {method}")
        print(f"{'='*50}")

        results[method] = {}
        
        # 按天进行十折交叉验证
        print(f"\n  日期范围: {start_date} ~ {end_date}")
        metrics = run_stage_validation(method, start_date, end_date, n_jobs=CONFIG['n_jobs'])
        results[method]['daily_cv'] = metrics

    # 保存结果
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    import json
    output_file = os.path.join(OUTPUT_DIR, "benchmark_daily_cv.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*70}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*70}")

    # 打印汇总表
    print("\nSummary:")
    print("-" * 80)
    header = f"{'Method':<12} {'Stage':<15} {'R2':<10} {'RMSE':<10} {'MAE':<10} {'MB':<10}"
    print(header)
    print("-" * 80)

    for method in methods:
        m = results[method]['daily_cv']
        r2 = f"{m['R2']:.4f}" if not np.isnan(m['R2']) else "N/A"
        rmse = f"{m['RMSE']:.2f}" if not np.isnan(m['RMSE']) else "N/A"
        mae = f"{m['MAE']:.2f}" if not np.isnan(m['MAE']) else "N/A"
        mb = f"{m['MB']:.2f}" if not np.isnan(m['MB']) else "N/A"
        print(f"{method:<12} {'daily_cv':<15} {r2:<10} {rmse:<10} {mae:<10} {mb:<10}")

    # 找出最优基准
    print("\nOptimal Baseline (excluding CMAQ):")
    print("-" * 80)
    best_method = None
    best_r2 = -np.inf
    for method in ['VNA', 'aVNA', 'eVNA', 'AdvancedRK']:  # 排除CMAQ
        r2 = results[method]['daily_cv']['R2']
        if not np.isnan(r2) and r2 > best_r2:
            best_r2 = r2
            best_method = method
    if best_method:
        m = results[best_method]['daily_cv']
        print(f"Best: {best_method}")
        print(f"  R2 = {m['R2']:.4f}")
        print(f"  RMSE = {m['RMSE']:.2f}")
        print(f"  MAE = {m['MAE']:.2f}")
        print(f"  MB = {m['MB']:.2f}")

    return results


if __name__ == '__main__':
    main()
