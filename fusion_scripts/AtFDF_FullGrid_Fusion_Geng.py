# -*- coding: utf-8 -*-
"""
全网格数据融合脚本
================
生成全网格预测结果，支持AtF和FtA两种融合模式

AtF (Aggregation-then-Fusion): 先聚合再融合
    - 将日期范围内的监测数据聚合（如 Annual 2014）
    - 融合结果输出为单行/多行Period（如 Annual_2014, Jan_2014 等）

FtA (Fusion-to-Aggregation): 先融合再聚合
    - 每天独立融合，然后聚合结果
    - 输出为Period列（如 2014-01-01）

输出格式: Period_ROW_COL_AtF(或者FtA)_Geng_multimethod.csv
包含列: ROW, COL, Period, model, vna, evna, avna, advancedrk
"""

import sys
import os

# 路径配置
MONITOR_DIR = "/data/workspace/DataFusion_China_CleanAir/Data/Monitor_Data"
MODEL_DIR = "/data/workspace/DataFusion_China_CleanAir/Data/Model_Data/Geng_QH_20251031_AnnualMeanCMAQ"
OUTPUT_DIR = "/data/workspace/DataFusion_China_CleanAir/Data/AtFOutput_CV_Geng"

# NNA和AdvancedRK路径
sys.path.insert(0, 'E:/CodeProject/ClaudeRoom/Data_Fusion_AutoResearch/Code')
from Code.VNAeVNAaVNA.nna_methods import NNA

sys.path.insert(0, 'E:/CodeProject/ClaudeRoom/Data_Fusion_AutoResearch/Innovation/success/AdvancedRK')
try:
    from AdvancedRK import AdvancedRK
except ImportError:
    print("警告: AdvancedRK 模块导入失败，将不支持该方法")
    AdvancedRK = None

try:
    import pyrsig
except ImportError:
    pyrsig = None
import pyproj
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


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


def _get_full_grid_coords(proj, COL_vals, ROW_vals, model_2d):
    """获取全网格经纬度坐标（批量投影转换）"""
    nrows, ncols = model_2d.shape
    row_idx_all = np.repeat(np.arange(nrows), ncols)
    col_idx_all = np.tile(np.arange(ncols), nrows)
    x_all = COL_vals[row_idx_all, col_idx_all]
    y_all = ROW_vals[row_idx_all, col_idx_all]
    lon_all, lat_all = proj(x_all, y_all, inverse=True)
    lon_grid_2d = lon_all.reshape(nrows, ncols)
    lat_grid_2d = lat_all.reshape(nrows, ncols)
    return lon_grid_2d, lat_grid_2d


def fusion_and_evaluate(proj, model_2d, full_grid_df, COL_vals, ROW_vals,
                         lon_grid_2d, lat_grid_2d, day_df, train_df,
                         X_eval=None, y_eval=None, cmaq_eval=None):
    """
    融合预测 + 监测站评估

    参数:
        proj, model_2d, full_grid_df, COL_vals, ROW_vals, lon_grid_2d, lat_grid_2d: 模型数据
        day_df: 当天监测数据
        train_df: 训练数据
        X_eval, y_eval, cmaq_eval: 评估用监测站坐标、真实值、CMAQ值（可选，默认用day_df）

    返回:
        results: 全网格预测结果 dict
        metrics: 评估指标 dict
        grid_row_flat, grid_col_flat: 网格行列索引
    """
    results = {}
    metrics = {}

    if len(train_df) < 10:
        return results, metrics, None, None

    # 获取全网格坐标用于预测
    nrows, ncols = model_2d.shape
    grid_rows = np.arange(nrows)
    grid_cols = np.arange(ncols)
    ROW_all, COL_all = np.meshgrid(grid_rows, grid_cols, indexing='ij')
    X_grid = np.column_stack([
        lon_grid_2d.ravel(),
        lat_grid_2d.ravel()
    ])
    grid_row_flat = ROW_all.ravel()
    grid_col_flat = COL_all.ravel()

    # 准备训练数据
    train_df['mod'] = train_df['matched_model']
    train_df['bias'] = train_df['Conc'] - train_df['mod']
    train_df['r_n'] = train_df['Conc'] / train_df['mod']

    # ============== VNA / eVNA / aVNA ==============
    nn = NNA(method='voronoi', k=30, power=-2)
    nn.fit(
        train_df[['Lon', 'Lat']],
        train_df[['Conc', 'mod', 'bias', 'r_n']]
    )

    # 全网格预测
    zdf = nn.predict(X_grid)
    vna_pred = zdf[:, 0]
    vna_bias_pred = zdf[:, 2]
    vna_rn_pred = zdf[:, 3]

    model_flat = model_2d.ravel()

    results['vna'] = vna_pred
    results['evna'] = model_flat * vna_rn_pred
    results['avna'] = model_flat + vna_bias_pred

    # ============== AdvancedRK ==============
    rk_pred = None
    if AdvancedRK is not None:
        X_train = train_df[['cmaq_lon', 'cmaq_lat']].values
        y_train = train_df['Conc'].values
        CMAQ_train = train_df['matched_model'].values

        X_grid_gpr = np.column_stack([
            lon_grid_2d.ravel(),
            lat_grid_2d.ravel()
        ])

        rk_model = AdvancedRK(poly_degree=2, nu=1.5, kernel_optimize=True)
        rk_model.fit(X_train, y_train, CMAQ_train)
        results['advancedrk'] = rk_model.predict(X_grid_gpr, model_flat)

    # ============== 监测站评估 ==============
    if X_eval is None:
        X_eval = day_df[['cmaq_lon', 'cmaq_lat']].values
        y_eval = day_df['Conc'].values
        cmaq_eval = day_df['matched_model'].values

    if len(y_eval) > 0:
        # 在监测站位置预测（用于评估）
        zdf_eval = nn.predict(X_eval)
        vna_eval = zdf_eval[:, 0]
        evna_eval = cmaq_eval * zdf_eval[:, 3]
        avna_eval = cmaq_eval + zdf_eval[:, 2]

        metrics['vna'] = compute_metrics(y_eval, vna_eval)
        metrics['evna'] = compute_metrics(y_eval, evna_eval)
        metrics['avna'] = compute_metrics(y_eval, avna_eval)
        metrics['model'] = compute_metrics(y_eval, cmaq_eval)

        if AdvancedRK is not None and 'advancedrk' in results:
            ar_eval = rk_model.predict(X_eval, cmaq_eval)
            metrics['advancedrk'] = compute_metrics(y_eval, ar_eval)

    return results, metrics, grid_row_flat, grid_col_flat


def run_fusion_fused_to_aggregate(date_range, period_name=None):
    """
    FtA模式 (Fusion-to-Aggregation): 先融合再聚合
    每天独立融合，然后聚合（平均）结果

    参数:
        date_range: (start_date, end_date) 元组
        period_name: 聚合后的期间名称（如 "2014_Annual"）
    """
    print("="*70)
    print("FtA模式: 先融合再聚合")
    print(f"日期范围: {date_range[0]} ~ {date_range[1]}")
    print("="*70)

    start_date, end_date = date_range
    if period_name is None:
        period_name = f"{start_date}_{end_date}"

    # 生成日期列表
    date_list = []
    current_date = datetime.strptime(start_date, '%Y-%m-%d')
    end_date_obj = datetime.strptime(end_date, '%Y-%m-%d')
    while current_date <= end_date_obj:
        date_list.append(current_date.strftime('%Y-%m-%d'))
        current_date += timedelta(days=1)

    print(f"共 {len(date_list)} 天")

    # 读取模型数据（只读一次）
    first_date = datetime.strptime(date_list[0], '%Y-%m-%d')
    first_year = first_date.year
    annual_model_file = os.path.join(MODEL_DIR, f"Annualmean.BASE{first_year}.nc")

    if pyrsig is None:
        print("错误: pyrsig 未安装")
        return None

    ds_model = pyrsig.open_ioapi(annual_model_file)
    proj = pyproj.Proj(ds_model.crs_proj4)
    model_2d = ds_model['PM25'].values.squeeze()

    if np.sum(model_2d == 0) > 0:
        model_2d[model_2d == 0] = 0.1

    full_grid_df = ds_model[["ROW", "COL"]].to_dataframe().reset_index()[["ROW", "COL"]].copy()
    nrows, ncols = model_2d.shape

    # 获取全网格坐标
    COL_vals = full_grid_df['COL'].values.reshape(model_2d.shape)
    ROW_vals = full_grid_df['ROW'].values.reshape(model_2d.shape)
    lon_grid_2d, lat_grid_2d = _get_full_grid_coords(proj, COL_vals, ROW_vals, model_2d)

    # 初始化累积结果
    total_vna = np.zeros_like(model_2d, dtype=np.float64)
    total_evna = np.zeros_like(model_2d, dtype=np.float64)
    total_avna = np.zeros_like(model_2d, dtype=np.float64)
    total_advancedrk = np.zeros_like(model_2d, dtype=np.float64)
    day_count = 0

    for date_str in date_list:
        print(f"  处理: {date_str}")

        # 读取当天监测数据
        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
        year = date_obj.year
        monitor_file = os.path.join(MONITOR_DIR, f"{year}_DailyPM2.5Monitor.csv")

        if not os.path.exists(monitor_file):
            print(f"    警告: 文件不存在 {monitor_file}")
            continue

        df_obs = pd.read_csv(monitor_file)
        if not pd.api.types.is_numeric_dtype(df_obs.get("Lat", pd.Series(dtype=float))):
            df_obs["Lat"] = pd.to_numeric(df_obs["Lat"], errors="coerce")
        if not pd.api.types.is_numeric_dtype(df_obs.get("Lon", pd.Series(dtype=float))):
            df_obs["Lon"] = pd.to_numeric(df_obs["Lon"], errors="coerce")
        df_obs["Date"] = pd.to_datetime(df_obs["Date"])

        day_df = df_obs[df_obs["Date"] == date_str].copy()
        day_df = day_df.dropna(subset=['Lat', 'Lon', 'Conc'])

        if len(day_df) < 10:
            print(f"    跳过: 站点数不足 ({len(day_df)})")
            continue

        # 站点投影与网格匹配
        day_df["x"], day_df["y"] = proj(day_df["Lon"], day_df["Lat"])
        row_idx, col_idx, _, _ = _map_obs_to_model_indices(day_df, model_2d, full_grid_df)

        # 获取站点对应的CMAQ值
        matched_model = model_2d[row_idx, col_idx]
        valid_mask = np.isfinite(matched_model)
        day_df = day_df.loc[valid_mask].copy().reset_index(drop=True)
        row_idx = row_idx[valid_mask]
        col_idx = col_idx[valid_mask]
        day_df['matched_model'] = matched_model[valid_mask]
        day_df['row_idx'] = row_idx
        day_df['col_idx'] = col_idx

        # 添加CMAQ网格坐标
        day_df['cmaq_lon'] = lon_grid_2d[day_df['row_idx'].values, day_df['col_idx'].values]
        day_df['cmaq_lat'] = lat_grid_2d[day_df['row_idx'].values, day_df['col_idx'].values]

        # 融合 + 评估（每天独立评估）
        fusion_results, day_metrics, grid_row_flat, grid_col_flat = fusion_and_evaluate(
            proj, model_2d, full_grid_df, COL_vals, ROW_vals,
            lon_grid_2d, lat_grid_2d, day_df, day_df
        )

        if not fusion_results:
            continue

        # 累积结果
        total_vna += fusion_results['vna'].reshape(nrows, ncols)
        total_evna += fusion_results['evna'].reshape(nrows, ncols)
        total_avna += fusion_results['avna'].reshape(nrows, ncols)
        if 'advancedrk' in fusion_results:
            total_advancedrk += fusion_results['advancedrk'].reshape(nrows, ncols)

        # 打印当天评估
        if day_metrics:
            r2_str = ' | '.join([f"{k}=R²{m['R2']:.4f}" for k, m in day_metrics.items()])
            print(f"    评估: {r2_str}")

        day_count += 1
        print(f"    完成: {len(day_df)} 站点, 累计 {day_count} 天")

    ds_model.close()

    if day_count == 0:
        print("错误: 没有有效数据")
        return None

    # 平均
    avg_vna = total_vna / day_count
    avg_evna = total_evna / day_count
    avg_avna = total_avna / day_count
    avg_advancedrk = total_advancedrk / day_count

    # 生成输出DataFrame
    print("生成输出表格...")
    rows_list = []
    for r in range(nrows):
        for c in range(ncols):
            rows_list.append({
                'ROW': r,
                'COL': c,
                'Period': period_name,
                'model': model_2d[r, c],
                'vna': avg_vna[r, c],
                'evna': avg_evna[r, c],
                'avna': avg_avna[r, c],
                'advancedrk': avg_advancedrk[r, c] if AdvancedRK else np.nan
            })

    result_df = pd.DataFrame(rows_list)

    return result_df


def run_fusion_aggregation_to_fuse(period_configs):
    """
    AtF模式 (Aggregation-then-Fusion): 先聚合再融合
    对每个配置的日期范围聚合监测数据，然后融合

    参数:
        period_configs: list of dict
            [
                {'name': 'Annual_2014', 'start': '2014-01-01', 'end': '2014-12-31'},
                {'name': 'Winter_2014', 'start': '2014-12-01', 'end': '2015-02-28'},
                ...
            ]
    """
    print("="*70)
    print("AtF模式: 先聚合再融合")
    print("="*70)

    results_list = []

    for config in period_configs:
        period_name = config['name']
        start_date = config['start']
        end_date = config['end']

        print(f"\n处理: {period_name} ({start_date} ~ {end_date})")

        # 生成日期列表
        date_list = []
        current_date = datetime.strptime(start_date, '%Y-%m-%d')
        end_date_obj = datetime.strptime(end_date, '%Y-%m-%d')
        while current_date <= end_date_obj:
            date_list.append(current_date.strftime('%Y-%m-%d'))
            current_date += timedelta(days=1)

        print(f"  共 {len(date_list)} 天")

        # 读取模型数据
        first_date = datetime.strptime(date_list[0], '%Y-%m-%d')
        first_year = first_date.year
        annual_model_file = os.path.join(MODEL_DIR, f"Annualmean.BASE{first_year}.nc")

        if pyrsig is None:
            print("错误: pyrsig 未安装")
            continue

        ds_model = pyrsig.open_ioapi(annual_model_file)
        proj = pyproj.Proj(ds_model.crs_proj4)
        model_2d = ds_model['PM25'].values.squeeze()

        if np.sum(model_2d == 0) > 0:
            model_2d[model_2d == 0] = 0.1

        full_grid_df = ds_model[["ROW", "COL"]].to_dataframe().reset_index()[["ROW", "COL"]].copy()
        nrows, ncols = model_2d.shape

        COL_vals = full_grid_df['COL'].values.reshape(model_2d.shape)
        ROW_vals = full_grid_df['ROW'].values.reshape(model_2d.shape)
        lon_grid_2d, lat_grid_2d = _get_full_grid_coords(proj, COL_vals, ROW_vals, model_2d)

        # 聚合监测数据
        agg_data = []

        for date_str in date_list:
            date_obj = datetime.strptime(date_str, '%Y-%m-%d')
            year = date_obj.year
            monitor_file = os.path.join(MONITOR_DIR, f"{year}_DailyPM2.5Monitor.csv")

            if not os.path.exists(monitor_file):
                continue

            df_obs = pd.read_csv(monitor_file)
            if not pd.api.types.is_numeric_dtype(df_obs.get("Lat", pd.Series(dtype=float))):
                df_obs["Lat"] = pd.to_numeric(df_obs["Lat"], errors="coerce")
            if not pd.api.types.is_numeric_dtype(df_obs.get("Lon", pd.Series(dtype=float))):
                df_obs["Lon"] = pd.to_numeric(df_obs["Lon"], errors="coerce")
            df_obs["Date"] = pd.to_datetime(df_obs["Date"])

            day_df = df_obs[df_obs["Date"] == date_str].copy()
            day_df = day_df.dropna(subset=['Lat', 'Lon', 'Conc'])

            if len(day_df) < 10:
                continue

            # 站点投影与网格匹配
            day_df["x"], day_df["y"] = proj(day_df["Lon"], day_df["Lat"])
            row_idx, col_idx, _, _ = _map_obs_to_model_indices(day_df, model_2d, full_grid_df)

            # 获取站点对应的CMAQ值
            matched_model = model_2d[row_idx, col_idx]
            valid_mask = np.isfinite(matched_model)
            day_df = day_df.loc[valid_mask].copy().reset_index(drop=True)
            row_idx = row_idx[valid_mask]
            col_idx = col_idx[valid_mask]
            day_df['matched_model'] = matched_model[valid_mask]
            day_df['row_idx'] = row_idx
            day_df['col_idx'] = col_idx

            # 添加CMAQ网格坐标
            day_df['cmaq_lon'] = lon_grid_2d[day_df['row_idx'].values, day_df['col_idx'].values]
            day_df['cmaq_lat'] = lat_grid_2d[day_df['row_idx'].values, day_df['col_idx'].values]

            agg_data.append(day_df)

        if len(agg_data) == 0:
            print(f"  警告: 没有有效数据")
            continue

        # 合并所有天数据
        all_data = pd.concat(agg_data, ignore_index=True)
        print(f"  总站点数: {len(all_data)}")

        # 按站点聚合（取平均值）
        agg_df = all_data.groupby(['Site', 'row_idx', 'col_idx']).agg({
            'Lon': 'mean',
            'Lat': 'mean',
            'Conc': 'mean',
            'matched_model': 'mean',
            'cmaq_lon': 'mean',
            'cmaq_lat': 'mean'
        }).reset_index()

        print(f"  聚合后站点数: {len(agg_df)}")

        # 使用 fusion_and_evaluate 进行全网格融合 + 监测站评估
        X_eval = agg_df[['cmaq_lon', 'cmaq_lat']].values
        y_eval = agg_df['Conc'].values
        cmaq_eval = agg_df['matched_model'].values

        fusion_results, atf_metrics, grid_row_flat, grid_col_flat = fusion_and_evaluate(
            proj, model_2d, full_grid_df, COL_vals, ROW_vals,
            lon_grid_2d, lat_grid_2d, agg_df, agg_df,
            X_eval=X_eval, y_eval=y_eval, cmaq_eval=cmaq_eval
        )

        model_flat = model_2d.ravel()

        if not fusion_results:
            print(f"  警告: 融合失败")
            ds_model.close()
            continue

        # 打印评估
        if atf_metrics:
            r2_str = ' | '.join([f"{k}=R²{m['R2']:.4f}" for k, m in atf_metrics.items()])
            print(f"  评估: {r2_str}")

        # 生成输出
        for r in range(nrows):
            for c in range(ncols):
                idx = r * ncols + c
                results_list.append({
                    'ROW': r,
                    'COL': c,
                    'Period': period_name,
                    'model': model_flat[idx],
                    'vna': fusion_results['vna'][idx],
                    'evna': fusion_results['evna'][idx],
                    'avna': fusion_results['avna'][idx],
                    'advancedrk': fusion_results.get('advancedrk', np.full_like(model_flat, np.nan))[idx]
                })

        ds_model.close()
        print(f"  完成: {period_name}")

    if not results_list:
        return None

    result_df = pd.DataFrame(results_list)
    return result_df


def main():
    """主函数 - 配置融合任务"""
    print("="*70)
    print("全网格数据融合脚本")
    print("="*70)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ==================== FtA模式配置 ====================
    # 示例：2014年全年 FtA 融合
    fta_config = {
        'enabled': False,  # 设为 True 启用
        'date_range': ('2014-01-01', '2014-12-31'),
        'period_name': 'Annual_2014'
    }

    # ==================== AtF模式配置 ====================
    # 示例：多个期间分别融合
    atf_config = {
        'enabled': True,  # 设为 True 启用
        'periods': [
            {'name': 'Annual_2014', 'start': '2014-01-01', 'end': '2014-12-31'},
            {'name': 'Winter_2014_2015', 'start': '2014-12-01', 'end': '2015-02-28'},
            {'name': 'Summer_2014', 'start': '2014-06-01', 'end': '2014-08-31'},
            {'name': 'Annual_2015', 'start': '2015-01-01', 'end': '2015-12-31'},
            {'name': 'Annual_2020', 'start': '2020-01-01', 'end': '2020-12-31'},
        ]
    }

    # ==================== 执行 ====================
    if fta_config['enabled']:
        print("\n执行 FtA 模式...")
        result_df = run_fusion_fused_to_aggregate(
            fta_config['date_range'],
            fta_config['period_name']
        )

        if result_df is not None:
            output_file = os.path.join(
                OUTPUT_DIR,
                f"{fta_config['period_name']}_FtA_Geng_multimethod.csv"
            )
            result_df.to_csv(output_file, index=False)
            print(f"保存完成: {output_file}")
            print(f"行数: {len(result_df)}")

    if atf_config['enabled']:
        print("\n执行 AtF 模式...")
        result_df = run_fusion_aggregation_to_fuse(atf_config['periods'])

        if result_df is not None:
            # 合并所有Period的AtF结果
            output_file = os.path.join(
                OUTPUT_DIR,
                f"AtF_Geng_multimethod.csv"
            )
            result_df.to_csv(output_file, index=False)
            print(f"保存完成: {output_file}")
            print(f"总行数: {len(result_df)}")
            print(f"包含期间: {result_df['Period'].unique()}")

    print("\n" + "="*70)
    print("全部完成!")
    print("="*70)


if __name__ == '__main__':
    main()
