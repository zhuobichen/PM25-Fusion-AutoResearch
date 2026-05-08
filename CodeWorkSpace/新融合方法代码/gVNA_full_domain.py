# -*- coding: utf-8 -*-
"""
gVNA 全域插值脚本
=================
将全部监测站点数据插值到整个 CMAQ 网格

原理：
- 训练：全部监测点 → 学习空间偏差场
- λ：通过标准变异函数从偏差场估计空间相关距离
- 预测：CMAQ 网格每点 → 加性偏差校正

输出：
- 融合后的全域 PM2.5 网格 (NetCDF)
- 每日 λ 估计值

用法：
--------
python gVNA_full_domain.py                          # 单日演示（2020-01-01）
python gVNA_full_domain.py --date 2020-01-15       # 指定日期
python gVNA_full_domain.py --stage Jan             # 全月（Jan/Jul/Dec）
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import argparse
import numpy as np
import pandas as pd
import netCDF4 as nc
from datetime import datetime, timedelta
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from gVNA import gVNA

# ============ 数据路径 ============
CMAQ_FILE = data_path('test_data/raw/CMAQ/2020_PM25.nc')
MONITOR_FILE = data_path('test_data/raw/Monitor/2020_DailyPM2.5Monitor.csv')
OUTPUT_DIR = 'E:/CodeProject/ClaudeRoom/Data_Fusion_AutoResearch/test_result/gVNA_full_domain'


def load_day(day_str):
    """
    加载指定日期的监测数据和 CMAQ 数据

    参数:
        day_str: 日期字符串，格式 'YYYY-MM-DD'

    返回:
        day_df: 包含 Lon, Lat, Conc, CMAQ 的 DataFrame
        lon_cmaq, lat_cmaq, cmaq_grid: CMAQ 网格坐标和模拟值
    """
    monitor_df = pd.read_csv(MONITOR_FILE)
    day_df = monitor_df[monitor_df['Date'] == day_str].copy()
    day_df = day_df.dropna(subset=['Lat', 'Lon', 'Conc'])

    if len(day_df) < 100:
        print(f"  [警告] {day_str} 有效站点不足 (<100)，跳过")
        return None, None, None, None

    # 读取 CMAQ 网格
    ds = nc.Dataset(CMAQ_FILE, 'r')
    lon_cmaq = ds.variables['lon'][:]
    lat_cmaq = ds.variables['lat'][:]
    pred_pm25 = ds.variables['pred_PM25'][:]
    ds.close()

    day_idx = (datetime.strptime(day_str, '%Y-%m-%d') - datetime(2020, 1, 1)).days
    if day_idx >= pred_pm25.shape[0]:
        print(f"  [警告] {day_str} 超出 CMAQ 数据范围，跳过")
        return None, None, None, None

    cmaq_grid = pred_pm25[day_idx]

    # 将 CMAQ 匹配到监测站点
    cmaq_at_site = []
    for _, r in day_df.iterrows():
        idx = np.argmin(np.sqrt((lon_cmaq - r['Lon'])**2 + (lat_cmaq - r['Lat'])**2))
        ny, nx = lon_cmaq.shape
        cmaq_at_site.append(cmaq_grid[idx // nx, idx % nx])

    day_df = day_df.copy()
    day_df['CMAQ'] = cmaq_at_site
    day_df = day_df.dropna(subset=['CMAQ'])

    return day_df, lon_cmaq, lat_cmaq, cmaq_grid


def predict_full_domain(day_df, lon_cmaq, lat_cmaq, cmaq_grid):
    """
    使用 gVNA 将监测数据插值到 CMAQ 全域网格

    参数:
        day_df: 监测数据 DataFrame
        lon_cmaq, lat_cmaq: CMAQ 网格坐标
        cmaq_grid: CMAQ 模拟值网格 (ny, nx)

    返回:
        fused_grid: 融合后的 PM2.5 网格 (ny, nx)
        lambda_used: 当日 λ 估计值
    """
    train_lon = day_df['Lon'].values
    train_lat = day_df['Lat'].values
    train_Conc = day_df['Conc'].values
    train_CMAQ = day_df['CMAQ'].values

    # 训练 gVNA（变异函数估计 λ）
    model = gVNA(k=30, p=2, adaptive=False, lambda_method='variogram')
    model.fit(train_lon, train_lat, train_Conc, train_CMAQ)
    lambda_used = model.lambda_bg_estimated

    print(f"  站点数: {len(train_lon)}, λ (变异函数) = {lambda_used:.2f}")

    # 构建 CMAQ 网格坐标
    ny, nx = lon_cmaq.shape
    grid_coords = np.column_stack([
        lon_cmaq.ravel(),
        lat_cmaq.ravel()
    ])
    cmaq_flat = cmaq_grid.ravel()

    # 批量预测
    fused_flat = model.predict(grid_coords, mod=cmaq_flat)
    fused_grid = fused_flat.reshape(ny, nx)

    # 监测站点交叉验证 R2（训练集内）
    pred_at_stations = model.predict(
        np.column_stack([train_lon, train_lat]),
        mod=train_CMAQ
    )
    r2 = r2_score(train_Conc, pred_at_stations)
    mae = mean_absolute_error(train_Conc, pred_at_stations)
    rmse = np.sqrt(mean_squared_error(train_Conc, pred_at_stations))

    print(f"  训练集 R2 = {r2:.4f}, MAE = {mae:.2f}, RMSE = {rmse:.2f}")

    return fused_grid, lambda_used, r2


def save_netcdf(fused_grid, lon_cmaq, lat_cmaq, day_str, lambda_used, r2, output_path):
    """
    保存融合结果为 NetCDF 文件
    """
    ny, nx = fused_grid.shape

    with nc.Dataset(output_path, 'w', format='NETCDF4') as ds:
        # 维度
        ds.createDimension('lon', nx)
        ds.createDimension('lat', ny)

        # 坐标变量
        lon_var = ds.createVariable('lon', 'f4', ('lat', 'lon'))
        lon_var[:] = lon_cmaq
        lon_var.units = 'degrees_east'
        lon_var.long_name = 'longitude'

        lat_var = ds.createVariable('lat', 'f4', ('lat', 'lon'))
        lat_var[:] = lat_cmaq
        lat_var.units = 'degrees_north'
        lat_var.long_name = 'latitude'

        # 融合结果
        fused_var = ds.createVariable('PM25_fused', 'f4', ('lat', 'lon'))
        fused_var[:] = fused_grid
        fused_var.units = 'μg/m³'
        fused_var.long_name = 'gVNA fused PM2.5 concentration'

        # 原始 CMAQ
        cmaq_var = ds.createVariable('PM25_cmaq', 'f4', ('lat', 'lon'))
        cmaq_var[:] = np.nan  # 需要上层传入

        # 全局属性
        ds.title = 'gVNA Full-Domain PM2.5 Fusion'
        ds.date = day_str
        ds.lambda_variogram = float(lambda_used)
        ds.train_r2 = float(r2)
        ds.method = 'gVNA (variogram-based lambda)'


def run_single_day(day_str, output_dir=None):
    """
    处理单个日期
    """
    print(f"\n{'='*60}")
    print(f"处理日期: {day_str}")
    print('='*60)

    # 加载数据
    day_df, lon_cmaq, lat_cmaq, cmaq_grid = load_day(day_str)
    if day_df is None:
        return

    print(f"  有效监测站点: {len(day_df)}")

    # 全域预测
    fused_grid, lambda_used, r2 = predict_full_domain(
        day_df, lon_cmaq, lat_cmaq, cmaq_grid
    )

    # 保存
    if output_dir is None:
        output_dir = OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    safe_date = day_str.replace('-', '')
    output_path = os.path.join(output_dir, f'gVNA_fused_{safe_date}.nc')

    # 重新读取CMAQ用于保存
    ds_in = nc.Dataset(CMAQ_FILE, 'r')
    pred_pm25 = ds_in.variables['pred_PM25'][:]
    ds_in.close()
    day_idx = (datetime.strptime(day_str, '%Y-%m-%d') - datetime(2020, 1, 1)).days
    cmaq_day = pred_pm25[day_idx]

    ny, nx = fused_grid.shape
    with nc.Dataset(output_path, 'w', format='NETCDF4') as ds:
        ds.createDimension('lon', nx)
        ds.createDimension('lat', ny)

        lon_var = ds.createVariable('lon', 'f4', ('lat', 'lon'))
        lon_var[:] = lon_cmaq
        lon_var.units = 'degrees_east'

        lat_var = ds.createVariable('lat', 'f4', ('lat', 'lon'))
        lat_var[:] = lat_cmaq
        lat_var.units = 'degrees_north'

        fused_var = ds.createVariable('PM25_fused', 'f4', ('lat', 'lon'))
        fused_var[:] = fused_grid
        fused_var.units = 'μg/m³'
        fused_var.long_name = 'gVNA fused PM2.5 (variogram lambda)'

        cmaq_var = ds.createVariable('PM25_cmaq', 'f4', ('lat', 'lon'))
        cmaq_var[:] = cmaq_day
        cmaq_var.units = 'μg/m³'
        cmaq_var.long_name = 'CMAQ predicted PM2.5'

        ds.title = 'gVNA Full-Domain PM2.5 Fusion'
        ds.date = day_str
        ds.history = f'Created {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
        ds.lambda_variogram = float(lambda_used)
        ds.train_r2 = float(r2)

    print(f"  已保存: {output_path}")
    return lambda_used, r2


def run_stage(stage_name, output_dir=None):
    """
    处理整月数据
    """
    stage_config = {
        'Jan': ('2020-01-01', '2020-01-31'),
        'Jul': ('2020-07-01', '2020-07-31'),
        'Dec': ('2020-12-01', '2020-12-31'),
    }

    if stage_name not in stage_config:
        print(f"未知月份: {stage_name}，可用: {list(stage_config.keys())}")
        return

    start_str, end_str = stage_config[stage_name]
    dates = [(datetime.strptime(start_str, '%Y-%m-%d') + timedelta(days=i)).strftime('%Y-%m-%d')
             for i in range((datetime.strptime(end_str, '%Y-%m-%d') - datetime.strptime(start_str, '%Y-%m-%d')).days + 1)]

    print(f"\n{'='*60}")
    print(f"处理月份: {stage_name} ({len(dates)} 天)")
    print('='*60)

    lambda_list = []
    r2_list = []

    for i, d in enumerate(dates):
        day_df, lon_cmaq, lat_cmaq, cmaq_grid = load_day(d)
        if day_df is None:
            continue

        fused_grid, lam, r2 = predict_full_domain(day_df, lon_cmaq, lat_cmaq, cmaq_grid)
        lambda_list.append(lam)
        r2_list.append(r2)

        if output_dir is None:
            out = OUTPUT_DIR
        else:
            out = output_dir
        os.makedirs(out, exist_ok=True)

        safe_date = d.replace('-', '')
        output_path = os.path.join(out, f'gVNA_fused_{safe_date}.nc')

        ds_in = nc.Dataset(CMAQ_FILE, 'r')
        pred_pm25 = ds_in.variables['pred_PM25'][:]
        ds_in.close()
        day_idx = (datetime.strptime(d, '%Y-%m-%d') - datetime(2020, 1, 1)).days
        cmaq_day = pred_pm25[day_idx]

        ny, nx = fused_grid.shape
        with nc.Dataset(output_path, 'w', format='NETCDF4') as ds:
            ds.createDimension('lon', nx)
            ds.createDimension('lat', ny)
            ds.createVariable('lon', 'f4', ('lat', 'lon'))[:] = lon_cmaq
            ds.createVariable('lat', 'f4', ('lat', 'lon'))[:] = lat_cmaq
            fused_var = ds.createVariable('PM25_fused', 'f4', ('lat', 'lon'))
            fused_var[:] = fused_grid
            fused_var.units = 'μg/m³'
            ds.createVariable('PM25_cmaq', 'f4', ('lat', 'lon'))[:] = cmaq_day
            ds.title = 'gVNA Full-Domain PM2.5 Fusion'
            ds.date = d
            ds.lambda_variogram = float(lam)
            ds.train_r2 = float(r2)

        if (i + 1) % 10 == 0:
            print(f"  进度: {i+1}/{len(dates)}")

    print(f"\n{'='*60}")
    print(f"{stage_name} 月统计:")
    print(f"  平均 λ (变异函数) = {np.mean(lambda_list):.2f} (std={np.std(lambda_list):.2f})")
    print(f"  平均训练 R2       = {np.mean(r2_list):.4f}")
    print(f"  λ 范围            = [{np.min(lambda_list):.2f}, {np.max(lambda_list):.2f}]")
    print('='*60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='gVNA 全域 PM2.5 插值')
    parser.add_argument('--date', type=str, default='2020-01-01',
                        help='指定日期 (YYYY-MM-DD)')
    parser.add_argument('--stage', type=str, default=None,
                        help='处理整月: Jan / Jul / Dec')
    parser.add_argument('--output', type=str, default=None,
                        help='输出目录')

    args = parser.parse_args()

    if args.stage:
        run_stage(args.stage, args.output)
    else:
        run_single_day(args.date, args.output)
