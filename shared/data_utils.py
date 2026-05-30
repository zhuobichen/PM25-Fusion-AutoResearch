# -*- coding: utf-8 -*-
"""
数据加载和处理工具
================
提供统一的数据加载、处理和常用函数，避免代码重复。
"""

import os
import numpy as np
import pandas as pd
import netCDF4 as nc
from datetime import datetime
from shared.paths import get_project_root, data_path


def get_project_paths():
    """获取项目常用路径"""
    root_dir = str(get_project_root())
    paths = {
        'root': root_dir,
        'cmaq_file': data_path('test_data/raw/CMAQ/2020_PM25.nc'),
        'monitor_file': data_path('test_data/raw/Monitor/2020_DailyPM25Monitor.csv'),
        'fold_file': data_path('test_data/fold_split_table_daily.csv'),
        'output_dir': os.path.join(root_dir, 'test_result', '创新方法')
    }
    os.makedirs(paths['output_dir'], exist_ok=True)
    return paths


def get_cmaq_at_site(lon, lat, lon_grid, lat_grid, pm25_grid):
    """
    获取站点位置的 CMAQ 值
    
    Parameters
    ----------
    lon : float
        站点经度
    lat : float
        站点纬度
    lon_grid : ndarray
        CMAQ 经度网格
    lat_grid : ndarray
        CMAQ 纬度网格
    pm25_grid : ndarray
        CMAQ PM2.5 预测网格
    
    Returns
    -------
    float
        站点位置的 CMAQ 值
    """
    dist = np.sqrt((lon_grid - lon)**2 + (lat_grid - lat)**2)
    idx = np.argmin(dist)
    ny, nx = lon_grid.shape
    row, col = idx // nx, idx % nx
    return pm25_grid[row, col]


def load_daily_data(selected_day, paths=None):
    """
    加载单日数据
    
    Parameters
    ----------
    selected_day : str
        日期字符串，格式 'YYYY-MM-DD'
    paths : dict, optional
        路径字典，默认调用 get_project_paths()
    
    Returns
    -------
    tuple
        (day_df, lon_cmaq, lat_cmaq, pred_day)
    """
    if paths is None:
        paths = get_project_paths()
    
    # 加载监测数据
    monitor_df = pd.read_csv(paths['monitor_file'])
    fold_df = pd.read_csv(paths['fold_file'])
    
    day_df = monitor_df[monitor_df['Date'] == selected_day].copy()
    day_df = day_df.merge(fold_df, on=['Date', 'Site'], how='left')
    day_df = day_df.dropna(subset=['Lat', 'Lon', 'Conc'])
    
    # 加载CMAQ数据
    ds = nc.Dataset(paths['cmaq_file'], 'r')
    lon_cmaq = ds.variables['lon'][:]
    lat_cmaq = ds.variables['lat'][:]
    pred_pm25 = ds.variables['pred_PM25'][:]
    ds.close()
    
    date_obj = datetime.strptime(selected_day, '%Y-%m-%d')
    day_idx = (date_obj - datetime(2020, 1, 1)).days
    if day_idx >= pred_pm25.shape[0]:
        return None, None, None, None
    
    pred_day = pred_pm25[day_idx]
    
    # 提取站点CMAQ值
    cmaq_values = []
    for _, row in day_df.iterrows():
        val = get_cmaq_at_site(row['Lon'], row['Lat'], lon_cmaq, lat_cmaq, pred_day)
        cmaq_values.append(val)
    day_df['CMAQ'] = cmaq_values
    
    return day_df, lon_cmaq, lat_cmaq, pred_day


def extract_cmaq_for_sites(day_df, lon_cmaq, lat_cmaq, pred_day):
    """
    为所有站点提取 CMAQ 值（批量处理）
    
    Parameters
    ----------
    day_df : DataFrame
        包含站点位置的 DataFrame
    lon_cmaq : ndarray
        CMAQ 经度网格
    lat_cmaq : ndarray
        CMAQ 纬度网格
    pred_day : ndarray
        当天的 CMAQ 预测
    
    Returns
    -------
    DataFrame
        添加了 CMAQ 列的 DataFrame
    """
    cmaq_values = []
    for _, row in day_df.iterrows():
        val = get_cmaq_at_site(row['Lon'], row['Lat'], lon_cmaq, lat_cmaq, pred_day)
        cmaq_values.append(val)
    day_df['CMAQ'] = cmaq_values
    return day_df


def get_cmaq_grid(lon_cmaq, lat_cmaq):
    """
    获取完整的 CMAQ 网格坐标
    
    Parameters
    ----------
    lon_cmaq : ndarray
        CMAQ 经度网格
    lat_cmaq : ndarray
        CMAQ 纬度网格
    
    Returns
    -------
    ndarray
        形状为 (N, 2) 的网格坐标数组
    """
    ny, nx = lon_cmaq.shape
    return np.column_stack([lon_cmaq.ravel(), lat_cmaq.ravel()])
