# -*- coding: utf-8 -*-
"""
地理空间工具函数
================
提供项目中复用的地理空间计算函数。

使用方式：
    from shared.geo_utils import get_cmaq_at_site, haversine_distance, idw_interpolate
"""

import numpy as np
from scipy.spatial.distance import cdist


def get_cmaq_at_site(lon, lat, lon_grid, lat_grid, pm25_grid):
    """
    获取监测站点位置的 CMAQ 模型值（最近邻查找）。

    参数:
        lon, lat: 站点经纬度 (标量)
        lon_grid, lat_grid: CMAQ 网格经纬度 (2D array)
        pm25_grid: CMAQ PM2.5 预测值 (2D array)

    返回:
        float: 最近网格点的 CMAQ 值
    """
    dist = np.sqrt((lon_grid - lon) ** 2 + (lat_grid - lat) ** 2)
    idx = np.argmin(dist)
    ny, nx = lon_grid.shape
    row, col = idx // nx, idx % nx
    return pm25_grid[row, col]


def get_cmaq_grid_coord(lon, lat, lon_grid, lat_grid):
    """
    获取监测站点对应的 CMAQ 网格中心坐标（最近邻）。

    参数:
        lon, lat: 站点经纬度 (标量)
        lon_grid, lat_grid: CMAQ 网格经纬度 (2D array)

    返回:
        tuple: (lon, lat) 最近网格点的经纬度
    """
    dist = np.sqrt((lon_grid - lon) ** 2 + (lat_grid - lat) ** 2)
    idx = np.argmin(dist)
    ny, nx = lon_grid.shape
    row, col = idx // nx, idx % nx
    return lon_grid[row, col], lat_grid[row, col]


def haversine_distance(lon1, lat1, lon2, lat2):
    """
    使用 Haversine 公式计算两点间的球面距离。

    参数:
        lon1, lat1: 第一个点的经纬度 (度)
        lon2, lat2: 第二个点的经纬度 (度)

    返回:
        float: 距离 (km)
    """
    R = 6371.0  # 地球半径 (km)

    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)

    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return R * c


def idw_interpolate(x_obs, values, x_pred, power=-2, k=None):
    """
    通用 IDW (Inverse Distance Weighting) 插值。

    参数:
        x_obs: 观测点坐标 (n_obs, 2)
        values: 观测点值 (n_obs,)
        x_pred: 预测点坐标 (n_pred, 2)
        power: 距离权重指数 (默认 -2，即 1/d^2)
        k: 近邻数量 (默认 None，使用全部观测点)

    返回:
        ndarray: 预测值 (n_pred,)
    """
    n_pred = x_pred.shape[0]
    n_obs = x_obs.shape[0]
    result = np.zeros(n_pred)

    if k is None:
        k = n_obs

    for i in range(n_pred):
        dists = cdist([x_pred[i]], x_obs, 'euclidean').ravel()

        if n_obs > k:
            idx = np.argpartition(dists, k)[:k]
        else:
            idx = np.arange(n_obs)

        dists_k = dists[idx]
        dists_k = np.maximum(dists_k, 1e-6)  # 避免除零

        # 处理正/负 power
        if power < 0:
            weights = dists_k ** (-power)
        else:
            weights = 1.0 / (dists_k ** power)

        weights /= weights.sum()
        result[i] = np.sum(weights * values[idx])

    return result
