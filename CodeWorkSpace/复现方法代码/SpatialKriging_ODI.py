"""
SpatialKriging, ODI 融合方法实现
================================

SpatialKriging - 空间克里金偏差校正
ODI - Observation Departure Indicator (观测偏差指示器融合)
"""

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel


def calculate_metrics(y_true, y_pred):
    """计算R2、MAE、RMSE、MB"""
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if len(y_true) == 0:
        return {'R2': np.nan, 'MAE': np.nan, 'RMSE': np.nan, 'MB': np.nan}

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mb = np.mean(y_pred - y_true)

    return {'R2': r2, 'MAE': mae, 'RMSE': rmse, 'MB': mb}


def spatial_kriging_fuse(cmaq_grid_values, station_obs, station_cmaq, station_coords, grid_coords, params):
    """
    空间克里金偏差校正

    步骤:
    1. 计算站点偏差 B = O - M
    2. 对偏差进行克里金插值
    3. 融合: P = M + B_kriged

    Parameters:
    -----------
    cmaq_grid_values : np.ndarray
        CMAQ网格值
    station_obs : np.ndarray
        站点观测值
    station_cmaq : np.ndarray
        站点处CMAQ值
    station_coords : np.ndarray
        站点坐标
    grid_coords : np.ndarray
        网格坐标
    params : dict
        参数: variogram_model, n_neighbors

    Returns:
    --------
    fused : np.ndarray
        融合结果
    """
    n_neighbors = params.get('n_neighbors', 30)

    # 步骤1: 计算站点偏差
    bias = station_obs - station_cmaq

    # 步骤2: 使用GPR进行偏差克里金
    kernel = ConstantKernel(1.0, (1e-2, 1e2)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(noise_level=0.1)

    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, alpha=0.1, normalize_y=True)
    gpr.fit(station_coords, bias)

    # 步骤3: 预测网格点偏差
    bias_pred, bias_std = gpr.predict(grid_coords, return_std=True)

    # 步骤4: 融合
    fused = cmaq_grid_values + bias_pred

    return fused


def odi_fuse(cmaq_grid_values, station_obs, station_cmaq, station_coords, grid_coords, params):
    """
    ODI - 观测偏差指示器融合

    核心思想: 使用CMAQ作为偏差指示器，建立观测与模型的统计关系

    步骤:
    1. 计算偏差指示器: DI = O / M (或 O - M)
    2. 对DI进行空间插值
    3. 融合: P = M * DI_interp (乘性) 或 P = M + DI_interp (加性)

    Parameters:
    -----------
    cmaq_grid_values : np.ndarray
        CMAQ网格值
    station_obs : np.ndarray
        站点观测值
    station_cmaq : np.ndarray
        站点处CMAQ值
    station_coords : np.ndarray
        站点坐标
    grid_coords : np.ndarray
        网格坐标
    params : dict
        参数: indicator_type ('multiplicative'/'additive'), k, power

    Returns:
    --------
    fused : np.ndarray
        融合结果
    """
    indicator_type = params.get('indicator_type', 'additive')
    k = params.get('k', 30)
    power = params.get('power', -2)

    # 步骤1: 计算偏差指示器
    if indicator_type == 'multiplicative':
        # 乘性指示器: DI = O / M
        # 避免除零
        station_cmaq_safe = np.maximum(station_cmaq, 1e-6)
        indicator = station_obs / station_cmaq_safe
    else:
        # 加性指示器: DI = O - M
        indicator = station_obs - station_cmaq

    # 步骤2: 对指示器进行IDW插值
    n_grid = grid_coords.shape[0]
    indicator_interp = np.zeros(n_grid)

    for i, x0 in enumerate(grid_coords):
        dists = cdist([x0], station_coords, 'euclidean').ravel()
        dists = np.maximum(dists, 1e-6)

        if len(dists) > k:
            idx = np.argpartition(dists, k)[:k]
        else:
            idx = np.arange(len(dists))

        dists_k = dists[idx]
        weights = dists_k ** (-power) if power < 0 else 1.0 / (dists_k ** power)
        weights /= weights.sum()

        indicator_interp[i] = np.sum(weights * indicator[idx])

    # 步骤3: 融合
    if indicator_type == 'multiplicative':
        fused = cmaq_grid_values * indicator_interp
    else:
        fused = cmaq_grid_values + indicator_interp

    return fused


def fuse_method(cmaq_data, station_data, station_coords, params):
    """
    统一融合接口

    Parameters:
    -----------
    cmaq_data : dict
        CMAQ数据
    station_data : dict
        站点数据
    station_coords : np.ndarray
        站点坐标
    params : dict
        参数，包含 method ('SpatialKriging'/'ODI')

    Returns:
    --------
    fused : np.ndarray
        融合结果
    """
    method = params.get('method', 'SpatialKriging')

    cmaq_grid_values = cmaq_data['grid_values']
    grid_coords = cmaq_data['coords']
    station_obs = station_data['obs']
    station_cmaq = station_data['cmaq']

    if method == 'SpatialKriging':
        return spatial_kriging_fuse(cmaq_grid_values, station_obs, station_cmaq, station_coords, grid_coords, params)
    elif method == 'ODI':
        return odi_fuse(cmaq_grid_values, station_obs, station_cmaq, station_coords, grid_coords, params)
    else:
        raise ValueError(f"Unknown method: {method}")
