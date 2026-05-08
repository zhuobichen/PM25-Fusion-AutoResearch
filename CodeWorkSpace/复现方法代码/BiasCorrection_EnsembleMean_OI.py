"""
BiasCorrection, EnsembleMean, OptimumInterpolation 融合方法实现
================================================================

BiasCorrection - 偏差校正法家族 (Mean/Spatial/Scale/Linear)
EnsembleMean - 集合平均
OptimumInterpolation - 最优插值
"""

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.linear_model import LinearRegression


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


def bias_correction_fuse(cmaq_grid_values, station_obs, station_cmaq, station_coords, grid_coords, params):
    """
    偏差校正法家族

    支持4种变体:
    1. Mean BC: P = M + mean(O - M)
    2. Spatial BC: P = M + IDW(O - M)
    3. Scale BC: P = M * mean(O/M)
    4. Linear BC: P = a + b*M (OLS)

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
        参数: bc_type ('mean'/'spatial'/'scale'/'linear'), k, power

    Returns:
    --------
    fused : np.ndarray
        融合结果
    """
    bc_type = params.get('bc_type', 'spatial')
    k = params.get('k', 30)
    power = params.get('power', -2)

    if bc_type == 'mean':
        # Mean BC: P = M + mean(O - M)
        bias = station_obs - station_cmaq
        mean_bias = np.mean(bias)
        fused = cmaq_grid_values + mean_bias

    elif bc_type == 'spatial':
        # Spatial BC: P = M + IDW(O - M)
        bias = station_obs - station_cmaq
        n_grid = grid_coords.shape[0]
        bias_interp = np.zeros(n_grid)

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

            bias_interp[i] = np.sum(weights * bias[idx])

        fused = cmaq_grid_values + bias_interp

    elif bc_type == 'scale':
        # Scale BC: P = M * mean(O/M)
        station_cmaq_safe = np.maximum(station_cmaq, 1e-6)
        ratio = station_obs / station_cmaq_safe
        mean_ratio = np.mean(ratio)
        fused = cmaq_grid_values * mean_ratio

    elif bc_type == 'linear':
        # Linear BC: P = a + b*M
        model = LinearRegression()
        model.fit(station_cmaq.reshape(-1, 1), station_obs)
        fused = model.predict(cmaq_grid_values.reshape(-1, 1))

    else:
        raise ValueError(f"Unknown bc_type: {bc_type}")

    return fused


def ensemble_mean_fuse(cmaq_grid_values, station_obs, station_cmaq, station_coords, grid_coords, params):
    """
    集合平均

    核心思想: 简单平均多个模型/方法的预测

    这里实现为: P = (M + IDW(O)) / 2
    即CMAQ模型和空间插值的平均

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
        参数: weights (可选权重)

    Returns:
    --------
    fused : np.ndarray
        融合结果
    """
    k = params.get('k', 30)
    power = params.get('power', -2)

    # 方法1: IDW插值观测值
    n_grid = grid_coords.shape[0]
    idw_interp = np.zeros(n_grid)

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

        idw_interp[i] = np.sum(weights * station_obs[idx])

    # 集合平均: P = (M + IDW(O)) / 2
    fused = (cmaq_grid_values + idw_interp) / 2

    return fused


def optimum_interpolation_fuse(cmaq_grid_values, station_obs, station_cmaq, station_coords, grid_coords, params):
    """
    最优插值 (OI)

    核心公式:
    P(s0) = M(s0) + K * (O - M(s_obs))
    其中 K = B * H^T * (H * B * H^T + R)^{-1}

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
        参数: correlation_length, obs_error_var

    Returns:
    --------
    fused : np.ndarray
        融合结果
    """
    correlation_length = params.get('correlation_length', 1.0)
    obs_error_var = params.get('obs_error_var', 1.0)

    n_stations = station_coords.shape[0]
    n_grid = grid_coords.shape[0]

    # 计算背景误差协方差 B (使用指数相关函数)
    # B(i,j) = exp(-d(i,j)/L)
    B = np.zeros((n_stations, n_stations))
    for i in range(n_stations):
        for j in range(n_stations):
            d = np.sqrt(np.sum((station_coords[i] - station_coords[j]) ** 2))
            B[i, j] = np.exp(-d / correlation_length)

    # 观测误差协方差 R
    R = obs_error_var * np.eye(n_stations)

    # 计算增量
    innovation = station_obs - station_cmaq

    # 计算最优增益 K
    # K = B * (B + R)^{-1}
    try:
        K = B @ np.linalg.inv(B + R)
    except np.linalg.LinAlgError:
        K = B @ np.linalg.pinv(B + R)

    # 分析增量
    analysis_innovation = K @ innovation

    # 对网格点进行插值
    fused = np.zeros(n_grid)

    for i, x0 in enumerate(grid_coords):
        # 计算网格点与站点的相关性
        corr = np.zeros(n_stations)
        for j in range(n_stations):
            d = np.sqrt(np.sum((x0 - station_coords[j]) ** 2))
            corr[j] = np.exp(-d / correlation_length)

        # 插值增量
        increment = corr @ analysis_innovation
        fused[i] = cmaq_grid_values[i] + increment

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
        参数，包含 method ('BiasCorrection'/'EnsembleMean'/'OI')

    Returns:
    --------
    fused : np.ndarray
        融合结果
    """
    method = params.get('method', 'BiasCorrection')

    cmaq_grid_values = cmaq_data['grid_values']
    grid_coords = cmaq_data['coords']
    station_obs = station_data['obs']
    station_cmaq = station_data['cmaq']

    if method == 'BiasCorrection':
        return bias_correction_fuse(cmaq_grid_values, station_obs, station_cmaq, station_coords, grid_coords, params)
    elif method == 'EnsembleMean':
        return ensemble_mean_fuse(cmaq_grid_values, station_obs, station_cmaq, station_coords, grid_coords, params)
    elif method == 'OI':
        return optimum_interpolation_fuse(cmaq_grid_values, station_obs, station_cmaq, station_coords, grid_coords, params)
    else:
        raise ValueError(f"Unknown method: {method}")
