"""
NC, Ki-CDPM, BSMFM 融合方法实现
================================

NC - 华北WRF-Chem多源融合方法
Ki-CDPM - 克里金信息扩散降尺度法
BSMFM - 贝叶斯多源融合模型
"""

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
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


def nc_fuse(cmaq_grid_values, station_obs, station_cmaq, station_coords, grid_coords, params):
    """
    华北多源融合方法 (NC)

    核心思想: 基于贝叶斯加权融合，考虑空间相关性和模型偏差的空间变化

    公式: P_NC = M + w * (O - M)
    权重w通过高斯核平滑确定

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
        参数: bandwidth (带宽), k_neighbors

    Returns:
    --------
    fused : np.ndarray
        融合结果
    """
    bandwidth = params.get('bandwidth', None)
    k = params.get('k', 30)

    # 计算站点偏差
    bias = station_obs - station_cmaq

    # 自动选择带宽 (通过交叉验证)
    if bandwidth is None:
        # 简化: 使用平均最近邻距离的2倍
        if len(station_coords) > 1:
            dists = cdist(station_coords, station_coords, 'euclidean')
            np.fill_diagonal(dists, np.inf)
            min_dists = np.min(dists, axis=1)
            bandwidth = np.mean(min_dists) * 2
        else:
            bandwidth = 1.0

    # 使用高斯核进行加权插值
    n_grid = grid_coords.shape[0]
    fused = np.zeros(n_grid)

    for i, x0 in enumerate(grid_coords):
        dists = cdist([x0], station_coords, 'euclidean').ravel()

        # 选择k个最近邻
        if len(dists) > k:
            idx = np.argpartition(dists, k)[:k]
        else:
            idx = np.arange(len(dists))

        dists_k = dists[idx]

        # 高斯核权重
        weights = np.exp(-dists_k ** 2 / (2 * bandwidth ** 2))
        weights /= weights.sum()

        # 加权偏差
        weighted_bias = np.sum(weights * bias[idx])
        fused[i] = cmaq_grid_values[i] + weighted_bias

    return fused


def kicdpm_fuse(cmaq_grid_values, station_obs, station_cmaq, station_coords, grid_coords, params):
    """
    Ki-CDPM - 克里金信息扩散降尺度法 (简化版)

    核心思想: 使用克里金空间结构信息引导降尺度

    简化实现:
    1. 计算空间变异函数
    2. 使用克里金进行空间插值
    3. 结合CMAQ进行条件融合

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
        参数: guidance_weight, range_param

    Returns:
    --------
    fused : np.ndarray
        融合结果
    """
    guidance_weight = params.get('guidance_weight', 0.3)

    # 计算站点偏差
    bias = station_obs - station_cmaq

    # 使用GPR进行空间克里金插值
    kernel = ConstantKernel(1.0, (1e-2, 1e2)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(noise_level=0.1)

    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, alpha=0.1, normalize_y=True)
    gpr.fit(station_coords, bias)

    # 预测网格点偏差
    bias_pred, bias_std = gpr.predict(grid_coords, return_std=True)

    # 条件融合: 结合CMAQ和克里金
    # P = (1-w) * (M + bias_kriging) + w * M
    # 简化: P = M + (1-w) * bias_kriging
    fused = cmaq_grid_values + (1 - guidance_weight) * bias_pred

    return fused


def bsmfm_fuse(cmaq_grid_values, station_obs, station_cmaq, station_coords, grid_coords, params):
    """
    BSMFM - 贝叶斯多源融合模型 (简化版)

    核心思想: 贝叶斯层次模型，考虑共享随机场和源特异性偏差

    简化实现:
    1. 全局偏差校正: O = a + b*M
    2. 残差空间建模: 使用GPR
    3. 贝叶斯融合: 加权平均

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
        参数: spatial_range, temporal_range

    Returns:
    --------
    fused : np.ndarray
        融合结果
    """
    # 步骤1: 全局偏差校正 (线性回归)
    model = LinearRegression()
    model.fit(station_cmaq.reshape(-1, 1), station_obs)
    a, b = model.intercept_, model.coef_[0]

    # 计算校正后的CMAQ
    cmaq_corrected = a + b * cmaq_grid_values

    # 步骤2: 计算残差
    residual = station_obs - (a + b * station_cmaq)

    # 步骤3: 残差空间建模 (GPR)
    kernel = ConstantKernel(1.0, (1e-2, 1e2)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(noise_level=0.1)

    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, alpha=0.1, normalize_y=True)
    gpr.fit(station_coords, residual)

    # 预测网格点残差
    residual_pred, residual_std = gpr.predict(grid_coords, return_std=True)

    # 步骤4: 贝叶斯融合 (简化为加权平均)
    # 考虑不确定性，对残差预测进行收缩
    uncertainty_weight = 1.0 / (1.0 + residual_std ** 2)

    # 最终融合
    fused = cmaq_corrected + uncertainty_weight * residual_pred

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
        参数，包含 method ('NC'/'KiCDPM'/'BSMFM')

    Returns:
    --------
    fused : np.ndarray
        融合结果
    """
    method = params.get('method', 'NC')

    cmaq_grid_values = cmaq_data['grid_values']
    grid_coords = cmaq_data['coords']
    station_obs = station_data['obs']
    station_cmaq = station_data['cmaq']

    if method == 'NC':
        return nc_fuse(cmaq_grid_values, station_obs, station_cmaq, station_coords, grid_coords, params)
    elif method == 'KiCDPM':
        return kicdpm_fuse(cmaq_grid_values, station_obs, station_cmaq, station_coords, grid_coords, params)
    elif method == 'BSMFM':
        return bsmfm_fuse(cmaq_grid_values, station_obs, station_cmaq, station_coords, grid_coords, params)
    else:
        raise ValueError(f"Unknown method: {method}")
