"""
Quantile Mapping (QM) - 分位数映射偏差校正
==========================================

通过映射模型输出分布到观测分布进行非参数偏差校正
"""

import numpy as np
from scipy.spatial.distance import cdist
from scipy import stats


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


def quantile_mapping_fuse(cmaq_grid_values, station_obs, station_cmaq, station_coords, grid_coords, params):
    """
    分位数映射融合

    步骤:
    1. 计算O和M的均值、标准差
    2. 全局QM: P = mean_O + (std_O/std_M) * (M - mean_M)
    3. 分位数映射: 对每个分位数q计算映射关系

    Parameters:
    -----------
    cmaq_grid_values : np.ndarray
        CMAQ网格值，shape (n_grid,)
    station_obs : np.ndarray
        站点观测值，shape (n_stations,)
    station_cmaq : np.ndarray
        站点处CMAQ值，shape (n_stations,)
    station_coords : np.ndarray
        站点坐标，shape (n_stations, 2)
    grid_coords : np.ndarray
        网格坐标，shape (n_grid, 2)
    params : dict
        参数: n_quantiles (分位数数量), method ('linear'/'spline'/'local')

    Returns:
    --------
    fused : np.ndarray
        融合结果，shape (n_grid,)
    """
    n_quantiles = params.get('n_quantiles', 10)
    method = params.get('method', 'linear')
    k = params.get('k', 30)
    power = params.get('power', -2)

    # 步骤1: 计算统计量
    mean_obs = np.mean(station_obs)
    mean_cmaq = np.mean(station_cmaq)
    std_obs = np.std(station_obs)
    std_cmaq = np.std(station_cmaq)

    # 步骤2: 全局QM (均值-方差校正)
    if std_cmaq > 0:
        global_qm = mean_obs + (std_obs / std_cmaq) * (cmaq_grid_values - mean_cmaq)
    else:
        global_qm = np.full_like(cmaq_grid_values, mean_obs)

    if method == 'global':
        return global_qm

    # 步骤3: 分位数映射
    # 计算分位数
    quantiles = np.linspace(0, 1, n_quantiles + 1)

    # 计算观测和模型的分位数
    obs_quantiles = np.quantile(station_obs, quantiles)
    cmaq_quantiles = np.quantile(station_cmaq, quantiles)

    # 对每个分位数区间拟合线性映射
    # F_O^{-1}(q) = a_q + b_q * F_M^{-1}(q)
    a_q = np.zeros(n_quantiles)
    b_q = np.zeros(n_quantiles)

    for i in range(n_quantiles):
        # 使用分位数点拟合
        if i == 0:
            x_pts = [cmaq_quantiles[i], cmaq_quantiles[i + 1]]
            y_pts = [obs_quantiles[i], obs_quantiles[i + 1]]
        else:
            x_pts = [cmaq_quantiles[i], cmaq_quantiles[i + 1]]
            y_pts = [obs_quantiles[i], obs_quantiles[i + 1]]

        # 线性拟合
        if x_pts[1] - x_pts[0] > 1e-6:
            b_q[i] = (y_pts[1] - y_pts[0]) / (x_pts[1] - x_pts[0])
            a_q[i] = y_pts[0] - b_q[i] * x_pts[0]
        else:
            a_q[i] = (y_pts[0] + y_pts[1]) / 2
            b_q[i] = 1.0

    # 步骤4: 对网格点应用分位数映射
    n_grid = cmaq_grid_values.shape[0]
    fused = np.zeros(n_grid)

    for i, m_val in enumerate(cmaq_grid_values):
        # 找到M值对应的分位数
        q_idx = np.searchsorted(cmaq_quantiles[1:], m_val)
        q_idx = min(q_idx, n_quantiles - 1)

        # 应用映射
        fused[i] = a_q[q_idx] + b_q[q_idx] * m_val

    # 边界处理: 限制在观测范围内
    fused = np.clip(fused, np.min(station_obs), np.max(station_obs))

    return fused


def local_quantile_mapping_fuse(cmaq_grid_values, station_obs, station_cmaq, station_coords, grid_coords, params):
    """
    局部分位数映射

    对每个网格点，使用近邻站点的分位数映射
    """
    k = params.get('k', 30)
    power = params.get('power', -2)
    n_quantiles = params.get('n_quantiles', 10)

    n_grid = grid_coords.shape[0]
    fused = np.zeros(n_grid)

    for i, x0 in enumerate(grid_coords):
        # 计算到所有站点的距离
        dists = cdist([x0], station_coords, 'euclidean').ravel()
        dists = np.maximum(dists, 1e-6)

        # 选择k个最近邻
        if len(dists) > k:
            idx = np.argpartition(dists, k)[:k]
        else:
            idx = np.arange(len(dists))

        dists_k = dists[idx]
        weights = dists_k ** (-power) if power < 0 else 1.0 / (dists_k ** power)
        weights /= weights.sum()

        # 局部统计量
        local_obs = station_obs[idx]
        local_cmaq = station_cmaq[idx]

        mean_obs = np.mean(local_obs)
        mean_cmaq = np.mean(local_cmaq)
        std_obs = np.std(local_obs)
        std_cmaq = np.std(local_cmaq)

        # 局部QM
        if std_cmaq > 0:
            fused[i] = mean_obs + (std_obs / std_cmaq) * (cmaq_grid_values[i] - mean_cmaq)
        else:
            fused[i] = mean_obs

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
        参数，包含 method ('global'/'local'/'quantile')

    Returns:
    --------
    fused : np.ndarray
        融合结果
    """
    method = params.get('method', 'global')

    cmaq_grid_values = cmaq_data['grid_values']
    grid_coords = cmaq_data['coords']
    station_obs = station_data['obs']
    station_cmaq = station_data['cmaq']

    if method == 'local':
        return local_quantile_mapping_fuse(cmaq_grid_values, station_obs, station_cmaq, station_coords, grid_coords, params)
    else:
        return quantile_mapping_fuse(cmaq_grid_values, station_obs, station_cmaq, station_coords, grid_coords, params)
