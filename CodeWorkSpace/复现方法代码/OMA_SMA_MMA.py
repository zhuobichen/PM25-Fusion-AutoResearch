"""
OMA, SMA, MMA 融合方法实现
==========================

OMA - Observational Model Aggregation (观测模型聚合)
SMA - Statistical Model Aggregation (统计模型聚合)
MMA - Mixed Model Aggregation (混合模型聚合)

文献来源: Shao et al. "融合观测数据与化学传输模型模拟以估算时空分辨率的环境空气污染的方法"
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


def oma_fuse(cmaq_grid_values, station_obs, station_cmaq, station_coords, grid_coords, params):
    """
    OMA - 观测模型聚合

    公式: P_OMA(s0) = M(s0) + alpha * [O(si) - M(si)]
    alpha 通过最小化目标函数优化

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
        参数: method ('global'/'local'), k (近邻数), power (距离指数)

    Returns:
    --------
    fused : np.ndarray
        融合结果，shape (n_grid,)
    """
    method = params.get('method', 'global')
    k = params.get('k', 30)
    power = params.get('power', -2)

    # 计算站点偏差
    bias = station_obs - station_cmaq

    if method == 'global':
        # 全局OMA: 优化单一alpha
        # min sum[(O - alpha*O - (1-alpha)*M)^2] = min sum[(bias - alpha*bias)^2]
        # 解析解: alpha = sum(bias^2) / sum(bias^2) = 1 (退化)
        # 使用偏差形式: P = M + alpha * mean(bias)
        alpha = params.get('alpha', 1.0)
        mean_bias = np.mean(bias)
        fused = cmaq_grid_values + alpha * mean_bias
    else:
        # 局部OMA: 对每个网格点使用近邻加权偏差
        n_grid = grid_coords.shape[0]
        fused = np.zeros(n_grid)

        for i, x0 in enumerate(grid_coords):
            dists = cdist([x0], station_coords, 'euclidean').ravel()
            dists = np.maximum(dists, 1e-6)

            # 选择k个最近邻
            if len(dists) > k:
                idx = np.argpartition(dists, k)[:k]
            else:
                idx = np.arange(len(dists))

            dists_k = dists[idx]
            # IDW权重
            weights = dists_k ** (-power) if power < 0 else 1.0 / (dists_k ** power)
            weights /= weights.sum()

            # 加权偏差
            weighted_bias = np.sum(weights * bias[idx])
            fused[i] = cmaq_grid_values[i] + weighted_bias

    return fused


def sma_fuse(cmaq_grid_values, station_obs, station_cmaq, params):
    """
    SMA - 统计模型聚合

    公式: P_SMA(s0) = a + b * M(s0)
    其中 a, b 由OLS拟合 O = a + b*M

    Parameters:
    -----------
    cmaq_grid_values : np.ndarray
        CMAQ网格值，shape (n_grid,)
    station_obs : np.ndarray
        站点观测值，shape (n_stations,)
    station_cmaq : np.ndarray
        站点处CMAQ值，shape (n_stations,)
    params : dict
        参数: regression_type ('linear'/'polynomial'), poly_degree

    Returns:
    --------
    fused : np.ndarray
        融合结果，shape (n_grid,)
    """
    regression_type = params.get('regression_type', 'linear')
    poly_degree = params.get('poly_degree', 1)

    # OLS拟合
    model = LinearRegression()

    if regression_type == 'polynomial' and poly_degree > 1:
        # 多项式回归
        X_train = np.column_stack([station_cmaq ** d for d in range(1, poly_degree + 1)])
        X_pred = np.column_stack([cmaq_grid_values ** d for d in range(1, poly_degree + 1)])
    else:
        # 线性回归
        X_train = station_cmaq.reshape(-1, 1)
        X_pred = cmaq_grid_values.reshape(-1, 1)

    model.fit(X_train, station_obs)
    fused = model.predict(X_pred)

    return fused


def mma_fuse(cmaq_grid_values, station_obs, station_cmaq, station_coords, grid_coords, params):
    """
    MMA - 混合模型聚合

    公式: P_MMA(s0) = M(s0) + beta * D_SMA(s0) + (1-beta) * D_VNA(s0)
    其中:
    - D_SMA = a + b*M - M (全局偏差校正)
    - D_VNA = sum(wi * (Oi - Mi)) (局部偏差插值)

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
        参数: beta (混合参数), k, power

    Returns:
    --------
    fused : np.ndarray
        融合结果，shape (n_grid,)
    """
    beta = params.get('beta', 0.5)
    k = params.get('k', 30)
    power = params.get('power', -2)

    # 步骤1: OLS回归拟合 O = a + b*M
    model = LinearRegression()
    model.fit(station_cmaq.reshape(-1, 1), station_obs)
    a, b = model.intercept_, model.coef_[0]

    # 步骤2: 计算局部偏差插值 D_VNA
    bias = station_obs - station_cmaq
    n_grid = grid_coords.shape[0]
    d_vna = np.zeros(n_grid)

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

        d_vna[i] = np.sum(weights * bias[idx])

    # 步骤3: 计算全局偏差校正 D_SMA
    d_sma = a + (b - 1) * cmaq_grid_values

    # 步骤4: 混合融合
    fused = cmaq_grid_values + beta * d_sma + (1 - beta) * d_vna

    return fused


def fuse_method(cmaq_data, station_data, station_coords, params):
    """
    统一融合接口

    Parameters:
    -----------
    cmaq_data : dict
        CMAQ数据，包含:
        - grid_values: 网格值 (n_grid,)
        - coords: 网格坐标 (n_grid, 2)
    station_data : dict
        站点数据，包含:
        - obs: 观测值 (n_stations,)
        - cmaq: 站点处CMAQ值 (n_stations,)
    station_coords : np.ndarray
        站点坐标 (n_stations, 2)
    params : dict
        参数，包含 method ('OMA'/'SMA'/'MMA')

    Returns:
    --------
    fused : np.ndarray
        融合结果 (n_grid,)
    """
    method = params.get('method', 'OMA')

    cmaq_grid_values = cmaq_data['grid_values']
    grid_coords = cmaq_data['coords']
    station_obs = station_data['obs']
    station_cmaq = station_data['cmaq']

    if method == 'OMA':
        return oma_fuse(cmaq_grid_values, station_obs, station_cmaq, station_coords, grid_coords, params)
    elif method == 'SMA':
        return sma_fuse(cmaq_grid_values, station_obs, station_cmaq, params)
    elif method == 'MMA':
        return mma_fuse(cmaq_grid_values, station_obs, station_cmaq, station_coords, grid_coords, params)
    else:
        raise ValueError(f"Unknown method: {method}")
