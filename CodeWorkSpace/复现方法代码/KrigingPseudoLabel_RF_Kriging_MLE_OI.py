"""
KrigingPseudoLabel, RF-Kriging, MLE-OI 融合方法实现
===================================================

KrigingPseudoLabel - 克里金伪标签增强法
RF-Kriging - 随机森林-克里金残差校正法
MLE-OI - 最大似然估计最优插值法
"""

import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.ensemble import RandomForestRegressor
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


def kriging_pseudo_label_fuse(cmaq_grid_values, station_obs, station_cmaq, station_coords, grid_coords, params):
    """
    克里金伪标签增强法 (简化版)

    核心思想: 使用克里金生成伪标签，增强训练数据

    简化实现:
    1. 使用GPR进行克里金插值
    2. 对高置信度网格点生成伪标签
    3. 融合CMAQ和插值结果

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
        参数: augmentation_ratio, confidence_threshold

    Returns:
    --------
    fused : np.ndarray
        融合结果
    """
    confidence_threshold = params.get('confidence_threshold', 0.8)

    # 使用GPR进行克里金插值
    kernel = ConstantKernel(1.0, (1e-2, 1e2)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(noise_level=0.1)

    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, alpha=0.1, normalize_y=True)
    gpr.fit(station_coords, station_obs)

    # 预测网格点
    pred_mean, pred_std = gpr.predict(grid_coords, return_std=True)

    # 计算置信度 (标准化置信度)
    confidence = 1.0 / (1.0 + pred_std)

    # 融合: 高置信度区域使用克里金，低置信度区域使用CMAQ
    high_conf = confidence > confidence_threshold
    fused = np.where(high_conf, pred_mean, cmaq_grid_values)

    return fused


def rf_kriging_fuse(cmaq_grid_values, station_obs, station_cmaq, station_coords, grid_coords, params):
    """
    RF-Kriging - 随机森林-克里金残差校正法

    步骤:
    1. 随机森林学习CMAQ→监测的非线性映射
    2. 计算训练残差
    3. 克里金插值校正残差

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
        参数: n_estimators, max_depth

    Returns:
    --------
    fused : np.ndarray
        融合结果
    """
    n_estimators = params.get('n_estimators', 100)
    max_depth = params.get('max_depth', 10)

    # 步骤1: 构建特征 (CMAQ值 + 坐标)
    X_train = np.column_stack([station_cmaq, station_coords])
    X_pred = np.column_stack([cmaq_grid_values, grid_coords])

    # 步骤2: 训练随机森林
    rf = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    rf.fit(X_train, station_obs)

    # 步骤3: RF预测
    rf_pred_train = rf.predict(X_train)
    rf_pred_grid = rf.predict(X_pred)

    # 步骤4: 计算训练残差
    residual = station_obs - rf_pred_train

    # 步骤5: 克里金插值残差 (使用GPR)
    kernel = ConstantKernel(1.0, (1e-2, 1e2)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(noise_level=0.1)

    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, alpha=0.1, normalize_y=True)
    gpr.fit(station_coords, residual)

    # 步骤6: 预测网格点残差
    residual_pred, _ = gpr.predict(grid_coords, return_std=True)

    # 步骤7: 最终融合
    fused = rf_pred_grid + residual_pred

    return fused


def mle_oi_fuse(cmaq_grid_values, station_obs, station_cmaq, station_coords, grid_coords, params):
    """
    MLE-OI - 最大似然估计最优插值法

    核心公式:
    y_analysis = y_CMAQ + k^T * (y_obs - H * y_CMAQ)
    k = B * H^T * (H * B * H^T + R)^{-1}

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
        参数: sigma_b_init, sigma_o_init, Lc_init

    Returns:
    --------
    fused : np.ndarray
        融合结果
    """
    sigma_b_init = params.get('sigma_b_init', 5.0)
    sigma_o_init = params.get('sigma_o_init', 2.0)
    Lc_init = params.get('Lc_init', 50.0)

    n_stations = station_coords.shape[0]
    n_grid = grid_coords.shape[0]

    # 计算观测增量
    innovation = station_obs - station_cmaq

    # 计算站点间距离矩阵
    dist_matrix = cdist(station_coords, station_coords, 'euclidean')

    # 定义负对数似然函数
    def neg_log_likelihood(params):
        sigma_b2, sigma_o2, Lc = params

        # 确保参数为正
        if sigma_b2 <= 0 or sigma_o2 <= 0 or Lc <= 0:
            return 1e10

        # 背景误差协方差矩阵 B
        B = sigma_b2 * np.exp(-dist_matrix / Lc)

        # 观测误差协方差矩阵 R
        R = sigma_o2 * np.eye(n_stations)

        # H*B*H^T + R (这里H=I，因为站点就是观测点)
        HBHT_R = B + R

        # 计算对数似然
        try:
            L = np.linalg.cholesky(HBHT_R)
            log_det = 2 * np.sum(np.log(np.diag(L)))
            inv_HBHT_R = np.linalg.solve(L.T, np.linalg.solve(L, innovation))
            mahal_dist = innovation @ inv_HBHT_R
            nll = 0.5 * (log_det + mahal_dist)
        except np.linalg.LinAlgError:
            nll = 1e10

        return nll

    # MLE参数估计
    x0 = [sigma_b_init ** 2, sigma_o_init ** 2, Lc_init]
    bounds = [(1e-6, None), (1e-6, None), (1e-6, None)]

    try:
        result = minimize(neg_log_likelihood, x0, bounds=bounds, method='L-BFGS-B')
        sigma_b2_opt, sigma_o2_opt, Lc_opt = result.x
    except:
        sigma_b2_opt, sigma_o2_opt, Lc_opt = x0

    # 构建最优协方差矩阵
    B = sigma_b2_opt * np.exp(-dist_matrix / Lc_opt)
    R = sigma_o2_opt * np.eye(n_stations)

    # 计算网格点与站点的距离
    dist_grid = cdist(grid_coords, station_coords, 'euclidean')

    # 背景误差协方差 (网格点与站点)
    B_grid = sigma_b2_opt * np.exp(-dist_grid / Lc_opt)

    # 最优增益 K = B_grid * (B + R)^{-1}
    try:
        inv_BR = np.linalg.inv(B + R)
        K = B_grid @ inv_BR
    except np.linalg.LinAlgError:
        K = B_grid @ np.linalg.pinv(B + R)

    # 分析更新
    fused = cmaq_grid_values + K @ innovation

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
        参数，包含 method ('KrigingPseudoLabel'/'RF-Kriging'/'MLE-OI')

    Returns:
    --------
    fused : np.ndarray
        融合结果
    """
    method = params.get('method', 'KrigingPseudoLabel')

    cmaq_grid_values = cmaq_data['grid_values']
    grid_coords = cmaq_data['coords']
    station_obs = station_data['obs']
    station_cmaq = station_data['cmaq']

    if method == 'KrigingPseudoLabel':
        return kriging_pseudo_label_fuse(cmaq_grid_values, station_obs, station_cmaq, station_coords, grid_coords, params)
    elif method == 'RF-Kriging':
        return rf_kriging_fuse(cmaq_grid_values, station_obs, station_cmaq, station_coords, grid_coords, params)
    elif method == 'MLE-OI':
        return mle_oi_fuse(cmaq_grid_values, station_obs, station_cmaq, station_coords, grid_coords, params)
    else:
        raise ValueError(f"Unknown method: {method}")
