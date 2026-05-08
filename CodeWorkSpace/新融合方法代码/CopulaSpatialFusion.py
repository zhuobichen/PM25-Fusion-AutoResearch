"""
CopulaSpatialFusion - Copula非高斯空间融合法
============================================

创新点: 使用Copula函数建模CMAQ与监测值的联合分布，显式处理PM2.5数据的非高斯特性

核心思想:
1. 边际分布拟合 (Gamma分布)
2. Gaussian Copula依赖建模
3. 条件期望融合
4. 空间残差克里金
"""

import numpy as np
from scipy import stats
from scipy.optimize import minimize
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


class CopulaSpatialFusion:
    """
    Copula非高斯空间融合法

    步骤:
    1. 边际分布拟合 (Gamma分布)
    2. Gaussian Copula依赖建模
    3. 条件期望融合
    4. 空间残差克里金
    """

    def __init__(self, params=None):
        self.params = params or {}
        self.marginal_dist = self.params.get('marginal_dist', 'gamma')
        self.copula_type = self.params.get('copula_type', 'gaussian')
        self.variogram_model = self.params.get('variogram_model', 'spherical')
        self.n_neighbors = self.params.get('n_neighbors', 12)

        # 拟合的参数
        self.alpha_c = None  # CMAQ Gamma分布参数
        self.beta_c = None
        self.alpha_o = None  # 观测Gamma分布参数
        self.beta_o = None
        self.rho = None  # Copula相关参数

    def fit_marginal(self, data):
        """
        拟合Gamma边际分布

        Parameters:
        -----------
        data : np.ndarray
            输入数据

        Returns:
        --------
        alpha, beta : float
            Gamma分布参数
        """
        # 确保数据为正
        data = data[data > 0]

        # 使用MLE估计Gamma分布参数
        try:
            alpha, loc, beta = stats.gamma.fit(data, floc=0)
        except:
            alpha, beta = 1.0, 1.0

        return alpha, beta

    def fit_copula(self, u, v):
        """
        估计Gaussian Copula参数rho

        Parameters:
        -----------
        u, v : np.ndarray
            均匀分布变量 (通过概率积分变换获得)

        Returns:
        --------
        rho : float
            Copula相关参数
        """
        # 转换为正态分布
        x = stats.norm.ppf(np.clip(u, 1e-6, 1 - 1e-6))
        y = stats.norm.ppf(np.clip(v, 1e-6, 1 - 1e-6))

        # MLE估计rho
        def neg_log_likelihood(rho):
            if abs(rho) >= 1:
                return 1e10
            # 二元正态分布的对数似然
            n = len(x)
            z = (x**2 + y**2 - 2*rho*x*y) / (1 - rho**2)
            log_lik = -0.5 * n * np.log(1 - rho**2) - 0.5 * np.sum(z) / (1 - rho**2)
            return -log_lik

        # 初始值为相关系数
        rho_init = np.corrcoef(x, y)[0, 1]
        rho_init = np.clip(rho_init, -0.99, 0.99)

        try:
            result = minimize(neg_log_likelihood, rho_init, bounds=[(-0.99, 0.99)])
            rho = result.x[0]
        except:
            rho = rho_init

        return rho

    def conditional_expectation(self, u0, rho):
        """
        计算条件期望 E[V|U=u0]

        Parameters:
        -----------
        u0 : float or np.ndarray
            CMAQ的均匀分布变量
        rho : float
            Copula相关参数

        Returns:
        --------
        ev : float or np.ndarray
            条件期望 E[V|U=u0]
        """
        # E[V|U=u0] = Phi(rho * Phi^{-1}(u0))
        x = stats.norm.ppf(np.clip(u0, 1e-6, 1 - 1e-6))
        ev = stats.norm.cdf(rho * x)
        return ev

    def fit(self, station_obs, station_cmaq):
        """
        拟合模型

        Parameters:
        -----------
        station_obs : np.ndarray
            站点观测值
        station_cmaq : np.ndarray
            站点处CMAQ值
        """
        # 步骤1: 拟合边际分布
        self.alpha_c, self.beta_c = self.fit_marginal(station_cmaq)
        self.alpha_o, self.beta_o = self.fit_marginal(station_obs)

        # 步骤2: 转换为均匀分布
        u = stats.gamma.cdf(station_cmaq, self.alpha_c, scale=self.beta_c)
        v = stats.gamma.cdf(station_obs, self.alpha_o, scale=self.beta_o)

        # 步骤3: 估计Copula参数
        self.rho = self.fit_copula(u, v)

    def predict(self, cmaq_grid_values, station_obs, station_cmaq, station_coords, grid_coords):
        """
        预测融合值

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

        Returns:
        --------
        fused : np.ndarray
            融合结果
        """
        # 步骤1: Copula条件期望
        u0 = stats.gamma.cdf(cmaq_grid_values, self.alpha_c, scale=self.beta_c)
        ev = self.conditional_expectation(u0, self.rho)
        copula_pred = stats.gamma.ppf(ev, self.alpha_o, scale=self.beta_o)

        # 步骤2: 计算站点残差
        u_train = stats.gamma.cdf(station_cmaq, self.alpha_c, scale=self.beta_c)
        ev_train = self.conditional_expectation(u_train, self.rho)
        copula_pred_train = stats.gamma.ppf(ev_train, self.alpha_o, scale=self.beta_o)
        residual = station_obs - copula_pred_train

        # 步骤3: 残差克里金 (使用GPR)
        kernel = ConstantKernel(1.0, (1e-2, 1e2)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(noise_level=0.1)

        gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, alpha=0.1, normalize_y=True)
        gpr.fit(station_coords, residual)

        # 步骤4: 预测网格点残差
        residual_pred, _ = gpr.predict(grid_coords, return_std=True)

        # 步骤5: 最终融合
        fused = copula_pred + residual_pred

        # 边界处理: 确保非负
        fused = np.maximum(fused, 0)

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
        参数

    Returns:
    --------
    fused : np.ndarray
        融合结果
    """
    cmaq_grid_values = cmaq_data['grid_values']
    grid_coords = cmaq_data['coords']
    station_obs = station_data['obs']
    station_cmaq = station_data['cmaq']

    # 创建模型
    model = CopulaSpatialFusion(params)

    # 拟合
    model.fit(station_obs, station_cmaq)

    # 预测
    fused = model.predict(cmaq_grid_values, station_obs, station_cmaq, station_coords, grid_coords)

    return fused
