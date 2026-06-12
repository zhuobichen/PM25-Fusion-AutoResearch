# -*- coding: utf-8 -*-
"""
SVCD - Spatially Varying Coefficient Downscaler
================================================
现代版 Downscaler：支持 log/sqrt 变换 + 空间变系数 + Matern核 + REML优化

方法概述:
1. 数据变换: Z(s)=sqrt(Y(s)) 或 Z(s)=log(Y(s)+c), U同
   - sqrt(默认O₃): 加性Jensen修正 μ²+σ², 适合值域窄的污染物
   - log(PM2.5备选): 指数级Jensen修正 exp(μ+σ²/2)-c
2. 空间变系数模型: Z(s) = β₀ + β₁U(s) + w₀(s) + w₁(s)U(s) + ε(s)
   - w₀(s) ~ GP(0, K₀): 空间变截距（捕捉CMAQ不同地区的系统偏差）
   - w₁(s) ~ GP(0, K₁): 空间变斜率（捕捉不同地区CMAQ与观测关系的差异）
3. Matérn协方差核(ν₀=ν₁=0.5 指数核) + REML优化(5个超参数, L-BFGS-B)
4. 偏差修正反变换（sqrt: μ²+σ², log: exp(μ+σ²/2)-c）

与原始 Downscaler 的区别:
- 原版: 1个GP偏差场 δ(s) + 全局斜率 β₁, MCMC推断, 原始尺度
  → 本版: 2个GP场(w₀截距+w₁斜率), REML优化, 支持log/sqrt变换

与 AdvancedRK 的区别:
- AdvancedRK: 多项式全局校正 + GPR残差克里金（两步法）
- SVCD: 统一的空间变系数贝叶斯模型（单步法，理论更完整）

参考文献:
- Berrocal et al. (2010) A Spatio-Temporal Downscaler for Output from Numerical Models
- Friberg et al. (2016) Method for Fusing Observational Data and CTM

作者: Data Fusion Auto Research
日期: 2026-06-04 | 更新: 2026-06-11 (sqrt变换 + ν=0.5)
"""

import numpy as np
from scipy.optimize import minimize
from scipy.linalg import cho_factor, cho_solve
from scipy.spatial.distance import cdist
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


class SVCD:
    """
    SVCD: Spatially Varying Coefficient Downscaler
    
    支持 log/sqrt 变换 + 空间变系数 + Matérn核 + REML优化。
    
    参数:
    ------
    c : float, default=1.0
        变换常数（仅 log 模式使用，sqrt 忽略）
    nu0 : float, default=0.5
        截距场 w₀(s) 的 Matérn 平滑参数（0.5=指数核，消融最优）
    nu1 : float, default=0.5
        斜率场 w₁(s) 的 Matérn 平滑参数（0.5=指数核）
    jitter : float, default=1e-6
        数值稳定性的微小噪声
    standardize_u : bool, default=True
        是否标准化 U（变换后的CMAQ值）
    bias_correction : str or bool, default='full'
        偏差修正模式:
        - log 变换: none=exp(μ)−c / full=exp(μ+½σ²)−c / partial=exp(μ+½(σ²−τ²))−c
        - sqrt变换: none=μ² / full=μ²+σ² / partial=max(0, μ²+max(0,σ²−τ²))
    transform : str, default='log'
        变换函数: 'log'=log(Y+c) / 'sqrt'=sqrt(Y)
        - log: 适合PM2.5等值域跨数量级的污染物
        - sqrt: 适合O₃等值域窄的污染物, 方差修正为加性(μ²+σ²)而非指数级
    """

    def __init__(self, c=1.0, nu0=0.5, nu1=0.5, jitter=1e-6,
                 standardize_u=True, bias_correction='full', transform='log'):
        self.c = c
        self.nu0 = nu0
        self.nu1 = nu1
        self.jitter = jitter
        self.standardize_u = standardize_u
        self.transform = transform
        # Normalize bias_correction to 'none' / 'full' / 'partial'
        if bias_correction in (True, 'full'):
            self.bias_correction = 'full'
        elif bias_correction in (False, 'none'):
            self.bias_correction = 'none'
        elif bias_correction == 'partial':
            self.bias_correction = 'partial'
        else:
            raise ValueError("bias_correction 必须是 'none'/'full'/'partial' 或 True/False")
        
        # 拟合后存储的内部状态
        self._coords_train = None
        self._z_train = None
        self._u_train = None
        self._u_mean = None
        self._u_std = None
        self._beta = None
        self._theta_log = None
        self._Xmat = None
        self._V_chol = None
        self._resid = None
        self._XtVinvX_inv = None
        self._fitted = False
    
    # =========================================================
    # 公开接口
    # =========================================================
    
    def fit(self, X, y, CMAQ_values, theta_init=None,
            maxiter=150, ftol=1e-6, gtol=1e-5):
        """
        训练 SVCD 模型
        
        参数:
        ------
        X : array-like, shape (n_samples, 2)
            训练点的坐标 [lon, lat]
        y : array-like, shape (n_samples,)
            监测站真实值（PM2.5浓度）
        CMAQ_values : array-like, shape (n_samples,)
            对应的CMAQ模型预测值
        theta_init : array-like or None
            初始超参数 (log空间)，用于warm start
        maxiter : int
            最大迭代次数
        ftol, gtol : float
            收敛阈值
        
        返回:
        ------
        self
        """
        coords = np.asarray(X, dtype=float)
        y_obs = np.asarray(y, dtype=float).reshape(-1)
        x_cmaq = np.asarray(CMAQ_values, dtype=float).reshape(-1)
        
        n = len(y_obs)
        if coords.shape[0] != n or len(x_cmaq) != n:
            raise ValueError("X, y, CMAQ_values 长度必须一致")
        if coords.ndim != 2 or coords.shape[1] != 2:
            raise ValueError("X 必须为 shape (n, 2) 的坐标数组")
        
        # 1. 变换
        z = self._transform(y_obs)
        u_raw = self._transform(x_cmaq)
        
        # 2. 标准化 U
        u_std = self._standardize_u_fit(u_raw)
        
        # 3. 设计矩阵
        Xmat = np.column_stack([np.ones(n), u_std])
        
        # 4. 预计算距离矩阵（关键优化：只算一次）
        dmat = cdist(coords, coords)
        uu_outer = np.outer(u_std, u_std)
        
        # 5. 初始参数
        if theta_init is not None:
            theta0 = np.asarray(theta_init, dtype=float)
        else:
            theta0 = self._init_theta(dmat, z)
        
        # 6. REML 优化
        obj = lambda th: self._reml_nll(th, dmat, uu_outer, z, Xmat)
        bounds = [(-6, 6)] * 5
        res = minimize(obj, theta0, method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol})
        
        # 7. 存储拟合结果
        self._theta_log = res.x
        self._coords_train = coords
        self._z_train = z
        self._u_train = u_std
        self._Xmat = Xmat
        self._dmat_train = dmat
        
        V = self._build_cov(dmat, uu_outer, self._theta_log)
        beta, chol, XtVinvX = self._gls_beta(Xmat, z, V)
        
        self._beta = beta
        self._V_chol = chol
        self._resid = z - Xmat @ beta
        self._XtVinvX_inv = np.linalg.inv(XtVinvX)
        self._fitted = True
        
        return self
    
    @property
    def theta(self):
        """返回优化后的超参数 (log空间)，可用于warm start"""
        return self._theta_log.copy() if self._theta_log is not None else None
    
    def predict(self, X_new, CMAQ_new):
        """
        预测新位置的 PM2.5 浓度
        
        参数:
        ------
        X_new : array-like, shape (n_samples, 2)
            新位置的坐标 [lon, lat]
        CMAQ_new : array-like, shape (n_samples,)
            新位置对应的 CMAQ 模型预测值
        
        返回:
        ------
        y_pred : ndarray, shape (n_samples,)
            融合后的 PM2.5 浓度预测值（原始尺度）
        """
        if not self._fitted:
            raise RuntimeError("模型未训练，请先调用 fit()")
        
        coords_new = np.asarray(X_new, dtype=float)
        x_cmaq_new = np.asarray(CMAQ_new, dtype=float).reshape(-1)
        
        if len(coords_new) != len(x_cmaq_new):
            raise ValueError("X_new 和 CMAQ_new 长度必须一致")
        
        # 变换 + 标准化
        u_new_raw = self._transform(x_cmaq_new)
        u_new_std = self._standardize_u_transform(u_new_raw)
        Xnew = np.column_stack([np.ones(len(u_new_std)), u_new_std])

        # 交叉协方差
        C = self._cross_covariance(coords_new, u_new_std)
        Vinv_resid = cho_solve(self._V_chol, self._resid, check_finite=False)

        # 变换尺度均值预测
        mean_trans = Xnew @ self._beta + C.T @ Vinv_resid
        var_trans = self._compute_pred_variance(coords_new, u_new_std, Xnew, C)
        y_pred = self._inverse_transform(mean_trans, var_trans)

        y_pred = np.maximum(y_pred, 0.0)
        return y_pred
    
    def predict_with_uncertainty(self, X_new, CMAQ_new):
        """
        预测 PM2.5 浓度并返回不确定性估计
        
        返回:
        ------
        y_pred : ndarray - 融合后的预测值
        pred_std : ndarray - 预测的标准差（原始尺度近似）
        """
        if not self._fitted:
            raise RuntimeError("模型未训练，请先调用 fit()")
        
        coords_new = np.asarray(X_new, dtype=float)
        x_cmaq_new = np.asarray(CMAQ_new, dtype=float).reshape(-1)
        
        u_new_raw = self._transform(x_cmaq_new)
        u_new_std = self._standardize_u_transform(u_new_raw)
        Xnew = np.column_stack([np.ones(len(u_new_std)), u_new_std])

        C = self._cross_covariance(coords_new, u_new_std)
        Vinv_resid = cho_solve(self._V_chol, self._resid, check_finite=False)
        mean_trans = Xnew @ self._beta + C.T @ Vinv_resid
        var_trans = self._compute_pred_variance(coords_new, u_new_std, Xnew, C)

        y_pred = self._inverse_transform(mean_trans, var_trans)
        y_pred = np.maximum(y_pred, 0.0)

        # Delta method 标准差（始终用全方差）
        if self.transform == 'sqrt':
            pred_std = 2.0 * np.abs(mean_trans) * np.sqrt(var_trans)
        else:
            pred_std = np.exp(mean_trans) * np.sqrt(var_trans)
        
        return y_pred, pred_std
    
    def score(self, X_test, y_test, CMAQ_test):
        """计算 R² 分数"""
        y_pred = self.predict(X_test, CMAQ_test)
        return r2_score(y_test, y_pred)
    
    # =========================================================
    # 内部方法
    # =========================================================
    
    def _transform(self, arr):
        """前向变换: log 或 sqrt"""
        arr = np.asarray(arr, dtype=float)
        if self.transform == 'sqrt':
            return np.sqrt(np.maximum(arr, 0.0))
        else:
            val = arr + self.c
            val = np.maximum(val, 1e-10)
            return np.log(val)

    def _inverse_transform(self, mean_log, var_log):
        """反变换 + 偏差修正

        log 模式: Ŷ = exp(μ + k·σ²) − c
        sqrt模式: Ŷ = μ² + k·σ²
        其中 k 由 bias_correction 决定 (0, 1, 或 fractional)
        """
        if self.transform == 'sqrt':
            mu2 = mean_log ** 2
            if self.bias_correction == 'full':
                return mu2 + var_log
            elif self.bias_correction == 'partial':
                tau2 = np.exp(self._theta_log[4])**2
                return mu2 + np.maximum(var_log - tau2, 0.0)
            else:  # 'none'
                return mu2
        else:  # 'log'
            if self.bias_correction == 'full':
                return np.exp(mean_log + 0.5 * var_log) - self.c
            elif self.bias_correction == 'partial':
                tau2 = np.exp(self._theta_log[4])**2
                var_corrected = np.maximum(var_log - tau2, 1e-12)
                return np.exp(mean_log + 0.5 * var_corrected) - self.c
            else:  # 'none'
                return np.exp(mean_log) - self.c
    
    def _standardize_u_fit(self, u):
        """标准化 U 并记录统计量"""
        if not self.standardize_u:
            return u
        self._u_mean = np.mean(u)
        self._u_std = np.std(u)
        if self._u_std < 1e-12:
            self._u_std = 1.0
        return (u - self._u_mean) / self._u_std
    
    def _standardize_u_transform(self, u):
        """使用已有统计量标准化 U"""
        if not self.standardize_u:
            return u
        return (u - self._u_mean) / self._u_std
    
    def _init_theta(self, dmat, z):
        """初始化超参数（log空间）"""
        positive_dist = dmat[dmat > 0]
        median_dist = np.median(positive_dist) if len(positive_dist) > 0 else 1.0
        z_sd = max(np.std(z), 1e-6)
        
        # theta = [log_sigma0, log_rho0, log_sigma1, log_rho1, log_tau]
        return np.log([
            z_sd * 0.5,      # sigma0: 截距场标准差
            median_dist,     # rho0: 截距场空间范围
            z_sd * 0.2,      # sigma1: 斜率场标准差
            median_dist,     # rho1: 斜率场空间范围
            z_sd * 0.3       # tau: 观测噪声标准差
        ])
    
    def _matern_kernel(self, d, sigma2, rho, nu):
        """Matern 协方差核 (nu in {0.5, 1.5, 2.5})"""
        d = np.maximum(d, 1e-12)
        if rho <= 0:
            rho = 1e-6
        
        if nu == 0.5:
            base = np.exp(-d / rho)
        elif nu == 1.5:
            r = np.sqrt(3.0) * d / rho
            base = (1.0 + r) * np.exp(-r)
        elif nu == 2.5:
            r = np.sqrt(5.0) * d / rho
            base = (1.0 + r + r**2 / 3.0) * np.exp(-r)
        else:
            r = np.sqrt(3.0) * d / rho
            base = (1.0 + r) * np.exp(-r)
        
        return sigma2 * base
    
    def _build_cov(self, dmat, uu_outer, theta_log):
        """
        构建边际协方差矩阵 V = Sigma_0 + (uu^T . Sigma_1) + tau^2 I
        
        使用 Hadamard 积: D_U Sigma_1 D_U = outer(u,u) * Sigma_1, O(n^2)
        """
        log_sigma0, log_rho0, log_sigma1, log_rho1, log_tau = theta_log
        
        sigma0 = np.exp(log_sigma0)
        rho0 = np.exp(log_rho0)
        sigma1 = np.exp(log_sigma1)
        rho1 = np.exp(log_rho1)
        tau = np.exp(log_tau)
        
        Sigma0 = self._matern_kernel(dmat, sigma0**2, rho0, self.nu0)
        Sigma1 = self._matern_kernel(dmat, sigma1**2, rho1, self.nu1)
        
        n = dmat.shape[0]
        V = Sigma0 + uu_outer * Sigma1 + (tau**2 + self.jitter) * np.eye(n)
        
        return V
    
    def _gls_beta(self, Xmat, z, V):
        """广义最小二乘估计 beta"""
        cF, low = cho_factor(V, lower=True, check_finite=False)
        
        VinvX = cho_solve((cF, low), Xmat, check_finite=False)
        Vinvz = cho_solve((cF, low), z, check_finite=False)
        
        XtVinvX = Xmat.T @ VinvX
        XtVinvz = Xmat.T @ Vinvz
        
        beta = np.linalg.solve(XtVinvX, XtVinvz)
        return beta, (cF, low), XtVinvX
    
    def _reml_nll(self, theta_log, dmat, uu_outer, z, Xmat):
        """REML 负对数似然（Cholesky logdet 优化版）"""
        try:
            V = self._build_cov(dmat, uu_outer, theta_log)
            cF, low = cho_factor(V, lower=True, check_finite=False)
            
            logdetV = 2.0 * np.sum(np.log(np.diag(cF)))
            
            VinvX = cho_solve((cF, low), Xmat, check_finite=False)
            Vinvz = cho_solve((cF, low), z, check_finite=False)
            
            XtVinvX = Xmat.T @ VinvX
            XtVinvz = Xmat.T @ Vinvz
            
            beta = np.linalg.solve(XtVinvX, XtVinvz)
            resid = z - Xmat @ beta
            Vinv_resid = cho_solve((cF, low), resid, check_finite=False)
            
            sign2, logdetXtVinvX = np.linalg.slogdet(XtVinvX)
            if sign2 <= 0:
                return 1e25
            
            nll = 0.5 * (logdetV + resid.T @ Vinv_resid + logdetXtVinvX)
            
            if not np.isfinite(nll):
                return 1e25
            return float(nll)
        
        except (np.linalg.LinAlgError, ValueError, FloatingPointError):
            return 1e25
    
    def _cross_covariance(self, coords_new, u_new_std):
        """计算训练点与新点的交叉协方差"""
        sigma0 = np.exp(self._theta_log[0])
        rho0 = np.exp(self._theta_log[1])
        sigma1 = np.exp(self._theta_log[2])
        rho1 = np.exp(self._theta_log[3])
        
        d_cross = cdist(self._coords_train, coords_new)
        
        C0 = self._matern_kernel(d_cross, sigma0**2, rho0, self.nu0)
        C1 = self._matern_kernel(d_cross, sigma1**2, rho1, self.nu1)
        
        # c(s0) = c0(s0) + u_new * (u_train * c1(s0))
        C = C0 + np.outer(self._u_train, u_new_std) * C1
        
        return C
    
    def _compute_pred_variance(self, coords_new, u_new_std, Xnew, C):
        """计算预测方差（含固定效应不确定性修正）"""
        VinvC = cho_solve(self._V_chol, C, check_finite=False)
        
        sigma0 = np.exp(self._theta_log[0])
        sigma1 = np.exp(self._theta_log[2])
        tau = np.exp(self._theta_log[4])
        
        m = len(u_new_std)
        var_log = np.empty(m, dtype=float)
        
        for j in range(m):
            prior_var = sigma0**2 + (u_new_std[j]**2) * sigma1**2 + tau**2
            conditional_var = prior_var - C[:, j].T @ VinvC[:, j]
            correction = Xnew[j, :] - self._Xmat.T @ VinvC[:, j]
            correction_var = correction.T @ self._XtVinvX_inv @ correction
            var_log[j] = max(conditional_var + correction_var, 1e-12)
        
        return var_log


# =========================================================
# 工具函数
# =========================================================

def compute_metrics(y_true, y_pred):
    """计算评估指标: R2, MAE, RMSE, MB"""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = ~(np.isnan(y_true) | np.isnan(y_pred) |
             np.isinf(y_true) | np.isinf(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    
    if len(y_true) == 0:
        return {'R2': np.nan, 'MAE': np.nan, 'RMSE': np.nan, 'MB': np.nan}
    
    return {
        'R2': float(r2_score(y_true, y_pred)),
        'MAE': float(mean_absolute_error(y_true, y_pred)),
        'RMSE': float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'MB': float(np.mean(y_pred - y_true))
    }
