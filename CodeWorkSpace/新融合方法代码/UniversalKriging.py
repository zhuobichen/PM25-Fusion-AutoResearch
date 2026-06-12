# -*- coding: utf-8 -*-
"""
Universal Kriging — Berrocal 2020 实现
=========================================
基于 Berrocal et al. (2020) Atmospheric Environment 中的 UK 实现:
- 指数半变异函数: γ(d) = τ² + σ²[1 - exp(-d/φ)]
- 两阶段估计: WLS 初估 (gstat 风格) → ML 精估 (geoR 风格)
- BLUP 预测 + 解析预测方差
- API 与 SVCD 完全兼容: fit(X, y, CMAQ) / predict(X_new, CMAQ_new) / score()

参考文献:
- Berrocal et al. (2020) Atmospheric Environment, 222, 117130
- Cressie (1993) Statistics for Spatial Data, Wiley

作者: Data Fusion Auto Research
日期: 2026-06-11
"""

import numpy as np
from scipy.optimize import minimize
from scipy.linalg import cho_factor, cho_solve
from scipy.spatial.distance import cdist
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


class UniversalKriging:
    """
    Universal Kriging — 指数协方差 + ML 参数估计。

    与 Berrocal 2020 实现的对应:
    - 半变异函数拟合 (WLS) → _fit_variogram_wls()
    - 最大似然精估 (ML)    → _fit_ml()
    - UK 预测              → predict()

    参数
    ------
    covariate_mode : str, default='CMAQ'
        - 'CMAQ':       μ(s) = β₀ + β₁·CMAQ(s)
        - 'CMAQ+Covs':  μ(s) = β₀ + β₁·CMAQ(s) + β₂·X₁(s) + ...  (需传入 extra_covariates)
    standardize_cmaq : bool, default=True
        是否标准化 CMAQ (与 SVCD 一致)
    """

    def __init__(self, covariate_mode='CMAQ', standardize_cmaq=True):
        if covariate_mode not in ('CMAQ', 'CMAQ+Covs'):
            raise ValueError("covariate_mode 必须是 'CMAQ' 或 'CMAQ+Covs'")
        self.covariate_mode = covariate_mode
        self.standardize_cmaq = standardize_cmaq

        # 拟合后状态
        self._fitted = False
        self._coords_train = None
        self._Xmat = None          # 设计矩阵
        self._beta = None          # 回归系数
        self._tau2 = None          # 块金值 (nugget)
        self._sigma2 = None        # 偏基台值 (partial sill)
        self._phi = None           # 变程 (range, km)
        self._V_chol = None        # V 的 Cholesky 分解
        self._resid = None         # Y - Xβ
        self._XtVinvX_inv = None   # (XᵀV⁻¹X)⁻¹

        # CMAQ 标准化统计量
        self._cmaq_mean = None
        self._cmaq_std = None

    # =========================================================
    # 公开接口 (与 SVCD 完全兼容)
    # =========================================================

    def fit(self, X, y, CMAQ_values, extra_covariates=None):
        """
        训练 Universal Kriging 模型。

        参数
        ------
        X : array-like, shape (n, 2)
            训练点坐标 [lon, lat]
        y : array-like, shape (n,)
            监测站真实值 (PM2.5 浓度)
        CMAQ_values : array-like, shape (n,)
            CMAQ 模型预测值
        extra_covariates : array-like, shape (n, k) or None
            额外协变量 (仅 covariate_mode='CMAQ+Covs' 时使用)

        返回
        ------
        self
        """
        coords = np.asarray(X, dtype=float)
        y_obs = np.asarray(y, dtype=float).reshape(-1)
        cmaq = np.asarray(CMAQ_values, dtype=float).reshape(-1)
        n = len(y_obs)

        if coords.shape[0] != n or len(cmaq) != n:
            raise ValueError("X, y, CMAQ_values 长度必须一致")

        # 1. 构建设计矩阵
        cmaq_std = self._standardize_cmaq_fit(cmaq)
        Xmat = self._build_design_matrix(cmaq_std, extra_covariates)

        # 2. 预计算距离矩阵
        dmat = cdist(coords, coords)

        # 3. 两阶段参数估计
        # Stage 1: WLS 初估
        theta_wls = self._fit_variogram_wls(dmat, y_obs, Xmat)
        # Stage 2: ML 精估
        theta_ml = self._fit_ml(dmat, y_obs, Xmat, theta_wls)

        # 4. 存储拟合结果
        self._tau2 = theta_ml[0]
        self._sigma2 = theta_ml[1]
        self._phi = theta_ml[2]
        self._coords_train = coords
        self._Xmat = Xmat

        # 5. GLS 估计 β
        V = self._build_covariance(dmat)
        self._beta, self._V_chol, XtVinvX = self._gls_beta(Xmat, y_obs, V)
        self._resid = y_obs - Xmat @ self._beta
        self._XtVinvX_inv = np.linalg.inv(XtVinvX)
        self._fitted = True

        return self

    def predict(self, X_new, CMAQ_new, extra_covariates_new=None):
        """
        UK 预测新位置的 PM2.5 浓度。

        返回
        ------
        y_pred : ndarray, shape (m,)
        """
        if not self._fitted:
            raise RuntimeError("模型未训练，请先调用 fit()")

        coords_new = np.asarray(X_new, dtype=float)
        cmaq_new = np.asarray(CMAQ_new, dtype=float).reshape(-1)
        cmaq_std = self._standardize_cmaq_transform(cmaq_new)
        Xnew = self._build_design_matrix(cmaq_std, extra_covariates_new)

        # 交叉协方差
        C = self._cross_covariance(coords_new)
        Vinv_resid = cho_solve(self._V_chol, self._resid, check_finite=False)

        # UK 预测
        y_pred = Xnew @ self._beta + C.T @ Vinv_resid
        return y_pred

    def predict_with_uncertainty(self, X_new, CMAQ_new, extra_covariates_new=None):
        """
        UK 预测 + 预测标准差。

        返回
        ------
        y_pred : ndarray
        pred_std : ndarray (预测标准差, √v̂)
        """
        if not self._fitted:
            raise RuntimeError("模型未训练，请先调用 fit()")

        coords_new = np.asarray(X_new, dtype=float)
        cmaq_new = np.asarray(CMAQ_new, dtype=float).reshape(-1)
        cmaq_std = self._standardize_cmaq_transform(cmaq_new)
        Xnew = self._build_design_matrix(cmaq_std, extra_covariates_new)

        C = self._cross_covariance(coords_new)
        Vinv_resid = cho_solve(self._V_chol, self._resid, check_finite=False)
        y_pred = Xnew @ self._beta + C.T @ Vinv_resid

        # 预测方差
        pred_var = self._compute_pred_variance(Xnew, C)
        pred_std = np.sqrt(np.maximum(pred_var, 1e-12))

        return y_pred, pred_std

    def score(self, X_test, y_test, CMAQ_test, extra_covariates_test=None):
        """计算 R²"""
        y_pred = self.predict(X_test, CMAQ_test, extra_covariates_test)
        return r2_score(y_test, y_pred)

    # =========================================================
    # 内部: 设计矩阵
    # =========================================================

    def _standardize_cmaq_fit(self, cmaq):
        if not self.standardize_cmaq:
            return cmaq
        self._cmaq_mean = np.mean(cmaq)
        self._cmaq_std = np.std(cmaq)
        if self._cmaq_std < 1e-12:
            self._cmaq_std = 1.0
        return (cmaq - self._cmaq_mean) / self._cmaq_std

    def _standardize_cmaq_transform(self, cmaq):
        if not self.standardize_cmaq:
            return cmaq
        return (cmaq - self._cmaq_mean) / self._cmaq_std

    def _build_design_matrix(self, cmaq_std, extra_covariates=None):
        """构建设计矩阵 X (n × p)"""
        n = len(cmaq_std)
        cols = [np.ones(n), cmaq_std]
        if self.covariate_mode == 'CMAQ+Covs' and extra_covariates is not None:
            extra = np.asarray(extra_covariates, dtype=float)
            if extra.ndim == 1:
                extra = extra.reshape(-1, 1)
            cols.append(extra)
        return np.column_stack(cols) if len(cols) > 1 else cols[0].reshape(-1, 1)

    # =========================================================
    # 内部: 协方差
    # =========================================================

    def _build_covariance(self, dmat):
        """
        V = σ²·exp(-d/φ) + τ²·I
        与 Berrocal 2020 eq.(7) 对应: Cov = σ²₀·exp(-d_ij/φ₀)
        """
        n = dmat.shape[0]
        V = self._sigma2 * np.exp(-dmat / self._phi)
        V.flat[::n + 1] += self._tau2 + 1e-8  # 对角加 nugget + jitter
        return V

    def _cross_covariance(self, coords_new):
        """
        c(s₀)_i = σ²·exp(-‖s₀ - s_i‖/φ)
        纯距离协方差, 无 CMAQ 依赖 (与 SVCD 的关键区别)
        """
        d_cross = cdist(self._coords_train, coords_new)
        return self._sigma2 * np.exp(-d_cross / self._phi)

    # =========================================================
    # 内部: 两阶段参数估计
    # =========================================================

    def _fit_variogram_wls(self, dmat, y, Xmat):
        """
        Stage 1: WLS 拟合指数半变异函数 (模拟 gstat 行为)

        步骤:
        1. OLS 去趋势, 提取残差
        2. 计算经验半变异函数
        3. WLS 拟合 γ(d) = τ² + σ²[1 - exp(-d/φ)]
        """
        # OLS 残差
        beta_ols = np.linalg.lstsq(Xmat, y, rcond=None)[0]
        resid = y - Xmat @ beta_ols
        sd_resid = max(np.std(resid), 1e-6)

        # 经验半变异函数: 按距离分 bin
        n = dmat.shape[0]
        dist_flat = dmat[np.triu_indices(n, k=1)]
        resid_flat = resid.reshape(-1)
        diff2_flat = np.zeros(len(dist_flat))
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                diff2_flat[idx] = 0.5 * (resid[i] - resid[j]) ** 2
                idx += 1

        # 分 15 个距离 bin
        n_bins = 15
        bin_edges = np.percentile(dist_flat, np.linspace(0, 100, n_bins + 1))
        bin_edges[0] = 0
        bin_centers = []
        bin_gamma = []
        bin_weights = []

        for b in range(n_bins):
            mask = (dist_flat >= bin_edges[b]) & (dist_flat < bin_edges[b + 1])
            if mask.sum() >= 5:
                bin_centers.append(np.mean(dist_flat[mask]))
                bin_gamma.append(np.mean(diff2_flat[mask]))
                bin_weights.append(mask.sum())  # WLS: 更多点 → 更高权重

        if len(bin_centers) < 3:
            # 数据不足, 返回启发式默认值
            return np.array([sd_resid**2 * 0.3, sd_resid**2 * 0.7, np.median(dist_flat) / 3.0])

        bin_centers = np.array(bin_centers)
        bin_gamma = np.array(bin_gamma)
        bin_weights = np.array(bin_weights, dtype=float)
        bin_weights /= bin_weights.sum()

        # WLS 拟合: min Σ w_b · [γ̂_b - (τ² + σ²(1 - exp(-d_b/φ)))]²
        def wls_obj(theta):
            tau2, sigma2, phi = theta
            if tau2 <= 0 or sigma2 <= 0 or phi <= 0:
                return 1e25
            gamma_fit = tau2 + sigma2 * (1.0 - np.exp(-bin_centers / phi))
            return np.sum(bin_weights * (bin_gamma - gamma_fit) ** 2)

        # 初始值: nugget = 0.1*sill, sill = 0.9*var, range = median_dist/3
        total_var = np.var(resid)
        theta0 = np.array([total_var * 0.1, total_var * 0.9, np.median(dist_flat) / 3.0])
        bounds = [(1e-4, total_var * 2), (1e-4, total_var * 2), (0.1, np.max(dist_flat))]

        res = minimize(wls_obj, theta0, method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': 100, 'ftol': 1e-8})
        return np.maximum(res.x, 1e-6)

    def _fit_ml(self, dmat, y, Xmat, theta_init):
        """
        Stage 2: ML 精估 (模拟 geoR 行为)

        以 WLS 估计为初值, 最大化多元正态似然。
        优化变量: log(τ²), log(σ²), log(φ) — log 空间保证正值约束。
        """
        theta0_log = np.log(np.maximum(theta_init, 1e-6))
        # 仅优化协方差参数, β 在每次似然计算中通过 GLS 获得

        def ml_nll(theta_log):
            tau2 = np.exp(theta_log[0])
            sigma2 = np.exp(theta_log[1])
            phi = np.exp(theta_log[2])

            try:
                V = sigma2 * np.exp(-dmat / phi)
                n = dmat.shape[0]
                V.flat[::n + 1] += tau2 + 1e-8
                cF, low = cho_factor(V, lower=True, check_finite=False)

                logdetV = 2.0 * np.sum(np.log(np.diag(cF)))
                VinvX = cho_solve((cF, low), Xmat, check_finite=False)
                Vinvy = cho_solve((cF, low), y, check_finite=False)

                XtVinvX = Xmat.T @ VinvX
                XtVinvy = Xmat.T @ Vinvy
                beta = np.linalg.solve(XtVinvX, XtVinvy)
                resid = y - Xmat @ beta
                Vinv_resid = cho_solve((cF, low), resid, check_finite=False)

                sign, logdetXtVinvX = np.linalg.slogdet(XtVinvX)
                if sign <= 0:
                    return 1e25

                nll = 0.5 * (logdetV + resid.T @ Vinv_resid + logdetXtVinvX)
                return float(nll) if np.isfinite(nll) else 1e25

            except (np.linalg.LinAlgError, ValueError):
                return 1e25

        bounds = [(-10, 10), (-10, 10), (-5, 8)]
        res = minimize(ml_nll, theta0_log, method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': 150, 'ftol': 1e-6})

        return np.exp(res.x)

    # =========================================================
    # 内部: GLS + 预测
    # =========================================================

    def _gls_beta(self, Xmat, y, V):
        """广义最小二乘 (与 SVCD._gls_beta 相同)"""
        cF, low = cho_factor(V, lower=True, check_finite=False)
        VinvX = cho_solve((cF, low), Xmat, check_finite=False)
        Vinvy = cho_solve((cF, low), y, check_finite=False)
        XtVinvX = Xmat.T @ VinvX
        XtVinvy = Xmat.T @ Vinvy
        beta = np.linalg.solve(XtVinvX, XtVinvy)
        return beta, (cF, low), XtVinvX

    def _compute_pred_variance(self, Xnew, C):
        """UK 预测方差 (与 SVCD 结构一致, 但 v_prior 不含 Ũ²σ₁²)"""
        VinvC = cho_solve(self._V_chol, C, check_finite=False)

        m = Xnew.shape[0]
        var_pred = np.empty(m, dtype=float)

        for j in range(m):
            prior_var = self._sigma2 + self._tau2        # 无 CMAQ 依赖!
            conditional_var = prior_var - C[:, j].T @ VinvC[:, j]
            delta = Xnew[j, :] - self._Xmat.T @ VinvC[:, j]
            correction_var = delta.T @ self._XtVinvX_inv @ delta
            var_pred[j] = max(conditional_var + correction_var, 1e-12)

        return var_pred


# =========================================================
# 工具函数
# =========================================================

def compute_metrics(y_true, y_pred):
    """计算评估指标"""
    mask = ~(np.isnan(y_true) | np.isnan(y_pred) | np.isinf(y_true) | np.isinf(y_pred))
    yt, yp = y_true[mask], y_pred[mask]
    if len(yt) == 0:
        return {'R2': np.nan, 'MAE': np.nan, 'RMSE': np.nan, 'MB': np.nan}
    return {
        'R2': float(r2_score(yt, yp)),
        'MAE': float(mean_absolute_error(yt, yp)),
        'RMSE': float(np.sqrt(mean_squared_error(yt, yp))),
        'MB': float(np.mean(yp - yt))
    }
