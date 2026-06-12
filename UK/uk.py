# -*- coding: utf-8 -*-
"""
UK - Universal Kriging
=======================
基于 Berrocal et al. (2020) Atmospheric Environment 中的 UK 实现:

模型:
  Y(s) = x(s)ᵀβ + ε(s),  ε ~ GP(0, C)
  C(d) = σ²·exp(-d/φ) + τ²·I

参数估计:
  Stage 1: WLS 拟合指数半变异函数 (gstat 风格)
  Stage 2: ML 精估协方差参数 (geoR 风格)

预测:
  Universal Kriging BLUP + 解析预测方差

与 SVCD 的 API 完全兼容: fit(X, y, CMAQ) / predict(X_new, CMAQ_new)

可选变换:
  transform=None (默认): 原始尺度建模 (Berrocal 2020 原版)
  transform='sqrt':       sqrt(Y) 尺度建模 (与 SVCD sqrt 模式一致)
  transform='log':        log(Y+c) 尺度建模

参考文献:
- Berrocal et al. (2020) Atmospheric Environment, 222, 117130
- Cressie (1993) Statistics for Spatial Data, Wiley
"""

import numpy as np
from scipy.optimize import minimize
from scipy.linalg import cho_factor, cho_solve
from scipy.spatial.distance import cdist
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


class UniversalKriging:
    """
    Universal Kriging — 指数协方差 + WLS/ML 两阶段估计。

    参数
    ------
    transform : str or None, default=None
        数据变换:
        - None:  原始尺度 (Berrocal 2020 原版)
        - 'sqrt': sqrt(Y), sqrt(CMAQ) — 反变换 Ŷ = max(0, μ̂² + σ̂²)
        - 'log':  log(Y+c), log(CMAQ+c) — 反变换 Ŷ = exp(μ̂ + σ̂²/2) - c
    c : float, default=1.0
        log 变换常数 (仅 transform='log' 时使用)
    standardize_cmaq : bool, default=True
        是否标准化 CMAQ 值
    bias_correction : bool, default=True
        反变换时是否偏差修正 (仅 transform='sqrt'/'log' 生效)
    """

    def __init__(self, transform=None, c=1.0,
                 standardize_cmaq=True, bias_correction=True):
        if transform not in (None, 'sqrt', 'log'):
            raise ValueError("transform 必须是 None / 'sqrt' / 'log'")
        self.transform = transform
        self.c = c
        self.standardize_cmaq = standardize_cmaq
        self.bias_correction = bias_correction

        self._fitted = False
        self._coords_train = None
        self._Xmat = None
        self._beta = None
        self._tau2 = None          # nugget
        self._sigma2 = None        # partial sill
        self._phi = None           # range (km)
        self._V_chol = None
        self._resid = None
        self._XtVinvX_inv = None
        self._cmaq_mean = None
        self._cmaq_std = None

    # =========================================================
    # 公开接口
    # =========================================================

    def fit(self, X, y, CMAQ_values):
        """
        训练 UK 模型。

        参数
        ------
        X : array-like, shape (n, 2)
            训练点坐标 [lon, lat]
        y : array-like, shape (n,)
            监测站真实值
        CMAQ_values : array-like, shape (n,)
            CMAQ 模型预测值

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

        # 1. 数据变换
        z = self._transform(y_obs)
        u = self._transform(cmaq)

        # 2. 标准化 CMAQ
        u_std = self._standardize_fit(u)

        # 3. 设计矩阵 X = [1, ũ]
        self._Xmat = np.column_stack([np.ones(n), u_std])

        # 4. 距离矩阵
        dmat = cdist(coords, coords)

        # 5. 两阶段参数估计
        theta_wls = self._fit_variogram_wls(dmat, z)
        theta_ml = self._fit_ml(dmat, z, theta_wls)

        self._tau2 = theta_ml[0]
        self._sigma2 = theta_ml[1]
        self._phi = theta_ml[2]
        self._coords_train = coords

        # 6. GLS 估计 β
        V = self._build_covariance(dmat)
        self._beta, self._V_chol, XtVinvX = self._gls_beta(self._Xmat, z, V)
        self._resid = z - self._Xmat @ self._beta
        self._XtVinvX_inv = np.linalg.inv(XtVinvX)
        self._fitted = True

        return self

    def predict(self, X_new, CMAQ_new):
        """
        UK 预测。

        返回
        ------
        y_pred : ndarray (原始尺度)
        """
        if not self._fitted:
            raise RuntimeError("模型未训练，请先调用 fit()")

        coords_new = np.asarray(X_new, dtype=float)
        cmaq_new = np.asarray(CMAQ_new, dtype=float).reshape(-1)
        u_new = self._transform(cmaq_new)
        u_std = self._standardize_transform(u_new)
        Xnew = np.column_stack([np.ones(len(u_std)), u_std])

        C = self._cross_covariance(coords_new)
        Vinv_resid = cho_solve(self._V_chol, self._resid, check_finite=False)

        mean_t = Xnew @ self._beta + C.T @ Vinv_resid

        if self.transform is not None and self.bias_correction:
            var_t = self._compute_pred_variance(Xnew, C)
            y_pred = self._inverse_transform(mean_t, var_t)
        elif self.transform is not None:
            y_pred = self._inverse_transform_no_bias(mean_t)
        else:
            y_pred = mean_t

        return np.maximum(y_pred, 0.0)

    def predict_with_uncertainty(self, X_new, CMAQ_new):
        """预测 + 标准差"""
        if not self._fitted:
            raise RuntimeError("模型未训练，请先调用 fit()")

        coords_new = np.asarray(X_new, dtype=float)
        cmaq_new = np.asarray(CMAQ_new, dtype=float).reshape(-1)
        u_new = self._transform(cmaq_new)
        u_std = self._standardize_transform(u_new)
        Xnew = np.column_stack([np.ones(len(u_std)), u_std])

        C = self._cross_covariance(coords_new)
        Vinv_resid = cho_solve(self._V_chol, self._resid, check_finite=False)
        mean_t = Xnew @ self._beta + C.T @ Vinv_resid
        var_t = self._compute_pred_variance(Xnew, C)

        if self.transform is not None:
            y_pred = self._inverse_transform(
                mean_t, var_t if self.bias_correction else np.zeros_like(var_t))
        else:
            y_pred = mean_t
        y_pred = np.maximum(y_pred, 0.0)

        if self.transform == 'sqrt':
            pred_std = 2.0 * np.abs(mean_t) * np.sqrt(var_t)
        elif self.transform == 'log':
            pred_std = np.exp(mean_t) * np.sqrt(var_t)
        else:
            pred_std = np.sqrt(var_t)

        return y_pred, pred_std

    def score(self, X_test, y_test, CMAQ_test):
        return r2_score(y_test, self.predict(X_test, CMAQ_test))

    @property
    def params(self):
        """返回协方差参数"""
        return {'nugget': self._tau2, 'sill': self._sigma2, 'range_km': self._phi}

    # =========================================================
    # 变换
    # =========================================================

    def _transform(self, arr):
        arr = np.asarray(arr, dtype=float)
        if self.transform == 'sqrt':
            return np.sqrt(np.maximum(arr, 0.0))
        elif self.transform == 'log':
            return np.log(np.maximum(arr + self.c, 1e-10))
        return arr

    def _inverse_transform(self, mean_t, var_t):
        if self.transform == 'sqrt':
            return mean_t ** 2 + var_t
        elif self.transform == 'log':
            return np.exp(mean_t + 0.5 * var_t) - self.c
        return mean_t

    def _inverse_transform_no_bias(self, mean_t):
        if self.transform == 'sqrt':
            return mean_t ** 2
        elif self.transform == 'log':
            return np.exp(mean_t) - self.c
        return mean_t

    def _standardize_fit(self, u):
        if not self.standardize_cmaq:
            return u
        self._cmaq_mean = np.mean(u)
        self._cmaq_std = np.std(u)
        if self._cmaq_std < 1e-12:
            self._cmaq_std = 1.0
        return (u - self._cmaq_mean) / self._cmaq_std

    def _standardize_transform(self, u):
        if not self.standardize_cmaq:
            return u
        return (u - self._cmaq_mean) / self._cmaq_std

    # =========================================================
    # 协方差
    # =========================================================

    def _build_covariance(self, dmat):
        """V = σ²·exp(-d/φ) + τ²·I"""
        n = dmat.shape[0]
        V = self._sigma2 * np.exp(-dmat / self._phi)
        V.flat[::n + 1] += self._tau2 + 1e-8
        return V

    def _cross_covariance(self, coords_new):
        """c(s₀)_i = σ²·exp(-‖s₀ - s_i‖/φ)   (纯距离协方差)"""
        d_cross = cdist(self._coords_train, coords_new)
        return self._sigma2 * np.exp(-d_cross / self._phi)

    # =========================================================
    # 两阶段参数估计
    # =========================================================

    def _fit_variogram_wls(self, dmat, z):
        """Stage 1: WLS 拟合指数半变异函数"""
        n = dmat.shape[0]

        # OLS 去趋势 (UK 的关键: 半变异函数必须从回归残差计算, 非均值中心化)
        beta_ols = np.linalg.lstsq(self._Xmat, z, rcond=None)[0]
        resid = z - self._Xmat @ beta_ols

        # 提取上三角距离和半方差
        iu = np.triu_indices(n, k=1)
        dist_flat = dmat[iu]
        diff2_flat = 0.5 * (resid[iu[0]] - resid[iu[1]]) ** 2

        # 分 15 个距离 bin
        n_bins = 15
        edges = np.percentile(dist_flat, np.linspace(0, 100, n_bins + 1))
        edges[0] = 0

        centers, gamma, weights = [], [], []
        for b in range(n_bins):
            mask = (dist_flat >= edges[b]) & (dist_flat < edges[b + 1])
            if mask.sum() >= 5:
                centers.append(np.mean(dist_flat[mask]))
                gamma.append(np.mean(diff2_flat[mask]))
                weights.append(float(mask.sum()))

        if len(centers) < 3:
            var = np.var(z)
            return np.array([var * 0.1, var * 0.9, np.median(dist_flat) / 3.0])

        centers = np.array(centers)
        gamma = np.array(gamma)
        w = np.array(weights) / sum(weights)

        def wls_obj(theta):
            tau2, sigma2, phi = theta
            if tau2 <= 0 or sigma2 <= 0 or phi <= 0:
                return 1e25
            fit = tau2 + sigma2 * (1.0 - np.exp(-centers / phi))
            return np.sum(w * (gamma - fit) ** 2)

        var = np.var(z)
        x0 = np.array([var * 0.1, var * 0.9, np.median(dist_flat) / 3.0])
        bnds = [(1e-4, var * 2), (1e-4, var * 2), (0.1, np.max(dist_flat))]
        res = minimize(wls_obj, x0, method='L-BFGS-B', bounds=bnds,
                       options={'maxiter': 100, 'ftol': 1e-8})
        return np.maximum(res.x, 1e-6)

    def _fit_ml(self, dmat, z, theta_init):
        """Stage 2: ML 精估 (log 空间优化)"""
        theta0_log = np.log(np.maximum(theta_init, 1e-6))

        def ml_nll(theta_log):
            tau2, sigma2, phi = np.exp(theta_log)
            try:
                n = dmat.shape[0]
                V = sigma2 * np.exp(-dmat / phi)
                V.flat[::n + 1] += tau2 + 1e-8
                cF, low = cho_factor(V, lower=True, check_finite=False)

                logdetV = 2.0 * np.sum(np.log(np.diag(cF)))
                VinvX = cho_solve((cF, low), self._Xmat, check_finite=False)
                Vinvz = cho_solve((cF, low), z, check_finite=False)

                XtVinvX = self._Xmat.T @ VinvX
                XtVinvz = self._Xmat.T @ Vinvz
                beta = np.linalg.solve(XtVinvX, XtVinvz)
                resid = z - self._Xmat @ beta
                Vinv_r = cho_solve((cF, low), resid, check_finite=False)

                sign, logdetX = np.linalg.slogdet(XtVinvX)
                if sign <= 0:
                    return 1e25

                nll = 0.5 * (logdetV + resid.T @ Vinv_r + logdetX)
                return float(nll) if np.isfinite(nll) else 1e25
            except (np.linalg.LinAlgError, ValueError):
                return 1e25

        bnds = [(-10, 10), (-10, 10), (-5, 8)]
        res = minimize(ml_nll, theta0_log, method='L-BFGS-B', bounds=bnds,
                       options={'maxiter': 150, 'ftol': 1e-6})
        return np.exp(res.x)

    # =========================================================
    # GLS + 预测方差
    # =========================================================

    def _gls_beta(self, Xmat, z, V):
        cF, low = cho_factor(V, lower=True, check_finite=False)
        VinvX = cho_solve((cF, low), Xmat, check_finite=False)
        Vinvz = cho_solve((cF, low), z, check_finite=False)
        XtVinvX = Xmat.T @ VinvX
        XtVinvz = Xmat.T @ Vinvz
        beta = np.linalg.solve(XtVinvX, XtVinvz)
        return beta, (cF, low), XtVinvX

    def _compute_pred_variance(self, Xnew, C):
        """UK 预测方差: σ²+τ² − cᵀV⁻¹c + δᵀ(XᵀV⁻¹X)⁻¹δ"""
        VinvC = cho_solve(self._V_chol, C, check_finite=False)
        m = Xnew.shape[0]
        var_pred = np.empty(m)

        for j in range(m):
            prior = self._sigma2 + self._tau2
            cond = prior - C[:, j].T @ VinvC[:, j]
            delta = Xnew[j, :] - self._Xmat.T @ VinvC[:, j]
            corr = delta.T @ self._XtVinvX_inv @ delta
            var_pred[j] = max(cond + corr, 1e-12)

        return var_pred


def compute_metrics(y_true, y_pred):
    """R², MAE, RMSE, MB"""
    mask = ~(np.isnan(y_true) | np.isnan(y_pred) |
             np.isinf(y_true) | np.isinf(y_pred))
    yt, yp = y_true[mask], y_pred[mask]
    if len(yt) == 0:
        return {'R2': np.nan, 'MAE': np.nan, 'RMSE': np.nan, 'MB': np.nan}
    return {
        'R2': float(r2_score(yt, yp)),
        'MAE': float(mean_absolute_error(yt, yp)),
        'RMSE': float(np.sqrt(mean_squared_error(yt, yp))),
        'MB': float(np.mean(yp - yt))
    }
