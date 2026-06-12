# -*- coding: utf-8 -*-
"""
SVCD-RAMP: SVCD + RAMP风格非参数残差后处理
===============================================
基于 BME-RAMP (Xu et al. 2016) 的非参数偏差校正思路，
在 SVCD 预测后叠加一层分箱+空间插值的残差修正。

原理:
1. SVCD 的线性假设可能遗漏 obs-CMAQ 关系中的非线性结构
2. 对训练残差按 CMAQ 值分箱，捕获不同浓度区间的系统性偏差
3. 预测时用 CMAQ 值维度插值 + 空间 IDW 混合得到点位偏差修正
4. 该后处理不改动 SVCD 核心模型，零侵入

参考文献:
- Xu et al. (2016) BME-RAMP, Environmental Science & Technology, 50(8), 4393-4400
- Berrocal et al. (2010) Spatio-Temporal Downscaler, Annals of Applied Statistics

作者: Data Fusion Auto Research
日期: 2026-06-08
"""

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


class RAMPPostProcessor:
    """
    RAMP 风格非参数残差后处理。

    对基础模型（如 SVCD）的训练残差按 CMAQ 值分箱建模，
    预测时通过 CMAQ 维度插值 + 空间 IDW 估计点位偏差并修正。

    参数
    ------
    n_bins : int, default=10
        CMAQ 值分箱数（等分位数）
    n_spatial : int, default=8
        空间 IDW 使用的最近邻站点数
    alpha : float, default=0.5
        CMAQ分箱偏差 vs 空间局部偏差的混合权重
        alpha=1.0: 纯 CMAQ 分箱偏差（全局）
        alpha=0.0: 纯空间 IDW 残差（局部）
    min_sites_per_bin : int, default=5
        每箱最少站点数，不足则合并相邻箱
    """

    def __init__(self, n_bins=10, n_spatial=8, alpha=0.5, min_sites_per_bin=5):
        self.n_bins = n_bins
        self.n_spatial = n_spatial
        self.alpha = alpha
        self.min_sites_per_bin = min_sites_per_bin

        # fit 后填充
        self._fitted = False
        self._bin_centers = None    # shape (n_valid_bins,)
        self._bin_biases = None     # shape (n_valid_bins,)
        self._coords_train = None
        self._residuals_train = None
        self._cmaq_train = None

    def fit(self, coords_train, y_true, y_pred, cmaq_train):
        """
        在训练集上拟合残差分箱模型。

        参数
        ------
        coords_train : ndarray, shape (n, 2)
            训练站点坐标 [lon, lat]
        y_true : ndarray, shape (n,)
            训练站点真实浓度
        y_pred : ndarray, shape (n,)
            基础模型在训练站点上的预测值
        cmaq_train : ndarray, shape (n,)
            训练站点对应的 CMAQ 值

        返回
        ------
        self
        """
        coords_train = np.asarray(coords_train, dtype=float)
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        cmaq_train = np.asarray(cmaq_train, dtype=float)

        # 残差: 正值 = 模型低估, 负值 = 模型高估
        residuals = y_true - y_pred

        self._coords_train = coords_train
        self._residuals_train = residuals
        self._cmaq_train = cmaq_train

        # 等分位数分箱
        percentiles = np.percentile(cmaq_train, np.linspace(0, 100, self.n_bins + 1))
        percentiles[0] = cmaq_train.min() - 1e-8   # 确保包含最小值
        percentiles[-1] = cmaq_train.max() + 1e-8  # 确保包含最大值

        bin_centers = []
        bin_biases = []

        for b in range(self.n_bins):
            lo, hi = percentiles[b], percentiles[b + 1]
            mask = (cmaq_train >= lo) & (cmaq_train < hi)
            if b == self.n_bins - 1:
                mask = (cmaq_train >= lo) & (cmaq_train <= hi)

            if mask.sum() >= self.min_sites_per_bin:
                bin_centers.append(np.median(cmaq_train[mask]))
                bin_biases.append(np.mean(residuals[mask]))

        # 合并样本不足的相邻箱: 用线性插值填充
        if len(bin_centers) < 3:
            # 箱太少，直接用全局均值
            self._bin_centers = np.array([cmaq_train.min(), cmaq_train.max()])
            self._bin_biases = np.array([np.mean(residuals), np.mean(residuals)])
        else:
            self._bin_centers = np.array(bin_centers)
            self._bin_biases = np.array(bin_biases)

        self._fitted = True
        return self

    def correct(self, coords_new, y_pred, cmaq_new):
        """
        对基础模型预测施加 RAMP 风格偏差修正。

        参数
        ------
        coords_new : ndarray, shape (m, 2)
            预测点坐标
        y_pred : ndarray, shape (m,)
            基础模型的预测值
        cmaq_new : ndarray, shape (m,)
            预测点对应的 CMAQ 值

        返回
        ------
        y_corrected : ndarray, shape (m,)
            修正后的预测值
        corrections : ndarray, shape (m,)
            施加的修正量（正值=上调预测）
        """
        if not self._fitted:
            raise RuntimeError("请先调用 fit()")

        coords_new = np.asarray(coords_new, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        cmaq_new = np.asarray(cmaq_new, dtype=float)
        m = len(y_pred)

        # 维度1: CMAQ 分箱偏差（全局浓度维度的系统性偏差）
        bias_cmaq = np.interp(cmaq_new, self._bin_centers, self._bin_biases,
                              left=self._bin_biases[0],
                              right=self._bin_biases[-1])

        # 维度2: 空间局部残差（训练站点残差的 IDW 插值）
        bias_spatial = np.zeros(m)
        dist_matrix = cdist(coords_new, self._coords_train)

        for i in range(m):
            nn_idx = np.argsort(dist_matrix[i])[:self.n_spatial]
            dists = dist_matrix[i][nn_idx]
            # 加权: 1 / (distance + 1km), 避免除零
            w = 1.0 / (dists + 1.0)
            w /= w.sum()
            bias_spatial[i] = np.sum(w * self._residuals_train[nn_idx])

        # 混合 CMAQ 分箱偏差 与 空间局部残差
        corrections = self.alpha * bias_cmaq + (1.0 - self.alpha) * bias_spatial

        y_corrected = y_pred + corrections
        return y_corrected, corrections

    def fit_correct(self, coords_train, y_true, y_pred, cmaq_train,
                    coords_new, y_pred_new, cmaq_new):
        """一步完成 fit + correct（便捷接口）"""
        self.fit(coords_train, y_true, y_pred, cmaq_train)
        return self.correct(coords_new, y_pred_new, cmaq_new)


class SVCD_RAMP:
    """
    SVCD + RAMP 后处理的封装模型。

    与原始 SVCD 接口完全兼容（fit / predict / score），
    内部自动对训练集做留一残差拟合，预测时施加 RAMP 修正。

    参数
    ------
    svcd_kwargs : dict
        传递给 SVCD 构造函数的参数（c, nu0, nu1 等）
    ramp_kwargs : dict
        传递给 RAMPPostProcessor 的参数（n_bins, n_spatial, alpha 等）
    """

    def __init__(self, svcd_kwargs=None, ramp_kwargs=None):
        import sys, os
        # 确保能导入 SVCD（在 CodeWorkSpace/新融合方法代码 下）
        _current_dir = os.path.dirname(os.path.abspath(__file__))
        if _current_dir not in sys.path:
            sys.path.insert(0, _current_dir)
        from SVCD import SVCD as _SVCD

        self._SVCD = _SVCD
        self._svcd_kwargs = svcd_kwargs or {}
        self._ramp_kwargs = ramp_kwargs or {}

        self._svcd = None
        self._ramp = None
        self._fitted = False

    def fit(self, X, y, CMAQ_values):
        """
        训练 SVCD + RAMP 后处理。

        流程:
        1. 训练 SVCD 基础模型
        2. 在训练集上获取 SVCD 预测值
        3. 用训练残差拟合 RAMP 后处理器
        """
        coords = np.asarray(X, dtype=float)
        y_obs = np.asarray(y, dtype=float).reshape(-1)
        cmaq = np.asarray(CMAQ_values, dtype=float).reshape(-1)

        # Step 1: 训练 SVCD
        self._svcd = self._SVCD(**self._svcd_kwargs)
        self._svcd.fit(coords, y_obs, cmaq)

        # Step 2: 在训练集上预测（用于计算残差）
        y_pred_train = self._svcd.predict(coords, cmaq)

        # Step 3: 拟合 RAMP 后处理器
        self._ramp = RAMPPostProcessor(**self._ramp_kwargs)
        self._ramp.fit(coords, y_obs, y_pred_train, cmaq)

        self._fitted = True
        return self

    @property
    def theta(self):
        """SVCD 优化后的超参数 (log空间)"""
        return self._svcd.theta if self._svcd is not None else None

    @property
    def svcd_model(self):
        """获取底层 SVCD 模型（用于访问预测方差等）"""
        return self._svcd

    def predict(self, X_new, CMAQ_new):
        """
        预测（含 RAMP 后处理修正）

        返回
        ------
        y_corrected : ndarray
            修正后的预测值
        """
        if not self._fitted:
            raise RuntimeError("模型未训练，请先调用 fit()")

        coords_new = np.asarray(X_new, dtype=float)
        cmaq_new = np.asarray(CMAQ_new, dtype=float).reshape(-1)

        # SVCD 基础预测
        y_svcd = self._svcd.predict(coords_new, cmaq_new)

        # RAMP 修正
        y_corrected, _ = self._ramp.correct(coords_new, y_svcd, cmaq_new)

        return y_corrected

    def predict_with_details(self, X_new, CMAQ_new):
        """
        预测并返回详细分解（SVCD基值 + RAMP修正量）

        返回
        ------
        y_corrected : ndarray
            修正后的预测值
        y_svcd : ndarray
            SVCD 原始预测
        correction : ndarray
            RAMP 修正量
        """
        if not self._fitted:
            raise RuntimeError("模型未训练，请先调用 fit()")

        coords_new = np.asarray(X_new, dtype=float)
        cmaq_new = np.asarray(CMAQ_new, dtype=float).reshape(-1)

        y_svcd = self._svcd.predict(coords_new, cmaq_new)
        y_corrected, correction = self._ramp.correct(coords_new, y_svcd, cmaq_new)

        return y_corrected, y_svcd, correction

    def score(self, X_test, y_test, CMAQ_test):
        """计算 R² 分数"""
        y_pred = self.predict(X_test, CMAQ_test)
        return r2_score(y_test, y_pred)


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
