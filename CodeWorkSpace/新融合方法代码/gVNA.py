import os
# -*- coding: utf-8 -*-
"""
gVNA - Generalized Voronoi Neighbor Averaging
==============================================
统一VNA/eVNA/aVNA的确定性空间融合方法

核心思想：
1. 相似性增强权重: w = d^(-p) * exp(-|M-M'|/|λ|)
2. 加性偏差校正: y = M(s) + bias_interp
3. 自适应λ: 基于每日CMAQ站点R²动态调节λ

方法定位：
- 主方法：similarity-informed aVNA (bias-only)
- 理论框架：统一VNA/eVNA/aVNA的确定性融合框架
- λ选择：标准变异函数（地统计学优雅方法）

作者: Data Fusion Auto Research
日期: 2026-04-23
"""

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


def estimate_lambda_variogram(lons, lats, biases, n_bins=15, max_dist=None):
    """
    基于标准变异函数的空间相关距离估计

    原理：
    γ(h) = 0.5 * E[(b(x) - b(x+h))²]  是空间偏差的半方差函数
    γ(h) → σ² 当 h 增大（空间独立）
    λ = 实用变程（practical range）：γ(h) = 0.85 * σ² 时的距离

    这是标准地统计学方法，从数据中直接学习空间相关尺度。

    参数:
    ------
    lons, lats : array
        站点坐标
    biases : array
        偏差值（观测值 - 模拟值）
    n_bins : int
        距离分层数
    max_dist : float
        最大搜索距离（度），默认自动

    返回:
    ------
    lambda_bg : float
        变异函数实用变程（空间相关距离）
    """
    n = len(biases)
    if n < 20:
        return 8.0

    coords = np.column_stack([lons, lats])

    if max_dist is None:
        max_dist = np.sqrt(np.ptp(coords[:, 0])**2 + np.ptp(coords[:, 1])**2) / 4
        max_dist = min(max_dist, 20.0)

    bin_edges = np.linspace(0, max_dist, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    gamma_vals = np.zeros(n_bins)
    counts = np.zeros(n_bins)

    for i in range(n):
        for j in range(i + 1, min(i + 30, n)):
            d = np.sqrt((coords[i, 0] - coords[j, 0])**2 + (coords[i, 1] - coords[j, 1])**2)
            if d < max_dist:
                bin_idx = np.searchsorted(bin_edges[1:], d)
                if bin_idx < n_bins:
                    gamma_vals[bin_idx] += 0.5 * (biases[i] - biases[j])**2
                    counts[bin_idx] += 1

    with np.errstate(divide='ignore', invalid='ignore'):
        gamma_mean = np.where(counts > 0, gamma_vals / counts, np.nan)

    sill = np.var(biases)
    if sill < 1e-6:
        return 8.0

    valid = counts > 0
    if valid.sum() < 3:
        return 8.0

    bc = bin_centers[valid]
    gv = gamma_mean[valid]

    # 指数变异函数模型: γ(h) = σ² * (1 - exp(-3h/λ))
    # => λ = -3h / ln(1 - γ/σ²)
    ranges_est = -3 * bc / np.log(np.maximum(1e-6, 1 - gv / (sill + 1e-6)))
    ranges_est = ranges_est[np.isfinite(ranges_est) & (ranges_est > 0) & (ranges_est < 50)]

    if len(ranges_est) == 0:
        return 8.0

    return float(np.clip(np.median(ranges_est), 1.0, 50.0))


def adaptive_lambda(cmaq_r2):
    """
    基于每日 CMAQ 站点 R² 自适应选择 λ

    原理：当日 CMAQ 模型质量好时，可信任更多邻近站点（λ较大）；
          CMAQ 质量差时，需严格过滤不一致邻近站点（λ较小）。

    参数:
    ------
    cmaq_r2 : float
        当日 CMAQ 在监测站点的 R²（由训练集计算）

    返回:
    ------
    lambda_bg : int
        推荐使用的 λ 值

    示例:
    ------
    >>> adaptive_lambda(0.5)   # CMAQ很好 -> 12
    >>> adaptive_lambda(0.2)   # CMAQ一般 -> 8
    >>> adaptive_lambda(0.05)  # CMAQ很差 -> 3
    """
    if np.isnan(cmaq_r2):
        return 8  # 默认值
    if cmaq_r2 < 0.05:
        return 3
    elif cmaq_r2 < 0.20:
        return 5
    elif cmaq_r2 < 0.35:
        return 8
    else:
        return 12


class gVNA:
    """
    gVNA: Similarity-informed Voronoi Neighbor Averaging

    使用背景相似性增强权重的广义VNA方法

    参数:
    ------
    k : int, default=30
        预测插值的邻近站点数
    p : float, default=2
        距离衰减指数
    lambda_bg : float, default=None
        背景相似性尺度参数（当 adaptive=False 时使用）
        - 如果为 None，则自动估计
        - 如果为数值，则使用固定值
    lambda_method : str, default='median'
        自动估计方法（当 adaptive=False 且 lambda_bg=None 时使用）
        - 'median': 中位数
        - 'std': 标准差
        - 'rmse': 均方根误差
        - 'variogram': 标准变异函数（地统计学空间相关距离）
    adaptive : bool, default=False
        是否使用自适应 λ 方案（基于每日 CMAQ R²）
        - True: 使用 adaptive_lambda() 动态确定 λ
        - False: 使用固定的 lambda_bg 或自动估计值
    clip_nonnegative : bool, default=True
        是否将预测值裁剪为非负（PM2.5物理上不能为负）

    Note:
    -----
    本实现为 bias-only 版本，即加性偏差校正：
        y = M(s) + sum(w_i * (O_i - M_i))

    λ 选择：标准变异函数法（默认），通过计算偏差场的空间半方差函数
    估计空间相关距离——经典地统计学方法，无需调参。
    """

    def __init__(self, k=30, p=2, lambda_bg=None, lambda_method='variogram',
                 adaptive=False, clip_nonnegative=True):
        """
        参数:
        ------
        k : int, default=30
            预测插值的邻近站点数
        p : float, default=2
            距离衰减指数
        lambda_bg : float, default=None
            背景相似性尺度参数（当 adaptive=False 时）
        lambda_method : str, default='variogram'
            自动估计方法：'variogram'（标准变异函数）、'median'、'std'、'rmse'
        adaptive : bool, default=False
            是否使用自适应 λ
        clip_nonnegative : bool, default=True
            是否将预测值裁剪为非负
        """
        self.k = k
        self.p = p
        self.lambda_bg = lambda_bg
        self.lambda_method = lambda_method
        self.adaptive = adaptive
        self.clip_nonnegative = clip_nonnegative

        self.train_lon = None
        self.train_lat = None
        self.train_Conc = None
        self.train_mod = None
        self.lambda_bg_estimated = None
        self.cmaq_r2_used = None

    def _estimate_lambda(self):
        """
        自动估计lambda_bg
        """
        if self.lambda_method == 'variogram':
            # 标准变异函数法：基于空间相关距离
            biases = self.train_Conc - self.train_mod
            return estimate_lambda_variogram(
                self.train_lon, self.train_lat, biases)

        n = len(self.train_mod)
        all_diffs = []
        max_pairs = 5000
        for i in range(n):
            for j in range(i+1, min(i+50, n)):
                all_diffs.append(abs(self.train_mod[i] - self.train_mod[j]))
            if len(all_diffs) >= max_pairs:
                break

        if not all_diffs:
            return 15.0

        all_diffs = np.array(all_diffs)

        if self.lambda_method == 'median':
            return np.median(all_diffs)
        elif self.lambda_method == 'std':
            return np.std(all_diffs)
        elif self.lambda_method == 'rmse':
            return np.sqrt(np.mean(all_diffs**2))
        else:
            return np.median(all_diffs)

    def _compute_distance(self, coord1, coord2):
        """
        计算两点间距离（简化版本，使用经纬度差值）
        对于全国范围虽不精确，但与VNA/eVNA/aVNA保持一致
        """
        return np.sqrt((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2)

    def _find_k_nearest(self, target_coord, coords_array, k):
        """找到k个最近邻"""
        n = len(coords_array)
        distances = np.array([
            self._compute_distance(target_coord, coords_array[i])
            for i in range(n)
        ])

        indices = np.argsort(distances)[:k]
        dists = distances[indices]
        return indices, dists

    def _compute_similarity_weights(self, target_mod, station_mods, indices):
        """
        计算背景相似性权重
        """
        dists = np.abs(target_mod - station_mods[indices])
        sim_weights = np.exp(-dists / self.lambda_bg_estimated)
        return sim_weights

    def fit(self, train_lon, train_lat, train_Conc, train_mod, cmaq_r2=None):
        """
        训练gVNA模型

        参数:
        ------
        train_lon, train_lat : array-like
            训练站点坐标
        train_Conc : array-like
            训练站点观测值（监测值）
        train_mod : array-like
            训练站点CMAQ模拟值
        cmaq_r2 : float, optional
            当日CMAQ在训练集上的R²（用于自适应λ选择）
            如果不提供且adaptive=True，则自动计算
        """
        # 数据清洗
        train_lon = np.asarray(train_lon, dtype=np.float64)
        train_lat = np.asarray(train_lat, dtype=np.float64)
        train_Conc = np.asarray(train_Conc, dtype=np.float64)
        train_mod = np.asarray(train_mod, dtype=np.float64)

        mask = ~(
            np.isnan(train_lon) | np.isnan(train_lat) |
            np.isnan(train_Conc) | np.isnan(train_mod) |
            np.isinf(train_lon) | np.isinf(train_lat) |
            np.isinf(train_Conc) | np.isinf(train_mod)
        )

        self.train_lon = train_lon[mask]
        self.train_lat = train_lat[mask]
        self.train_Conc = train_Conc[mask]
        self.train_mod = train_mod[mask]

        # 确定 λ
        if self.adaptive:
            # 自适应方案：基于 CMAQ R² 确定 λ
            if cmaq_r2 is None:
                # 自动计算训练集上的 CMAQ R²
                valid = ~np.isnan(self.train_mod) & ~np.isnan(self.train_Conc)
                if valid.sum() > 10:
                    cmaq_r2 = r2_score(self.train_Conc[valid], self.train_mod[valid])
                else:
                    cmaq_r2 = np.nan
            self.cmaq_r2_used = cmaq_r2
            self.lambda_bg_estimated = adaptive_lambda(cmaq_r2)
        elif self.lambda_bg is None:
            # 非自适应模式：自动估计
            self.lambda_bg_estimated = self._estimate_lambda()
        else:
            # 非自适应模式：使用固定值
            self.lambda_bg_estimated = self.lambda_bg

        return self

    def predict_single(self, lon, lat, mod):
        """
        预测单点

        使用相似性增强权重插值加性偏差
        """
        coords = np.column_stack([self.train_lon, self.train_lat])
        target_coord = np.array([lon, lat])

        k_eff = min(self.k, len(coords))
        indices, dists = self._find_k_nearest(target_coord, coords, k_eff)

        if len(indices) == 0:
            return np.nan

        # 距离权重
        dist_weights = np.power(dists + 1e-10, -self.p)

        # 相似性权重
        sim_weights = self._compute_similarity_weights(mod, self.train_mod, indices)

        # 组合权重
        weights = dist_weights * sim_weights
        weights = weights / (weights.sum() + 1e-10)

        # 加性偏差校正: y = M + bias_interp
        biases = self.train_Conc - self.train_mod
        bias_interp = np.sum(weights * biases[indices])

        y_pred = mod + bias_interp

        if self.clip_nonnegative:
            y_pred = max(0.0, y_pred)

        return y_pred

    def predict(self, X, mod):
        """
        批量预测

        参数:
        ------
        X : array-like, shape (n, 2)
            预测点坐标 [lon, lat]
        mod : array-like, shape (n,)
            预测点对应的CMAQ值（必须提供）

        返回:
        ------
        y_pred : array, shape (n,)
        """
        if mod is None:
            raise ValueError("mod must be provided for prediction.")

        X = np.asarray(X)
        mod = np.asarray(mod)

        if X.ndim != 2 or X.shape[1] != 2:
            raise ValueError("X must have shape (n, 2)")
        if mod.ndim != 1:
            raise ValueError("mod must be a 1D array")
        if len(X) != len(mod):
            raise ValueError("X and mod must have the same length")

        return np.array([
            self.predict_single(X[i, 0], X[i, 1], mod[i])
            for i in range(len(X))
        ])


def compute_metrics(y_true, y_pred):
    """计算评估指标"""
    mask = ~(np.isnan(y_true) | np.isnan(y_pred) | np.isinf(y_true) | np.isinf(y_pred))
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


if __name__ == '__main__':
    print("gVNA 方法已定义 (bias-only + similarity weight + variogram lambda)")
    print()
    print("变异函数模式（默认，推荐）:")
    print("  model = gVNA(adaptive=False, lambda_method='variogram')")
    print("  model.fit(train_lon, train_lat, train_Conc, train_mod)")
    print("  pred = model.predict(X_test, mod_test)")
    print()
    print("固定lambda模式:")
    print("  model = gVNA(adaptive=False, lambda_bg=8)     # 固定lambda=8")
    print()
    print("其他自动估计方法:")
    print("  lambda_method='median': 站点CMAQ差异中位数")
    print("  lambda_method='std':    站点CMAQ差异标准差")
    print("  lambda_method='rmse':   站点CMAQ差异均方根")
    print()
    print("自适应模式:")
    print("  model = gVNA(adaptive=True)                    # 基于CMAQ R²")
    print("  model.fit(train_lon, train_lat, train_Conc, train_mod, cmaq_r2=0.2)")
    print()
    print("变异函数原理:")
    print("  γ(h) = 0.5 * E[(b(x) - b(x+h))²]")
    print("  λ = 实用变程: γ(h) = 0.85 * σ² 时的距离")
