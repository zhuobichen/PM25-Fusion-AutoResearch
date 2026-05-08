import os
# -*- coding: utf-8 -*-
"""
GD-IDW - Gradient-Direction Inverse Distance Weighting
=======================================================
基于CMAQ梯度方向信息的确定性空间插值方法

创新点：
1. 利用CMAQ局部梯度计算上风/下风方向
2. 上风方向站点权重增强，下风方向减弱
3. 纯确定性方法，无统计学习

作者: Data Fusion Auto Research
日期: 2026-04-22
"""

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

sys.path.insert(0, 'E:/CodeProject/ClaudeRoom/Data_Fusion_AutoResearch/Code/VNAeVNAaVNA')

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from Code.VNAeVNAaVNA.nna_methods import NNA


class GDIDW:
    """
    GD-IDW: Gradient-Direction Inverse Distance Weighting

    参数:
    ------
    k : int, default=30
        近邻数量
    power : float, default=-2
        距离权重指数
    beta : float, default=0.5
        方向敏感系数（β=0时退化为标准IDW）
    use_gradient : bool, default=True
        是否使用梯度方向调整
    """

    def __init__(self, k=30, power=-2, beta=0.5, use_gradient=True):
        self.k = k
        self.power = power
        self.beta = beta
        self.use_gradient = use_gradient

        self.train_lon = None
        self.train_lat = None
        self.train_Conc = None
        self.train_mod = None
        self.train_bias = None
        self.train_rn = None
        self.gradient_direction = None

    def _compute_gradient_direction(self, lon_grid, lat_grid, pm25_grid):
        """
        计算CMAQ网格的局部梯度方向

        返回:
            gradient_lon, gradient_lat: 梯度向量场（与pm25_grid同形状）
        """
        ny, nx = pm25_grid.shape

        # 计算经度方向梯度
        gradient_lon = np.zeros_like(pm25_grid)
        gradient_lon[:, 1:-1] = (pm25_grid[:, 2:] - pm25_grid[:, :-2]) / 2
        gradient_lon[:, 0] = gradient_lon[:, 1]
        gradient_lon[:, -1] = gradient_lon[:, -2]

        # 计算纬度方向梯度
        gradient_lat = np.zeros_like(pm25_grid)
        gradient_lat[1:-1, :] = (pm25_grid[2:, :] - pm25_grid[:-2, :]) / 2
        gradient_lat[0, :] = gradient_lat[1, :]
        gradient_lat[-1, :] = gradient_lat[-2, :]

        return gradient_lon, gradient_lat

    def _get_gradient_at_point(self, lon, lat, lon_grid, lat_grid,
                                gradient_lon, gradient_lat):
        """
        获取某点的CMAQ梯度值
        """
        dist = np.sqrt((lon_grid - lon)**2 + (lat_grid - lat)**2)
        idx = np.argmin(dist)
        ny, nx = lon_grid.shape
        row, col = idx // nx, idx % nx
        return gradient_lon[row, col], gradient_lat[row, col]

    def fit(self, train_lon, train_lat, train_Conc, train_mod,
            lon_grid=None, lat_grid=None, pm25_grid=None):
        """
        训练GD-IDW模型

        参数:
        ------
        train_lon, train_lat : array-like
            训练点经纬度
        train_Conc : array-like
            监测值
        train_mod : array-like
            CMAQ模型值
        lon_grid, lat_grid, pm25_grid : 可选
            CMAQ全网格数据（用于计算梯度）
        """
        self.train_lon = np.asarray(train_lon)
        self.train_lat = np.asarray(train_lat)
        self.train_Conc = np.asarray(train_Conc)
        self.train_mod = np.asarray(train_mod)
        self.train_bias = self.train_Conc - self.train_mod
        self.train_rn = self.train_Conc / self.train_mod

        # 如果提供了CMAQ网格数据，计算梯度方向
        if self.use_gradient and lon_grid is not None:
            grad_lon, grad_lat = self._compute_gradient_direction(
                lon_grid, lat_grid, pm25_grid
            )
            self.gradient_lon_grid = grad_lon
            self.gradient_lat_grid = grad_lat
            self.lon_grid = lon_grid
            self.lat_grid = lat_grid
        else:
            self.gradient_lon_grid = None
            self.gradient_lat_grid = None

        return self

    def _compute_direction_weight(self, lon, lat):
        """
        计算方向调整因子α

        对于预测点(lon, lat)：
        - 如果有CMAQ梯度，计算该点的梯度方向
        - 计算每个训练点相对于预测点的方向
        - 上风方向（与梯度方向相反）权重增强
        """
        if not self.use_gradient or self.gradient_lon_grid is None:
            return np.ones(len(self.train_lon))

        # 获取预测点处的CMAQ梯度
        gx, gy = self._get_gradient_at_point(
            lon, lat, self.lon_grid, self.lat_grid,
            self.gradient_lon_grid, self.gradient_lat_grid
        )

        # 如果梯度很小（接近均匀场），返回均匀权重
        grad_mag = np.sqrt(gx**2 + gy**2)
        if grad_mag < 1e-10:
            return np.ones(len(self.train_lon))

        # 梯度方向（从低浓度指向高浓度）
        theta_grad = np.arctan2(gy, gx)

        # 计算每个训练点相对于预测点的方向
        dx = self.train_lon - lon
        dy = self.train_lat - lat
        theta_points = np.arctan2(dy, dx)

        # 计算角度差（点到梯度源的方向 vs 梯度方向）
        # 上风方向：点沿着梯度相反方向，即污染来向
        # cos(θ_grad - θ_point) 接近1表示上风方向，接近-1表示下风方向
        angle_diff = theta_grad - theta_points

        # 方向调整因子：exp(β * cos(θ_diff))
        # 上风方向 cos≈1 → α>1 增强权重
        # 下风方向 cos≈-1 → α<1 减弱权重
        alpha = np.exp(self.beta * np.cos(angle_diff))

        return alpha

    def predict_single(self, lon, lat, method='vna'):
        """
        预测单点

        参数:
        ------
        lon, lat : float
            预测点坐标
        method : str
            'vna' - 直接插值监测值
            'gd' - CMAQ值 + 插值bias
            'ge' - CMAQ值 × 插值r_n
        """
        # 计算距离
        dist = np.sqrt(
            (self.train_lon - lon)**2 +
            (self.train_lat - lat)**2
        )

        # 距离权重
        dist_power = np.power(dist, self.power)

        # 方向调整因子
        alpha = self._compute_direction_weight(lon, lat)

        # 组合权重
        weights = dist_power * alpha

        # 归一化
        weights = weights / np.sum(weights)

        if method == 'vna':
            return np.sum(weights * self.train_Conc)
        elif method == 'gd':
            # 需要先插值bias，然后 CMAQ + bias
            bias_interp = np.sum(weights * self.train_bias)
            return bias_interp
        elif method == 'ge':
            # 需要先插值r_n，然后 CMAQ × r_n
            rn_interp = np.sum(weights * self.train_rn)
            return rn_interp

    def predict(self, X, method='vna'):
        """
        批量预测

        参数:
        ------
        X : array-like, shape (n, 2)
            预测点坐标 [lon, lat]
        method : str
            'vna' - 直接插值监测值
            'gd' - CMAQ值 + 插值bias
            'ge' - CMAQ值 × 插值r_n

        返回:
        ------
        y_pred : array, shape (n,)
        """
        X = np.asarray(X)
        return np.array([
            self.predict_single(lon, lat, method)
            for lon, lat in X
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
    # 演示代码
    print("GD-IDW 方法已定义")
    print("使用示例:")
    print("  model = GDIDW(k=30, power=-2, beta=0.5)")
    print("  model.fit(train_lon, train_lat, train_Conc, train_mod, lon_grid, lat_grid, pm25_grid)")
    print("  pred = model.predict(X_test)")
