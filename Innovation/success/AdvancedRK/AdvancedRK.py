import os
from shared.paths import get_project_root, data_path
# -*- coding: utf-8 -*-
"""
AdvancedRK - Advanced Residual Kriging with Matern Kernel
=========================================================
AdvancedRK: 基于Matern核的高斯过程回归残差克里金方法

方法概述:
1. 二阶多项式趋势校正: 建立CMAQ与监测值的非线性偏差关系 O = f(CMAQ)
2. GPR残差空间建模: 使用Matern(ν=1.5)核函数对残差进行克里金插值
3. 融合预测: 多项式校正值 + GPR残差预测值

与PolyRK的区别:
- PolyRK使用RBF核
- AdvancedRK使用Matern(ν=1.5)核，更符合PM2.5空间扩散的物理规律

参考文献:
- PolyRK: 多项式残差克里金基础方法
- Matern kernel: 更灵活的的空间相关性建模

作者: Data Fusion Auto Research
日期: 2026-04-16
"""

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


class AdvancedRK:
    """
    AdvancedRK: 二阶多项式校正 + GPR-Matern残差克里金

    参数:
    ------
    poly_degree : int, default=2
        多项式阶数（建议使用2，二阶多项式在大多数情况下最优）
    nu : float, default=1.5
        Matern核的平滑参数（ν=1.5时等价于指数核的平滑版本）
    kernel_optimize : bool, default=True
        是否自动优化GPR核函数参数
    """

    def __init__(self, poly_degree=2, nu=1.5, kernel_optimize=True):
        self.poly_degree = poly_degree
        self.nu = nu
        self.kernel_optimize = kernel_optimize

        self.poly_features = None
        self.ols_model = None
        self.gpr_model = None
        self.X_train = None

    def _build_kernel(self):
        """构建GPR的Matern核函数"""
        kernel = (
            ConstantKernel(10.0, (1e-2, 1e3)) *
            Matern(
                length_scale=1.0,
                length_scale_bounds=(1e-2, 1e2),
                nu=self.nu
            ) +
            WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e1))
        )
        return kernel

    def fit(self, X, y, CMAQ_values):
        """
        训练AdvancedRK模型

        参数:
        ------
        X : array-like, shape (n_samples, 2)
            训练点的坐标 [lon, lat]
        y : array-like, shape (n_samples,)
            监测站真实值（PM2.5浓度）
        CMAQ_values : array-like, shape (n_samples,)
            对应的CMAQ模型预测值

        返回:
        ------
        self
        """
        # Step 1: 多项式校正 - 拟合 CMAQ -> Obs 的非线性关系
        self.poly_features = PolynomialFeatures(
            degree=self.poly_degree,
            include_bias=False
        )
        CMAQ_poly = self.poly_features.fit_transform(CMAQ_values.reshape(-1, 1))

        self.ols_model = LinearRegression()
        self.ols_model.fit(CMAQ_poly, y)

        # 计算残差
        residual = y - self.ols_model.predict(CMAQ_poly)

        # Step 2: GPR空间建模 - 使用Matern核
        self.X_train = np.array(X)
        kernel = self._build_kernel()

        self.gpr_model = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=2,
            alpha=0.1,
            normalize_y=True
        )
        self.gpr_model.fit(self.X_train, residual)

        return self

    def predict(self, X_new, CMAQ_new):
        """
        预测新位置的PM2.5浓度

        参数:
        ------
        X_new : array-like, shape (n_samples, 2)
            新位置的坐标 [lon, lat]
        CMAQ_new : array-like, shape (n_samples,)
            新位置对应的CMAQ模型预测值

        返回:
        ------
        y_pred : array-like, shape (n_samples,)
            融合后的预测值
        residual_std : array-like, shape (n_samples,), optional
            预测的标准差（当return_std=True时返回）
        """
        # 多项式校正
        CMAQ_poly = self.poly_features.transform(CMAQ_new.reshape(-1, 1))
        poly_pred = self.ols_model.predict(CMAQ_poly)

        # GPR残差预测
        X_new = np.array(X_new)
        gpr_pred = self.gpr_model.predict(X_new)

        # 融合
        y_pred = poly_pred + gpr_pred

        return y_pred

    def predict_with_uncertainty(self, X_new, CMAQ_new):
        """
        预测新位置的PM2.5浓度，并返回不确定性估计

        参数:
        ------
        X_new : array-like, shape (n_samples, 2)
            新位置的坐标 [lon, lat]
        CMAQ_new : array-like, shape (n_samples,)
            新位置对应的CMAQ模型预测值

        返回:
        ------
        y_pred : array-like, shape (n_samples,)
            融合后的预测值
        pred_std : array-like, shape (n_samples,)
            预测的标准差（不确定性）
        """
        # 多项式校正
        CMAQ_poly = self.poly_features.transform(CMAQ_new.reshape(-1, 1))
        poly_pred = self.ols_model.predict(CMAQ_poly)

        # GPR残差预测（带不确定性）
        X_new = np.array(X_new)
        gpr_pred, gpr_std = self.gpr_model.predict(X_new, return_std=True)

        # 融合
        y_pred = poly_pred + gpr_pred

        return y_pred, gpr_std

    def get_residual_std(self, X_new, CMAQ_new):
        """
        获取预测不确定性（标准差）

        参数:
        ------
        X_new : array-like, shape (n_samples, 2)
            新位置的坐标 [lon, lat]
        CMAQ_new : array-like, shape (n_samples,)
            新位置对应的CMAQ模型预测值

        返回:
        ------
        pred_std : array-like, shape (n_samples,)
            预测的标准差（不确定性）

        物理意义:
        - 标准差大：该位置附近监测站稀疏，预测不可靠
        - 标准差小：该位置附近监测站密集，预测可靠
        """
        X_new = np.array(X_new)
        CMAQ_poly = self.poly_features.transform(CMAQ_new.reshape(-1, 1))
        poly_pred = self.ols_model.predict(CMAQ_poly)
        gpr_pred, gpr_std = self.gpr_model.predict(X_new, return_std=True)
        return gpr_std

    def score(self, X_test, y_test, CMAQ_test):
        """
        计算预测的R²分数

        参数:
        ------
        X_test : array-like, shape (n_samples, 2)
            测试点坐标
        y_test : array-like, shape (n_samples,)
            测试点真实值
        CMAQ_test : array-like, shape (n_samples,)
            测试点CMAQ预测值

        返回:
        ------
        r2 : float
            R²分数
        """
        y_pred = self.predict(X_test, CMAQ_test)
        return r2_score(y_test, y_pred)


def compute_metrics(y_true, y_pred):
    """
    计算评估指标

    参数:
    ------
    y_true : array-like
        真实值
    y_pred : array-like
        预测值

    返回:
    ------
    metrics : dict
        包含R²、MAE、RMSE、MB的字典
    """
    mask = ~(
        np.isnan(y_true) |
        np.isnan(y_pred) |
        np.isinf(y_true) |
        np.isinf(y_pred)
    )
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
    # 使用示例
    import netCDF4 as nc
    from datetime import datetime

    ROOT_DIR = str(get_project_root())
    CMAQ_FILE = data_path('test_data/raw/CMAQ/2020_PM25.nc')
    MONITOR_FILE = data_path('test_data/raw/Monitor/2020_DailyPM2.5Monitor.csv')

    # 加载数据
    monitor_df = pd.read_csv(MONITOR_FILE)
    monitor_df = monitor_df[monitor_df['Date'] == '2020-01-01'].dropna(
        subset=['Lat', 'Lon', 'Conc']
    )

    ds = nc.Dataset(CMAQ_FILE, 'r')
    lon_cmaq = ds.variables['lon'][:]
    lat_cmaq = ds.variables['lat'][:]
    pred_pm25 = ds.variables['pred_PM25'][:]
    ds.close()

    # 获取CMAQ值
    def get_cmaq_at_site(lon, lat, lon_grid, lat_grid, pm25_grid):
        dist = np.sqrt((lon_grid - lon)**2 + (lat_grid - lat)**2)
        idx = np.argmin(dist)
        ny, nx = lon_grid.shape
        row, col = idx // nx, idx % nx
        return pm25_grid[row, col]

    cmaq_values = []
    for _, row in monitor_df.iterrows():
        val = get_cmaq_at_site(
            row['Lon'], row['Lat'],
            lon_cmaq, lat_cmaq,
            pred_pm25[0]
        )
        cmaq_values.append(val)
    monitor_df['CMAQ'] = cmaq_values

    # 划分训练测试集
    from sklearn.model_selection import train_test_split
    train_df, test_df = train_test_split(
        monitor_df, test_size=0.2, random_state=42
    )

    # 训练模型
    X_train = train_df[['Lon', 'Lat']].values
    y_train = train_df['Conc'].values
    CMAQ_train = train_df['CMAQ'].values

    X_test = test_df[['Lon', 'Lat']].values
    y_test = test_df['Conc'].values
    CMAQ_test = test_df['CMAQ'].values

    # 创建并训练AdvancedRK
    model = AdvancedRK(poly_degree=2, nu=1.5)
    model.fit(X_train, y_train, CMAQ_train)

    # 预测
    y_pred = model.predict(X_test, CMAQ_test)

    # 评估
    metrics = compute_metrics(y_test, y_pred)
    print("AdvancedRK 演示结果:")
    print(f"  R²   = {metrics['R2']:.4f}")
    print(f"  MAE  = {metrics['MAE']:.2f}")
    print(f"  RMSE = {metrics['RMSE']:.2f}")
    print(f"  MB   = {metrics['MB']:.2f}")

    # 带不确定性的预测
    y_pred_unc, pred_std = model.predict_with_uncertainty(X_test, CMAQ_test)
    print(f"\n不确定性统计:")
    print(f"  平均标准差 = {np.mean(pred_std):.2f}")
    print(f"  最大标准差 = {np.max(pred_std):.2f}")
    print(f"  最小标准差 = {np.min(pred_std):.2f}")
