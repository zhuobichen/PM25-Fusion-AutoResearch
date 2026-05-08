"""
WaveletGPR - 小波多尺度GPR残差融合法
=====================================

创新点: 使用离散小波变换将CMAQ残差分解为多个空间尺度，对每个尺度独立建模后重构

核心思想:
1. 全局多项式校正
2. 残差网格化
3. 小波分解
4. 各尺度GPR建模
5. 小波重构
"""

import numpy as np
from scipy.spatial.distance import cdist
from scipy.interpolate import griddata
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.linear_model import LinearRegression

# 尝试导入PyWavelets
try:
    import pywt
    HAS_PYWAVELETS = True
except ImportError:
    HAS_PYWAVELETS = False


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


class WaveletGPR:
    """
    小波多尺度GPR残差融合法

    步骤:
    1. 全局多项式校正
    2. 残差网格化 (IDW插值)
    3. 小波分解 (2D-DWT)
    4. 各尺度GPR建模
    5. 小波重构
    6. 最终融合
    """

    def __init__(self, params=None):
        self.params = params or {}
        self.wavelet = self.params.get('wavelet', 'db4')
        self.decomposition_level = self.params.get('decomposition_level', 3)
        self.grid_resolution = self.params.get('grid_resolution', 0.1)
        self.poly_degree = self.params.get('poly_degree', 2)

        # 拟合的参数
        self.poly_model = None
        self.gpr_models = {}

    def fit_polynomial(self, cmaq, obs):
        """
        拟合全局多项式

        Parameters:
        -----------
        cmaq : np.ndarray
            CMAQ值
        obs : np.ndarray
            观测值

        Returns:
        --------
        pred : np.ndarray
            多项式预测值
        """
        # 构建多项式特征
        X = np.column_stack([cmaq ** d for d in range(1, self.poly_degree + 1)])

        # OLS拟合
        self.poly_model = LinearRegression()
        self.poly_model.fit(X, obs)

        pred = self.poly_model.predict(X)
        return pred

    def grid_residuals(self, coords, residuals, grid_resolution=None):
        """
        将散点残差插值到规则网格

        Parameters:
        -----------
        coords : np.ndarray
            站点坐标 (n, 2)
        residuals : np.ndarray
            残差值 (n,)
        grid_resolution : float
            网格分辨率

        Returns:
        --------
        grid_x, grid_y : np.ndarray
            网格坐标
        grid_residuals : np.ndarray
            网格化残差
        """
        if grid_resolution is None:
            grid_resolution = self.grid_resolution

        # 创建规则网格
        x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
        y_min, y_max = coords[:, 1].min(), coords[:, 1].max()

        # 扩展边界
        x_min -= grid_resolution
        x_max += grid_resolution
        y_min -= grid_resolution
        y_max += grid_resolution

        grid_x = np.arange(x_min, x_max, grid_resolution)
        grid_y = np.arange(y_min, y_max, grid_resolution)
        grid_x, grid_y = np.meshgrid(grid_x, grid_y)

        # IDW插值
        grid_residuals = griddata(coords, residuals, (grid_x, grid_y), method='nearest')

        return grid_x, grid_y, grid_residuals

    def wavelet_decompose(self, grid_data):
        """
        小波分解

        Parameters:
        -----------
        grid_data : np.ndarray
            2D网格数据

        Returns:
        --------
        coefficients : list
            小波系数 [cA_n, cH_n, cV_n, cD_n, ..., cH_1, cV_1, cD_1]
        """
        if not HAS_PYWAVELETS:
            # 如果没有PyWavelets，返回简单分解
            return [grid_data]

        # 2D小波分解
        coefficients = pywt.wavedec2(grid_data, self.wavelet, level=self.decomposition_level)

        return coefficients

    def wavelet_reconstruct(self, coefficients):
        """
        小波重构

        Parameters:
        -----------
        coefficients : list
            小波系数

        Returns:
        --------
        reconstructed : np.ndarray
            重构数据
        """
        if not HAS_PYWAVELETS:
            return coefficients[0]

        # 2D小波重构
        reconstructed = pywt.waverec2(coefficients, self.wavelet)

        return reconstructed

    def fit_scale_gpr(self, scale_data, coords):
        """
        对单个尺度拟合GPR

        Parameters:
        -----------
        scale_data : np.ndarray
            尺度数据
        coords : np.ndarray
            坐标

        Returns:
        --------
        gpr : GaussianProcessRegressor
            拟合的GPR模型
        """
        kernel = ConstantKernel(1.0, (1e-2, 1e2)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(noise_level=0.1)

        gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, alpha=0.1, normalize_y=True)
        gpr.fit(coords, scale_data.ravel())

        return gpr

    def fit(self, station_obs, station_cmaq, station_coords):
        """
        拟合模型

        Parameters:
        -----------
        station_obs : np.ndarray
            站点观测值
        station_cmaq : np.ndarray
            站点处CMAQ值
        station_coords : np.ndarray
            站点坐标
        """
        # 步骤1: 全局多项式校正
        poly_pred = self.fit_polynomial(station_cmaq, station_obs)

        # 步骤2: 计算残差
        residual = station_obs - poly_pred

        # 步骤3: 残差网格化
        grid_x, grid_y, grid_residuals = self.grid_residuals(station_coords, residual)

        # 步骤4: 小波分解
        if HAS_PYWAVELETS:
            coefficients = self.wavelet_decompose(grid_residuals)

            # 步骤5: 对每个尺度拟合GPR
            self.gpr_models = {}
            for i, coeff in enumerate(coefficients):
                if isinstance(coeff, np.ndarray) and coeff.size > 0:
                    # 创建该尺度的坐标
                    ny, nx = coeff.shape
                    x = np.linspace(station_coords[:, 0].min(), station_coords[:, 0].max(), nx)
                    y = np.linspace(station_coords[:, 1].min(), station_coords[:, 1].max(), ny)
                    xx, yy = np.meshgrid(x, y)
                    coords = np.column_stack([xx.ravel(), yy.ravel()])

                    # 拟合GPR
                    gpr = self.fit_scale_gpr(coeff, coords)
                    self.gpr_models[i] = (gpr, coords.shape)
        else:
            # 简化: 不使用小波，直接对残差拟合GPR
            kernel = ConstantKernel(1.0, (1e-2, 1e2)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(noise_level=0.1)
            self.gpr_residual = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, alpha=0.1, normalize_y=True)
            self.gpr_residual.fit(station_coords, residual)

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
        # 步骤1: 多项式预测
        X_pred = np.column_stack([cmaq_grid_values ** d for d in range(1, self.poly_degree + 1)])
        poly_pred = self.poly_model.predict(X_pred)

        # 步骤2: 残差预测
        if HAS_PYWAVELETS and self.gpr_models:
            # 小波重构残差
            predicted_coeffs = []
            for i, (gpr, shape) in self.gpr_models.items():
                # 创建预测坐标
                ny, nx = shape
                x = np.linspace(grid_coords[:, 0].min(), grid_coords[:, 0].max(), nx)
                y = np.linspace(grid_coords[:, 1].min(), grid_coords[:, 1].max(), ny)
                xx, yy = np.meshgrid(x, y)
                pred_coords = np.column_stack([xx.ravel(), yy.ravel()])

                # 预测
                pred = gpr.predict(pred_coords)
                predicted_coeffs.append(pred.reshape(ny, nx))

            # 小波重构
            residual_grid = self.wavelet_reconstruct(predicted_coeffs)

            # 从网格提取到预测点
            from scipy.interpolate import RegularGridInterpolator
            x = np.linspace(grid_coords[:, 0].min(), grid_coords[:, 0].max(), residual_grid.shape[1])
            y = np.linspace(grid_coords[:, 1].min(), grid_coords[:, 1].max(), residual_grid.shape[0])
            interpolator = RegularGridInterpolator((y, x), residual_grid, method='linear', bounds_error=False, fill_value=0)
            residual_pred = interpolator((grid_coords[:, 1], grid_coords[:, 0]))
        else:
            # 简化: 直接GPR预测残差
            residual_pred, _ = self.gpr_residual.predict(grid_coords, return_std=True)

        # 步骤3: 最终融合
        fused = poly_pred + residual_pred

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
    model = WaveletGPR(params)

    # 拟合
    model.fit(station_obs, station_cmaq, station_coords)

    # 预测
    fused = model.predict(cmaq_grid_values, station_obs, station_cmaq, station_coords, grid_coords)

    return fused
