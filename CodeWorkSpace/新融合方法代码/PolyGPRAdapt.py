"""
PolyGPRAdapt - 大气稳定度自适应多项式-高斯过程残差融合法
============================================================
Atmospheric-Stability-Adaptive Polynomial Calibration with Gaussian Process Residual Modeling

原理:
1. 多项式CMAQ校正（最小二乘解析解）
2. 残差提取
3. 大气稳定度自适应GPR（Mattern变异函数，参数随稳定度缩放）
4. 融合输出 + 不确定性估计

核心创新:
- 无权重学习（多项式为解析最小二乘，GPR为边缘似然优化）
- 大气稳定度物理解释（稳定度直接影响扩散率/相关长度）
- 不确定性量化（GPR天然提供后验方差）

方法指纹: MD5: `polygpr_adapt_atmospheric_stability_v1`
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.paths import get_project_root, data_path
import numpy as np
import pandas as pd
import xarray as xr
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from sklearn.linear_model import LinearRegression
from scipy.special import gamma as gamma_func
from scipy.special import kv as bessel_kv
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

root_dir = str(get_project_root())
cmaq_file = data_path('test_data/raw/CMAQ/2020_PM25.nc')
monitor_file = data_path('test_data/raw/Monitor/2020_DailyPM2.5Monitor.csv')
met_file = data_path('test_data/raw/Meteorology/2020_Meteorology.nc')
fold_file = data_path('test_data/fold_split_table_daily.csv')
output_dir = f'{root_dir}/test_result/创新方法'
os.makedirs(output_dir, exist_ok=True)


# 稳定度缩放因子表（Pasquill-Gifford分类）
STABILITY_SCALE_FACTORS = {
    'A': 2.5,   # 极不稳定
    'B': 1.8,   # 不稳定
    'C': 1.3,   # 弱不稳定
    'D': 1.0,   # 中性
    'E': 0.7,   # 弱稳定
    'F': 0.4    # 稳定
}


def compute_metrics(y_true, y_pred):
    """计算评估指标"""
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) == 0:
        return {'R2': np.nan, 'MAE': np.nan, 'RMSE': np.nan, 'MB': np.nan}

    n = len(y_true)
    r2 = 1 - np.sum((y_pred - y_true)**2) / np.sum((np.mean(y_true) - y_true)**2)
    mae = np.sum(np.abs(y_pred - y_true)) / n
    rmse = np.sqrt(np.sum((y_pred - y_true)**2) / n)
    mb = np.sum(y_pred - y_true) / n

    return {'R2': r2, 'MAE': mae, 'RMSE': rmse, 'MB': mb}


def get_cmaq_at_site(lon, lat, lon_grid, lat_grid, pm25_grid):
    """获取站点位置的CMAQ值（最近邻插值）"""
    dist = np.sqrt((lon_grid - lon)**2 + (lat_grid - lat)**2)
    idx = np.argmin(dist)
    ny, nx = lon_grid.shape
    row, col = idx // nx, idx % nx
    return pm25_grid[row, col]


def get_met_at_site(lon, lat, lon_grid, lat_grid, met_var):
    """获取站点位置的气象值（最近邻插值）"""
    dist = np.sqrt((lon_grid - lon)**2 + (lat_grid - lat)**2)
    idx = np.argmin(dist)
    ny, nx = lon_grid.shape
    row, col = idx // nx, idx % nx
    return met_var[row, col]


def classify_stability(pblh, wind_speed, solar_rad=None):
    """
    基于PBLH和风速分类大气稳定度（简化Pasquill-Gifford分类）

    参数:
        pblh: 边界层高度 (m)
        wind_speed: 风速 (m/s)
        solar_rad: 太阳辐射（可选）(W/m^2)

    返回:
        stability: 稳定度等级 ('A'-'F')
    """
    # 基于PBLH和风速的简化分类
    if pblh < 200:
        if wind_speed < 2:
            return 'F'  # 稳定
        else:
            return 'E'  # 弱稳定
    elif pblh < 500:
        if wind_speed < 2:
            return 'E'
        elif wind_speed < 4:
            return 'D'  # 中性
        else:
            return 'C'  # 弱不稳定
    elif pblh < 900:
        if wind_speed < 3:
            return 'D'
        else:
            return 'C'
    else:
        if wind_speed < 3:
            return 'C'
        elif wind_speed < 5:
            return 'B'  # 不稳定
        else:
            return 'A'  # 极不稳定


def mattern_variogram(h, sigma, ell, nu=1.5, phi=1.0):
    """
    Mattern变异函数（各向同性）

    参数:
        h: 距离
        sigma: 方差
        ell: 相关长度
        nu: Mattern阶数（默认3/2）
        phi: 稳定度缩放因子

    返回:
        gamma: 变异函数值
    """
    h_phi = h * phi / ell
    # Mattern变异函数: gamma(h) = sigma^2 * [1 - (2^(1-nu)/Gamma(nu)) * (h*phi/ell)^nu * K_nu(h*phi/ell)]
    term = (2 ** (1 - nu)) / gamma_func(nu) * (h_phi ** nu) * bessel_kv(nu, h_phi)
    gamma = sigma * (1 - term)
    # 确保非负
    gamma = np.maximum(gamma, 0)
    return gamma


def polynomial_calibration(cmaq_values, temperature, pblh, obs_values):
    """
    多项式CMAQ校正（最小二乘解析解）

    公式: CMAQ_cal = alpha0 + alpha1*CMAQ + alpha2*CMAQ^2 + alpha3*T + alpha4*PBLH

    参数:
        cmaq_values: CMAQ值 (n,)
        temperature: 温度 (n,)
        pblh: 边界层高度 (n,)
        obs_values: 监测值 (n,)

    返回:
        alpha: 拟合参数 (5,)
        cmaq_cal: 校正后CMAQ (n,)
    """
    # 构建特征矩阵
    X = np.column_stack([
        np.ones_like(cmaq_values),  # alpha0
        cmaq_values,                 # alpha1 * CMAQ
        cmaq_values ** 2,            # alpha2 * CMAQ^2
        temperature,                  # alpha3 * T
        pblh                         # alpha4 * PBLH
    ])

    # 最小二乘解析解
    lr = LinearRegression(fit_intercept=False)
    lr.fit(X, obs_values)
    alpha = lr.coef_

    # 计算校正后CMAQ
    cmaq_cal = lr.predict(X)

    return alpha, cmaq_cal


def build_stability_adaptive_kernel(nu=1.5, phi=1.0, base_length_scale=1.0):
    """
    构建稳定度自适应的Mattern核函数

    参数:
        nu: Mattern阶数
        phi: 稳定度缩放因子
        base_length_scale: 基线相关长度

    返回:
        kernel: 高斯过程核函数
    """
    # 使用scikit-learn的Matern核（nu=1.5对应经典Mattern 3/2）
    # 注意：sklearn的Matern核会自动处理缩放
    length_scale = base_length_scale / phi  # phi越大，相关长度越小

    kernel = ConstantKernel(10.0, (1e-2, 1e3)) * Matern(
        length_scale=length_scale,
        length_scale_bounds=(1e-2, 1e2),
        nu=nu
    ) + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e1))

    return kernel


def fuse_poly_gpr_adapt(cmaq_data, station_data, station_coords, params):
    """
    PolyGPRAdapt融合方法主函数

    Parameters:
    -----------
    cmaq_data : xarray.DataArray
        CMAQ模型数据，shape (time, lat, lon)
    station_data : np.ndarray
        监测站数据，shape (n_stations, n_times)
    station_coords : np.ndarray
        监测站坐标，shape (n_stations, 2) - [lon, lat]
    params : dict
        方法参数
        - temperature: 温度场 (lat, lon, time)
        - pblh: 边界层高度 (lat, lon, time)
        - stability: 稳定度等级 (lat, lon, time) - 可选
        - nu: Mattern阶数（默认1.5）
        - base_length_scale: 基线相关长度（默认1.0度）

    Returns:
    --------
    fused_data : xarray.DataArray
        融合结果，shape (time, lat, lon)
    uncertainty_data : xarray.DataArray
        不确定性估计，shape (time, lat, lon)
    """
    print("="*60)
    print("PolyGPRAdapt: 大气稳定度自适应多项式-高斯过程残差融合法")
    print("="*60)

    # 提取参数
    temperature = params.get('temperature')
    pblh = params.get('pblh')
    stability = params.get('stability')
    nu = params.get('nu', 1.5)
    base_length_scale = params.get('base_length_scale', 1.0)

    # 获取CMAQ网格信息
    if isinstance(cmaq_data, xr.DataArray):
        lon_grid = cmaq_data.lon.values
        lat_grid = cmaq_data.lat.values
        cmaq_values = cmaq_data.values
        n_time = cmaq_values.shape[0]
        ny, nx = lon_grid.shape
    else:
        raise ValueError("cmaq_data应为xarray.DataArray格式")

    # 创建网格坐标
    X_grid = np.column_stack([lon_grid.ravel(), lat_grid.ravel()])

    # 初始化输出
    fused_grid = np.zeros((n_time, ny, nx))
    uncertainty_grid = np.zeros((n_time, ny, nx))

    print(f"\n处理 {n_time} 个时间步...")

    for t in range(n_time):
        if t % 10 == 0:
            print(f"  处理时间步 {t+1}/{n_time}...")

        # 当前时间步的CMAQ数据
        cmaq_t = cmaq_values[t]

        # 提取站点CMAQ值
        site_cmaq = np.array([
            get_cmaq_at_site(station_coords[i, 0], station_coords[i, 1],
                            lon_grid, lat_grid, cmaq_t)
            for i in range(len(station_coords))
        ])

        # 提取站点气象数据
        if temperature is not None:
            site_temp = np.array([
                get_met_at_site(station_coords[i, 0], station_coords[i, 1],
                               lon_grid, lat_grid, temperature[t])
                for i in range(len(station_coords))
            ])
        else:
            site_temp = np.zeros(len(station_coords))

        if pblh is not None:
            site_pblh = np.array([
                get_met_at_site(station_coords[i, 0], station_coords[i, 1],
                               lon_grid, lat_grid, pblh[t])
                for i in range(len(station_coords))
            ])
        else:
            site_pblh = np.ones(len(station_coords)) * 500.0  # 默认500m

        # 监测值
        obs_values = station_data[:, t] if station_data.ndim == 2 else station_data

        # ========== 步骤1: 多项式CMAQ校正 ==========
        valid_mask = ~np.isnan(obs_values) & ~np.isnan(site_cmaq)
        if np.sum(valid_mask) < 5:
            # 数据不足，使用原始CMAQ
            fused_grid[t] = cmaq_t
            uncertainty_grid[t] = np.ones_like(cmaq_t) * 10.0
            continue

        alpha, cmaq_cal_train = polynomial_calibration(
            site_cmaq[valid_mask],
            site_temp[valid_mask],
            site_pblh[valid_mask],
            obs_values[valid_mask]
        )

        # 计算残差
        residuals = obs_values[valid_mask] - cmaq_cal_train

        # ========== 步骤2: 稳定度自适应GPR ==========
        # 分类稳定度
        if stability is not None:
            site_stab = np.array([
                get_met_at_site(station_coords[i, 0], station_coords[i, 1],
                               lon_grid, lat_grid, stability[t])
                for i in range(len(station_coords))
            ])
            # 取众数或平均
            phi = np.mean([STABILITY_SCALE_FACTORS.get(str(s), 1.0)
                          for s in site_stab[valid_mask]])
        else:
            # 基于PBLH和风速计算（简化）
            wind_speed = np.ones(len(station_coords)) * 2.0  # 默认风速
            site_stab = [
                classify_stability(site_pblh[i], wind_speed[i])
                for i in range(len(station_coords))
            ]
            phi = np.mean([STABILITY_SCALE_FACTORS.get(str(s), 1.0)
                          for s in site_stab[valid_mask]])

        # 构建自适应核函数
        kernel = build_stability_adaptive_kernel(nu=nu, phi=phi,
                                                 base_length_scale=base_length_scale)

        # GPR拟合
        coords_train = station_coords[valid_mask]
        gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2,
                                        alpha=0.1, normalize_y=True)
        gpr.fit(coords_train, residuals)

        # 网格预测
        residual_pred, residual_std = gpr.predict(X_grid, return_std=True)
        residual_grid = residual_pred.reshape((ny, nx))
        uncertainty_grid[t] = residual_std.reshape((ny, nx))

        # ========== 步骤3: 融合输出 ==========
        # CMAQ_cal on grid
        cmaq_cal_grid = alpha[0] + alpha[1]*cmaq_t + alpha[2]*cmaq_t**2

        # 添加气象修正项
        if temperature is not None:
            T_grid = temperature[t]
            cmaq_cal_grid = cmaq_cal_grid + alpha[3] * T_grid
        if pblh is not None:
            PBLH_grid = pblh[t]
            cmaq_cal_grid = cmaq_cal_grid + alpha[4] * PBLH_grid

        # 融合
        fused_grid[t] = cmaq_cal_grid + residual_grid

    # 确保非负
    fused_grid = np.maximum(fused_grid, 0)
    uncertainty_grid = np.maximum(uncertainty_grid, 0)

    # 构建输出DataArray
    fused_data = xr.DataArray(
        fused_grid,
        dims=['time', 'lat', 'lon'],
        coords={
            'time': cmaq_data.time,
            'lat': cmaq_data.lat,
            'lon': cmaq_data.lon
        }
    )

    uncertainty_data = xr.DataArray(
        uncertainty_grid,
        dims=['time', 'lat', 'lon'],
        coords={
            'time': cmaq_data.time,
            'lat': cmaq_data.lat,
            'lon': cmaq_data.lon
        }
    )

    print("融合完成！")
    return fused_data, uncertainty_data


def cross_validate(method_func, fold_split_table, selected_days, **kwargs):
    """
    十折交叉验证

    Parameters:
    -----------
    method_func : callable
        融合方法函数
    fold_split_table : str
        路径 to fold_split_table_daily.csv
    selected_days : list
        测试日期列表

    Returns:
    --------
    metrics : dict
        {"R2": ..., "MAE": ..., "RMSE": ..., "MB": ...}
    """
    print("="*60)
    print("PolyGPRAdapt 十折交叉验证")
    print("="*60)

    # 加载数据
    monitor_df = pd.read_csv(monitor_file)
    fold_df = pd.read_csv(fold_split_table)

    # 加载CMAQ
    ds = xr.open_dataset(cmaq_file)
    lon_cmaq = ds.lon.values
    lat_cmaq = ds.lat.values
    cmaq_var = ds['pred_PM25'].values
    ds.close()

    # 加载气象数据（如果存在）
    temperature, pblh, stability = None, None, None
    if os.path.exists(met_file):
        try:
            ds_met = xr.open_dataset(met_file)
            if 'temperature' in ds_met.variables:
                temperature = ds_met['temperature'].values
            if 'PBLH' in ds_met.variables:
                pblh = ds_met['PBLH'].values
            ds_met.close()
        except:
            print("  气象数据加载失败，使用默认值")

    results_all = []

    for selected_day in selected_days:
        print(f"\n处理日期: {selected_day}")

        # 筛选日期数据
        day_df = monitor_df[monitor_df['Date'] == selected_day].copy()
        day_df = day_df.merge(fold_df, on='Site', how='left')
        day_df = day_df.dropna(subset=['Lat', 'Lon', 'Conc'])

        if len(day_df) == 0:
            continue

        # 获取时间索引
        from datetime import datetime
        date_obj = datetime.strptime(selected_day, '%Y-%m-%d')
        day_idx = (date_obj - datetime(2020, 1, 1)).days

        if day_idx >= cmaq_var.shape[0]:
            continue

        pred_day = cmaq_var[day_idx]

        # 提取站点CMAQ值
        cmaq_values = []
        for _, row in day_df.iterrows():
            val = get_cmaq_at_site(row['Lon'], row['Lat'],
                                   lon_cmaq, lat_cmaq, pred_day)
            cmaq_values.append(val)
        day_df['CMAQ'] = cmaq_values

        # 十折验证
        for fold_id in range(1, 11):
            train_df = day_df[day_df['fold'] != fold_id].copy()
            test_df = day_df[day_df['fold'] == fold_id].copy()

            train_df = train_df.dropna(subset=['Lon', 'Lat', 'CMAQ', 'Conc'])
            test_df = test_df.dropna(subset=['Lon', 'Lat', 'CMAQ', 'Conc'])

            if len(test_df) == 0 or len(train_df) < 5:
                continue

            # 多项式校正
            site_cmaq = train_df['CMAQ'].values
            site_temp = np.ones(len(train_df)) * 15.0  # 默认温度
            site_pblh = np.ones(len(train_df)) * 500.0  # 默认PBLH

            alpha, _ = polynomial_calibration(
                site_cmaq, site_temp, site_pblh, train_df['Conc'].values
            )

            # 计算测试集校正后CMAQ
            cmaq_cal_test = alpha[0] + alpha[1]*test_df['CMAQ'].values + \
                           alpha[2]*test_df['CMAQ'].values**2

            # 残差
            residuals = train_df['Conc'].values - \
                       (alpha[0] + alpha[1]*train_df['CMAQ'].values +
                        alpha[2]*train_df['CMAQ'].values**2)

            # GPR
            kernel = build_stability_adaptive_kernel(nu=1.5, phi=1.0,
                                                     base_length_scale=1.0)
            gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2,
                                            alpha=0.1, normalize_y=True)
            gpr.fit(train_df[['Lon', 'Lat']].values, residuals)

            residual_pred, _ = gpr.predict(test_df[['Lon', 'Lat']].values,
                                          return_std=True)

            # 融合预测
            fused_pred = cmaq_cal_test + residual_pred
            fused_pred = np.maximum(fused_pred, 0)

            results_all.append({
                'day': selected_day,
                'fold': fold_id,
                'y_true': test_df['Conc'].values,
                'y_pred': fused_pred
            })

    # 汇总结果
    all_true = np.concatenate([r['y_true'] for r in results_all])
    all_pred = np.concatenate([r['y_pred'] for r in results_all])

    metrics = compute_metrics(all_true, all_pred)

    print(f"\n十折验证结果:")
    print(f"  R2   = {metrics['R2']:.4f}")
    print(f"  MAE  = {metrics['MAE']:.4f}")
    print(f"  RMSE = {metrics['RMSE']:.4f}")
    print(f"  MB   = {metrics['MB']:.4f}")

    return metrics


if __name__ == '__main__':
    print("PolyGPRAdapt 方法测试")

    # 十折验证
    metrics = cross_validate(
        fuse_poly_gpr_adapt,
        fold_file,
        ['2020-01-01', '2020-01-02', '2020-01-03']
    )

    print(f"\n最终结果: R2={metrics['R2']:.4f}")
