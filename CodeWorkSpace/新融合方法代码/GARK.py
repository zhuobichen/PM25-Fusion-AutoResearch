"""
GARK - Gradient Anisotropic Residual Kriging
=============================================
利用CMAQ格点计算局地浓度梯度方向，在梯度主轴方向进行各向异性克里金插值

创新点:
1. 利用CMAQ格点数据计算局地浓度梯度方向
2. 在梯度主轴方向进行各向异性克里金插值
3. 垂直梯度方向和沿梯度方向使用不同的相关长度

核心参数:
- a_min: 垂直梯度方向相关长度 (km), default: 8.0
- a_max: 沿梯度方向相关长度 (km), default: 20.0
- alpha: 各向异性指数, default: 2.0
- n_neighbor: 邻域站点数, default: 12
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.paths import get_project_root, data_path
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
import netCDF4 as nc

root_dir = str(get_project_root())
cmaq_file = data_path('test_data/raw/CMAQ/2020_PM25.nc')
monitor_file = data_path('test_data/raw/Monitor/2020_DailyPM2.5Monitor.csv')
fold_file = data_path('test_data/fold_split_table_daily.csv')
output_dir = f'{root_dir}/test_result/创新方法'
os.makedirs(output_dir, exist_ok=True)


def compute_metrics(y_true, y_pred):
    """计算评估指标"""
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) == 0:
        return {'R2': np.nan, 'MAE': np.nan, 'RMSE': np.nan, 'MB': np.nan}

    return {
        'R2': r2_score(y_true, y_pred),
        'MAE': mean_absolute_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MB': np.mean(y_pred - y_true)
    }


def get_cmaq_at_site(lon, lat, lon_grid, lat_grid, pm25_grid):
    """获取站点位置的CMAQ值（最近邻）"""
    dist = np.sqrt((lon_grid - lon)**2 + (lat_grid - lat)**2)
    idx = np.argmin(dist)
    ny, nx = lon_grid.shape
    row, col = idx // nx, idx % nx
    return pm25_grid[row, col]


def compute_gradient_direction(lon, lat, lon_grid, lat_grid, pm25_grid):
    """
    计算给定位置的局地浓度梯度方向

    Parameters:
    -----------
    lon, lat: float
        给定位置的经纬度
    lon_grid, lat_grid: 2D arrays
        CMAQ网格经纬度
    pm25_grid: 2D array
        CMAQ PM2.5浓度网格

    Returns:
    --------
    gradient_magnitude: float
        梯度幅值
    gradient_direction: float
        梯度方向（弧度，从北顺时针）
    """
    # 找到最近邻的4个格点来计算梯度
    dist = np.sqrt((lon_grid - lon)**2 + (lat_grid - lat)**2)
    idx = np.argmin(dist)
    ny, nx = lon_grid.shape
    row, col = idx // nx, idx % nx

    # 确保不超出边界
    row = np.clip(row, 1, ny - 2)
    col = np.clip(col, 1, nx - 2)

    # 使用相邻格点计算梯度
    pm_xp = pm25_grid[row, min(col + 1, nx - 1)]
    pm_xm = pm25_grid[row, max(col - 1, 0)]
    pm_yp = pm25_grid[min(row + 1, ny - 1), col]
    pm_ym = pm25_grid[max(row - 1, 0), col]

    lon_step = np.mean(np.diff(lon_grid, axis=1))
    lat_step = np.mean(np.diff(lat_grid, axis=0))

    # 计算梯度分量
    dpm_dlon = (pm_xp - pm_xm) / (2 * lon_step) if lon_step != 0 else 0
    dpm_dlat = (pm_yp - pm_ym) / (2 * lat_step) if lat_step != 0 else 0

    # 梯度幅值和方向
    gradient_magnitude = np.sqrt(dpm_dlon**2 + dpm_dlat**2)
    gradient_direction = np.arctan2(dpm_dlon, dpm_dlat)  # 从北顺时针

    return gradient_magnitude, gradient_direction


def anisotropic_distance(x1, x2, direction, a_min, a_max, alpha=2.0):
    """
    计算各向异性距离

    Parameters:
    -----------
    x1, x2: arrays of shape (2,)
        两点坐标 [lon, lat]
    direction: float
        主方向（弧度）
    a_min: float
        垂直方向的相关长度
    a_max: float
        沿主方向的相关长度
    alpha: float
        各向异性指数

    Returns:
    --------
    dist: float
        各向异性距离
    """
    # 坐标差
    dx = x1[0] - x2[0]
    dy = x1[1] - x2[1]

    # 旋转到主轴坐标系
    cos_d = np.cos(direction)
    sin_d = np.sin(direction)

    # 在主轴方向和垂直方向的分量
    d_along = dx * cos_d + dy * sin_d
    d_perp = -dx * sin_d + dy * cos_d

    # 各向异性距离（椭圆形）
    dist = np.sqrt((d_along / a_max)**2 + (d_perp / a_min)**2)

    return dist


def gark_predict(x_obs, y_obs, values, x_pred, direction_grid, a_min=8.0, a_max=20.0, alpha=2.0, n_neighbor=12):
    """
    GARK预测 - 各向异性残差插值

    Parameters:
    -----------
    x_obs: array (n_obs, 2)
        观测点坐标
    y_obs: array (n_obs,)
        观测值
    x_pred: array (n_pred, 2)
        预测点坐标
    direction_grid: array (n_pred,)
        每个预测点的梯度方向
    a_min, a_max: float
        垂直/沿梯度方向的相关长度
    alpha: float
        各向异性指数
    n_neighbor: int
        使用的最近邻数量

    Returns:
    --------
    pred_values: array (n_pred,)
        预测值
    """
    n_pred = x_pred.shape[0]
    pred_values = np.zeros(n_pred)

    for i in range(n_pred):
        # 计算到所有观测点的各向异性距离
        dists = np.array([
            anisotropic_distance(x_pred[i], x_obs[j], direction_grid[i], a_min, a_max, alpha)
            for j in range(x_obs.shape[0])
        ])

        # 选择最近的n_neighbor个点
        if n_neighbor < len(dists):
            idx = np.argpartition(dists, n_neighbor)[:n_neighbor]
        else:
            idx = np.arange(len(dists))

        dists_k = dists[idx]
        values_k = y_obs[idx]

        # 避免除零
        dists_k = np.maximum(dists_k, 1e-10)

        # 高斯相关函数权重
        weights = np.exp(-0.5 * (dists_k / (a_min / 3.0))**2)
        weights = weights / weights.sum()

        pred_values[i] = np.sum(weights * values_k)

    return pred_values


def run_gark_ten_fold(selected_day='2020-01-01', a_min=8.0, a_max=20.0, alpha=2.0, n_neighbor=12):
    """
    运行GARK十折交叉验证
    """
    print("="*60)
    print("GARK Ten-Fold Cross Validation")
    print(f"Parameters: a_min={a_min}, a_max={a_max}, alpha={alpha}, n_neighbor={n_neighbor}")
    print("="*60)

    # 加载数据
    print("\n=== Loading Data ===")
    monitor_df = pd.read_csv(monitor_file)
    fold_df = pd.read_csv(fold_file)

    day_df = monitor_df[monitor_df['Date'] == selected_day].copy()
    day_df = day_df.merge(fold_df, on='Site', how='left')
    day_df = day_df.dropna(subset=['Lat', 'Lon', 'Conc'])

    # 加载CMAQ数据
    ds = nc.Dataset(cmaq_file, 'r')
    lon_cmaq = ds.variables['lon'][:]
    lat_cmaq = ds.variables['lat'][:]
    pred_pm25 = ds.variables['pred_PM25'][:]
    ds.close()

    from datetime import datetime
    date_obj = datetime.strptime(selected_day, '%Y-%m-%d')
    day_idx = (date_obj - datetime(2020, 1, 1)).days
    pred_day = pred_pm25[day_idx]

    # 提取站点CMAQ值
    print("=== Extracting CMAQ at Sites ===")
    cmaq_values = []
    gradient_dirs = []
    for _, row in day_df.iterrows():
        val = get_cmaq_at_site(row['Lon'], row['Lat'], lon_cmaq, lat_cmaq, pred_day)
        cmaq_values.append(val)
        _, direction = compute_gradient_direction(row['Lon'], row['Lat'], lon_cmaq, lat_cmaq, pred_day)
        gradient_dirs.append(direction)
    day_df['CMAQ'] = cmaq_values
    day_df['gradient_direction'] = gradient_dirs

    print(f"Data loaded: {len(day_df)} monitoring records")

    # 运行十折验证
    print("\n=== Running 10-fold Cross Validation ===")
    results = {fold_id: {} for fold_id in range(1, 11)}

    for fold_id in range(1, 11):
        train_df = day_df[day_df['fold'] != fold_id].copy()
        test_df = day_df[day_df['fold'] == fold_id].copy()

        train_df = train_df.dropna(subset=['Lon', 'Lat', 'CMAQ', 'Conc'])
        test_df = test_df.dropna(subset=['Lon', 'Lat', 'CMAQ', 'Conc'])

        if len(test_df) == 0:
            continue

        # 计算残差
        train_df['residual'] = train_df['Conc'] - train_df['CMAQ']

        X_train = train_df[['Lon', 'Lat']].values
        X_test = test_df[['Lon', 'Lat']].values
        residual_train = train_df['residual'].values
        direction_test = test_df['gradient_direction'].values

        # GARK预测
        gark_residual_pred = gark_predict(
            X_train, residual_train, residual_train, X_test,
            direction_test, a_min, a_max, alpha, n_neighbor
        )

        # 融合预测
        cmaq_test = test_df['CMAQ'].values
        gark_pred = cmaq_test + gark_residual_pred

        # 对比：标准RK（各向同性GPR）
        kernel = ConstantKernel(10.0, (1e-2, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e1))
        gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, alpha=0.1, normalize_y=True)
        gpr.fit(X_train, residual_train)
        gpr_residual_pred, _ = gpr.predict(X_test, return_std=True)
        rk_pred = cmaq_test + gpr_residual_pred

        results[fold_id] = {
            'y_true': test_df['Conc'].values,
            'gark': gark_pred,
            'rk': rk_pred
        }

        print(f"  Fold {fold_id}: completed")

    # 汇总
    gark_all = np.concatenate([results[f]['gark'] for f in range(1, 11) if results[f]])
    rk_all = np.concatenate([results[f]['rk'] for f in range(1, 11) if results[f]])
    true_all = np.concatenate([results[f]['y_true'] for f in range(1, 11) if results[f]])

    # 计算R2
    print("\n=== Results ===")
    gark_metrics = compute_metrics(true_all, gark_all)
    rk_metrics = compute_metrics(true_all, rk_all)

    print(f"  GARK: R2={gark_metrics['R2']:.4f}, MAE={gark_metrics['MAE']:.2f}, RMSE={gark_metrics['RMSE']:.2f}")
    print(f"  RK:   R2={rk_metrics['R2']:.4f}, MAE={rk_metrics['MAE']:.2f}, RMSE={rk_metrics['RMSE']:.2f}")

    # 保存结果
    result_df = pd.DataFrame([{
        'Method': 'GARK',
        'R2': gark_metrics['R2'],
        'MAE': gark_metrics['MAE'],
        'RMSE': gark_metrics['RMSE'],
        'MB': gark_metrics['MB']
    }])
    result_df.to_csv(f'{output_dir}/GARK_summary.csv', index=False)

    print(f"\nResults saved to: {output_dir}/GARK_summary.csv")

    return gark_metrics


if __name__ == '__main__':
    metrics = run_gark_ten_fold('2020-01-01', a_min=8.0, a_max=20.0, alpha=2.0, n_neighbor=12)
    print(f"\nFinal: R2={metrics['R2']:.4f}")
