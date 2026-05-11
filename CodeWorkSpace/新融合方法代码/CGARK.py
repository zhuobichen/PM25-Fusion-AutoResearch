"""

# === 预测值缓存（由 patch_methods.py 自动添加）===
_last_y_true = None
_last_y_pred = None

CGARK - CMAQ Gradient Anisotropic Residual Kriging
====================================================
利用CMAQ梯度引导各向异性GPR进行残差克里金插值

创新点:
1. 全局 OLS 多项式校正: O = a + b*M + c*M² + ε
2. CMAQ 梯度引导各向异性 GPR:
   - 沿梯度方向：相关长度 = 25km
   - 垂直梯度方向：相关长度 = 10km
3. 融合

参数:
- lambda_along: 25.0 (km)
- lambda_across: 10.0 (km)
- poly_degree: 2
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.paths import get_project_root, data_path
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
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
    dist = np.sqrt((lon_grid - lon)**2 + (lat_grid - lat)**2)
    idx = np.argmin(dist)
    ny, nx = lon_grid.shape
    row, col = idx // nx, idx % nx

    row = np.clip(row, 1, ny - 2)
    col = np.clip(col, 1, nx - 2)

    pm_xp = pm25_grid[row, min(col + 1, nx - 1)]
    pm_xm = pm25_grid[row, max(col - 1, 0)]
    pm_yp = pm25_grid[min(row + 1, ny - 1), col]
    pm_ym = pm25_grid[max(row - 1, 0), col]

    lon_step = np.mean(np.diff(lon_grid, axis=1))
    lat_step = np.mean(np.diff(lat_grid, axis=0))

    dpm_dlon = (pm_xp - pm_xm) / (2 * lon_step) if lon_step != 0 else 0
    dpm_dlat = (pm_yp - pm_ym) / (2 * lat_step) if lat_step != 0 else 0

    gradient_magnitude = np.sqrt(dpm_dlon**2 + dpm_dlat**2)
    gradient_direction = np.arctan2(dpm_dlon, dpm_dlat)  # 从北顺时针

    return gradient_magnitude, gradient_direction


def anisotropic_distance(x1, x2, direction, lambda_along, lambda_across):
    """
    计算各向异性距离

    Parameters:
    -----------
    x1, x2: arrays of shape (2,)
        两点坐标 [lon, lat]
    direction: float
        主方向（弧度）
    lambda_along: float
        沿梯度方向的相关长度
    lambda_across: float
        垂直梯度方向的相关长度

    Returns:
    --------
    dist: float
        各向异性距离
    """
    dx = x1[0] - x2[0]
    dy = x1[1] - x2[1]

    cos_d = np.cos(direction)
    sin_d = np.sin(direction)

    d_along = dx * cos_d + dy * sin_d
    d_perp = -dx * sin_d + dy * cos_d

    dist = np.sqrt((d_along / lambda_along)**2 + (d_perp / lambda_across)**2)

    return dist


def cgark_predict(x_obs, y_obs, x_pred, direction_pred, lambda_along=25.0, lambda_across=10.0, n_neighbor=12):
    """
    CGARK预测 - 各向异性残差插值

    Parameters:
    -----------
    x_obs: array (n_obs, 2)
        观测点坐标
    y_obs: array (n_obs,)
        观测值
    x_pred: array (n_pred, 2)
        预测点坐标
    direction_pred: array (n_pred,)
        每个预测点的梯度方向
    lambda_along: float
        沿梯度方向的相关长度
    lambda_across: float
        垂直梯度方向的相关长度
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
        dists = np.array([
            anisotropic_distance(x_pred[i], x_obs[j], direction_pred[i], lambda_along, lambda_across)
            for j in range(x_obs.shape[0])
        ])

        if n_neighbor < len(dists):
            idx = np.argpartition(dists, n_neighbor)[:n_neighbor]
        else:
            idx = np.arange(len(dists))

        dists_k = dists[idx]
        values_k = y_obs[idx]

        dists_k = np.maximum(dists_k, 1e-10)

        # 高斯相关函数权重
        weights = np.exp(-0.5 * (dists_k / (lambda_across / 3.0))**2)
        weights = weights / weights.sum()

        pred_values[i] = np.sum(weights * values_k)

    return pred_values


def run_cgark_ten_fold(selected_day='2020-01-01', lambda_along=25.0, lambda_across=10.0, poly_degree=2, n_neighbor=12):
    """
    运行CGARK十折交叉验证
    """
    print("="*60)
    print("CGARK Ten-Fold Cross Validation")
    print(f"Parameters: lambda_along={lambda_along}, lambda_across={lambda_across}, poly_degree={poly_degree}")
    print("="*60)

    # 加载数据
    print("\n=== Loading Data ===")
    monitor_df = pd.read_csv(monitor_file)
    fold_df = pd.read_csv(fold_file)

    day_df = monitor_df[monitor_df['Date'] == selected_day].copy()
    day_df = day_df.merge(fold_df, on=['Date', 'Site'], how='left')
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

    # 提取站点CMAQ值和梯度方向
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

    print("\n=== Running 10-fold Cross Validation ===")
    results = {fold_id: {} for fold_id in range(1, 11)}

    for fold_id in range(1, 11):
        train_df = day_df[day_df['fold'] != fold_id].copy()
        test_df = day_df[day_df['fold'] == fold_id].copy()

        train_df = train_df.dropna(subset=['Lon', 'Lat', 'CMAQ', 'Conc'])
        test_df = test_df.dropna(subset=['Lon', 'Lat', 'CMAQ', 'Conc'])

        if len(test_df) == 0:
            continue

        X_train = train_df[['Lon', 'Lat']].values
        X_test = test_df[['Lon', 'Lat']].values
        y_train = train_df['Conc'].values
        y_test = test_df['Conc'].values
        m_train = train_df['CMAQ'].values
        m_test = test_df['CMAQ'].values

        # 多项式特征
        poly = PolynomialFeatures(degree=poly_degree, include_bias=False)
        m_train_poly = poly.fit_transform(m_train.reshape(-1, 1))
        m_test_poly = poly.transform(m_test.reshape(-1, 1))

        # OLS 多项式校正
        ols = LinearRegression()
        ols.fit(m_train_poly, y_train)
        pred_ols = ols.predict(m_test_poly)
        residual = y_train - ols.predict(m_train_poly)

        # 测试站点梯度方向
        direction_test = test_df['gradient_direction'].values

        # CGARK 预测 (各向异性 GPR)
        cgark_residual_pred = cgark_predict(
            X_train, residual, X_test, direction_test,
            lambda_along, lambda_across, n_neighbor
        )

        # 标准 RK (各向同性 GPR)
        kernel = (ConstantKernel(10.0, (1e-2, 1e3)) *
                  RBF(length_scale=20.0, length_scale_bounds=(1e-2, 1e2)) +
                  WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e1)))
        gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, alpha=0.1, normalize_y=True)
        gpr.fit(X_train, residual)
        gpr_residual_pred, _ = gpr.predict(X_test, return_std=True)

        # 融合预测
        cgark_pred = pred_ols + cgark_residual_pred
        rk_pred = pred_ols + gpr_residual_pred

        results[fold_id] = {
            'y_true': y_test,
            'cgark': cgark_pred,
            'rk': rk_pred
        }

        print(f"  Fold {fold_id}: completed")

    # 汇总
    cgark_all = np.concatenate([results[f]['cgark'] for f in range(1, 11) if results[f]])
    rk_all = np.concatenate([results[f]['rk'] for f in range(1, 11) if results[f]])
    true_all = np.concatenate([results[f]['y_true'] for f in range(1, 11) if results[f]])

    # 计算指标
    print("\n=== Results ===")
    cgark_metrics = compute_metrics(true_all, cgark_all)
    rk_metrics = compute_metrics(true_all, rk_all)
    # 缓存预测值供多天聚合使用
    global _last_y_true, _last_y_pred
    _last_y_true = true_all
    _last_y_pred = rk_all


    print(f"  CGARK: R2={cgark_metrics['R2']:.4f}, MAE={cgark_metrics['MAE']:.2f}, RMSE={cgark_metrics['RMSE']:.2f}, MB={cgark_metrics['MB']:.2f}")
    print(f"  RK:    R2={rk_metrics['R2']:.4f}, MAE={rk_metrics['MAE']:.2f}, RMSE={rk_metrics['RMSE']:.2f}, MB={rk_metrics['MB']:.2f}")

    # 保存结果
    result_df = pd.DataFrame([{
        'Method': 'CGARK',
        'R2': cgark_metrics['R2'],
        'MAE': cgark_metrics['MAE'],
        'RMSE': cgark_metrics['RMSE'],
        'MB': cgark_metrics['MB']
    }, {
        'Method': 'RK_Isotropic',
        'R2': rk_metrics['R2'],
        'MAE': rk_metrics['MAE'],
        'RMSE': rk_metrics['RMSE'],
        'MB': rk_metrics['MB']
    }])
    result_df.to_csv(f'{output_dir}/CGARK_summary.csv', index=False)

    print(f"\nResults saved to: {output_dir}/CGARK_summary.csv")

    return cgark_metrics


if __name__ == '__main__':
    metrics = run_cgark_ten_fold('2020-01-01')
    print(f"\nFinal: R2={metrics['R2']:.4f}")
