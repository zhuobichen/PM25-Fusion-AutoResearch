"""

# === 预测值缓存（由 patch_methods.py 自动添加）===
_last_y_true = None
_last_y_pred = None

CMAQ梯度各向异性残差克里金 (CMAQ-Gradient-Anisotropic Residual Kriging, CGARK)
================================================================================
在PolyRK架构基础上，添加CMAQ梯度方向引导的各向异性GPR。
计算CMAQ格点场的局地浓度梯度，污染扩散沿梯度方向具有更长相关长度。
使用各向异性RBF核替代各向同性RBF核，使空间相关性建模更符合PM2.5传输物理。

核心创新:
1. 全局OLS多项式校正: O = a + b*M + c*M^2 + ε
2. CMAQ梯度引导各向异性GPR:
   - 沿梯度方向：相关长度 = 25km
   - 垂直梯度方向：相关长度 = 10km
3. 各向异性距离度量

核心公式:
1. 梯度方向: θ_g = arctan2(dCMAQ/dy, dCMAQ/dx)
2. 各向异性距离: d^2 = (h_parallel/l_parallel)^2 + (h_perp/l_perp)^2
3. 融合: P = M_cal + R* (各向异性GPR残差)

参数:
- lambda_along: 沿梯度方向相关长度 25.0 km
- lambda_across: 垂直梯度方向相关长度 10.0 km
- poly_degree: 多项式阶数 2
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
import netCDF4 as nc

root_dir = str(get_project_root())
cmaq_file = data_path('test_data/raw/CMAQ/2020_PM25.nc')
monitor_file = data_path('test_data/raw/Monitor/2020_DailyPM2.5Monitor.csv')
fold_file = data_path('test_data/fold_split_table_daily.csv')
output_dir = f'{root_dir}/test_result/创新方法'
os.makedirs(output_dir, exist_ok=True)


def compute_metrics(y_true, y_pred):
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true, y_pred = y_true[mask], y_pred[mask]
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

    使用CMAQ格点中心差分计算梯度:
    dCMAQ/dx = (CMAQ[i,j+1] - CMAQ[i,j-1]) / (2*dx)
    dCMAQ/dy = (CMAQ[i+1,j] - CMAQ[i-1,j]) / (2*dy)
    θ_g = arctan2(dCMAQ/dy, dCMAQ/dx)
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
    gradient_direction = np.arctan2(dpm_dlon, dpm_dlat)

    return gradient_magnitude, gradient_direction


def anisotropic_distance(x1, x2, direction, lambda_along, lambda_across):
    """
    计算各向异性距离

    d^2 = (d_parallel / lambda_along)^2 + (d_perp / lambda_across)^2
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

    使用高斯相关函数权重:
    w_i = exp(-0.5 * (d_i / (lambda_across/3))^2)
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


def run_CMAQ梯度各向异性克里金_ten_fold(selected_day='2020-01-01'):
    """
    运行CMAQ梯度各向异性残差克里金(CGARK)十折交叉验证

    创新点:
    - PolyRK架构 + CMAQ梯度引导各向异性GPR
    - 沿梯度方向相关长度更长（平流主导）
    - 垂直梯度方向相关长度更短（湍流混合主导）
    - 各向异性比 2.5:1

    Parameters:
    -----------
    selected_day: str
        选择验证的日期 (默认: '2020-01-01')

    Returns:
    --------
    metrics: dict
        包含R2, MAE, RMSE, MB的评估指标
    """
    # CGARK参数
    lambda_along = 25.0   # 沿梯度方向相关长度 (km)
    lambda_across = 10.0  # 垂直梯度方向相关长度 (km)
    poly_degree = 2
    n_neighbor = 12

    print("=" * 60)
    print("CMAQ梯度各向异性残差克里金 (CGARK) Ten-Fold Cross Validation")
    print(f"Parameters: lambda_along={lambda_along}, lambda_across={lambda_across}, poly_degree={poly_degree}")
    print("=" * 60)

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

        # Step 1: 二次多项式OLS校正
        poly = PolynomialFeatures(degree=poly_degree, include_bias=False)
        m_train_poly = poly.fit_transform(m_train.reshape(-1, 1))
        m_test_poly = poly.transform(m_test.reshape(-1, 1))

        ols = LinearRegression()
        ols.fit(m_train_poly, y_train)
        pred_ols = ols.predict(m_test_poly)
        residual = y_train - ols.predict(m_train_poly)

        # Step 2: CMAQ梯度引导的各向异性残差插值
        direction_test = test_df['gradient_direction'].values

        cgark_residual_pred = cgark_predict(
            X_train, residual, X_test, direction_test,
            lambda_along, lambda_across, n_neighbor
        )

        # Step 3: 融合预测
        cgark_pred = pred_ols + cgark_residual_pred

        results[fold_id] = {
            'y_true': y_test,
            'cgark': cgark_pred
        }

        print(f"  Fold {fold_id}: completed")

    # 汇总
    cgark_all = np.concatenate([results[f]['cgark'] for f in range(1, 11) if results[f]])
    true_all = np.concatenate([results[f]['y_true'] for f in range(1, 11) if results[f]])

    # 计算指标
    metrics = compute_metrics(true_all, cgark_all)
    # 缓存预测值供多天聚合使用
    global _last_y_true, _last_y_pred
    _last_y_true = true_all
    _last_y_pred = cgark_all


    print(f"\n=== CMAQ梯度各向异性残差克里金 Results ===")
    print(f"  R2={metrics['R2']:.4f}, MAE={metrics['MAE']:.2f}, RMSE={metrics['RMSE']:.2f}, MB={metrics['MB']:.2f}")

    # 保存结果
    result_df = pd.DataFrame([{
        'Method': 'CMAQ梯度各向异性克里金',
        'R2': metrics['R2'],
        'MAE': metrics['MAE'],
        'RMSE': metrics['RMSE'],
        'MB': metrics['MB']
    }])
    result_df.to_csv(f'{output_dir}/CMAQ梯度各向异性克里金_summary.csv', index=False)

    print(f"\nResults saved to: {output_dir}/CMAQ梯度各向异性克里金_summary.csv")
    return metrics


if __name__ == '__main__':
    metrics = run_CMAQ梯度各向异性克里金_ten_fold('2020-01-01')
    print(f"\nFinal: R2={metrics['R2']:.4f}")
