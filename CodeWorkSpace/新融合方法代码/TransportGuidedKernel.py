"""

# === 预测值缓存（由 patch_methods.py 自动添加）===
_last_y_true = None
_last_y_pred = None

TransportGuidedKernel - 传输引导核融合法
==========================================
TGK: Transport-Guided Kernel Fusion

创新点:
1. 利用CMAQ梯度场构建各向异性空间核
2. 梯度方向决定核形状，沿传输方向相关性增强
3. 物理引导的空间插值，非纯统计方法
4. 二次多项式偏差校正 + 各向异性核残差克里金

核心公式:
- CMAQ梯度: grad_C = (dC/dx, dC/dy)
- 各向异性比: lambda = 1 + alpha * |grad_C| / sigma_grad
- 各向异性距离: d_A = sqrt((si-sj)^T A (si-sj))
- 传输引导核: K(si,sj) = exp(-d_A^2 / (2*ell^2))
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
from scipy.linalg import solve
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


def compute_gradient_field(pm25_grid, lon_grid, lat_grid):
    """
    计算CMAQ网格上的梯度场

    Returns:
        grad_x: x方向梯度 (dC/dlon)
        grad_y: y方向梯度 (dC/dlat)
        grad_magnitude: 梯度强度
        grad_direction: 梯度方向 (弧度)
    """
    # 计算网格间距 (km)
    lat_mean = np.mean(lat_grid)
    km_per_deg_lon = 111.0 * np.cos(np.radians(lat_mean))
    km_per_deg_lat = 111.0

    # 使用numpy gradient计算梯度
    # np.gradient返回 [dC/drow, dC/dcol]，对应 [dC/dlat, dC/dlon]
    grad_lat, grad_lon = np.gradient(pm25_grid)

    # 转换为km单位
    dy_km = np.abs(lat_grid[1, 0] - lat_grid[0, 0]) * km_per_deg_lat
    dx_km = np.abs(lon_grid[0, 1] - lon_grid[0, 0]) * km_per_deg_lon

    grad_y = grad_lat / (dy_km + 1e-10)  # dC/dy (km)
    grad_x = grad_lon / (dx_km + 1e-10)  # dC/dx (km)

    grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    grad_direction = np.arctan2(grad_y, grad_x)

    return grad_x, grad_y, grad_magnitude, grad_direction


def get_gradient_at_site(lon, lat, lon_grid, lat_grid, grad_x, grad_y, grad_mag, grad_dir):
    """获取站点位置的梯度值"""
    dist = np.sqrt((lon_grid - lon)**2 + (lat_grid - lat)**2)
    idx = np.argmin(dist)
    ny, nx = lon_grid.shape
    row, col = idx // nx, idx % nx
    return grad_x[row, col], grad_y[row, col], grad_mag[row, col], grad_dir[row, col]


def build_anisotropic_kernel(X_train, grad_mag_train, grad_dir_train, alpha=2.0, ell=15.0, sigma_n=1.0):
    """
    构建各向异性核矩阵

    对任意两点 si, sj:
    1. 计算局部梯度方向 theta = mean(grad_dir(si), grad_dir(sj))
    2. 计算各向异性比 lambda = 1 + alpha * min(|grad(si)|, |grad(sj)|) / sigma_grad
    3. 构建旋转矩阵 R
    4. 各向异性矩阵 A = R^T diag(1, lambda^2) R
    5. 各向异性距离 d_A = sqrt((si-sj)^T A (si-sj))
    6. 核值 K = exp(-d_A^2 / (2*ell^2))
    """
    n = len(X_train)
    K = np.zeros((n, n))

    # 梯度强度的标准差(用于归一化)
    grad_std = np.std(grad_mag_train) + 1e-10

    for i in range(n):
        for j in range(i, n):
            # 局部梯度方向
            theta = (grad_dir_train[i] + grad_dir_train[j]) / 2.0

            # 各向异性比
            grad_strength = min(grad_mag_train[i], grad_mag_train[j])
            lam = 1.0 + alpha * grad_strength / grad_std

            # 旋转矩阵
            cos_t = np.cos(theta)
            sin_t = np.sin(theta)
            R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])

            # 各向异性矩阵 A = R^T diag(1, lambda^2) R
            D = np.diag([1.0, lam**2])
            A = R.T @ D @ R

            # 各向异性距离
            diff = X_train[i] - X_train[j]
            d_A_sq = diff @ A @ diff

            # 核值
            K[i, j] = np.exp(-d_A_sq / (2.0 * ell**2))
            K[j, i] = K[i, j]

    # 添加噪声项
    K_reg = K + sigma_n**2 * np.eye(n)
    return K, K_reg


def predict_anisotropic(X_test, X_train, K_reg, residuals, grad_mag_train, grad_dir_train,
                         grad_mag_test, grad_dir_test, alpha=2.0, ell=15.0):
    """
    各向异性核预测

    k_star = kernel(test_point, train_points)
    pred = k_star @ K_reg^{-1} @ residuals
    """
    n_test = len(X_test)
    n_train = len(X_train)
    grad_std = np.std(grad_mag_train) + 1e-10

    # 求解权重
    weights = solve(K_reg, residuals, assume_a='pos')

    pred = np.zeros(n_test)
    for i in range(n_test):
        k_star = np.zeros(n_train)
        for j in range(n_train):
            theta = (grad_dir_test[i] + grad_dir_train[j]) / 2.0
            grad_strength = min(grad_mag_test[i], grad_mag_train[j])
            lam = 1.0 + alpha * grad_strength / grad_std

            cos_t = np.cos(theta)
            sin_t = np.sin(theta)
            R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
            D = np.diag([1.0, lam**2])
            A = R.T @ D @ R

            diff = X_test[i] - X_train[j]
            d_A_sq = diff @ A @ diff
            k_star[j] = np.exp(-d_A_sq / (2.0 * ell**2))

        pred[i] = k_star @ weights

    return pred


def run_TransportGuidedKernel_ten_fold(selected_day='2020-01-01', alpha=2.0, ell=15.0, sigma_n=1.0):
    print("=" * 60)
    print("TransportGuidedKernel Ten-Fold Cross Validation")
    print(f"Parameters: alpha={alpha}, ell={ell}, sigma_n={sigma_n}")
    print("=" * 60)

    print("\n=== Loading Data ===")
    monitor_df = pd.read_csv(monitor_file)
    fold_df = pd.read_csv(fold_file)

    day_df = monitor_df[monitor_df['Date'] == selected_day].copy()
    day_df = day_df.merge(fold_df, on=['Date', 'Site'], how='left')
    day_df = day_df.dropna(subset=['Lat', 'Lon', 'Conc'])

    ds = nc.Dataset(cmaq_file, 'r')
    lon_cmaq = ds.variables['lon'][:]
    lat_cmaq = ds.variables['lat'][:]
    pred_pm25 = ds.variables['pred_PM25'][:]
    ds.close()

    from datetime import datetime
    date_obj = datetime.strptime(selected_day, '%Y-%m-%d')
    day_idx = (date_obj - datetime(2020, 1, 1)).days
    pred_day = pred_pm25[day_idx]

    print("=== Extracting CMAQ at Sites ===")
    cmaq_values = []
    for _, row in day_df.iterrows():
        val = get_cmaq_at_site(row['Lon'], row['Lat'], lon_cmaq, lat_cmaq, pred_day)
        cmaq_values.append(val)
    day_df['CMAQ'] = cmaq_values

    # 计算CMAQ梯度场
    print("=== Computing CMAQ Gradient Field ===")
    grad_x, grad_y, grad_mag, grad_dir = compute_gradient_field(pred_day, lon_cmaq, lat_cmaq)

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

        # Step 1: 二次多项式偏差校正
        poly = PolynomialFeatures(degree=2, include_bias=False)
        m_train_poly = poly.fit_transform(m_train.reshape(-1, 1))
        m_test_poly = poly.transform(m_test.reshape(-1, 1))

        ols = LinearRegression()
        ols.fit(m_train_poly, y_train)
        pred_ols_train = ols.predict(m_train_poly)
        pred_ols_test = ols.predict(m_test_poly)
        residuals = y_train - pred_ols_train

        # Step 2: 获取站点梯度信息
        grad_mag_train = np.zeros(len(X_train))
        grad_dir_train = np.zeros(len(X_train))
        for i, (lon, lat) in enumerate(X_train):
            _, _, gm, gd = get_gradient_at_site(lon, lat, lon_cmaq, lat_cmaq, grad_x, grad_y, grad_mag, grad_dir)
            grad_mag_train[i] = gm
            grad_dir_train[i] = gd

        grad_mag_test = np.zeros(len(X_test))
        grad_dir_test = np.zeros(len(X_test))
        for i, (lon, lat) in enumerate(X_test):
            _, _, gm, gd = get_gradient_at_site(lon, lat, lon_cmaq, lat_cmaq, grad_x, grad_y, grad_mag, grad_dir)
            grad_mag_test[i] = gm
            grad_dir_test[i] = gd

        # Step 3: 构建各向异性核矩阵
        K, K_reg = build_anisotropic_kernel(X_train, grad_mag_train, grad_dir_train, alpha, ell, sigma_n)

        # Step 4: 各向异性核预测残差
        residual_pred = predict_anisotropic(
            X_test, X_train, K_reg, residuals,
            grad_mag_train, grad_dir_train,
            grad_mag_test, grad_dir_test,
            alpha, ell
        )

        # Step 5: 融合
        tgk_pred = pred_ols_test + residual_pred

        results[fold_id] = {
            'y_true': y_test,
            'tgk': tgk_pred
        }
        print(f"  Fold {fold_id}: completed")

    # 汇总
    tgk_all = np.concatenate([results[f]['tgk'] for f in range(1, 11) if results[f]])
    true_all = np.concatenate([results[f]['y_true'] for f in range(1, 11) if results[f]])

    print("\n=== Results ===")
    tgk_metrics = compute_metrics(true_all, tgk_all)
    # 缓存预测值供多天聚合使用
    global _last_y_true, _last_y_pred
    _last_y_true = true_all
    _last_y_pred = tgk_all

    print(f"  TGK: R2={tgk_metrics['R2']:.4f}, MAE={tgk_metrics['MAE']:.2f}, RMSE={tgk_metrics['RMSE']:.2f}")

    result_df = pd.DataFrame([{'method': 'TransportGuidedKernel', **tgk_metrics}])
    result_df.to_csv(f'{output_dir}/TransportGuidedKernel_summary.csv', index=False)
    print(f"\nResults saved to: {output_dir}/TransportGuidedKernel_summary.csv")

    return tgk_metrics


if __name__ == '__main__':
    metrics = run_TransportGuidedKernel_ten_fold('2020-01-01')
    print(f"\nFinal: R2={metrics['R2']:.4f}")
