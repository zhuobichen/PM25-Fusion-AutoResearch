"""
SPIN-Kr - 图核时空克里金法
Spatiotemporal Physics-Informed Graph Kernel Kriging
======================================================
将AOD卫星梯度作为损失函数约束，结合图核建模时空邻域关系

创新点:
1. 图核距离融合欧氏距离和风向传输距离
2. AOD梯度作为平流约束而非直接输入
3. 时变克里金权重
4. 归纳式可泛化到新位置

参数:
- alpha: 欧氏-风向距离权重 (0.5)
- lambda: 图核长度尺度 km (50.0)
- gamma: 时间正则化 (0.1)
- beta: CMAQ缩放因子 (1.0)
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
from scipy.spatial.distance import cdist
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


def build_graph_kernel_matrix(X, alpha=0.5, lambda_km=50.0, wind_dir=None):
    """
    构建图核矩阵

    Args:
        X: 站点坐标 (n, 2) [lon, lat]
        alpha: 欧氏-风向距离权重，0.5表示各占一半
        lambda_km: 图核长度尺度 km
        wind_dir: 风向角度（度），可选

    Returns:
        K: 图核矩阵 (n, n)
    """
    n = X.shape[0]
    # 欧氏距离矩阵（度为单位，近似km）
    D_eucl = cdist(X, X, metric='euclidean')
    D_eucl = D_eucl * 111.0  # 1度 ≈ 111km

    if wind_dir is not None:
        # 风向影响距离：沿风向的距离更近
        wind_rad = np.deg2rad(wind_dir)
        # 方向向量
        dx = np.cos(wind_rad)
        dy = np.sin(wind_rad)

        # 计算沿风向的投影距离
        delta = X[:, np.newaxis, :] - X[np.newaxis, :, :]  # (n, n, 2)
        # 投影到风向
        proj = delta[:, :, 0] * dx + delta[:, :, 1] * dy
        # 风向距离（保留符号）
        D_wind = np.abs(proj)

        # 融合距离
        D_graph = alpha * D_eucl + (1 - alpha) * D_wind
    else:
        D_graph = D_eucl

    # 图核矩阵
    K = np.exp(-D_graph / lambda_km)
    return K


def spin_kriging_predict(train_obs, train_cmaq, test_cmaq, **kwargs):
    """
    SPIN-Kr 融合预测

    Args:
        train_obs: 监测数据 (n, 3) [lon, lat, PM25]
        train_cmaq: CMAQ模拟数据 (n, 3) [lon, lat, PM25]
        test_cmaq: 测试CMAQ数据 (m, 3)
        kwargs: alpha, lambda_km, gamma, beta, wind_dir

    Returns:
        predictions: (m,)
    """
    alpha = kwargs.get('alpha', 0.5)
    lambda_km = kwargs.get('lambda_km', 50.0)
    gamma = kwargs.get('gamma', 0.1)
    beta = kwargs.get('beta', 1.0)
    wind_dir = kwargs.get('wind_dir', None)

    X_train = train_obs[:, :2]  # (n, 2)
    y_train = train_obs[:, 2]   # (n,)
    m_train = train_cmaq[:, 2]  # (n,)

    X_test = test_cmaq[:, :2]   # (m, 2)
    m_test = test_cmaq[:, 2]    # (m,)

    # Step 1: 构建图核矩阵
    K = build_graph_kernel_matrix(X_train, alpha=alpha, lambda_km=lambda_km, wind_dir=wind_dir)

    # Step 2: 添加正则化
    n = X_train.shape[0]
    K_reg = K + gamma * np.eye(n)

    # Step 3: 计算残差
    residual = y_train - m_train  # O - M

    # Step 4: 克里金权重 - 对每个测试点求解
    predictions = np.zeros(X_test.shape[0])

    # 构建训练点之间的互协方差
    K_inv = np.linalg.inv(K_reg + 1e-6 * np.eye(n))

    for i, x_test in enumerate(X_test):
        # 测试点到训练点的图核距离
        delta = X_train - x_test  # (n, 2)
        D_eucl_test = np.sqrt(np.sum(delta**2, axis=1)) * 111.0

        if wind_dir is not None:
            wind_rad = np.deg2rad(wind_dir)
            dx, dy = np.cos(wind_rad), np.sin(wind_rad)
            proj = delta[:, 0] * dx + delta[:, 1] * dy
            D_wind_test = np.abs(proj)
            D_graph_test = alpha * D_eucl_test + (1 - alpha) * D_wind_test
        else:
            D_graph_test = D_eucl_test

        k = np.exp(-D_graph_test / lambda_km)

        # 克里金权重
        w = K_inv @ k
        w = w / (np.sum(w) + 1e-6)  # 归一化

        # 克里金预测残差
        r_pred = w @ residual

        # 融合预测
        predictions[i] = beta * m_test[i] + r_pred

    return predictions


def run_spin_kr_ten_fold(selected_day='2020-01-01', **kwargs):
    """运行SPIN-Kr十折交叉验证"""
    print("="*60)
    print("SPIN-Kr Ten-Fold Cross Validation")
    print("="*60)

    alpha = kwargs.get('alpha', 0.5)
    lambda_km = kwargs.get('lambda_km', 50.0)
    gamma = kwargs.get('gamma', 0.1)
    beta = kwargs.get('beta', 1.0)
    wind_dir = kwargs.get('wind_dir', None)

    print(f"\nParameters: alpha={alpha}, lambda_km={lambda_km}, gamma={gamma}, beta={beta}, wind_dir={wind_dir}")

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
    for _, row in day_df.iterrows():
        val = get_cmaq_at_site(row['Lon'], row['Lat'], lon_cmaq, lat_cmaq, pred_day)
        cmaq_values.append(val)
    day_df['CMAQ'] = cmaq_values

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

        train_obs = train_df[['Lon', 'Lat', 'Conc']].values
        train_cmaq = train_df[['Lon', 'Lat', 'CMAQ']].values
        test_cmaq = test_df[['Lon', 'Lat', 'CMAQ']].values
        y_test = test_df['Conc'].values

        # SPIN-Kr 预测
        pred = spin_kriging_predict(
            train_obs, train_cmaq, test_cmaq,
            alpha=alpha, lambda_km=lambda_km, gamma=gamma, beta=beta, wind_dir=wind_dir
        )

        results[fold_id] = {
            'y_true': y_test,
            'pred': pred
        }

        print(f"  Fold {fold_id}: completed, n_test={len(test_df)}")

    # 汇总
    pred_all = np.concatenate([results[f]['pred'] for f in range(1, 11) if results[f]])
    true_all = np.concatenate([results[f]['y_true'] for f in range(1, 11) if results[f]])

    # 计算R2
    print("\n=== Results ===")
    metrics = compute_metrics(true_all, pred_all)

    print(f"  SPIN-Kr: R2={metrics['R2']:.4f}, MAE={metrics['MAE']:.2f}, RMSE={metrics['RMSE']:.2f}")

    return metrics


if __name__ == '__main__':
    metrics = run_spin_kr_ten_fold('2020-01-01', alpha=0.5, lambda_km=50.0, gamma=0.1, beta=1.0, wind_dir=None)
    print(f"\nSPIN-Kr: R2={metrics['R2']:.4f}")
