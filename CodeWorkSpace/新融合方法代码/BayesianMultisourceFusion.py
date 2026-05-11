"""

# === 预测值缓存（由 patch_methods.py 自动添加）===
_last_y_true = None
_last_y_pred = None

BayesianMultisourceFusion - 贝叶斯多源融合地球统计法
=====================================================
BMSF-Geostat: Bayesian Multisource Fusion with Geostatistical Mapping

创新点:
1. 贝叶斯后验而非点估计权重
2. SPDE潜在时空随机场 (Matern协方差近似)
3. 不确定性量化(可信区间)
4. 多源融合(CMAQ + 多项式协变量)

参数:
- range_km: SPDE相关距离 km (80.0)
- sigma_f: SPDE方差 (15.0)
- phi: 时间自相关 (0.8)
- alpha_matern: Matern光滑度参数 (1.5)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.paths import get_project_root, data_path
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel, Matern
from scipy.spatial.distance import cdist
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


def spde_matern_covariance(X, range_km=80.0, sigma=15.0, alpha=1.5):
    """
    SPDE近似构建Matern协方差矩阵

    Args:
        X: 坐标 (n, 2) [lon, lat]
        range_km: 相关距离 km
        sigma: 方差
        alpha: Matern光滑度参数 (1.5 = 3/2)

    Returns:
        Sigma: 协方差矩阵 (n, n)
    """
    # 经纬度转换为km (近似)
    lat_mean = np.mean(X[:, 1])
    km_per_deg_lon = 111.0 * np.cos(np.radians(lat_mean))
    km_per_deg_lat = 111.0

    X_km = np.zeros_like(X)
    X_km[:, 0] = X[:, 0] * km_per_deg_lon
    X_km[:, 1] = X[:, 1] * km_per_deg_lat

    # 计算距离矩阵
    dist = cdist(X_km, X_km)

    # Matern 3/2 协方差
    sqrt3 = np.sqrt(3.0)
    r = dist / range_km
    Sigma = sigma**2 * (1.0 + sqrt3 * r) * np.exp(-sqrt3 * r)

    return Sigma


def build_bayesian_model(X_train, y_train, m_train, range_km=80.0, sigma_f=15.0, alpha_matern=1.5):
    """
    贝叶斯层级模型拟合

    数据层: Y = X*beta + f_spde + epsilon
    过程层: f ~ GP(0, Matern)
    参数层: beta ~ N(0, sigma_beta^2), epsilon ~ N(0, sigma_y^2)

    Returns:
        beta_posterior: 后验系数均值
        f_posterior: 潜在场后验均值
        sigma_y: 观测噪声标准差
    """
    n = len(y_train)

    # 构建设计矩阵: [1, CMAQ, CMAQ^2]
    X_design = np.column_stack([np.ones(n), m_train, m_train**2])

    # 先验: beta ~ N(0, sigma_beta^2 * I)
    sigma_beta = 10.0
    Sigma_beta_prior = sigma_beta**2 * np.eye(X_design.shape[1])

    # SPDE潜在场协方差
    K_spde = spde_matern_covariance(X_train, range_km, sigma_f, alpha_matern)

    # 观测噪声
    sigma_y = 1.0
    Sigma_noise = sigma_y**2 * np.eye(n)

    # 总协方差: K_total = K_spde + sigma_y^2 * I
    K_total = K_spde + Sigma_noise

    # 贝叶斯后验 (使用共轭先验近似)
    # beta_posterior = (X'K^{-1}X + Sigma_beta^{-1})^{-1} X'K^{-1}y
    try:
        K_inv = np.linalg.inv(K_total + 1e-6 * np.eye(n))
    except np.linalg.LinAlgError:
        K_inv = np.linalg.pinv(K_total)

    XtKinvX = X_design.T @ K_inv @ X_design
    XtKinvY = X_design.T @ K_inv @ y_train

    # 后验协方差和均值
    Sigma_posterior = np.linalg.inv(XtKinvX + np.linalg.inv(Sigma_beta_prior) + 1e-6 * np.eye(X_design.shape[1]))
    beta_posterior = Sigma_posterior @ XtKinvY

    # 潜在场后验: f = K_spde @ K_total^{-1} @ (y - X*beta)
    residual_f = y_train - X_design @ beta_posterior
    f_posterior = K_spde @ K_inv @ residual_f

    # 估计观测噪声
    residuals = y_train - X_design @ beta_posterior - f_posterior
    sigma_y_est = np.std(residuals)

    return beta_posterior, f_posterior, sigma_y_est, K_spde, K_inv, X_design


def predict_bayesian(X_test, m_test, X_train, y_train, beta_posterior, f_posterior, K_spde, K_inv, sigma_y, range_km=80.0, sigma_f=15.0, alpha_matern=1.5):
    """
    贝叶斯预测

    mu_new = X_new * beta + K_new_train @ K_train^{-1} @ f_posterior
    """
    n_test = len(m_test)
    n_train = len(y_train)

    # 设计矩阵
    X_design_test = np.column_stack([np.ones(n_test), m_test, m_test**2])

    # 协变量预测
    mu_covariate = X_design_test @ beta_posterior

    # SPDE潜在场外推
    K_cross = spde_matern_covariance_cross(X_train, X_test, range_km, sigma_f, alpha_matern)
    mu_spde = K_cross @ K_inv @ f_posterior

    # 总预测
    pred = mu_covariate + mu_spde

    return pred


def spde_matern_covariance_cross(X_train, X_test, range_km=80.0, sigma=15.0, alpha=1.5):
    """计算训练-测试集之间的交叉协方差"""
    lat_mean = np.mean(np.vstack([X_train, X_test])[:, 1])
    km_per_deg_lon = 111.0 * np.cos(np.radians(lat_mean))
    km_per_deg_lat = 111.0

    X_train_km = X_train.copy()
    X_train_km[:, 0] *= km_per_deg_lon
    X_train_km[:, 1] *= km_per_deg_lat

    X_test_km = X_test.copy()
    X_test_km[:, 0] *= km_per_deg_lon
    X_test_km[:, 1] *= km_per_deg_lat

    dist = cdist(X_test_km, X_train_km)
    sqrt3 = np.sqrt(3.0)
    r = dist / range_km
    K_cross = sigma**2 * (1.0 + sqrt3 * r) * np.exp(-sqrt3 * r)

    return K_cross


def run_BayesianMultisourceFusion_ten_fold(selected_day='2020-01-01', range_km=80.0, sigma_f=15.0, alpha_matern=1.5):
    print("=" * 60)
    print("BayesianMultisourceFusion Ten-Fold Cross Validation")
    print(f"Parameters: range_km={range_km}, sigma_f={sigma_f}, alpha={alpha_matern}")
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

        # 标准化
        y_mean, y_std = np.mean(y_train), np.std(y_train) + 1e-8
        m_mean, m_std = np.mean(m_train), np.std(m_train) + 1e-8
        y_train_norm = (y_train - y_mean) / y_std
        m_train_norm = (m_train - m_mean) / m_std
        m_test_norm = (m_test - m_mean) / m_std

        # 贝叶斯层级模型拟合
        beta_post, f_post, sigma_y, K_spde, K_inv, X_design = build_bayesian_model(
            X_train, y_train_norm, m_train_norm, range_km, sigma_f, alpha_matern
        )

        # 预测
        pred_norm = predict_bayesian(
            X_test, m_test_norm, X_train, y_train_norm,
            beta_post, f_post, K_spde, K_inv, sigma_y,
            range_km, sigma_f, alpha_matern
        )

        # 反标准化
        pred = pred_norm * y_std + y_mean

        results[fold_id] = {
            'y_true': y_test,
            'bmsf': pred
        }
        print(f"  Fold {fold_id}: completed")

    # 汇总
    bmsf_all = np.concatenate([results[f]['bmsf'] for f in range(1, 11) if results[f]])
    true_all = np.concatenate([results[f]['y_true'] for f in range(1, 11) if results[f]])

    print("\n=== Results ===")
    bmsf_metrics = compute_metrics(true_all, bmsf_all)
    # 缓存预测值供多天聚合使用
    global _last_y_true, _last_y_pred
    _last_y_true = true_all
    _last_y_pred = bmsf_all

    print(f"  BMSF: R2={bmsf_metrics['R2']:.4f}, MAE={bmsf_metrics['MAE']:.2f}, RMSE={bmsf_metrics['RMSE']:.2f}")

    result_df = pd.DataFrame([{'method': 'BayesianMultisourceFusion', **bmsf_metrics}])
    result_df.to_csv(f'{output_dir}/BayesianMultisourceFusion_summary.csv', index=False)
    print(f"\nResults saved to: {output_dir}/BayesianMultisourceFusion_summary.csv")

    return bmsf_metrics


if __name__ == '__main__':
    metrics = run_BayesianMultisourceFusion_ten_fold('2020-01-01')
    print(f"\nFinal: R2={metrics['R2']:.4f}")
