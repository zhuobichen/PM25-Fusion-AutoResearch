"""
CorrDiff-3km - 残差修正扩散降尺度法
Residual Corrective Diffusion for 3km Downscaling
===================================================
使用扩散概率模型学习从25km到3km的高频残差校正

注意：完整的扩散模型需要PyTorch训练，此处提供框架实现
对于无GPU环境，使用简化的残差扩散近似

创新点:
1. 扩散模型学习高频残差分布
2. 生成式不确定性量化
3. 条件引导保持物理一致性
4. 多采样路径统计

参数:
- T: 扩散步数 (1000)
- w: 引导强度 (1.0)
- n_samples: 采样数量 (10)
- lr: 学习率 (1e-4)
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
from scipy.ndimage import gaussian_filter
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


def extract_high_frequency_residual(cmaq_fine, sigma_km=5.0):
    """
    提取高频残差成分
    高频 = 原始 - 高斯平滑（代表粗尺度趋势）

    Args:
        cmaq_fine: 3km高分辨率CMAQ数据
        sigma_km: 平滑尺度 km

    Returns:
        high_freq: 高频残差
    """
    # sigma从km转换为网格点数（假设3km分辨率）
    sigma_grid = sigma_km / 3.0
    smooth = gaussian_filter(cmaq_fine, sigma=sigma_grid)
    high_freq = cmaq_fine - smooth
    return high_freq


def diffusion_denoise_step(x_t, t, T, noise_schedule='cosine'):
    """
    单步去噪（简化版）

    Args:
        x_t: 当前噪声状态
        t: 当前步
        T: 总步数
        noise_schedule: 噪声调度

    Returns:
        x_{t-1}: 去噪后状态
    """
    alpha_t = 1.0 - (t / T) * 0.9  # 从1到0.1

    if noise_schedule == 'linear':
        beta_t = 0.1 / T
        alpha_t = np.prod(1 - beta_t * np.arange(1, t+1))
    elif noise_schedule == 'cosine':
        # cosine schedule
        s = 0.008
        steps = np.arange(T + 1)
        alpha_bar = np.cos(((steps / T) + s) / (1 + s) * np.pi * 0.5) ** 2
        alpha_bar = alpha_bar / alpha_bar[0]
        alpha_t = alpha_bar[t] / alpha_bar[t-1] if t > 0 else alpha_bar[0]

    # 简化去噪
    x_prev = x_t / np.sqrt(alpha_t)

    return x_prev


def corr_diff_predict(train_obs, train_cmaq, test_cmaq, **kwargs):
    """
    CorrDiff-3km 融合预测

    简化实现：使用GPR学习高频残差模式，模拟扩散模型的去噪效果

    Args:
        train_obs: 监测数据 (n, 3) [lon, lat, PM25]
        train_cmaq: CMAQ模拟数据 (n, 3) [lon, lat, PM25]
        test_cmaq: 测试CMAQ数据 (m, 3)
        kwargs: T, w, n_samples

    Returns:
        predictions: (m,)
    """
    T = kwargs.get('T', 1000)  # 扩散步数
    w = kwargs.get('w', 1.0)   # 引导强度
    n_samples = kwargs.get('n_samples', 10)

    X_train = train_obs[:, :2]  # (n, 2)
    y_train = train_obs[:, 2]  # (n,)
    m_train = train_cmaq[:, 2] # (n,)

    X_test = test_cmaq[:, :2]  # (m, 2)
    m_test = test_cmaq[:, 2]   # (m,)

    # Step 1: 估计粗尺度趋势
    # CMAQ本身已经提供了粗尺度信息，这里用线性模型校正
    from sklearn.linear_model import LinearRegression
    lr = LinearRegression()
    lr.fit(m_train.reshape(-1, 1), y_train)
    trend = lr.predict(m_train.reshape(-1, 1))
    residual_train = y_train - trend

    # Step 2: 高频残差建模
    # 使用GPR捕捉非均匀的高频残差模式（模拟扩散模型学习的残差分布）
    kernel = ConstantKernel(5.0, (1e-2, 1e2)) * RBF(length_scale=0.5, length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e1))
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, alpha=0.1, normalize_y=True)
    gpr.fit(X_train, residual_train)

    # Step 3: 多样本预测（模拟扩散模型多路径采样）
    predictions_list = []
    for s in range(n_samples):
        residual_pred, residual_std = gpr.predict(X_test, return_std=True)
        # 添加随机扰动模拟扩散采样不确定性
        noise = np.random.normal(0, np.abs(residual_std) * w / np.sqrt(n_samples))
        pred_s = lr.predict(m_test.reshape(-1, 1)) + residual_pred + noise
        predictions_list.append(pred_s)

    # Step 4: 融合多路径预测
    predictions = np.mean(predictions_list, axis=0)

    return predictions


def run_corr_diff_ten_fold(selected_day='2020-01-01', **kwargs):
    """运行CorrDiff-3km十折交叉验证"""
    print("="*60)
    print("CorrDiff-3km Ten-Fold Cross Validation")
    print("="*60)

    T = kwargs.get('T', 1000)
    w = kwargs.get('w', 1.0)
    n_samples = kwargs.get('n_samples', 10)

    print(f"\nParameters: T={T}, w={w}, n_samples={n_samples}")

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

        # CorrDiff 预测
        pred = corr_diff_predict(
            train_obs, train_cmaq, test_cmaq,
            T=T, w=w, n_samples=n_samples
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

    print(f"  CorrDiff-3km: R2={metrics['R2']:.4f}, MAE={metrics['MAE']:.2f}, RMSE={metrics['RMSE']:.2f}")

    return metrics


if __name__ == '__main__':
    metrics = run_corr_diff_ten_fold('2020-01-01', T=1000, w=1.0, n_samples=10)
    print(f"\nCorrDiff-3km: R2={metrics['R2']:.4f}")
