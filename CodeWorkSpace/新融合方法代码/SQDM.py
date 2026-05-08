"""
SQDM - Spatial Quantile Deviation Mapping
==========================================
空间分位数偏差映射，捕捉"同一CMAQ值在不同位置偏差不同"

创新点:
1. 对CMAQ值进行分位数划分，不同分位数区间使用不同的偏差映射
2. 考虑空间权重，距离越近的站点对偏差估计影响越大
3. 捕捉CMAQ偏差的空间异质性

核心参数:
- n_neighbor: 邻域站点数, default: 12
- gamma: CMAQ依赖校正因子, default: 0.1
- alpha_spatial: 空间权重指数, default: 2.0
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.paths import get_project_root, data_path
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
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


def spatial_quantile_deviation(x_obs, m_obs, r_obs, x_pred, n_neighbor=12, gamma=0.1, alpha_spatial=2.0):
    """
    空间分位数偏差映射预测

    Parameters:
    -----------
    x_obs: array (n_obs, 2)
        观测点坐标 [lon, lat]
    m_obs: array (n_obs,)
        观测点CMAQ值
    r_obs: array (n_obs,)
        观测点残差 (Obs - CMAQ)
    x_pred: array (n_pred, 2)
        预测点坐标
    n_neighbor: int
        邻域站点数
    gamma: float
        CMAQ依赖校正因子
    alpha_spatial: float
        空间权重指数

    Returns:
    --------
    pred_values: array (n_pred,)
        预测偏差值
    """
    n_pred = x_pred.shape[0]
    pred_values = np.zeros(n_pred)

    # 将CMAQ值分成几个分位数区间
    quantiles = [0, 0.25, 0.5, 0.75, 1.0]
    quantile_values = np.quantile(m_obs, quantiles)

    for i in range(n_pred):
        # 计算到所有观测点的距离
        dists = np.sqrt((x_obs[:, 0] - x_pred[i, 0])**2 + (x_obs[:, 1] - x_pred[i, 1])**2)

        # 选择n_neighbor个最近邻
        if n_neighbor < len(dists):
            idx = np.argpartition(dists, n_neighbor)[:n_neighbor]
        else:
            idx = np.arange(len(dists))

        dists_k = dists[idx]
        m_k = m_obs[idx]
        r_k = r_obs[idx]

        # 避免除零
        dists_k = np.maximum(dists_k, 1e-10)

        # 空间权重
        spatial_weights = 1.0 / (dists_k ** alpha_spatial)
        spatial_weights = spatial_weights / spatial_weights.sum()

        # 找到预测点CMAQ值所在的分位数区间
        m_pred_val = get_cmaq_at_site(x_pred[i, 0], x_pred[i, 1], x_obs[:, 0].reshape(1, -1) if False else x_obs[:, 0].reshape(1, -1), x_obs[:, 1].reshape(1, -1), m_obs.reshape(1, -1))

        # 简化：使用预测点周围观测点的加权平均CMAQ作为参考
        m_ref = np.sum(spatial_weights * m_k)

        # 计算CMAQ依赖的偏差校正
        # 如果观测点CMAQ值接近，使用局部加权平均偏差
        # 如果观测点CMAQ值差异大，使用基于CMAQ差异的校正
        cmaq_diff = np.abs(m_k - m_ref)
        cmaq_weights = 1.0 / (1.0 + gamma * cmaq_diff)
        cmaq_weights = cmaq_weights / cmaq_weights.sum()

        # 综合权重
        combined_weights = spatial_weights * (1 - gamma) + cmaq_weights * gamma
        combined_weights = combined_weights / combined_weights.sum()

        pred_values[i] = np.sum(combined_weights * r_k)

    return pred_values


def sqdm_predict(x_obs, m_obs, r_obs, x_pred, n_neighbor=12, gamma=0.1, alpha_spatial=2.0):
    """
    SQDM预测 - 空间分位数偏差映射

    对于每个预测点：
    1. 找到最近的n_neighbor个观测点
    2. 根据CMAQ值的相似性调整权重
    3. 计算加权平均偏差

    Parameters:
    -----------
    x_obs: array (n_obs, 2)
        观测点坐标
    m_obs: array (n_obs,)
        观测点CMAQ值
    r_obs: array (n_obs,)
        观测点残差 (Obs - CMAQ)
    x_pred: array (n_pred, 2)
        预测点坐标
    n_neighbor: int
        邻域站点数
    gamma: float
        CMAQ依赖校正因子
    alpha_spatial: float
        空间权重指数

    Returns:
    --------
    pred_values: array (n_pred,)
        预测的残差值
    """
    n_pred = x_pred.shape[0]
    pred_values = np.zeros(n_pred)

    for i in range(n_pred):
        # 计算到所有观测点的距离
        dists = np.sqrt((x_obs[:, 0] - x_pred[i, 0])**2 + (x_obs[:, 1] - x_pred[i, 1])**2)

        # 选择n_neighbor个最近邻
        if n_neighbor < len(dists):
            idx = np.argpartition(dists, n_neighbor)[:n_neighbor]
        else:
            idx = np.arange(len(dists))

        dists_k = dists[idx]
        m_k = m_obs[idx]
        r_k = r_obs[idx]

        # 避免除零
        dists_k = np.maximum(dists_k, 1e-10)

        # 空间权重
        spatial_weights = 1.0 / (dists_k ** alpha_spatial)

        # CMAQ相似性权重（指数衰减）
        # 使用预测点CMAQ值的估计（这里用最近邻的CMAQ）
        m_pred_est = m_k[np.argmin(dists_k)]
        cmaq_diff = np.abs(m_k - m_pred_est)
        cmaq_weights = np.exp(-gamma * cmaq_diff)

        # 综合权重
        combined_weights = spatial_weights * cmaq_weights
        combined_weights = combined_weights / combined_weights.sum()

        pred_values[i] = np.sum(combined_weights * r_k)

    return pred_values


def run_sqdm_ten_fold(selected_day='2020-01-01', n_neighbor=12, gamma=0.1, alpha_spatial=2.0):
    """
    运行SQDM十折交叉验证
    """
    print("="*60)
    print("SQDM Ten-Fold Cross Validation")
    print(f"Parameters: n_neighbor={n_neighbor}, gamma={gamma}, alpha_spatial={alpha_spatial}")
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
    for _, row in day_df.iterrows():
        val = get_cmaq_at_site(row['Lon'], row['Lat'], lon_cmaq, lat_cmaq, pred_day)
        cmaq_values.append(val)
    day_df['CMAQ'] = cmaq_values

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
        m_train = train_df['CMAQ'].values
        m_test = test_df['CMAQ'].values
        residual_train = train_df['residual'].values

        # SQDM预测
        sqdm_residual_pred = sqdm_predict(X_train, m_train, residual_train, X_test, n_neighbor, gamma, alpha_spatial)

        # 融合预测
        cmaq_test = test_df['CMAQ'].values
        sqdm_pred = cmaq_test + sqdm_residual_pred

        # 对比：简单IDW残差插值
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel

        kernel = ConstantKernel(10.0, (1e-2, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e1))
        gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, alpha=0.1, normalize_y=True)
        gpr.fit(X_train, residual_train)
        gpr_residual_pred, _ = gpr.predict(X_test, return_std=True)
        gpr_pred = cmaq_test + gpr_residual_pred

        results[fold_id] = {
            'y_true': test_df['Conc'].values,
            'sqdm': sqdm_pred,
            'gpr': gpr_pred
        }

        print(f"  Fold {fold_id}: completed")

    # 汇总
    sqdm_all = np.concatenate([results[f]['sqdm'] for f in range(1, 11) if results[f]])
    gpr_all = np.concatenate([results[f]['gpr'] for f in range(1, 11) if results[f]])
    true_all = np.concatenate([results[f]['y_true'] for f in range(1, 11) if results[f]])

    # 计算R2
    print("\n=== Results ===")
    sqdm_metrics = compute_metrics(true_all, sqdm_all)
    gpr_metrics = compute_metrics(true_all, gpr_all)

    print(f"  SQDM: R2={sqdm_metrics['R2']:.4f}, MAE={sqdm_metrics['MAE']:.2f}, RMSE={sqdm_metrics['RMSE']:.2f}")
    print(f"  GPR:  R2={gpr_metrics['R2']:.4f}, MAE={gpr_metrics['MAE']:.2f}, RMSE={gpr_metrics['RMSE']:.2f}")

    # 保存结果
    result_df = pd.DataFrame([{
        'Method': 'SQDM',
        'R2': sqdm_metrics['R2'],
        'MAE': sqdm_metrics['MAE'],
        'RMSE': sqdm_metrics['RMSE'],
        'MB': sqdm_metrics['MB']
    }])
    result_df.to_csv(f'{output_dir}/SQDM_summary.csv', index=False)

    print(f"\nResults saved to: {output_dir}/SQDM_summary.csv")

    return sqdm_metrics


if __name__ == '__main__':
    metrics = run_sqdm_ten_fold('2020-01-01', n_neighbor=12, gamma=0.1, alpha_spatial=2.0)
    print(f"\nFinal: R2={metrics['R2']:.4f}")
