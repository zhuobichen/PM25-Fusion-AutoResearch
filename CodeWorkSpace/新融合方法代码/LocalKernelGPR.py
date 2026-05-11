"""

# === 预测值缓存（由 patch_methods.py 自动添加）===
_last_y_true = None
_last_y_pred = None

LocalKernelGPR - 局部带宽高斯过程回归
======================================
使用局部自适应带宽的GPR替代全局GPR。

创新点:
1. 根据邻域站点密度自适应调整GPR核宽度
2. 密集区域使用较短的相关长度，稀疏区域使用较长的相关长度
3. 城市高密度区短相关，郊区低密度区长三角
4. 捕捉空间非平稳性

核心参数:
- ell_0: 基础相关长度 (km), default: 15.0
- ell_min: 最小相关长度 (km), default: 5.0
- ell_max: 最大相关长度 (km), default: 40.0
- n_neighbor: 邻域站点数, default: 10
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.paths import get_project_root, data_path
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
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


def compute_local_bandwidth(x_obs, x_target, n_neighbor=10, ell_0=15.0, ell_min=5.0, ell_max=40.0):
    """
    计算目标点处的局部带宽
    密度高(mean_dist小) -> 短带宽, 密度低 -> 长带宽
    """
    n_target = x_target.shape[0]
    local_bw = np.zeros(n_target)

    for i in range(n_target):
        dists = np.sqrt((x_obs[:, 0] - x_target[i, 0])**2 + (x_obs[:, 1] - x_target[i, 1])**2)
        if n_neighbor < len(dists):
            idx = np.argpartition(dists, n_neighbor)[:n_neighbor]
            dists_k = dists[idx]
        else:
            dists_k = dists

        mean_dist = np.mean(dists_k) + 1e-10
        ref_dist = ell_0
        local_bw_i = ell_0 * (mean_dist / ref_dist)
        local_bw[i] = np.clip(local_bw_i, ell_min, ell_max)

    return local_bw


def lbgpr_predict(x_obs, y_obs, x_pred, local_bw, n_neighbor=10):
    """
    使用局部带宽高斯权重的GPR预测
    """
    n_pred = x_pred.shape[0]
    pred_values = np.zeros(n_pred)

    for i in range(n_pred):
        dists = np.sqrt((x_obs[:, 0] - x_pred[i, 0])**2 + (x_obs[:, 1] - x_pred[i, 1])**2)
        if n_neighbor < len(dists):
            idx = np.argpartition(dists, n_neighbor)[:n_neighbor]
        else:
            idx = np.arange(len(dists))

        dists_k = dists[idx]
        y_k = y_obs[idx]
        dists_k = np.maximum(dists_k, 1e-10)

        bw = local_bw[i]
        weights = np.exp(-0.5 * (dists_k / bw)**2)
        weights = weights / weights.sum()

        pred_values[i] = np.sum(weights * y_k)

    return pred_values


def run_LocalKernelGPR_ten_fold(selected_day='2020-01-01', ell_0=15.0, ell_min=5.0, ell_max=40.0, n_neighbor=10):
    print("=" * 60)
    print("LocalKernelGPR Ten-Fold Cross Validation")
    print(f"Parameters: ell_0={ell_0}, ell_min={ell_min}, ell_max={ell_max}, n_neighbor={n_neighbor}")
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

        # 残差 = 观测 - CMAQ
        residual_train = y_train - m_train

        # 计算测试点的局部带宽
        local_bw = compute_local_bandwidth(X_train, X_test, n_neighbor, ell_0, ell_min, ell_max)

        # LB-GPR预测
        lbgpr_residual_pred = lbgpr_predict(X_train, residual_train, X_test, local_bw, n_neighbor)

        # 融合预测
        lbgpr_pred = m_test + lbgpr_residual_pred

        results[fold_id] = {
            'y_true': y_test,
            'lbgpr': lbgpr_pred,
            'local_bw': local_bw
        }
        print(f"  Fold {fold_id}: completed (mean local_bw: {local_bw.mean():.2f})")

    # 汇总
    lbgpr_all = np.concatenate([results[f]['lbgpr'] for f in range(1, 11) if results[f]])
    true_all = np.concatenate([results[f]['y_true'] for f in range(1, 11) if results[f]])

    print("\n=== Results ===")
    lbgpr_metrics = compute_metrics(true_all, lbgpr_all)
    # 缓存预测值供多天聚合使用
    global _last_y_true, _last_y_pred
    _last_y_true = true_all
    _last_y_pred = lbgpr_all

    print(f"  LB-GPR: R2={lbgpr_metrics['R2']:.4f}, MAE={lbgpr_metrics['MAE']:.2f}, RMSE={lbgpr_metrics['RMSE']:.2f}")

    result_df = pd.DataFrame([{'method': 'LocalKernelGPR', **lbgpr_metrics}])
    result_df.to_csv(f'{output_dir}/LocalKernelGPR_summary.csv', index=False)
    print(f"\nResults saved to: {output_dir}/LocalKernelGPR_summary.csv")

    return lbgpr_metrics


if __name__ == '__main__':
    metrics = run_LocalKernelGPR_ten_fold('2020-01-01')
    print(f"\nFinal: R2={metrics['R2']:.4f}")
