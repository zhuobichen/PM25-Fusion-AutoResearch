"""
LB-GPR - Local Bandwidth Gaussian Process Regression
======================================================
使用局部自适应带宽的GPR，根据邻域站点密度自适应调整核宽度

创新点:
1. 根据邻域站点密度自适应调整GPR核宽度
2. 密集区域使用较短的相关长度，稀疏区域使用较长的相关长度
3. 捕捉不同空间尺度的局部相关性

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


def compute_local_bandwidth(x_obs, x_target, n_neighbor=10, ell_0=15.0, ell_min=5.0, ell_max=40.0):
    """
    计算目标点处的局部带宽

    基于目标点周围观测点的密度自适应调整相关长度
    密度高 -> 短相关长度，密度低 -> 长相关长度

    Parameters:
    -----------
    x_obs: array (n_obs, 2)
        观测点坐标
    x_target: array (n_target, 2)
        目标点坐标
    n_neighbor: int
        用于计算密度的邻居数
    ell_0: float
        基础相关长度
    ell_min, ell_max: float
        最小/最大相关长度

    Returns:
    --------
    local_bw: array (n_target,)
        每个目标点的局部带宽
    """
    n_target = x_target.shape[0]
    local_bw = np.zeros(n_target)

    for i in range(n_target):
        # 计算到所有观测点的距离
        dists = np.sqrt((x_obs[:, 0] - x_target[i, 0])**2 + (x_obs[:, 1] - x_target[i, 1])**2)

        # 找到n_neighbor个最近邻
        if n_neighbor < len(dists):
            idx = np.argpartition(dists, n_neighbor)[:n_neighbor]
            dists_k = dists[idx]
        else:
            dists_k = dists

        # 计算平均距离（作为密度度量）
        mean_dist = np.mean(dists_k) + 1e-10

        # 密度越高（mean_dist越小），使用越短的带宽
        # 使用ell_0 * (mean_dist / 参考距离)作为局部带宽
        # 参考距离设为ell_0，此时使用ell_0
        ref_dist = ell_0
        local_bw_i = ell_0 * (mean_dist / ref_dist)

        # 限制在[ell_min, ell_max]范围内
        local_bw[i] = np.clip(local_bw_i, ell_min, ell_max)

    return local_bw


def lbgpr_predict_efficient(x_obs, y_obs, x_pred, local_bw, ell_0=15.0, n_neighbor=10):
    """
    使用局部带宽的GPR进行高效预测

    使用加权IDW结合自适应带宽的GPR

    Parameters:
    -----------
    x_obs: array (n_obs, 2)
        观测点坐标
    y_obs: array (n_obs,)
        观测值
    x_pred: array (n_pred, 2)
        预测点坐标
    local_bw: array (n_pred,)
        每个预测点的局部带宽
    ell_0: float
        基础相关长度（用于核函数）
    n_neighbor: int
        使用的邻居数量

    Returns:
    --------
    pred_values: array (n_pred,)
        预测值
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
        y_k = y_obs[idx]
        x_k = x_obs[idx]

        # 避免除零
        dists_k = np.maximum(dists_k, 1e-10)

        # 使用局部带宽的高斯权重
        bw = local_bw[i]
        weights = np.exp(-0.5 * (dists_k / bw)**2)
        weights = weights / weights.sum()

        # 使用加权平均作为预测
        pred_values[i] = np.sum(weights * y_k)

    return pred_values


def run_lbgpr_ten_fold(selected_day='2020-01-01', ell_0=15.0, ell_min=5.0, ell_max=40.0, n_neighbor=10):
    """
    运行LB-GPR十折交叉验证
    """
    print("="*60)
    print("LB-GPR Ten-Fold Cross Validation")
    print(f"Parameters: ell_0={ell_0}, ell_min={ell_min}, ell_max={ell_max}, n_neighbor={n_neighbor}")
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
        residual_train = train_df['residual'].values

        # 计算测试点的局部带宽
        local_bw = compute_local_bandwidth(X_train, X_test, n_neighbor, ell_0, ell_min, ell_max)

        # LB-GPR预测（高效版本）
        lbgpr_residual_pred = lbgpr_predict_efficient(X_train, residual_train, X_test, local_bw, ell_0, n_neighbor)

        # 融合预测
        cmaq_test = test_df['CMAQ'].values
        lbgpr_pred = cmaq_test + lbgpr_residual_pred

        # 对比：标准GPR（固定带宽）
        kernel = ConstantKernel(10.0, (1e-2, 1e3)) * RBF(length_scale=ell_0, length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e1))
        gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, alpha=0.1, normalize_y=True)
        gpr.fit(X_train, residual_train)
        gpr_residual_pred, _ = gpr.predict(X_test, return_std=True)
        gpr_pred = cmaq_test + gpr_residual_pred

        results[fold_id] = {
            'y_true': test_df['Conc'].values,
            'lbgpr': lbgpr_pred,
            'gpr': gpr_pred,
            'local_bw': local_bw
        }

        print(f"  Fold {fold_id}: completed (mean local_bw: {local_bw.mean():.2f})")

    # 汇总
    lbgpr_all = np.concatenate([results[f]['lbgpr'] for f in range(1, 11) if results[f]])
    gpr_all = np.concatenate([results[f]['gpr'] for f in range(1, 11) if results[f]])
    true_all = np.concatenate([results[f]['y_true'] for f in range(1, 11) if results[f]])

    # 计算R2
    print("\n=== Results ===")
    lbgpr_metrics = compute_metrics(true_all, lbgpr_all)
    gpr_metrics = compute_metrics(true_all, gpr_all)

    print(f"  LB-GPR: R2={lbgpr_metrics['R2']:.4f}, MAE={lbgpr_metrics['MAE']:.2f}, RMSE={lbgpr_metrics['RMSE']:.2f}")
    print(f"  GPR:    R2={gpr_metrics['R2']:.4f}, MAE={gpr_metrics['MAE']:.2f}, RMSE={gpr_metrics['RMSE']:.2f}")

    # 保存结果
    result_df = pd.DataFrame([{
        'Method': 'LBGPR',
        'R2': lbgpr_metrics['R2'],
        'MAE': lbgpr_metrics['MAE'],
        'RMSE': lbgpr_metrics['RMSE'],
        'MB': lbgpr_metrics['MB']
    }])
    result_df.to_csv(f'{output_dir}/LBGPR_summary.csv', index=False)

    print(f"\nResults saved to: {output_dir}/LBGPR_summary.csv")

    return lbgpr_metrics


if __name__ == '__main__':
    metrics = run_lbgpr_ten_fold('2020-01-01', ell_0=15.0, ell_min=5.0, ell_max=40.0, n_neighbor=10)
    print(f"\nFinal: R2={metrics['R2']:.4f}")
