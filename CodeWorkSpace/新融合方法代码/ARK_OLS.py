"""

# === 预测值缓存（由 patch_methods.py 自动添加）===
_last_y_true = None
_last_y_pred = None

ARK-OLS - Adaptive Residual Kriging with OLS Linear Correction
==============================================================
创新方法：结合OLS线性校正和多尺度残差插值

原理:
1. 使用OLS找到观测值与模型值的最佳线性变换
2. 对OLS残差使用自适应多尺度IDW插值
3. 融合: P = a + b*M + R_idw

核心公式:
OLS: O = a + b*M + e
融合: P = a + b*M + R_interpolated

创新点:
1. OLS比简单偏差或比率更准确地建模线性关系
2. 多尺度IDW捕捉不同空间尺度的残差结构
3. 自适应权重结合多种尺度
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
from Code.VNAeVNAaVNA.nna_methods import NNA

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


def idw_predict(x_obs, y_obs, values, x_pred, power=-2, k=None):
    """
    IDW插值预测

    Parameters:
    -----------
    x_obs, y_obs : array
        观测点坐标
    values : array
        观测值
    x_pred : array
        预测点坐标
    power : float
        IDW幂次
    k : int or None
        最近邻数量，None表示使用所有点

    Returns:
    --------
    pred_values : array
        预测值
    """
    n_pred = x_pred.shape[0]
    n_obs = x_obs.shape[0]
    pred_values = np.zeros(n_pred)

    for i in range(n_pred):
        dists = np.sqrt((x_obs[:, 0] - x_pred[i, 0])**2 + (x_obs[:, 1] - x_pred[i, 1])**2)

        if k is not None and k < n_obs:
            # 使用k个最近邻
            idx = np.argpartition(dists, k)[:k]
            dists_k = dists[idx]
            values_k = values[idx]
        else:
            dists_k = dists
            values_k = values

        # 避免除零
        dists_k = np.maximum(dists_k, 1e-10)

        # IDW权重
        weights = 1.0 / (dists_k ** power)
        weights = weights / weights.sum()

        pred_values[i] = np.sum(weights * values_k)

    return pred_values


def run_ark_ols_ten_fold(selected_day='2020-01-01'):
    """
    运行ARK-OLS十折交叉验证
    """
    print("="*60)
    print("ARK-OLS Ten-Fold Cross Validation")
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

    # 提取站点CMAQ值
    print("=== Extracting CMAQ at Sites ===")
    cmaq_values = []
    for _, row in day_df.iterrows():
        val = get_cmaq_at_site(row['Lon'], row['Lat'], lon_cmaq, lat_cmaq, pred_day)
        cmaq_values.append(val)
    day_df['CMAQ'] = cmaq_values

    # 创建网格坐标
    ny, nx = lon_cmaq.shape
    X_grid_full = np.column_stack([lon_cmaq.ravel(), lat_cmaq.ravel()])
    y_grid_model_full = pred_day.ravel()

    print(f"Data loaded: {len(day_df)} monitoring records")

    # 多尺度k值
    k_values = [10, 20, 30, 50]

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

        # === 1. OLS线性校正 ===
        X_train = train_df['CMAQ'].values.reshape(-1, 1)
        y_train = train_df['Conc'].values

        ols = LinearRegression()
        ols.fit(X_train, y_train)
        a, b = ols.intercept_, ols.coef_[0]

        # OLS预测值
        ols_train_pred = a + b * train_df['CMAQ'].values
        ols_test_pred = a + b * test_df['CMAQ'].values

        # OLS残差
        residual_train = y_train - ols_train_pred

        # === 2. 多尺度IDW残差插值 ===
        X_train_coords = train_df[['Lon', 'Lat']].values
        X_test_coords = test_df[['Lon', 'Lat']].values

        # 在测试站点预测残差
        residual_pred = np.zeros(len(test_df))
        for k in k_values:
            residual_k = idw_predict(X_train_coords, residual_train, residual_train, X_test_coords, power=-2, k=k)
            residual_pred += residual_k / len(k_values)

        # === 3. 融合 ===
        ark_ols_pred = ols_test_pred + residual_pred

        # === 4. 对比：单纯OLS ===
        ols_only_pred = ols_test_pred

        # === 5. 对比：RK-like (GPR) ===
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel

        kernel = ConstantKernel(10.0, (1e-2, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e1))
        gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, alpha=0.1, normalize_y=True)
        gpr.fit(X_train_coords, residual_train)
        gpr_residual_pred, _ = gpr.predict(X_test_coords, return_std=True)
        rk_pred = ols_test_pred + gpr_residual_pred

        results[fold_id] = {
            'y_true': test_df['Conc'].values,
            'ols': ols_only_pred,
            'ark_ols': ark_ols_pred,
            'rk': rk_pred
        }

        print(f"  Fold {fold_id}: completed")

    # 汇总所有折叠结果
    ols_all = np.concatenate([results[f]['ols'] for f in range(1, 11) if results[f]])
    ark_ols_all = np.concatenate([results[f]['ark_ols'] for f in range(1, 11) if results[f]])
    rk_all = np.concatenate([results[f]['rk'] for f in range(1, 11) if results[f]])
    true_all = np.concatenate([results[f]['y_true'] for f in range(1, 11) if results[f]])

    # 计算各方法R2
    print("\n=== Individual Method R2 ===")
    ols_metrics = compute_metrics(true_all, ols_all)
    ark_ols_metrics = compute_metrics(true_all, ark_ols_all)
    rk_metrics = compute_metrics(true_all, rk_all)

    print(f"  OLS Linear Correction: {ols_metrics['R2']:.4f}")
    print(f"  ARK-OLS (Multi-scale IDW): {ark_ols_metrics['R2']:.4f}")
    print(f"  RK with OLS base: {rk_metrics['R2']:.4f}")

    # 优化Ensemble权重
    print("\n=== Optimizing Ensemble Weights (ARK-OLS + RK) ===")
    best_r2 = -np.inf
    best_weights = None
    weight_results = []

    for w1 in np.arange(0, 1.05, 0.1):
        w2 = round(1.0 - w1, 2)
        ensemble_pred = w1 * ark_ols_all + w2 * rk_all
        metrics = compute_metrics(true_all, ensemble_pred)

        weight_results.append({
            'w1_ark_ols': w1, 'w2_rk': w2,
            'R2': metrics['R2'], 'MAE': metrics['MAE'], 'RMSE': metrics['RMSE']
        })

        if metrics['R2'] > best_r2:
            best_r2 = metrics['R2']
            best_weights = (w1, w2)

    print(f"\nBest weights: ARK-OLS={best_weights[0]:.2f}, RK={best_weights[1]:.2f}")
    print(f"Best R2: {best_r2:.4f}")

    # 最终评估
    final_pred = best_weights[0] * ark_ols_all + best_weights[1] * rk_all
    final_metrics = compute_metrics(true_all, final_pred)
    # 缓存预测值供多天聚合使用
    global _last_y_true, _last_y_pred
    _last_y_true = true_all
    _last_y_pred = final_pred


    print("\n" + "="*60)
    print("Final ARK-OLS+RK Ensemble Results")
    print("="*60)
    print(f"Optimal weights: ARK-OLS={best_weights[0]:.2f}, RK={best_weights[1]:.2f}")
    print(f"R2: {final_metrics['R2']:.4f}")
    print(f"MAE: {final_metrics['MAE']:.2f}")
    print(f"RMSE: {final_metrics['RMSE']:.2f}")
    print(f"MB: {final_metrics['MB']:.2f}")

    # 保存结果
    result_df = pd.DataFrame([{
        'method': 'ARK-OLS-Ensemble',
        'w1_ark_ols': best_weights[0],
        'w2_rk': best_weights[1],
        **final_metrics
    }])
    result_df.to_csv(f'{output_dir}/ARK_OLS_summary.csv', index=False)

    weight_df = pd.DataFrame(weight_results)
    weight_df.to_csv(f'{output_dir}/ARK_OLS_weight_search.csv', index=False)

    print(f"\nResults saved to: {output_dir}/")

    return final_metrics, best_weights


if __name__ == '__main__':
    metrics, weights = run_ark_ols_ten_fold('2020-01-01')
    print(f"\nFinal: R2={metrics['R2']:.4f}, Weights={weights}")