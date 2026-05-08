"""
EnsembleRK - 多模型集成残差克里金
===================================
将ResidualKriging、eVNA、aVNA三种方法进行加权集成

创新点:
1. 结合ResidualKriging的空间插值能力
2. 结合eVNA的比率法优点
3. 结合aVNA的偏差法优点
4. 通过网格搜索优化三者的权重

公式: P_Ensemble = w1 * RK + w2 * eVNA + w3 * aVNA
其中 w1 + w2 + w3 = 1
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


def run_ensemble_rk_ten_fold(selected_day='2020-01-01'):
    """
    运行EnsembleRK十折交叉验证
    """
    print("="*60)
    print("EnsembleRK Ten-Fold Cross Validation")
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

    # 创建网格坐标
    ny, nx = lon_cmaq.shape
    X_grid_full = np.column_stack([lon_cmaq.ravel(), lat_cmaq.ravel()])
    y_grid_model_full = pred_day.ravel()

    print(f"Data loaded: {len(day_df)} monitoring records")

    # 定义高斯过程核函数
    kernel = ConstantKernel(10.0, (1e-2, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e1))

    # 运行十折验证，收集三种方法的预测
    print("\n=== Running 10-fold Cross Validation ===")
    results = {fold_id: {} for fold_id in range(1, 11)}

    for fold_id in range(1, 11):
        train_df = day_df[day_df['fold'] != fold_id].copy()
        test_df = day_df[day_df['fold'] == fold_id].copy()

        train_df = train_df.dropna(subset=['Lon', 'Lat', 'CMAQ', 'Conc'])
        test_df = test_df.dropna(subset=['Lon', 'Lat', 'CMAQ', 'Conc'])

        if len(test_df) == 0:
            continue

        # === 1. Residual Kriging ===
        train_df_rk = train_df.copy()
        train_df_rk['residual'] = train_df_rk['Conc'] - train_df_rk['CMAQ']

        gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, alpha=0.1, normalize_y=True)
        gpr.fit(train_df_rk[['Lon', 'Lat']].values, train_df_rk['residual'].values)
        residual_pred, _ = gpr.predict(test_df[['Lon', 'Lat']].values, return_std=True)
        rk_pred = test_df['CMAQ'].values + residual_pred

        # === 2. eVNA ===
        train_df['x'] = train_df['Lon']
        train_df['y'] = train_df['Lat']
        train_df['mod'] = train_df['CMAQ']
        train_df['bias'] = train_df['Conc'] - train_df['CMAQ']
        train_df['rn'] = train_df['Conc'] / train_df['CMAQ']

        nn = NNA(method='voronoi', k=30, power=-2)
        nn.fit(train_df[['x', 'y']], train_df[['bias', 'rn']])

        # 在网格上预测
        zdf_grid = nn.predict(X_grid_full, njobs=4)
        bias_grid = zdf_grid[:, 0]
        rn_grid = zdf_grid[:, 1]

        # 在测试站点提取
        evna_pred = np.zeros(len(test_df))
        avna_pred = np.zeros(len(test_df))
        for i, (_, row) in enumerate(test_df.iterrows()):
            dist = np.sqrt((lon_cmaq - row['Lon'])**2 + (lat_cmaq - row['Lat'])**2)
            idx = np.argmin(dist)
            evna_pred[i] = y_grid_model_full[idx] * rn_grid[idx]  # eVNA = M * rn
            avna_pred[i] = y_grid_model_full[idx] + bias_grid[idx]  # aVNA = M + bias

        results[fold_id] = {
            'y_true': test_df['Conc'].values,
            'rk': rk_pred,
            'evna': evna_pred,
            'avna': avna_pred
        }

        print(f"  Fold {fold_id}: completed")

    # 汇总所有折叠结果
    rk_all = np.concatenate([results[f]['rk'] for f in range(1, 11) if results[f]])
    evna_all = np.concatenate([results[f]['evna'] for f in range(1, 11) if results[f]])
    avna_all = np.concatenate([results[f]['avna'] for f in range(1, 11) if results[f]])
    true_all = np.concatenate([results[f]['y_true'] for f in range(1, 11) if results[f]])

    # 计算各单一方法R2
    print("\n=== Individual Method R2 ===")
    print(f"  RK: {compute_metrics(true_all, rk_all)['R2']:.4f}")
    print(f"  eVNA: {compute_metrics(true_all, evna_all)['R2']:.4f}")
    print(f"  aVNA: {compute_metrics(true_all, avna_all)['R2']:.4f}")

    # 优化集成权重
    print("\n=== Optimizing Ensemble Weights ===")
    best_r2 = -np.inf
    best_weights = None
    weight_results = []

    # 网格搜索权重 (w1=RK, w2=eVNA, w3=aVNA)
    for w1 in np.arange(0, 1.05, 0.1):
        for w2 in np.arange(0, 1.05 - w1, 0.1):
            w3 = round(1.0 - w1 - w2, 2)
            if w3 < 0:
                continue

            ensemble_pred = w1 * rk_all + w2 * evna_all + w3 * avna_all
            metrics = compute_metrics(true_all, ensemble_pred)

            weight_results.append({
                'w1_rk': w1, 'w2_evna': w2, 'w3_avna': w3,
                'R2': metrics['R2'], 'MAE': metrics['MAE'], 'RMSE': metrics['RMSE']
            })

            if metrics['R2'] > best_r2:
                best_r2 = metrics['R2']
                best_weights = (w1, w2, w3)

    print(f"\nBest weights: RK={best_weights[0]:.2f}, eVNA={best_weights[1]:.2f}, aVNA={best_weights[2]:.2f}")
    print(f"Best R2: {best_r2:.4f}")

    # 最终评估
    final_pred = best_weights[0] * rk_all + best_weights[1] * evna_all + best_weights[2] * avna_all
    final_metrics = compute_metrics(true_all, final_pred)

    print("\n" + "="*60)
    print("Final EnsembleRK Results")
    print("="*60)
    print(f"Optimal weights: RK={best_weights[0]:.2f}, eVNA={best_weights[1]:.2f}, aVNA={best_weights[2]:.2f}")
    print(f"R2: {final_metrics['R2']:.4f}")
    print(f"MAE: {final_metrics['MAE']:.2f}")
    print(f"RMSE: {final_metrics['RMSE']:.2f}")
    print(f"MB: {final_metrics['MB']:.2f}")

    # 保存结果
    result_df = pd.DataFrame([{
        'method': 'EnsembleRK',
        'w1_rk': best_weights[0],
        'w2_evna': best_weights[1],
        'w3_avna': best_weights[2],
        **final_metrics
    }])
    result_df.to_csv(f'{output_dir}/EnsembleRK_summary.csv', index=False)

    weight_df = pd.DataFrame(weight_results)
    weight_df.to_csv(f'{output_dir}/EnsembleRK_weight_search.csv', index=False)

    print(f"\nResults saved to: {output_dir}/")

    return final_metrics, best_weights


if __name__ == '__main__':
    metrics, weights = run_ensemble_rk_ten_fold('2020-01-01')
    print(f"\nFinal: R2={metrics['R2']:.4f}, Weights={weights}")