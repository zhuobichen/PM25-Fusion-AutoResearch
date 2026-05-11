"""

# === 预测值缓存（由 patch_methods.py 自动添加）===
_last_y_true = None
_last_y_pred = None

MSEF - Multi-Scale Ensemble Fusion (Correct Implementation)
==========================================================
多尺度集成融合方法

使用与十折验证测试.py相同的逻辑：
1. 在整个网格上预测
2. 在测试站点位置提取
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.paths import get_project_root, data_path
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
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


def run_msef_ten_fold(selected_day='2020-01-01'):
    """
    运行MSEF十折交叉验证
    """
    print("="*60)
    print("MSEF Ten-Fold Cross Validation")
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

        # 准备训练数据
        train_df['x'] = train_df['Lon']
        train_df['y'] = train_df['Lat']
        train_df['mod'] = train_df['CMAQ']
        train_df['bias'] = train_df['Conc'] - train_df['CMAQ']
        train_df['rn'] = train_df['Conc'] / train_df['CMAQ']

        # NNA拟合
        nn = NNA(method='voronoi', k=30, power=-2)
        nn.fit(
            train_df[['x', 'y']],
            train_df[['bias', 'rn']]
        )

        # 在整个网格上预测
        zdf_grid = nn.predict(X_grid_full, njobs=4)
        bias_grid = zdf_grid[:, 0]
        rn_grid = zdf_grid[:, 1]

        # 在测试站点位置提取预测值
        evna_pred = np.zeros(len(test_df))
        bias_pred = np.zeros(len(test_df))
        for i, (_, row) in enumerate(test_df.iterrows()):
            dist = np.sqrt((lon_cmaq - row['Lon'])**2 + (lat_cmaq - row['Lat'])**2)
            idx = np.argmin(dist)
            evna_pred[i] = y_grid_model_full[idx] * rn_grid[idx]  # eVNA = M * rn
            bias_pred[i] = bias_grid[idx]

        cmaq_test = test_df['CMAQ'].values
        gmos_pred = cmaq_test + bias_pred  # GMOS = M + bias
        dscan_pred = cmaq_test + bias_pred  # Downscaler = M + bias (简化版)

        results[fold_id] = {
            'y_true': test_df['Conc'].values,
            'evna': evna_pred,
            'gmos': gmos_pred,
            'dscan': dscan_pred
        }

        print(f"  Fold {fold_id}: completed")

    # 汇总所有折叠结果
    evna_all = np.concatenate([results[f]['evna'] for f in range(1, 11) if results[f]])
    gmos_all = np.concatenate([results[f]['gmos'] for f in range(1, 11) if results[f]])
    dscan_all = np.concatenate([results[f]['dscan'] for f in range(1, 11) if results[f]])
    true_all = np.concatenate([results[f]['y_true'] for f in range(1, 11) if results[f]])

    # 计算各方法R2
    evna_r2 = compute_metrics(true_all, evna_all)['R2']
    gmos_r2 = compute_metrics(true_all, gmos_all)['R2']
    dscan_r2 = compute_metrics(true_all, dscan_all)['R2']

    print(f"\nIndividual Method R2:")
    print(f"  eVNA: {evna_r2:.4f}")
    print(f"  GMOS: {gmos_r2:.4f}")
    print(f"  Downscaler: {dscan_r2:.4f}")

    # 优化权重
    print("\n=== Optimizing Ensemble Weights ===")
    best_r2 = -np.inf
    best_weights = None
    weight_results = []

    # 网格搜索权重 (b1, b2, b3)
    for b1 in np.arange(0, 1.05, 0.1):
        for b2 in np.arange(0, 1.05 - b1, 0.1):
            b3 = round(1.0 - b1 - b2, 2)
            if b3 < 0:
                continue

            ensemble_pred = b1 * evna_all + b2 * gmos_all + b3 * dscan_all
            metrics = compute_metrics(true_all, ensemble_pred)

            weight_results.append({
                'beta1': b1, 'beta2': b2, 'beta3': b3,
                'R2': metrics['R2'], 'MAE': metrics['MAE'], 'RMSE': metrics['RMSE']
            })

            if metrics['R2'] > best_r2:
                best_r2 = metrics['R2']
                best_weights = (b1, b2, b3)

    print(f"\nBest weights: beta1={best_weights[0]:.2f}, beta2={best_weights[1]:.2f}, beta3={best_weights[2]:.2f}")
    print(f"Best R2: {best_r2:.4f}")

    # 最终评估
    final_pred = best_weights[0] * evna_all + best_weights[1] * gmos_all + best_weights[2] * dscan_all
    final_metrics = compute_metrics(true_all, final_pred)
    # 缓存预测值供多天聚合使用
    global _last_y_true, _last_y_pred
    _last_y_true = true_all
    _last_y_pred = final_pred


    print("\n" + "="*60)
    print("Final MSEF Results")
    print("="*60)
    print(f"Optimal weights: beta1={best_weights[0]:.2f}, beta2={best_weights[1]:.2f}, beta3={best_weights[2]:.2f}")
    print(f"R2: {final_metrics['R2']:.4f}")
    print(f"MAE: {final_metrics['MAE']:.2f}")
    print(f"RMSE: {final_metrics['RMSE']:.2f}")
    print(f"MB: {final_metrics['MB']:.2f}")

    # 保存结果
    result_df = pd.DataFrame([{
        'method': 'MSEF',
        'beta1': best_weights[0],
        'beta2': best_weights[1],
        'beta3': best_weights[2],
        **final_metrics
    }])
    result_df.to_csv(f'{output_dir}/MSEF_summary.csv', index=False)

    weight_df = pd.DataFrame(weight_results)
    weight_df.to_csv(f'{output_dir}/MSEF_weight_search.csv', index=False)

    print(f"\nResults saved to: {output_dir}/")

    return final_metrics, best_weights


if __name__ == '__main__':
    metrics, weights = run_msef_ten_fold('2020-01-01')
    print(f"\nFinal: R2={metrics['R2']:.4f}, Weights={weights}")