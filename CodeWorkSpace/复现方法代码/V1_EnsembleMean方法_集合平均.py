"""
V1_EnsembleMean方法_集合平均 - Ensemble Mean
============================================
Reproduction of ensemble-based PM2.5 fusion

核心公式:
  集合平均: M_bar = mean(M_1, ..., M_Ne)
  带偏差校正: P_EM-BC(s0) = M_bar(s0) + sum_i w_i * (O(si) - M_bar(si))
  缩放形式:   P_EM-SC(s0) = M_bar(s0) * sum(w_i*O_i) / sum(w_i*M_bar_i)
  加权集合:   P_WEM(s0) = sum_k w_k * M_k(s0)

当Ne=1时退化为aVNA。
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.paths import get_project_root, data_path
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.spatial.distance import cdist
import netCDF4 as nc

root_dir = str(get_project_root())
cmaq_file = data_path('test_data/raw/CMAQ/2020_PM25.nc')
monitor_file = data_path('test_data/raw/Monitor/2020_DailyPM2.5Monitor.csv')
fold_file = data_path('test_data/fold_split_table_daily.csv')
output_dir = f'{root_dir}/test_result/复现方法'
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


class EnsembleMean:
    """
    集合平均融合方法

    当仅有单个CMAQ成员时（Ne=1），退化为aVNA加法偏差校正:
    P(s0) = M(s0) + sum_i w_i * (O_i - M_i)

    支持三种模式:
    - 'mean': 简单集合平均 + IDW偏差校正
    - 'weighted': 加权集合平均（基于历史表现）
    - 'bias_corrected': 带偏差校正的集合平均
    """

    def __init__(self, method='bias_corrected', k=30, power=-2):
        """
        Parameters:
        -----------
        method : str
            'mean', 'weighted', 'bias_corrected'
        k : int
            近邻数量
        power : float
            距离权重指数
        """
        self.method = method
        self.k = k
        self.power = power

    def predict(self, train_coords, train_obs, train_cmaq,
                pred_coords, pred_cmaq):
        """
        预测

        Parameters:
        -----------
        train_coords : array (n, 2) - 训练站点坐标
        train_obs : array (n,) - 训练站点观测值
        train_cmaq : array (n,) - 训练站点CMAQ值（单成员）
        pred_coords : array (m, 2) - 预测点坐标
        pred_cmaq : array (m,) - 预测点CMAQ值

        Returns:
        --------
        pred : array (m,) - 融合预测值
        """
        n_pred = len(pred_coords)
        pred = np.zeros(n_pred)

        # 计算站点偏差
        bias = train_obs - train_cmaq

        # 计算距离矩阵
        dist_matrix = cdist(pred_coords, train_coords)

        for i in range(n_pred):
            dists = dist_matrix[i]

            # 选择最近的k个站点
            k = min(self.k, len(train_coords))
            idx = np.argsort(dists)[:k]
            d_k = np.maximum(dists[idx], 1e-10)
            bias_k = bias[idx]
            obs_k = train_obs[idx]
            cmaq_k = train_cmaq[idx]

            # IDW权重
            weights = d_k ** self.power
            weights = weights / np.sum(weights)

            if self.method == 'mean':
                # 简单平均偏差校正
                pred[i] = pred_cmaq[i] + np.mean(bias_k)

            elif self.method == 'bias_corrected':
                # 加权偏差校正 (aVNA形式)
                pred[i] = pred_cmaq[i] + np.sum(weights * bias_k)

            elif self.method == 'weighted':
                # 缩放形式
                weighted_obs = np.sum(weights * obs_k)
                weighted_cmaq = np.sum(weights * cmaq_k)
                if weighted_cmaq > 1e-10:
                    pred[i] = pred_cmaq[i] * (weighted_obs / weighted_cmaq)
                else:
                    pred[i] = pred_cmaq[i]

        pred = np.maximum(pred, 0)
        return pred


def run_EnsembleMean方法_集合平均_ten_fold(selected_day='2020-01-01'):
    """
    运行EnsembleMean十折交叉验证
    """
    print("=" * 60)
    print("Ensemble Mean Ten-Fold Cross Validation")
    print("=" * 60)

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

    # 运行十折验证（三种方法）
    em_methods = ['bias_corrected', 'mean', 'weighted']
    all_results = {}

    for em_method in em_methods:
        print(f"\n--- EnsembleMean ({em_method}) ---")
        results = {fold_id: {} for fold_id in range(1, 11)}

        for fold_id in range(1, 11):
            train_df = day_df[day_df['fold'] != fold_id].copy()
            test_df = day_df[day_df['fold'] == fold_id].copy()

            train_df = train_df.dropna(subset=['Lon', 'Lat', 'CMAQ', 'Conc'])
            test_df = test_df.dropna(subset=['Lon', 'Lat', 'CMAQ', 'Conc'])

            if len(test_df) == 0:
                continue

            train_coords = train_df[['Lon', 'Lat']].values
            test_coords = test_df[['Lon', 'Lat']].values
            train_obs = train_df['Conc'].values
            test_obs = test_df['Conc'].values
            train_cmaq = train_df['CMAQ'].values
            test_cmaq = test_df['CMAQ'].values

            # 训练
            model = EnsembleMean(method=em_method, k=30, power=-2)

            # 预测
            y_pred = model.predict(
                train_coords, train_obs, train_cmaq,
                test_coords, test_cmaq
            )

            results[fold_id] = {
                'y_true': test_obs,
                'y_pred': y_pred
            }

        # 汇总
        y_pred_all = np.concatenate([results[f]['y_pred'] for f in range(1, 11) if results[f]])
        y_true_all = np.concatenate([results[f]['y_true'] for f in range(1, 11) if results[f]])

        metrics = compute_metrics(y_true_all, y_pred_all)
        all_results[em_method] = metrics

        print(f"  {em_method}: R2={metrics['R2']:.4f}, MAE={metrics['MAE']:.2f}, "
              f"RMSE={metrics['RMSE']:.2f}, MB={metrics['MB']:.4f}")

    # 保存结果
    result_list = []
    for method_name, metrics in all_results.items():
        result_list.append({
            'method': f'EnsembleMean_{method_name}',
            **metrics
        })
    result_df = pd.DataFrame(result_list)
    result_df.to_csv(f'{output_dir}/EnsembleMean_summary.csv', index=False)
    print(f"\nResults saved to: {output_dir}/EnsembleMean_summary.csv")

    # 返回最佳方法的结果
    best_method = max(all_results.keys(), key=lambda x: all_results[x]['R2'])
    return all_results[best_method]


if __name__ == '__main__':
    metrics = run_EnsembleMean方法_集合平均_ten_fold('2020-01-01')
    print(f"\nBest EnsembleMean: R2={metrics['R2']:.4f}")
