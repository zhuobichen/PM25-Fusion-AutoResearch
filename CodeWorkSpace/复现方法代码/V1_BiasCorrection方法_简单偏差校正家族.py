"""
V1_BiasCorrection方法_简单偏差校正家族 - Bias Correction Family
==============================================================
Reproduction of multiple bias correction methods

四种偏差校正方法:
  1. Mean BC:   P = M + mean(O - M)
  2. Spatial BC: P(s0) = M(s0) + IDW_interpolated_bias(s0)
  3. Scale BC:  P = M * mean(O / M)
  4. Linear BC: P = a + b*M  (OLS拟合)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.paths import get_project_root, data_path
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
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


class BiasCorrection:
    """
    偏差校正方法家族

    支持四种变体:
    - 'mean':    全局均值偏差校正
    - 'spatial': 空间IDW偏差校正
    - 'scale':   缩放偏差校正
    - 'linear':  线性OLS偏差校正
    """

    def __init__(self, method='spatial', k=30, power=-2):
        """
        Parameters:
        -----------
        method : str
            'mean', 'spatial', 'scale', 'linear'
        k : int
            近邻数量（spatial方法）
        power : float
            距离权重指数（spatial方法）
        """
        self.method = method
        self.k = k
        self.power = power
        self.mean_bias = None
        self.mean_ratio = None
        self.a = None
        self.b = None

    def fit(self, train_obs, train_cmaq, train_coords=None):
        """
        拟合偏差校正参数

        Parameters:
        -----------
        train_obs : array (n,) - 监测站观测值
        train_cmaq : array (n,) - CMAQ模型值
        train_coords : array (n, 2) - 站点坐标（spatial方法需要）
        """
        if self.method == 'mean':
            # Mean BC: mean bias = mean(O - M)
            self.mean_bias = np.mean(train_obs - train_cmaq)

        elif self.method == 'scale':
            # Scale BC: mean ratio = mean(O / M)
            # 避免除零
            valid = train_cmaq > 1e-10
            self.mean_ratio = np.mean(train_obs[valid] / train_cmaq[valid])

        elif self.method == 'linear':
            # Linear BC: O = a + b*M
            M = train_cmaq.reshape(-1, 1)
            model = LinearRegression()
            model.fit(M, train_obs)
            self.a = model.intercept_
            self.b = model.coef_[0]

        elif self.method == 'spatial':
            # Spatial BC: 存储训练数据用于IDW插值
            self.train_obs = train_obs
            self.train_cmaq = train_cmaq
            self.train_coords = train_coords
            self.train_bias = train_obs - train_cmaq

        return self

    def predict(self, pred_cmaq, pred_coords=None):
        """
        预测

        Parameters:
        -----------
        pred_cmaq : array (m,) - 预测点CMAQ值
        pred_coords : array (m, 2) - 预测点坐标（spatial方法需要）

        Returns:
        --------
        pred : array (m,) - 融合预测值
        """
        if self.method == 'mean':
            # P = M + mean_bias
            pred = pred_cmaq + self.mean_bias

        elif self.method == 'scale':
            # P = M * mean_ratio
            pred = pred_cmaq * self.mean_ratio

        elif self.method == 'linear':
            # P = a + b*M
            pred = self.a + self.b * pred_cmaq

        elif self.method == 'spatial':
            # P(s0) = M(s0) + IDW_bias(s0)
            n_pred = len(pred_coords)
            pred = np.zeros(n_pred)

            dist_matrix = cdist(pred_coords, self.train_coords)

            for i in range(n_pred):
                dists = dist_matrix[i]
                k = min(self.k, len(self.train_coords))
                idx = np.argsort(dists)[:k]
                d_k = np.maximum(dists[idx], 1e-10)
                bias_k = self.train_bias[idx]

                weights = d_k ** self.power
                weights = weights / np.sum(weights)
                pred[i] = pred_cmaq[i] + np.sum(weights * bias_k)

        pred = np.maximum(pred, 0)
        return pred


def run_BiasCorrection方法_简单偏差校正家族_ten_fold(selected_day='2020-01-01'):
    """
    运行所有偏差校正方法的十折交叉验证
    """
    print("=" * 60)
    print("Bias Correction Family Ten-Fold Cross Validation")
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

    # 对每种偏差校正方法运行十折验证
    bc_methods = ['mean', 'spatial', 'scale', 'linear']
    all_results = {}

    for bc_method in bc_methods:
        print(f"\n--- {bc_method.upper()} Bias Correction ---")
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
            model = BiasCorrection(method=bc_method, k=30, power=-2)
            model.fit(train_obs, train_cmaq, train_coords)

            # 预测
            if bc_method == 'spatial':
                y_pred = model.predict(test_cmaq, test_coords)
            else:
                y_pred = model.predict(test_cmaq)

            results[fold_id] = {
                'y_true': test_obs,
                'y_pred': y_pred
            }

        # 汇总
        y_pred_all = np.concatenate([results[f]['y_pred'] for f in range(1, 11) if results[f]])
        y_true_all = np.concatenate([results[f]['y_true'] for f in range(1, 11) if results[f]])

        metrics = compute_metrics(y_true_all, y_pred_all)
        all_results[bc_method] = metrics

        print(f"  {bc_method}: R2={metrics['R2']:.4f}, MAE={metrics['MAE']:.2f}, "
              f"RMSE={metrics['RMSE']:.2f}, MB={metrics['MB']:.4f}")

    # 保存结果
    result_list = []
    for method_name, metrics in all_results.items():
        result_list.append({
            'method': f'BiasCorrection_{method_name}',
            **metrics
        })
    result_df = pd.DataFrame(result_list)
    result_df.to_csv(f'{output_dir}/BiasCorrection_summary.csv', index=False)
    print(f"\nResults saved to: {output_dir}/BiasCorrection_summary.csv")

    # 返回最佳方法的结果
    best_method = max(all_results.keys(), key=lambda x: all_results[x]['R2'])
    return all_results[best_method]


if __name__ == '__main__':
    metrics = run_BiasCorrection方法_简单偏差校正家族_ten_fold('2020-01-01')
    print(f"\nBest BiasCorrection: R2={metrics['R2']:.4f}")
