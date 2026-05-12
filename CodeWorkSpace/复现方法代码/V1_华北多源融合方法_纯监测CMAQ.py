"""
V1_华北多源融合方法_纯监测CMAQ - North China Multi-source Fusion
================================================================
Reproduction of NC method (华北WRF-Chem多源方法)

核心公式:
  P_NC = w * O + (1-w) * M
  等价: P_NC = M + w * (O - M)

  权重 w 是空间位置的函数，通过高斯核平滑确定:
  w(s, s_i) = K(d(s,s_i)) / sum K(d(s_j, s_i))
  K(d) = exp(-d^2 / (2*h^2))

  带宽 h 通过留一法交叉验证选择。
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


class NCFusion:
    """
    华北多源融合方法 (North China WRF-Chem Multi-source Fusion)

    基于贝叶斯加权融合，使用高斯核平滑确定空间变化的权重:
    P(s0) = M(s0) + sum_i w_i * (O_i - M_i)

    w_i 由高斯核函数 K(d) = exp(-d^2/(2h^2)) 决定
    带宽 h 通过交叉验证选择
    """

    def __init__(self, k_neighbors=30, bandwidth=None, power=-2):
        """
        Parameters:
        -----------
        k_neighbors : int
            近邻数量
        bandwidth : float
            高斯核带宽（度），None则自动选择
        power : float
            IDW距离权重指数（备选方案）
        """
        self.k_neighbors = k_neighbors
        self.bandwidth = bandwidth
        self.power = power

    def _gaussian_kernel(self, dist, bandwidth):
        """高斯核函数 K(d) = exp(-d^2 / (2*h^2))"""
        return np.exp(-dist**2 / (2 * bandwidth**2))

    def _select_bandwidth(self, train_coords, train_obs, train_cmaq):
        """
        通过留一法交叉验证选择最优带宽

        目标: min sum_i (O_i - P_NC(x_i))^2
        """
        n = len(train_coords)
        # 候选带宽（度）
        dists = cdist(train_coords, train_coords)
        median_dist = np.median(dists[dists > 0])
        bandwidths = np.linspace(median_dist * 0.5, median_dist * 5.0, 20)

        best_mse = np.inf
        best_bw = bandwidths[len(bandwidths) // 2]

        for bw in bandwidths:
            mse = 0.0
            for i in range(n):
                # 留一: 用除i外的所有站点预测i
                dist_i = dists[i]
                idx = np.argsort(dist_i)
                # 排除自身(idx[0]是自身，距离=0)
                idx = idx[1:self.k_neighbors + 1]
                d_k = dist_i[idx]

                # 高斯核权重
                weights = self._gaussian_kernel(d_k, bw)
                weights = weights / (np.sum(weights) + 1e-10)

                # 偏差插值
                bias_k = train_obs[idx] - train_cmaq[idx]
                pred_i = train_cmaq[i] + np.sum(weights * bias_k)

                mse += (train_obs[i] - pred_i)**2

            mse /= n
            if mse < best_mse:
                best_mse = mse
                best_bw = bw

        self.bandwidth = best_bw
        return best_bw

    def predict(self, train_coords, train_obs, train_cmaq,
                pred_coords, pred_cmaq, select_bandwidth=True):
        """
        预测

        Parameters:
        -----------
        train_coords : array (n, 2) - 训练站点坐标
        train_obs : array (n,) - 训练站点观测值
        train_cmaq : array (n,) - 训练站点CMAQ值
        pred_coords : array (m, 2) - 预测点坐标
        pred_cmaq : array (m,) - 预测点CMAQ值
        select_bandwidth : bool - 是否自动选择带宽

        Returns:
        --------
        pred : array (m,) - 融合预测值
        """
        # 自动选择带宽
        if self.bandwidth is None and select_bandwidth:
            self._select_bandwidth(train_coords, train_obs, train_cmaq)

        n_pred = len(pred_coords)
        pred = np.zeros(n_pred)

        # 计算站点偏差
        bias = train_obs - train_cmaq

        # 计算距离矩阵
        dist_matrix = cdist(pred_coords, train_coords)

        for i in range(n_pred):
            dists = dist_matrix[i]

            # 选择最近的k个站点
            k = min(self.k_neighbors, len(train_coords))
            idx = np.argsort(dists)[:k]
            d_k = np.maximum(dists[idx], 1e-10)
            bias_k = bias[idx]

            # 高斯核权重
            if self.bandwidth is not None and self.bandwidth > 0:
                weights = self._gaussian_kernel(d_k, self.bandwidth)
            else:
                # 退化为IDW
                weights = d_k ** self.power

            weights = weights / (np.sum(weights) + 1e-10)

            # 融合
            pred[i] = pred_cmaq[i] + np.sum(weights * bias_k)

        pred = np.maximum(pred, 0)
        return pred


def run_华北多源融合方法_纯监测CMAQ_ten_fold(selected_day='2020-01-01'):
    """
    运行NC方法十折交叉验证
    """
    print("=" * 60)
    print("NC (North China Multi-source Fusion) Ten-Fold Cross Validation")
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

        train_coords = train_df[['Lon', 'Lat']].values
        test_coords = test_df[['Lon', 'Lat']].values
        train_obs = train_df['Conc'].values
        test_obs = test_df['Conc'].values
        train_cmaq = train_df['CMAQ'].values
        test_cmaq = test_df['CMAQ'].values

        # 训练
        model = NCFusion(k_neighbors=30)
        y_pred = model.predict(
            train_coords, train_obs, train_cmaq,
            test_coords, test_cmaq,
            select_bandwidth=True
        )

        results[fold_id] = {
            'y_true': test_obs,
            'y_pred': y_pred
        }
        print(f"  Fold {fold_id}: n_test={len(test_df)}, bandwidth={model.bandwidth:.4f}")

    # 汇总
    y_pred_all = np.concatenate([results[f]['y_pred'] for f in range(1, 11) if results[f]])
    y_true_all = np.concatenate([results[f]['y_true'] for f in range(1, 11) if results[f]])

    metrics = compute_metrics(y_true_all, y_pred_all)

    print("\n=== Results ===")
    print(f"  NC: R2={metrics['R2']:.4f}, MAE={metrics['MAE']:.2f}, "
          f"RMSE={metrics['RMSE']:.2f}, MB={metrics['MB']:.4f}")

    # 保存结果
    result_df = pd.DataFrame([{
        'method': 'NC_Fusion',
        **metrics
    }])
    result_df.to_csv(f'{output_dir}/NC_Fusion_summary.csv', index=False)
    print(f"\nResults saved to: {output_dir}/NC_Fusion_summary.csv")

    return metrics


if __name__ == '__main__':
    metrics = run_华北多源融合方法_纯监测CMAQ_ten_fold('2020-01-01')
    print(f"\nNC: R2={metrics['R2']:.4f}")
