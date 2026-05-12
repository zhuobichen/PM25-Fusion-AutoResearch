"""
V1_OMA方法_观测模型聚合 - Observational Model Aggregation
=========================================================
Reproduction of Shao et al. Section 2.2.1

核心公式:
  P_OMA(s0) = M(s0) + alpha * [O(si) - M(si)]
  alpha* = argmin sum_i [D(si) - alpha * D(si)]^2
  其中 D(si) = O(si) - M(si)

全局OMA: P_OMA(s0) = M(s0) + alpha * mean(D)
局部OMA: P_OMA(s0) = M(s0) + sum_i w_i * D(si), w_i 基于距离
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


class OMA:
    """
    Observational Model Aggregation (观测模型聚合)

    全局OMA: 优化一个全局alpha，使 P = M + alpha * mean(O-M)
    局部OMA: 对每个网格点用IDW插值偏差 P = M + sum(w_i * (O_i - M_i))
    """

    def __init__(self, method='local', k=30, power=-2):
        """
        Parameters:
        -----------
        method : str
            'global' - 全局权重alpha
            'local' - 局部IDW插值偏差
        k : int
            局部方法的近邻数量
        power : float
            距离权重指数（负值）
        """
        self.method = method
        self.k = k
        self.power = power
        self.alpha = None

    def _optimize_alpha(self, obs, cmaq):
        """
        全局OMA: 优化alpha
        最小化 sum_i [D(si) - alpha * D(si)]^2
        等价于: alpha* = argmin sum_i [(1-alpha)*D(si)]^2
        解: alpha = 1（当D不全为0时），但实际使用如下推导：

        公式 P = M + alpha * mean(D)
        目标: min sum (O_i - M_i - alpha * mean(D))^2
        解: alpha = sum(D_i * mean(D)) / sum(mean(D)^2) = 1 (平凡解)

        改用: min sum (O_i - (alpha*O_i + (1-alpha)*M_i))^2
        => min sum ((1-alpha)*(O_i - M_i))^2
        => alpha = 1 (退化)

        实际文献做法: 用偏差校正形式
        P = M + alpha * D_mean, 通过LOOCV选择最优alpha
        """
        D = obs - cmaq
        D_mean = np.mean(D)

        if abs(D_mean) < 1e-10:
            return 0.0

        # 最小化 sum (O_i - (M_i + alpha * D_mean))^2
        # = sum (D_i - alpha * D_mean)^2
        # d/d(alpha) = -2 * D_mean * sum(D_i - alpha * D_mean) = 0
        # alpha = sum(D_i) / (n * D_mean) = n * D_mean / (n * D_mean) = 1
        # 但这只对训练站点成立，对网格点需要用不同策略

        # 实际使用: alpha 使得全局偏差最小
        alpha = np.sum(D * D_mean) / (np.sum(D_mean**2) * len(D) + 1e-10)
        return np.clip(alpha, 0.0, 1.0)

    def fit_predict_local(self, train_coords, train_obs, train_cmaq,
                          pred_coords, pred_cmaq):
        """
        局部OMA: 对每个预测点用IDW插值站点偏差

        P(s0) = M(s0) + sum_i w_i * (O_i - M_i)
        w_i = (1/d_i^p) / sum(1/d_j^p)
        """
        n_pred = len(pred_coords)
        pred = np.zeros(n_pred)

        # 计算站点偏差
        bias = train_obs - train_cmaq

        # 计算预测点到所有训练点的距离
        dist_matrix = cdist(pred_coords, train_coords)

        for i in range(n_pred):
            dists = dist_matrix[i]

            # 选择最近的k个站点
            k = min(self.k, len(train_coords))
            idx = np.argsort(dists)[:k]
            d_k = dists[idx]
            bias_k = bias[idx]

            # 避免除零
            d_k = np.maximum(d_k, 1e-10)

            # IDW权重
            weights = d_k ** self.power
            weights = weights / np.sum(weights)

            # 加权偏差插值
            pred[i] = pred_cmaq[i] + np.sum(weights * bias_k)

        # 非负约束
        pred = np.maximum(pred, 0)
        return pred

    def fit_predict_global(self, train_obs, train_cmaq, pred_cmaq):
        """
        全局OMA: P(s0) = M(s0) + alpha * mean(O - M)
        """
        bias = train_obs - train_cmaq
        mean_bias = np.mean(bias)
        alpha = self._optimize_alpha(train_obs, train_cmaq)
        self.alpha = alpha

        pred = pred_cmaq + alpha * mean_bias
        pred = np.maximum(pred, 0)
        return pred

    def predict(self, train_coords, train_obs, train_cmaq,
                pred_coords, pred_cmaq):
        if self.method == 'global':
            return self.fit_predict_global(train_obs, train_cmaq, pred_cmaq)
        else:
            return self.fit_predict_local(
                train_coords, train_obs, train_cmaq,
                pred_coords, pred_cmaq
            )


def run_OMA方法_观测模型聚合_ten_fold(selected_day='2020-01-01'):
    """
    运行OMA十折交叉验证
    """
    print("=" * 60)
    print("OMA (Observational Model Aggregation) Ten-Fold Cross Validation")
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

        # 局部OMA
        model = OMA(method='local', k=30, power=-2)
        y_pred = model.predict(
            train_coords, train_obs, train_cmaq,
            test_coords, test_cmaq
        )

        results[fold_id] = {
            'y_true': test_obs,
            'y_pred': y_pred
        }
        print(f"  Fold {fold_id}: n_test={len(test_df)}")

    # 汇总
    y_pred_all = np.concatenate([results[f]['y_pred'] for f in range(1, 11) if results[f]])
    y_true_all = np.concatenate([results[f]['y_true'] for f in range(1, 11) if results[f]])

    metrics = compute_metrics(y_true_all, y_pred_all)

    print("\n=== Results ===")
    print(f"  OMA (local): R2={metrics['R2']:.4f}, MAE={metrics['MAE']:.2f}, "
          f"RMSE={metrics['RMSE']:.2f}, MB={metrics['MB']:.4f}")

    # 保存结果
    result_df = pd.DataFrame([{
        'method': 'OMA_local',
        **metrics
    }])
    result_df.to_csv(f'{output_dir}/OMA_summary.csv', index=False)
    print(f"\nResults saved to: {output_dir}/OMA_summary.csv")

    return metrics


if __name__ == '__main__':
    metrics = run_OMA方法_观测模型聚合_ten_fold('2020-01-01')
    print(f"\nOMA: R2={metrics['R2']:.4f}")
