"""
V1_MMA方法_混合模型聚合 - Mixed Model Aggregation
=================================================
Reproduction of Shao et al. Section 2.2.3

核心公式:
  P_MMA(s0) = M(s0) + beta * D_SMA(s0) + (1-beta) * D_VNA(s0)

其中:
  D_SMA(s0) = a_hat + (b_hat - 1) * M(s0)   (全局SMA偏差)
  D_VNA(s0) = sum_i w_i * (O(si) - M(si))   (局部IDW偏差)
  beta: 混合参数，0=局部主导，1=全局主导
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


class MMA:
    """
    Mixed Model Aggregation (混合模型聚合)

    结合SMA的全局统计校正和局部IDW偏差插值:
    P_MMA(s0) = M(s0) + beta * D_SMA(s0) + (1-beta) * D_VNA(s0)

    当beta=0: 退化为aVNA（纯局部）
    当beta=1: 退化为SMA（纯全局）
    """

    def __init__(self, beta=0.5, k=30, power=-2):
        """
        Parameters:
        -----------
        beta : float
            混合参数 (0=局部aVNA主导, 1=全局SMA主导)
        k : int
            近邻数量
        power : float
            距离权重指数
        """
        self.beta = beta
        self.k = k
        self.power = power
        self.a = None
        self.b = None

    def fit(self, train_obs, train_cmaq):
        """
        拟合全局OLS回归 O = a + b*M

        Parameters:
        -----------
        train_obs : array (n,) - 监测站观测值
        train_cmaq : array (n,) - CMAQ模型值
        """
        M = train_cmaq.reshape(-1, 1)
        model = LinearRegression()
        model.fit(M, train_obs)
        self.a = model.intercept_
        self.b = model.coef_[0]
        return self

    def predict(self, train_coords, train_obs, train_cmaq,
                pred_coords, pred_cmaq):
        """
        预测

        Parameters:
        -----------
        train_coords : array (n, 2) - 训练站点坐标
        train_obs : array (n,) - 训练站点观测值
        train_cmaq : array (n,) - 训练站点CMAQ值
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
            # 全局SMA偏差: D_SMA = a + (b-1)*M
            D_SMA = self.a + (self.b - 1) * pred_cmaq[i]

            # 局部IDW偏差: D_VNA = sum(w_j * bias_j)
            dists = dist_matrix[i]
            k = min(self.k, len(train_coords))
            idx = np.argsort(dists)[:k]
            d_k = np.maximum(dists[idx], 1e-10)
            bias_k = bias[idx]

            weights = d_k ** self.power
            weights = weights / np.sum(weights)
            D_VNA = np.sum(weights * bias_k)

            # 混合融合
            pred[i] = pred_cmaq[i] + self.beta * D_SMA + (1 - self.beta) * D_VNA

        pred = np.maximum(pred, 0)
        return pred


def run_MMA方法_混合模型聚合_ten_fold(selected_day='2020-01-01'):
    """
    运行MMA十折交叉验证
    """
    print("=" * 60)
    print("MMA (Mixed Model Aggregation) Ten-Fold Cross Validation")
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

        # 训练MMA
        model = MMA(beta=0.5, k=30, power=-2)
        model.fit(train_obs, train_cmaq)

        # 预测
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
    print(f"  MMA (beta=0.5): R2={metrics['R2']:.4f}, MAE={metrics['MAE']:.2f}, "
          f"RMSE={metrics['RMSE']:.2f}, MB={metrics['MB']:.4f}")

    # 保存结果
    result_df = pd.DataFrame([{
        'method': 'MMA_beta0.5',
        **metrics
    }])
    result_df.to_csv(f'{output_dir}/MMA_summary.csv', index=False)
    print(f"\nResults saved to: {output_dir}/MMA_summary.csv")

    return metrics


if __name__ == '__main__':
    metrics = run_MMA方法_混合模型聚合_ten_fold('2020-01-01')
    print(f"\nMMA: R2={metrics['R2']:.4f}")
