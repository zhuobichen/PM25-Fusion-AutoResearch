"""
GWR — 地理加权回归法
====================
(Geographically Weighted Regression)

核心思想：
回归系数不是全局固定的，而是随空间位置变化的。
每个预测位置使用附近的观测数据进行局部加权最小二乘回归，
权重由空间距离的核函数决定。

文献来源：
- "Geographically and temporally weighted neural networks for satellite-based mapping of ground-level PM2.5"
- arXiv: 1809.09860
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
output_dir = f'{root_dir}/test_result/创新方法'
os.makedirs(output_dir, exist_ok=True)


def compute_metrics(y_true, y_pred):
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
    dist = np.sqrt((lon_grid - lon)**2 + (lat_grid - lat)**2)
    idx = np.argmin(dist)
    ny, nx = lon_grid.shape
    row, col = idx // nx, idx % nx
    return pm25_grid[row, col]


class GWR:
    """
    地理加权回归

    在每个预测位置s0，求解局部加权最小二乘：
    β(s0) = (X^T W(s0) X)^{-1} X^T W(s0) Y

    其中W(s0)是对角权重矩阵，对角元素：
    w_j(s0) = exp(-d_{j0}^2 / (2b^2))
    """

    def __init__(self, kernel='gaussian', bandwidth=None,
                 bandwidth_min=0.05, bandwidth_max=5.0, bandwidth_n=30):
        self.kernel = kernel
        self.bandwidth = bandwidth
        self.bandwidth_min = bandwidth_min
        self.bandwidth_max = bandwidth_max
        self.bandwidth_n = bandwidth_n

    def _compute_weights(self, dist, bandwidth):
        """计算高斯核权重"""
        if self.kernel == 'gaussian':
            return np.exp(-dist**2 / (2 * bandwidth**2))
        elif self.kernel == 'bisquare':
            w = np.zeros_like(dist)
            mask = dist < bandwidth
            w[mask] = (1 - (dist[mask] / bandwidth)**2)**2
            return w
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")

    def _compute_aicc(self, X, y, bandwidth, dist_matrix):
        """计算AICc准则"""
        n = len(y)
        y_pred = np.zeros(n)
        trace_S = 0

        for i in range(n):
            weights = self._compute_weights(dist_matrix[i], bandwidth)
            W = np.diag(weights)

            # 局部加权最小二乘
            try:
                XtWX = X.T @ W @ X
                XtWy = X.T @ W @ y
                beta = np.linalg.solve(XtWX + 1e-8 * np.eye(X.shape[1]), XtWy)
                y_pred[i] = X[i] @ beta

                # 计算hat matrix的对角元素
                hat_row = X[i] @ np.linalg.solve(XtWX + 1e-8 * np.eye(X.shape[1]), X.T @ W)
                trace_S += hat_row[i]
            except np.linalg.LinAlgError:
                y_pred[i] = np.mean(y)

        # 计算sigma_hat
        residuals = y - y_pred
        sigma_hat = np.sqrt(np.sum(residuals**2) / n)

        # AICc
        if n - 2 - trace_S > 0:
            aicc = 2 * n * np.log(sigma_hat + 1e-10) + n * np.log(2 * np.pi) + \
                   (n + trace_S) / (n - 2 - trace_S)
        else:
            aicc = np.inf

        return aicc

    def fit(self, X_train, m_train, y_train):
        """
        训练GWR模型（选择最优带宽）

        Parameters:
        -----------
        X_train : array (n, 2) - 站点坐标 [lon, lat]
        m_train : array (n,) - CMAQ站点值
        y_train : array (n,) - 监测值
        """
        self.X_train = X_train
        self.m_train = m_train
        self.y_train = y_train

        # 构建设计矩阵 [1, CMAQ]
        self.X_design = np.column_stack([np.ones(len(m_train)), m_train])

        # 计算距离矩阵
        self.dist_matrix = cdist(X_train, X_train)

        # 如果没有指定带宽，搜索最优带宽
        if self.bandwidth is None:
            self._select_bandwidth()

        return self

    def _select_bandwidth(self):
        """使用AICc选择最优带宽"""
        # 候选带宽
        bandwidths = np.linspace(self.bandwidth_min, self.bandwidth_max, self.bandwidth_n)

        best_aicc = np.inf
        best_bw = bandwidths[len(bandwidths)//2]

        for bw in bandwidths:
            aicc = self._compute_aicc(self.X_design, self.y_train, bw, self.dist_matrix)
            if aicc < best_aicc:
                best_aicc = aicc
                best_bw = bw

        self.bandwidth = best_bw

    def predict(self, X_test, m_test):
        """
        预测

        Parameters:
        -----------
        X_test : array (m, 2) - 预测点坐标
        m_test : array (m,) - 预测点CMAQ值

        Returns:
        --------
        y_pred : array (m,) - 融合预测值
        """
        # 计算预测点到训练点的距离
        dist_test = cdist(X_test, self.X_train)

        n_pred = len(X_test)
        y_pred = np.zeros(n_pred)

        for i in range(n_pred):
            # 计算权重
            weights = self._compute_weights(dist_test[i], self.bandwidth)
            W = np.diag(weights)

            # 局部加权最小二乘
            try:
                XtWX = self.X_design.T @ W @ self.X_design
                XtWy = self.X_design.T @ W @ self.y_train
                beta = np.linalg.solve(XtWX + 1e-8 * np.eye(self.X_design.shape[1]), XtWy)

                # 预测
                x_new = np.array([1, m_test[i]])
                y_pred[i] = x_new @ beta
            except np.linalg.LinAlgError:
                y_pred[i] = np.mean(self.y_train)

        # 非负约束
        y_pred = np.maximum(y_pred, 0)

        return y_pred


def run_gwr_ten_fold(selected_day='2020-01-01'):
    """
    运行GWR十折交叉验证
    """
    print("="*60)
    print("GWR Ten-Fold Cross Validation")
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

        # 训练模型
        model = GWR(
            kernel='gaussian',
            bandwidth_min=0.05,
            bandwidth_max=3.0,
            bandwidth_n=20
        )
        model.fit(X_train, m_train, y_train)

        # 预测
        y_pred = model.predict(X_test, m_test)

        results[fold_id] = {
            'y_true': y_test,
            'y_pred': y_pred
        }

        print(f"  Fold {fold_id}: bandwidth={model.bandwidth:.3f}")

    # 汇总
    y_pred_all = np.concatenate([results[f]['y_pred'] for f in range(1, 11) if results[f]])
    y_true_all = np.concatenate([results[f]['y_true'] for f in range(1, 11) if results[f]])

    metrics = compute_metrics(y_true_all, y_pred_all)

    print("\n=== Results ===")
    print(f"  GWR: R2={metrics['R2']:.4f}, MAE={metrics['MAE']:.2f}, RMSE={metrics['RMSE']:.2f}, MB={metrics['MB']:.4f}")

    # 保存结果
    result_df = pd.DataFrame([{
        'method': 'GWR',
        **metrics
    }])
    result_df.to_csv(f'{output_dir}/GWR_summary.csv', index=False)

    print(f"\nResults saved to: {output_dir}/")

    return metrics


if __name__ == '__main__':
    metrics = run_gwr_ten_fold('2020-01-01')
    print(f"\nGWR: R2={metrics['R2']:.4f}")
