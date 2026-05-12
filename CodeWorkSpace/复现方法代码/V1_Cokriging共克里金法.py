"""
V1_Cokriging共克里金法 - Cokriging for Multi-variable Spatial Prediction
=======================================================================
Reproduction of Singh et al. (2011) Cokriging method

核心公式:
  共克里金估计:
  U_hat(s0) = sum_i lambda_i^U * U(si) + sum_j lambda_j^V * V(sj)

  无偏约束:
  sum(lambda_i^U) = 1, sum(lambda_j^V) = 0

  方程组:
  [C_UU  C_UV  1 0] [lambda^U]   [C_UU(s0)]
  [C_VU  C_VV  0 1] [lambda^V] = [C_VV(s0)]
  [1^T   0^T   0 0] [mu_1    ]   [1       ]
  [0^T   1^T   0 0] [mu_2    ]   [0       ]

  其中U=监测PM2.5（主变量），V=CMAQ（辅助变量）
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


def exponential_variogram(h, nugget, sill, range_param):
    """
    指数变异函数: gamma(h) = nugget + sill * (1 - exp(-3h/range))
    """
    return nugget + sill * (1.0 - np.exp(-3.0 * h / range_param))


def fit_variogram(coords, values, n_bins=15):
    """
    从数据拟合变异函数参数

    Parameters:
    -----------
    coords : array (n, 2) - 坐标
    values : array (n,) - 变量值
    n_bins : int - 距离分箱数

    Returns:
    --------
    nugget, sill, range_param : 变异函数参数
    """
    n = len(coords)
    if n < 3:
        return 0.1, np.var(values), 1.0

    # 计算实验变异函数
    dists = cdist(coords, coords)
    upper_idx = np.triu_indices(n, k=1)
    h_vals = dists[upper_idx]

    # 半方差
    gamma_vals = 0.5 * (values[upper_idx[0]] - values[upper_idx[1]])**2

    # 分箱
    h_max = np.max(h_vals) * 0.5
    bins = np.linspace(0, h_max, n_bins + 1)
    h_bin = []
    gamma_bin = []

    for k in range(n_bins):
        mask = (h_vals >= bins[k]) & (h_vals < bins[k + 1])
        if np.sum(mask) > 0:
            h_bin.append(np.mean(h_vals[mask]))
            gamma_bin.append(np.mean(gamma_vals[mask]))

    if len(h_bin) < 3:
        return 0.1, np.var(values), 1.0

    h_bin = np.array(h_bin)
    gamma_bin = np.array(gamma_bin)

    # 简单参数估计
    nugget_est = gamma_bin[0] if gamma_bin[0] > 0 else 0.1
    sill_est = np.max(gamma_bin)
    # 找到gamma首次达到0.95*sill的距离作为变程
    range_est = h_bin[-1]
    for k in range(len(gamma_bin)):
        if gamma_bin[k] >= 0.95 * sill_est:
            range_est = h_bin[k]
            break

    range_est = max(range_est, 0.1)
    sill_est = max(sill_est - nugget_est, 0.1)

    return nugget_est, sill_est, range_est


class Cokriging:
    """
    共克里金法 (Cokriging)

    利用主变量（监测PM2.5）和辅助变量（CMAQ）的空间互相关性
    进行联合插值预测。
    """

    def __init__(self, n_neighbors_obs=15, n_neighbors_cmaq=30):
        """
        Parameters:
        -----------
        n_neighbors_obs : int
            主变量（监测）近邻数量
        n_neighbors_cmaq : int
            辅助变量（CMAQ）近邻数量
        """
        self.n_neighbors_obs = n_neighbors_obs
        self.n_neighbors_cmaq = n_neighbors_cmaq

    def fit(self, U_obs, V_obs, coords_obs):
        """
        拟合变异函数

        Parameters:
        -----------
        U_obs : array (n,) - 主变量（监测PM2.5）
        V_obs : array (n,) - 辅助变量在观测点的值（CMAQ）
        coords_obs : array (n, 2) - 观测坐标
        """
        # 转换为km
        self.coords_obs = coords_obs
        self.U_obs = U_obs
        self.V_obs = V_obs

        # 拟合主变量变异函数
        self.nugget_U, self.sill_U, self.range_U = fit_variogram(
            coords_obs, U_obs
        )

        # 拟合辅助变量变异函数
        self.nugget_V, self.sill_V, self.range_V = fit_variogram(
            coords_obs, V_obs
        )

        # 拟合交叉变异函数（使用差值）
        cross_diff = U_obs - V_obs
        self.nugget_UV, self.sill_UV, self.range_UV = fit_variogram(
            coords_obs, cross_diff
        )

        return self

    def predict(self, V_pred, coords_pred):
        """
        共克里金预测

        Parameters:
        -----------
        V_pred : array (m,) - 辅助变量在预测点的值（CMAQ）
        coords_pred : array (m, 2) - 预测坐标

        Returns:
        --------
        pred : array (m,) - 共克里金估计
        """
        n_pred = len(coords_pred)
        n_obs = len(self.U_obs)
        pred = np.zeros(n_pred)

        for i in range(n_pred):
            # 计算到所有观测点的距离
            dist_to_obs = np.sqrt(
                np.sum((self.coords_obs - coords_pred[i])**2, axis=1)
            )

            # 选择最近邻
            n_use = min(self.n_neighbors_obs, n_obs)
            idx = np.argsort(dist_to_obs)[:n_use]

            coords_use = self.coords_obs[idx]
            U_use = self.U_obs[idx]
            V_use = self.V_obs[idx]
            dist_pred = dist_to_obs[idx]

            # 站点间距离
            dist_obs = cdist(coords_use, coords_use)

            # 转换为km（粗略）
            km_per_deg = 111.0

            # 构建协方差矩阵
            # C_UU: 主变量自身协方差
            gamma_UU = exponential_variogram(
                dist_obs * km_per_deg,
                self.nugget_U, self.sill_U, self.range_U
            )
            C_UU = (self.sill_U + self.nugget_U) - gamma_UU

            # C_VV: 辅助变量自身协方差
            gamma_VV = exponential_variogram(
                dist_obs * km_per_deg,
                self.nugget_V, self.sill_V, self.range_V
            )
            C_VV = (self.sill_V + self.nugget_V) - gamma_VV

            # C_UV: 互协方差
            gamma_UV = exponential_variogram(
                dist_obs * km_per_deg,
                self.nugget_UV, self.sill_UV, self.range_UV
            )
            C_UV = (self.sill_UV + self.nugget_UV) - gamma_UV

            # 预测点协方差向量
            gamma_UU_pred = exponential_variogram(
                dist_pred * km_per_deg,
                self.nugget_U, self.sill_U, self.range_U
            )
            c_UU = (self.sill_U + self.nugget_U) - gamma_UU_pred

            gamma_UV_pred = exponential_variogram(
                dist_pred * km_per_deg,
                self.nugget_UV, self.sill_UV, self.range_UV
            )
            c_UV = (self.sill_UV + self.nugget_UV) - gamma_UV_pred

            # 构建共克里金方程组
            # [C_UU  C_UV  1 0] [lambda_U]   [c_UU(s0)]
            # [C_UV^T C_VV 0 1] [lambda_V] = [c_UV(s0)]
            # [1^T   0^T   0 0] [mu_1    ]   [1       ]
            # [0^T   1^T   0 0] [mu_2    ]   [0       ]
            n = n_use
            A = np.zeros((2 * n + 2, 2 * n + 2))
            A[:n, :n] = C_UU
            A[:n, n:2*n] = C_UV
            A[n:2*n, :n] = C_UV.T
            A[n:2*n, n:2*n] = C_VV

            # 无偏约束
            A[:n, 2*n] = 1.0
            A[n:2*n, 2*n+1] = 1.0
            A[2*n, :n] = 1.0
            A[2*n+1, n:2*n] = 1.0

            # 右端向量
            b = np.zeros(2 * n + 2)
            b[:n] = c_UU
            b[n:2*n] = c_UV
            b[2*n] = 1.0
            b[2*n+1] = 0.0

            # 添加正则化
            A += 1e-6 * np.eye(2 * n + 2)

            # 求解
            try:
                x = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                x = np.linalg.lstsq(A, b, rcond=None)[0]

            lambda_U = x[:n]
            lambda_V = x[n:2*n]

            # 共克里金估计
            pred[i] = np.dot(lambda_U, U_use) + np.dot(lambda_V, V_use)

        pred = np.maximum(pred, 0)
        return pred


def run_Cokriging共克里金法_ten_fold(selected_day='2020-01-01'):
    """
    运行Cokriging十折交叉验证
    """
    print("=" * 60)
    print("Cokriging Ten-Fold Cross Validation")
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

        # 训练Cokriging
        model = Cokriging(n_neighbors_obs=15, n_neighbors_cmaq=30)
        model.fit(train_obs, train_cmaq, train_coords)

        # 预测
        y_pred = model.predict(test_cmaq, test_coords)

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
    print(f"  Cokriging: R2={metrics['R2']:.4f}, MAE={metrics['MAE']:.2f}, "
          f"RMSE={metrics['RMSE']:.2f}, MB={metrics['MB']:.4f}")

    # 保存结果
    result_df = pd.DataFrame([{
        'method': 'Cokriging',
        **metrics
    }])
    result_df.to_csv(f'{output_dir}/Cokriging_summary.csv', index=False)
    print(f"\nResults saved to: {output_dir}/Cokriging_summary.csv")

    return metrics


if __name__ == '__main__':
    metrics = run_Cokriging共克里金法_ten_fold('2020-01-01')
    print(f"\nCokriging: R2={metrics['R2']:.4f}")
