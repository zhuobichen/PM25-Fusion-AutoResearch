"""
V1_OptimumInterpolation方法_最优插值 - Optimum Interpolation
============================================================
Reproduction of OI method (Section 2.3)

核心公式:
  分析公式: P_a(s0) = P_b(s0) + sum_i w_i * (O(si) - P_b(si))
  权重求解: w = B^{-1} * h
  其中 B_ij = C_P(si,sj) + eps_ij (协方差+误差)
        h_i = C_P(s0, si) (背景场协方差)

与Kriging数学等价，当背景场协方差已知时。
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


def exponential_covariance(d, sill, range_param, nugget=0.0):
    """
    指数协方差函数
    C(h) = sill * exp(-3h/range) + nugget * I(h=0)
    """
    cov = sill * np.exp(-3.0 * d / range_param)
    # 块金效应仅在h=0时
    if nugget > 0:
        cov = cov + nugget * (d < 1e-10).astype(float)
    return cov


class OptimumInterpolation:
    """
    最优插值法 (OI)

    P_a(s0) = P_b(s0) + w^T * (O - P_b)

    通过协方差矩阵求解最优权重:
    B * w = h
    其中 B_ij = C(si,sj) + eps_ij
          h_i = C(s0,si)
    """

    def __init__(self, method='local', n_neighbors=30,
                 corr_scale=None, obs_error_var=None):
        """
        Parameters:
        -----------
        method : str
            'local' 使用最近邻构建局地化
        n_neighbors : int
            近邻数量
        corr_scale : float
            空间相关尺度（km），None则自动估计
        obs_error_var : float
            观测误差方差，None则自动估计
        """
        self.method = method
        self.n_neighbors = n_neighbors
        self.corr_scale = corr_scale
        self.obs_error_var = obs_error_var

    def _estimate_parameters(self, train_coords, residuals):
        """从数据中估计协方差参数"""
        n = len(train_coords)

        # 估计观测误差方差（残差方差）
        if self.obs_error_var is None:
            self.obs_error_var = np.var(residuals) * 0.1  # 假设10%为观测误差

        # 估计空间相关尺度
        if self.corr_scale is None:
            # 使用站点间平均距离作为参考
            dists = cdist(train_coords, train_coords)
            # 取非零距离的中位数
            upper_tri = dists[np.triu_indices(n, k=1)]
            self.corr_scale = np.median(upper_tri) * 2.0

        # 估计模型场方差
        self.model_var = np.var(residuals) - self.obs_error_var
        self.model_var = max(self.model_var, 1.0)

    def fit_predict(self, train_coords, train_obs, train_cmaq,
                    pred_coords, pred_cmaq):
        """
        OI分析

        Parameters:
        -----------
        train_coords : array (n, 2) - 训练站点坐标
        train_obs : array (n,) - 训练站点观测值
        train_cmaq : array (n,) - 训练站点CMAQ值（背景场）
        pred_coords : array (m, 2) - 预测点坐标
        pred_cmaq : array (m,) - 预测点CMAQ值（背景场）

        Returns:
        --------
        pred : array (m,) - 分析场（融合预测值）
        """
        residuals = train_obs - train_cmaq

        # 估计参数
        self._estimate_parameters(train_coords, residuals)

        n_pred = len(pred_coords)
        pred = np.zeros(n_pred)

        for i in range(n_pred):
            # 计算预测点到训练点的距离
            dist_to_pred = np.sqrt(
                np.sum((train_coords - pred_coords[i])**2, axis=1)
            )

            # 局地化: 选择最近的n_neighbors个站点
            n_use = min(self.n_neighbors, len(train_coords))
            idx = np.argsort(dist_to_pred)[:n_use]

            coords_use = train_coords[idx]
            resid_use = residuals[idx]
            cmaq_use = train_cmaq[idx]
            dist_pred_use = dist_to_pred[idx]

            # 计算站点间距离矩阵
            dist_obs = cdist(coords_use, coords_use)

            # 构建协方差矩阵 B = C_obs + eps*I
            # 将度转换为km (粗略)
            km_per_deg = 111.0
            C_obs = exponential_covariance(
                dist_obs * km_per_deg,
                self.model_var, self.corr_scale
            )
            B = C_obs + self.obs_error_var * np.eye(n_use)

            # 构建h向量: 预测点与观测点的协方差
            h = exponential_covariance(
                dist_pred_use * km_per_deg,
                self.model_var, self.corr_scale
            )

            # 求解权重: B * w = h
            try:
                w = np.linalg.solve(B, h)
            except np.linalg.LinAlgError:
                # 如果矩阵奇异，使用伪逆
                w = np.linalg.lstsq(B, h, rcond=None)[0]

            # OI分析: P_a = P_b + w^T * (O - P_b)
            pred[i] = pred_cmaq[i] + np.dot(w, resid_use)

        pred = np.maximum(pred, 0)
        return pred


def run_OptimumInterpolation方法_最优插值_ten_fold(selected_day='2020-01-01'):
    """
    运行OI十折交叉验证
    """
    print("=" * 60)
    print("Optimum Interpolation Ten-Fold Cross Validation")
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

        # 训练OI
        model = OptimumInterpolation(
            method='local',
            n_neighbors=30
        )

        # 预测
        y_pred = model.fit_predict(
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
    print(f"  OI: R2={metrics['R2']:.4f}, MAE={metrics['MAE']:.2f}, "
          f"RMSE={metrics['RMSE']:.2f}, MB={metrics['MB']:.4f}")

    # 保存结果
    result_df = pd.DataFrame([{
        'method': 'OptimumInterpolation',
        **metrics
    }])
    result_df.to_csv(f'{output_dir}/OptimumInterpolation_summary.csv', index=False)
    print(f"\nResults saved to: {output_dir}/OptimumInterpolation_summary.csv")

    return metrics


if __name__ == '__main__':
    metrics = run_OptimumInterpolation方法_最优插值_ten_fold('2020-01-01')
    print(f"\nOI: R2={metrics['R2']:.4f}")
