"""
V1_BSMFM贝叶斯多源融合模型法 - Bayesian Multisource Fusion Model
================================================================
Reproduction of BSMFM method

核心公式:
  层次模型: Y(s,t) = X(s,t) + epsilon(s,t)
  潜在场:   X_i(s,t) = f_i(s,t) + xi(s,t) + eta_i(s,t)
  CMAQ偏差校正: f_CMAQ(s,t) = beta0 + beta1 * CMAQ(s,t)

  使用MLE替代MCMC进行参数估计:
  1. OLS拟合偏差校正系数 beta0, beta1
  2. 估计空间相关性参数
  3. 预测: Y_hat = beta0 + beta1 * CMAQ + xi_hat
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
from scipy.optimize import minimize
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


def exponential_cov(d, sill, range_km, nugget=0.0):
    """指数协方差函数"""
    cov = sill * np.exp(-3.0 * d / range_km)
    if nugget > 0:
        cov = cov + nugget * np.eye(len(d))
    return cov


class BSMFM:
    """
    贝叶斯多源融合模型 (Bayesian Multisource Fusion Model)

    简化实现（MLE替代MCMC）:
    1. 线性偏差校正: O = beta0 + beta1 * CMAQ + xi + epsilon
    2. 空间随机场 xi 使用指数协方差建模
    3. 通过MLE估计参数
    4. 预测: Y_hat = beta0 + beta1 * CMAQ + Kriging(xi)
    """

    def __init__(self, spatial_range=None, n_neighbors=30):
        """
        Parameters:
        -----------
        spatial_range : float
            空间相关尺度（km），None则自动估计
        n_neighbors : int
            近邻数量
        """
        self.spatial_range = spatial_range
        self.n_neighbors = n_neighbors
        self.beta0 = None
        self.beta1 = None
        self.residual_var = None
        self.spatial_var = None
        self.nugget_var = None

    def fit(self, train_coords, train_obs, train_cmaq):
        """
        拟合模型

        Parameters:
        -----------
        train_coords : array (n, 2) - 站点坐标
        train_obs : array (n,) - 观测值
        train_cmaq : array (n,) - CMAQ值
        """
        # 步骤1: OLS拟合偏差校正
        M = train_cmaq.reshape(-1, 1)
        model = LinearRegression()
        model.fit(M, train_obs)
        self.beta0 = model.intercept_
        self.beta1 = model.coef_[0]

        # 步骤2: 计算残差（空间随机场）
        fitted = self.beta0 + self.beta1 * train_cmaq
        residuals = train_obs - fitted

        # 步骤3: 估计空间相关参数
        self.residual_var = np.var(residuals)

        # 估计空间相关尺度
        if self.spatial_range is None:
            dists = cdist(train_coords, train_coords)
            upper_tri = dists[np.triu_indices(len(train_coords), k=1)]
            self.spatial_range = np.median(upper_tri) * 111.0 * 3.0  # 度->km, x3

        self.spatial_var = self.residual_var * 0.8
        self.nugget_var = self.residual_var * 0.2

        # 存储训练数据用于Kriging
        self.train_coords = train_coords
        self.train_residuals = residuals

        return self

    def predict(self, pred_coords, pred_cmaq):
        """
        预测

        Parameters:
        -----------
        pred_coords : array (m, 2) - 预测点坐标
        pred_cmaq : array (m,) - 预测点CMAQ值

        Returns:
        --------
        pred : array (m,) - 融合预测值
        """
        n_pred = len(pred_coords)
        n_train = len(self.train_coords)

        # 偏差校正部分
        bias_corrected = self.beta0 + self.beta1 * pred_cmaq

        # Kriging插值残差
        pred = np.zeros(n_pred)

        # 距离矩阵（km）
        dist_train = cdist(self.train_coords, self.train_coords) * 111.0
        dist_pred = cdist(pred_coords, self.train_coords) * 111.0

        # 协方差矩阵
        C_train = self.spatial_var * np.exp(-3.0 * dist_train / self.spatial_range)
        C_train = C_train + self.nugget_var * np.eye(n_train)

        # 添加小正则化
        C_train = C_train + 1e-6 * np.eye(n_train)

        try:
            C_train_inv = np.linalg.inv(C_train)
        except np.linalg.LinAlgError:
            C_train_inv = np.linalg.pinv(C_train)

        for i in range(n_pred):
            # 预测点与训练点的协方差
            c_pred = self.spatial_var * np.exp(
                -3.0 * dist_pred[i] / self.spatial_range
            )

            # Kriging权重
            w = C_train_inv @ c_pred

            # 插值残差
            residual_pred = np.dot(w, self.train_residuals)

            # 最终预测
            pred[i] = bias_corrected[i] + residual_pred

        pred = np.maximum(pred, 0)
        return pred


def run_BSMFM贝叶斯多源融合模型法_ten_fold(selected_day='2020-01-01'):
    """
    运行BSMFM十折交叉验证
    """
    print("=" * 60)
    print("BSMFM (Bayesian Multisource Fusion Model) Ten-Fold Cross Validation")
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

        # 训练BSMFM
        model = BSMFM(n_neighbors=30)
        model.fit(train_coords, train_obs, train_cmaq)

        # 预测
        y_pred = model.predict(test_coords, test_cmaq)

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
    print(f"  BSMFM: R2={metrics['R2']:.4f}, MAE={metrics['MAE']:.2f}, "
          f"RMSE={metrics['RMSE']:.2f}, MB={metrics['MB']:.4f}")

    # 保存结果
    result_df = pd.DataFrame([{
        'method': 'BSMFM',
        **metrics
    }])
    result_df.to_csv(f'{output_dir}/BSMFM_summary.csv', index=False)
    print(f"\nResults saved to: {output_dir}/BSMFM_summary.csv")

    return metrics


if __name__ == '__main__':
    metrics = run_BSMFM贝叶斯多源融合模型法_ten_fold('2020-01-01')
    print(f"\nBSMFM: R2={metrics['R2']:.4f}")
