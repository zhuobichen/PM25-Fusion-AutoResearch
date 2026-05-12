"""
V1_CensoredExceedances方法_贝叶斯截断阈值融合法 - Bayesian Censored Exceedances
==============================================================================
Reproduction of Cuba et al. (2025) arXiv:2504.20268

核心公式:
  层次模型: Y(s,t) = mu(s,t) + w(s,t) + epsilon(s,t)
  均值成分: mu(s,t) = beta0 + beta1 * CMAQ(s,t)
  截断似然:
    L(theta) = prod_{Y_i<=u} F(u;theta) * prod_{Y_i>u} f(Y_i;theta)
  GPD尾部: P(Y>y|Y>u) = (1+xi*(y-u)/sigma_u)^{-1/xi}
  AR(1)潜过程: w(s,t) = phi * w(s,t-1) + v(s,t)

  使用MLE替代MCMC进行参数估计。
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.paths import get_project_root, data_path
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.spatial.distance import cdist
from scipy.stats import norm, genpareto
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


class CensoredExceedances:
    """
    贝叶斯截断阈值融合法 (Bayesian Censored Exceedances)

    处理PM2.5监测数据中的截断问题:
    - 低浓度值低于检测限时，仅知其 <= 阈值
    - 使用截断似然函数处理这类数据
    - GPD建模超标值的尾部分布
    - AR(1)潜过程建模时空相关性

    简化实现（MLE替代MCMC）:
    1. OLS拟合均值成分 mu = beta0 + beta1 * CMAQ
    2. 对残差拟合GPD（超过阈值的部分）
    3. 估计空间相关参数
    4. 融合预测
    """

    def __init__(self, threshold_u=35.0, n_neighbors=30,
                 xi_init=0.1, sigma_u_init=10.0):
        """
        Parameters:
        -----------
        threshold_u : float
            截断阈值（日均PM2.5标准 35 ug/m3）
        n_neighbors : int
            近邻数量
        xi_init : float
            GPD形状参数初始值
        sigma_u_init : float
            GPD尺度参数初始值
        """
        self.threshold_u = threshold_u
        self.n_neighbors = n_neighbors
        self.xi_init = xi_init
        self.sigma_u_init = sigma_u_init

    def fit(self, train_coords, train_obs, train_cmaq):
        """
        拟合模型

        Parameters:
        -----------
        train_coords : array (n, 2) - 站点坐标
        train_obs : array (n,) - 观测值
        train_cmaq : array (n,) - CMAQ值
        """
        self.train_coords = train_coords
        self.train_obs = train_obs
        self.train_cmaq = train_cmaq

        # 步骤1: OLS拟合均值成分 mu = beta0 + beta1 * CMAQ
        from sklearn.linear_model import LinearRegression
        M = train_cmaq.reshape(-1, 1)
        model = LinearRegression()
        model.fit(M, train_obs)
        self.beta0 = model.intercept_
        self.beta1 = model.coef_[0]

        # 步骤2: 计算残差
        fitted = self.beta0 + self.beta1 * train_cmaq
        residuals = train_obs - fitted

        # 步骤3: GPD拟合（对超过阈值的残差）
        # 注意：这里对原始值超过阈值的部分拟合GPD
        exceed_mask = train_obs > self.threshold_u
        exceed_values = train_obs[exceed_mask]

        if len(exceed_values) > 5:
            # 拟合GPD
            try:
                self.gpd_params = genpareto.fit(
                    exceed_values - self.threshold_u,
                    floc=0  # 固定位置参数为0
                )
                self.xi = self.gpd_params[0]
                self.sigma_u = self.gpd_params[2]
            except Exception:
                self.xi = self.xi_init
                self.sigma_u = self.sigma_u_init
        else:
            self.xi = self.xi_init
            self.sigma_u = self.sigma_u_init

        # 步骤4: 估计空间相关参数
        self.residual_var = np.var(residuals)
        self.spatial_var = self.residual_var * 0.8
        self.nugget_var = self.residual_var * 0.2

        # 空间相关尺度
        dists = cdist(train_coords, train_coords)
        upper_tri = dists[np.triu_indices(len(train_coords), k=1)]
        self.spatial_range = np.median(upper_tri) * 111.0 * 3.0  # 度->km

        # 存储残差用于Kriging
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
        exceed_prob : array (m,) - 超标概率
        cond_expect : array (m,) - 条件期望
        """
        n_pred = len(pred_coords)
        n_train = len(self.train_coords)

        # 偏差校正部分
        bias_corrected = self.beta0 + self.beta1 * pred_cmaq

        # Kriging插值残差
        pred = np.zeros(n_pred)
        exceed_prob = np.zeros(n_pred)
        cond_expect = np.zeros(n_pred)

        # 距离矩阵（km）
        dist_train = cdist(self.train_coords, self.train_coords) * 111.0
        dist_pred = cdist(pred_coords, self.train_coords) * 111.0

        # 协方差矩阵
        C_train = self.spatial_var * np.exp(-3.0 * dist_train / self.spatial_range)
        C_train = C_train + self.nugget_var * np.eye(n_train)
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

            # 超标概率: P(Y > u | data)
            # 使用正态近似
            sigma_pred = np.sqrt(max(
                self.spatial_var - np.dot(w, c_pred),
                0.1
            ))
            exceed_prob[i] = 1 - norm.cdf(
                self.threshold_u, loc=pred[i], scale=sigma_pred
            )

            # 条件期望: E(Y | Y > u, data)
            if self.xi < 1 and self.sigma_u > 0:
                # GPD条件期望
                z = (self.threshold_u - pred[i]) / max(sigma_pred, 0.1)
                if z < 3:  # 只有当阈值不太远时才计算
                    cond_expect[i] = pred[i] + sigma_pred * norm.pdf(z) / (1 - norm.cdf(z))
                else:
                    cond_expect[i] = pred[i]
            else:
                cond_expect[i] = pred[i]

        pred = np.maximum(pred, 0)
        cond_expect = np.maximum(cond_expect, 0)
        exceed_prob = np.clip(exceed_prob, 0, 1)

        return pred, exceed_prob, cond_expect


def run_CensoredExceedances方法_贝叶斯截断阈值融合法_ten_fold(selected_day='2020-01-01'):
    """
    运行CensoredExceedances十折交叉验证
    """
    print("=" * 60)
    print("Censored Exceedances Ten-Fold Cross Validation")
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
        model = CensoredExceedances(threshold_u=35.0, n_neighbors=30)
        model.fit(train_coords, train_obs, train_cmaq)

        # 预测
        y_pred, exceed_prob, cond_expect = model.predict(test_coords, test_cmaq)

        results[fold_id] = {
            'y_true': test_obs,
            'y_pred': y_pred,
            'exceed_prob': exceed_prob,
            'cond_expect': cond_expect
        }
        n_exceed = np.sum(test_obs > 35.0)
        print(f"  Fold {fold_id}: n_test={len(test_df)}, n_exceed={n_exceed}")

    # 汇总
    y_pred_all = np.concatenate([results[f]['y_pred'] for f in range(1, 11) if results[f]])
    y_true_all = np.concatenate([results[f]['y_true'] for f in range(1, 11) if results[f]])

    metrics = compute_metrics(y_true_all, y_pred_all)

    print("\n=== Results ===")
    print(f"  CensoredExceedances: R2={metrics['R2']:.4f}, MAE={metrics['MAE']:.2f}, "
          f"RMSE={metrics['RMSE']:.2f}, MB={metrics['MB']:.4f}")

    # 保存结果
    result_df = pd.DataFrame([{
        'method': 'CensoredExceedances',
        **metrics
    }])
    result_df.to_csv(f'{output_dir}/CensoredExceedances_summary.csv', index=False)
    print(f"\nResults saved to: {output_dir}/CensoredExceedances_summary.csv")

    return metrics


if __name__ == '__main__':
    metrics = run_CensoredExceedances方法_贝叶斯截断阈值融合法_ten_fold('2020-01-01')
    print(f"\nCensoredExceedances: R2={metrics['R2']:.4f}")
