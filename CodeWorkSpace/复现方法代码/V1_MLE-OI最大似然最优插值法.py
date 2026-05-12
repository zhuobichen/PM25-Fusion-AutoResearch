"""
V1_MLE-OI最大似然最优插值法 - 最大似然估计最优插值法
====================================================
Reproduction of MLE-OI (Maximum Likelihood Optimal Interpolation)

核心思想：
  贝叶斯框架下的最优插值：
  将CMAQ作为背景场（先验），监测作为观测，
  通过最小化分析误差方差确定最优权重。

  物理可解释：权重由背景误差协方差和观测误差协方差的比值决定

核心公式：
  y_analysis(s0) = y_CMAQ(s0) + k^T * (y_obs - H * y_CMAQ)
  k = B * H^T * (H * B * H^T + R)^{-1}
  B_ij = sigma_b^2 * exp(-d_ij / L_c)
  R_ij = sigma_o^2 * delta_ij

文献来源：
  Fuentes and Raftery (2005), "Model evaluation and spatial interpolation
  by Bayesian combination of observations with numerical models"
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.paths import get_project_root, data_path
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.optimize import minimize
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


class MLE_OI:
    """
    最大似然估计最优插值法

    步骤:
    1. 构建背景场：CMAQ网格值作为先验估计
    2. 计算观测增量：d_i = y_obs_i - y_CMAQ_i
    3. MLE参数估计：优化 sigma_b2, sigma_o2, Lc
    4. 构建协方差矩阵：B (指数模型), R (对角)
    5. 计算最优增益：k = B * H^T * (H*B*H^T + R)^{-1}
    6. 分析更新：y_analysis = y_CMAQ + k^T * d
    """

    def __init__(self, sigma_b_init=5.0, sigma_o_init=2.0, Lc_init=50.0,
                 max_neighbors=20):
        """
        Parameters:
        -----------
        sigma_b_init : float
            背景误差标准差初值 (ug/m3)
        sigma_o_init : float
            观测误差标准差初值 (ug/m3)
        Lc_init : float
            空间相关长度初值 (km)
        max_neighbors : int
            最大近邻数（降低计算成本）
        """
        self.sigma_b_init = sigma_b_init
        self.sigma_o_init = sigma_o_init
        self.Lc_init = Lc_init
        self.max_neighbors = max_neighbors

    def _compute_distance_km(self, coords1, coords2):
        """
        计算两组坐标间的距离矩阵 (km)

        使用Haversine公式将经纬度转换为km距离
        """
        R = 6371.0  # 地球半径 km
        n1 = len(coords1)
        n2 = len(coords2)
        dist = np.zeros((n1, n2))

        for i in range(n1):
            lon1, lat1 = np.radians(coords1[i])
            for j in range(n2):
                lon2, lat2 = np.radians(coords2[j])
                dlat = lat2 - lat1
                dlon = lon2 - lon1
                a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
                dist[i, j] = 2 * R * np.arcsin(np.sqrt(a))

        return dist

    def _neg_log_likelihood(self, params, dist_matrix, innovation):
        """
        负对数似然函数

        ell(sigma_b2, sigma_o2, Lc) =
            -0.5 * log|H*B*H^T + R| - 0.5 * d^T * (H*B*H^T + R)^{-1} * d

        Parameters:
        -----------
        params : array [sigma_b2, sigma_o2, Lc]
        dist_matrix : array (n, n) - 站点间距离矩阵 (km)
        innovation : array (n,) - 观测增量 d = y_obs - y_CMAQ

        Returns:
        --------
        nll : float - 负对数似然值
        """
        sigma_b2, sigma_o2, Lc = params

        # 确保参数为正
        if sigma_b2 <= 0 or sigma_o2 <= 0 or Lc <= 0:
            return 1e10

        n = len(innovation)

        # 背景误差协方差矩阵 B (指数模型)
        B = sigma_b2 * np.exp(-dist_matrix / Lc)

        # 观测误差协方差矩阵 R (对角)
        R = sigma_o2 * np.eye(n)

        # H*B*H^T + R (H=I，因为站点就是观测点)
        S = B + R

        # 计算对数似然（使用Cholesky分解提高数值稳定性）
        try:
            L = np.linalg.cholesky(S)
            log_det = 2 * np.sum(np.log(np.diag(L)))
            # S^{-1} * d = L^{-T} * L^{-1} * d
            inv_S_d = np.linalg.solve(L.T, np.linalg.solve(L, innovation))
            mahal_dist = innovation @ inv_S_d
            nll = 0.5 * (log_det + mahal_dist)
        except np.linalg.LinAlgError:
            nll = 1e10

        return nll

    def _mle_estimate(self, dist_matrix, innovation):
        """
        MLE参数估计

        通过最大化观测的边际似然函数估计 sigma_b2, sigma_o2, Lc

        Parameters:
        -----------
        dist_matrix : array (n, n) - 站点间距离矩阵 (km)
        innovation : array (n,) - 观测增量

        Returns:
        --------
        sigma_b2_opt, sigma_o2_opt, Lc_opt : float - 最优参数
        """
        x0 = [self.sigma_b_init**2, self.sigma_o_init**2, self.Lc_init]
        bounds = [(1e-6, None), (1e-6, None), (1e-6, None)]

        # 多次初始化避免局部最优
        best_result = None
        best_nll = np.inf

        for scale in [0.5, 1.0, 2.0]:
            x0_scaled = [x * scale for x in x0]
            try:
                result = minimize(
                    self._neg_log_likelihood,
                    x0_scaled,
                    args=(dist_matrix, innovation),
                    bounds=bounds,
                    method='L-BFGS-B'
                )
                if result.fun < best_nll:
                    best_nll = result.fun
                    best_result = result
            except Exception:
                continue

        if best_result is not None:
            sigma_b2_opt, sigma_o2_opt, Lc_opt = best_result.x
        else:
            sigma_b2_opt, sigma_o2_opt, Lc_opt = x0

        return sigma_b2_opt, sigma_o2_opt, Lc_opt

    def fit_predict(self, train_coords, train_obs, train_cmaq,
                    test_coords, test_cmaq):
        """
        训练并预测

        Parameters:
        -----------
        train_coords : array (n_train, 2) - 训练站坐标 [lon, lat]
        train_obs : array (n_train,) - 训练站观测值
        train_cmaq : array (n_train,) - 训练站CMAQ值
        test_coords : array (n_test, 2) - 测试站坐标 [lon, lat]
        test_cmaq : array (n_test,) - 测试站CMAQ值

        Returns:
        --------
        predictions : array (n_test,) - 融合预测值
        """
        n_train = len(train_coords)
        n_test = len(test_coords)

        # 如果训练站太多，限制近邻数
        if n_train > self.max_neighbors:
            # 对每个测试站选择最近的max_neighbors个训练站
            pass  # 在预测循环中处理

        # 步骤1: 计算观测增量
        innovation = train_obs - train_cmaq

        # 步骤2: 计算训练站间距离矩阵 (km)
        dist_train = self._compute_distance_km(train_coords, train_coords)

        # 步骤3: MLE参数估计
        sigma_b2_opt, sigma_o2_opt, Lc_opt = self._mle_estimate(dist_train, innovation)

        # 步骤4: 构建协方差矩阵
        B = sigma_b2_opt * np.exp(-dist_train / Lc_opt)
        R = sigma_o2_opt * np.eye(n_train)
        S = B + R  # H*B*H^T + R

        # 步骤5: 计算测试站与训练站的距离
        dist_test_train = self._compute_distance_km(test_coords, train_coords)

        # 步骤6: 背景误差协方差 (测试站与训练站)
        B_test_train = sigma_b2_opt * np.exp(-dist_test_train / Lc_opt)

        # 步骤7: 计算最优增益 K = B_test_train * S^{-1}
        try:
            inv_S = np.linalg.inv(S)
            K = B_test_train @ inv_S
        except np.linalg.LinAlgError:
            inv_S = np.linalg.pinv(S)
            K = B_test_train @ inv_S

        # 步骤8: 分析更新
        # y_analysis = y_CMAQ + K * d
        predictions = test_cmaq + K @ innovation

        # 非负约束
        predictions = np.maximum(predictions, 0)

        return predictions


def run_MLE_OI_ten_fold(selected_day='2020-01-01'):
    """
    运行MLE-OI十折交叉验证

    Parameters:
    -----------
    selected_day : str
        验证日期，格式 'YYYY-MM-DD'
    """
    print("=" * 60)
    print("MLE-OI Ten-Fold Cross Validation")
    print(f"Date: {selected_day}")
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

    # 十折交叉验证
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
        model = MLE_OI(
            sigma_b_init=5.0,
            sigma_o_init=2.0,
            Lc_init=50.0,
            max_neighbors=20
        )
        y_pred = model.fit_predict(X_train, y_train, m_train, X_test, m_test)

        results[fold_id] = {
            'y_true': y_test,
            'y_pred': y_pred
        }

        fold_metrics = compute_metrics(y_test, y_pred)
        print(f"  Fold {fold_id}: R2={fold_metrics['R2']:.4f}, "
              f"RMSE={fold_metrics['RMSE']:.2f}, N={len(test_df)}")

    # 汇总
    y_pred_all = np.concatenate([results[f]['y_pred'] for f in range(1, 11) if results[f]])
    y_true_all = np.concatenate([results[f]['y_true'] for f in range(1, 11) if results[f]])

    metrics = compute_metrics(y_true_all, y_pred_all)

    print("\n=== Results ===")
    print(f"  MLE-OI: R2={metrics['R2']:.4f}, MAE={metrics['MAE']:.2f}, "
          f"RMSE={metrics['RMSE']:.2f}, MB={metrics['MB']:.4f}")

    # 保存结果
    result_df = pd.DataFrame([{
        'method': 'MLE-OI',
        **metrics
    }])
    result_df.to_csv(f'{output_dir}/MLE-OI_folds.csv', index=False)

    print(f"\nResults saved to: {output_dir}/")

    return metrics


if __name__ == '__main__':
    metrics = run_MLE_OI_ten_fold('2020-01-01')
    print(f"\nMLE-OI: R2={metrics['R2']:.4f}")
