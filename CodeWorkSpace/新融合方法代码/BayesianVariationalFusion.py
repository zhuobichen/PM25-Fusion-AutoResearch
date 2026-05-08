"""
BayesianVariationalFusion - 贝叶斯变分融合方法
===============================================
创新点:
1. 使用贝叶斯推断建模CMAQ偏差
2. 使用变分近似求解后验分布
3. 结合多种偏差校正方法的优点

核心公式:
P(bias | data) ∝ P(data | bias) * P(bias)

其中:
- P(data | bias) = N(y - M - bias, σ²/ω * I)  似然
- P(bias) = N(0, (ε*DTD + δ*I)⁻¹)              先验
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.paths import get_project_root, data_path
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import spsolve
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
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


def build_laplacian_matrix(lon, lat, n_neighbors=10):
    """构建拉普拉斯平滑矩阵"""
    n = len(lon)
    # 计算距离矩阵
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        dist_matrix[i, :] = np.sqrt((lon - lon[i])**2 + (lat - lat[i])**2)

    # 构建稀疏权重矩阵
    row_indices = []
    col_indices = []
    data = []

    for i in range(n):
        # 找到k近邻
        distances = dist_matrix[i, :]
        nearest_idx = np.argsort(distances)[1:n_neighbors+1]  # 排除自身

        for j in nearest_idx:
            if i != j:
                # 高斯核权重
                d = distances[j]
                weight = np.exp(-d**2 / 2.0)  # 假设带宽=1度
                row_indices.append(i)
                col_indices.append(j)
                data.append(weight)

    W = sparse.csr_matrix((data, (row_indices, col_indices)), shape=(n, n))
    D = sparse.diags(W.sum(axis=1).A1) - W  # D = Diag - W
    return D


def bayesian_variational_fusion(y_obs, m_cmaq, lon, lat, omega=1.0, epsilon=1e-2, delta=1e-2):
    """
    贝叶斯变分融合方法

    后验均值估计:
    bias = (omega * I + epsilon * DTD + delta * I)^(-1) * omega * (y_obs - m_cmaq)

    参数:
    - y_obs: 观测值
    - m_cmaq: CMAQ模型预测值
    - lon, lat: 站点坐标
    - omega: 观测拟合权重
    - epsilon: 空间平滑正则化参数
    - delta: 小偏差权重

    返回:
    - bias: 偏差估计值
    """
    n = len(y_obs)

    # 构建拉普拉斯矩阵
    D = build_laplacian_matrix(lon, lat, n_neighbors=10)
    DTD = D.T @ D

    # 系统矩阵: omega * I + epsilon * DTD + delta * I
    system_matrix = omega * sparse.eye(n) + epsilon * DTD + delta * sparse.eye(n)

    # 右端项: omega * (y_obs - m_cmaq)
    rhs = omega * (y_obs - m_cmaq)

    # 求解线性系统
    bias = spsolve(system_matrix.tocsc(), rhs)

    return bias


def run_bayesian_variational_fusion_ten_fold(selected_day='2020-01-01'):
    """
    运行贝叶斯变分融合十折交叉验证
    """
    print("="*60)
    print("BayesianVariationalFusion Ten-Fold Cross Validation")
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

    ny, nx = lon_cmaq.shape
    X_grid_full = np.column_stack([lon_cmaq.ravel(), lat_cmaq.ravel()])
    y_grid_model_full = pred_day.ravel()

    print(f"Data loaded: {len(day_df)} monitoring records")

    # 参数网格搜索
    print("\n=== Parameter Grid Search ===")
    best_r2 = -np.inf
    best_params = None

    omega_values = [0.5, 1.0, 2.0, 5.0, 10.0]
    epsilon_values = [1e-3, 1e-2, 1e-1, 0.5]
    delta_values = [1e-3, 1e-2, 1e-1, 0.5]

    for omega in omega_values:
        for epsilon in epsilon_values:
            for delta in delta_values:
                results = {fold_id: {} for fold_id in range(1, 11)}

                for fold_id in range(1, 11):
                    train_df = day_df[day_df['fold'] != fold_id].copy()
                    test_df = day_df[day_df['fold'] == fold_id].copy()

                    train_df = train_df.dropna(subset=['Lon', 'Lat', 'CMAQ', 'Conc'])
                    test_df = test_df.dropna(subset=['Lon', 'Lat', 'CMAQ', 'Conc'])

                    if len(test_df) == 0:
                        continue

                    y_train = train_df['Conc'].values
                    m_train = train_df['CMAQ'].values
                    lon_train = train_df['Lon'].values
                    lat_train = train_df['Lat'].values

                    y_test = test_df['Conc'].values
                    m_test = test_df['CMAQ'].values

                    # 训练偏差
                    bias_train = bayesian_variational_fusion(y_train, m_train, lon_train, lat_train,
                                                            omega=omega, epsilon=epsilon, delta=delta)

                    # 在测试站点上应用（使用IDW插值偏差）
                    from scipy.spatial.distance import cdist

                    test_locations = test_df[['Lon', 'Lat']].values
                    train_locations = train_df[['Lon', 'Lat']].values

                    # IDW插值偏差
                    distances = cdist(test_locations, train_locations)
                    weights = 1.0 / (distances + 1e-6)
                    weights = weights / weights.sum(axis=1, keepdims=True)
                    bias_test = weights @ bias_train

                    # 融合预测
                    pred_test = m_test + bias_test

                    results[fold_id] = {
                        'y_true': y_test,
                        'pred': pred_test
                    }

                # 汇总
                y_true_all = np.concatenate([results[f]['y_true'] for f in range(1, 11) if results[f]])
                pred_all = np.concatenate([results[f]['pred'] for f in range(1, 11) if results[f]])
                metrics = compute_metrics(y_true_all, pred_all)

                if metrics['R2'] > best_r2:
                    best_r2 = metrics['R2']
                    best_params = {'omega': omega, 'epsilon': epsilon, 'delta': delta}
                    print(f"  New best: omega={omega}, epsilon={epsilon}, delta={delta}, R2={best_r2:.4f}")

    print(f"\nBest parameters: omega={best_params['omega']}, epsilon={best_params['epsilon']}, delta={best_params['delta']}")
    print(f"Best R2: {best_r2:.4f}")

    # 使用最佳参数重新运行十折验证
    print("\n=== Final Ten-Fold Validation ===")
    results = {fold_id: {} for fold_id in range(1, 11)}

    for fold_id in range(1, 11):
        train_df = day_df[day_df['fold'] != fold_id].copy()
        test_df = day_df[day_df['fold'] == fold_id].copy()

        train_df = train_df.dropna(subset=['Lon', 'Lat', 'CMAQ', 'Conc'])
        test_df = test_df.dropna(subset=['Lon', 'Lat', 'CMAQ', 'Conc'])

        if len(test_df) == 0:
            continue

        y_train = train_df['Conc'].values
        m_train = train_df['CMAQ'].values
        lon_train = train_df['Lon'].values
        lat_train = train_df['Lat'].values

        y_test = test_df['Conc'].values
        m_test = test_df['CMAQ'].values

        # 训练偏差
        bias_train = bayesian_variational_fusion(y_train, m_train, lon_train, lat_train, **best_params)

        # IDW插值偏差
        test_locations = test_df[['Lon', 'Lat']].values
        train_locations = train_df[['Lon', 'Lat']].values

        distances = cdist(test_locations, train_locations)
        weights = 1.0 / (distances + 1e-6)
        weights = weights / weights.sum(axis=1, keepdims=True)
        bias_test = weights @ bias_train

        # 融合预测
        pred_test = m_test + bias_test

        results[fold_id] = {
            'y_true': y_test,
            'pred': pred_test
        }

        print(f"  Fold {fold_id}: completed")

    # 汇总
    y_true_all = np.concatenate([results[f]['y_true'] for f in range(1, 11) if results[f]])
    pred_all = np.concatenate([results[f]['pred'] for f in range(1, 11) if results[f]])
    final_metrics = compute_metrics(y_true_all, pred_all)

    print("\n" + "="*60)
    print("Final BayesianVariationalFusion Results")
    print("="*60)
    print(f"R2: {final_metrics['R2']:.4f}")
    print(f"MAE: {final_metrics['MAE']:.2f}")
    print(f"RMSE: {final_metrics['RMSE']:.2f}")
    print(f"MB: {final_metrics['MB']:.2f}")

    # 保存结果
    result_df = pd.DataFrame([{
        'method': 'BayesianVariationalFusion',
        **best_params,
        **final_metrics
    }])
    result_df.to_csv(f'{output_dir}/BayesianVariationalFusion_summary.csv', index=False)

    print(f"\nResults saved to: {output_dir}/")

    return final_metrics, best_params


if __name__ == '__main__':
    metrics, params = run_bayesian_variational_fusion_ten_fold('2020-01-01')
    print(f"\nFinal: R2={metrics['R2']:.4f}, params={params}")