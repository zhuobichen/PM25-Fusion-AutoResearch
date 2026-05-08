"""
HybridEAVNA - 混合比率-偏差法
==============================
结合eVNA和aVNA的优点，通过混合系数beta融合两种方法的预测结果

公式: P_Hybrid = beta * eVNA + (1-beta) * aVNA
其中beta通过十折交叉验证优化确定

创新点:
- 结合eVNA(比率法)和aVNA(偏差法)的优点
- beta通过数据驱动确定，而非手动设置
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.paths import get_project_root, data_path
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import netCDF4 as nc
from Code.VNAeVNAaVNA.nna_methods import NNA

root_dir = str(get_project_root())
cmaq_file = data_path('test_data/raw/CMAQ/2020_PM25.nc')
monitor_file = data_path('test_data/raw/Monitor/2020_DailyPM2.5Monitor.csv')
fold_file = data_path('test_data/fold_split_table_daily.csv')
output_dir = f'{root_dir}/test_result/创新方法'
os.makedirs(output_dir, exist_ok=True)


def compute_metrics(y_true, y_pred):
    """计算评估指标"""
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
    """获取站点位置的CMAQ值（最近邻）"""
    dist = np.sqrt((lon_grid - lon)**2 + (lat_grid - lat)**2)
    idx = np.argmin(dist)
    ny, nx = lon_grid.shape
    row, col = idx // nx, idx % nx
    return pm25_grid[row, col]


class HybridEAVNA:
    """
    Hybrid eVNA-aVNA 混合融合方法

    通过优化混合系数beta，结合eVNA(比率法)和aVNA(偏差法)的优点
    """

    def __init__(self, k=30, power=-2, method='voronoi'):
        self.k = k
        self.power = power
        self.method = method
        self.beta = None
        self.nn = None

    def fit(self, X_obs, y_obs, y_model_obs):
        """
        拟合NNA模型（不优化beta，beta在十折验证中优化）

        Parameters:
        -----------
        X_obs : array (n_obs, 2)
            监测站点坐标 [lon, lat]
        y_obs : array
            监测站点观测值
        y_model_obs : array
            监测站点对应的CMAQ模型值
        """
        self.nn = NNA(method=self.method, k=self.k, power=self.power)

        # 构建训练数据DataFrame
        train_data = pd.DataFrame({
            'x': X_obs[:, 0],
            'y': X_obs[:, 1],
            'Conc': y_obs,
            'mod': y_model_obs,
            'bias': y_obs - y_model_obs,
            'r_n': y_obs / y_model_obs
        })

        self.nn.fit(
            train_data[['x', 'y']],
            train_data[['Conc', 'mod', 'bias', 'r_n']]
        )

    def predict_with_params(self, X_grid, y_grid_model, beta):
        """
        使用指定beta预测融合结果

        Parameters:
        -----------
        X_grid : array (n_grid, 2)
            网格点坐标
        y_grid_model : array
            网格点对应的CMAQ模型值
        beta : float
            混合系数 (0 <= beta <= 1)

        Returns:
        --------
        y_pred : array
            融合预测结果
        """
        # 在网格上预测
        zdf_grid = self.nn.predict(X_grid, njobs=4)
        vna_bias_grid = zdf_grid[:, 2]  # bias
        vna_rn_grid = zdf_grid[:, 3]  # ratio

        # eVNA = M * r_n
        evna_pred = y_grid_model * vna_rn_grid

        # aVNA = M + bias
        avna_pred = y_grid_model + vna_bias_grid

        # Hybrid = beta * eVNA + (1-beta) * aVNA
        y_pred = beta * evna_pred + (1 - beta) * avna_pred

        return y_pred

    def extract_at_sites(self, X_grid, y_pred, lon_sites, lat_sites, lon_grid, lat_grid):
        """
        在站点位置提取预测值（最近邻）

        Parameters:
        -----------
        X_grid : array
            网格坐标
        y_pred : array
            网格预测值
        lon_sites, lat_sites : array
            站点坐标
        lon_grid, lat_grid : array
            网格坐标

        Returns:
        --------
        y_at_sites : array
            站点预测值
        """
        y_at_sites = np.zeros(len(lon_sites))
        for i in range(len(lon_sites)):
            dist = np.sqrt((lon_grid - lon_sites[i])**2 + (lat_grid - lat_sites[i])**2)
            idx = np.argmin(dist)
            y_at_sites[i] = y_pred[idx]
        return y_at_sites


def run_hybrid_ten_fold(selected_day='2020-01-01'):
    """
    运行HybridEAVNA十折交叉验证

    通过十折验证优化beta系数
    """
    print("="*60)
    print("HybridEAVNA Ten-Fold Cross Validation")
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

    # 创建网格坐标
    ny, nx = lon_cmaq.shape
    X_grid_full = np.column_stack([lon_cmaq.ravel(), lat_cmaq.ravel()])
    y_grid_model_full = pred_day.ravel()

    print(f"Data loaded: {len(day_df)} monitoring records")

    # 优化beta: 测试多个候选值
    beta_candidates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    best_beta = None
    best_r2 = -np.inf
    beta_results = {}

    print("\n=== Optimizing Beta ===")

    for beta in beta_candidates:
        all_true = []
        all_pred = []

        for fold_id in range(1, 11):
            train_df = day_df[day_df['fold'] != fold_id].copy()
            test_df = day_df[day_df['fold'] == fold_id].copy()

            train_df = train_df.dropna(subset=['Lon', 'Lat', 'CMAQ', 'Conc'])
            test_df = test_df.dropna(subset=['Lon', 'Lat', 'CMAQ', 'Conc'])

            if len(test_df) == 0:
                continue

            # 训练
            model = HybridEAVNA(k=30, power=-2, method='voronoi')
            model.fit(
                train_df[['Lon', 'Lat']].values,
                train_df['Conc'].values,
                train_df['CMAQ'].values
            )

            # 预测
            y_pred_grid = model.predict_with_params(X_grid_full, y_grid_model_full, beta)
            y_pred = model.extract_at_sites(
                X_grid_full, y_pred_grid,
                test_df['Lon'].values, test_df['Lat'].values,
                lon_cmaq, lat_cmaq
            )

            all_true.extend(test_df['Conc'].values)
            all_pred.extend(y_pred)

        all_true = np.array(all_true)
        all_pred = np.array(all_pred)
        metrics = compute_metrics(all_true, all_pred)
        beta_results[beta] = metrics

        print(f"  beta={beta:.1f}: R2={metrics['R2']:.4f}, MAE={metrics['MAE']:.2f}, RMSE={metrics['RMSE']:.2f}")

        if metrics['R2'] > best_r2:
            best_r2 = metrics['R2']
            best_beta = beta

    print(f"\nBest beta: {best_beta} with R2={best_r2:.4f}")

    # 使用最优beta进行正式十折验证
    print(f"\n=== Final 10-Fold with beta={best_beta} ===")
    results = {fold_id: {} for fold_id in range(1, 11)}

    for fold_id in range(1, 11):
        train_df = day_df[day_df['fold'] != fold_id].copy()
        test_df = day_df[day_df['fold'] == fold_id].copy()

        train_df = train_df.dropna(subset=['Lon', 'Lat', 'CMAQ', 'Conc'])
        test_df = test_df.dropna(subset=['Lon', 'Lat', 'CMAQ', 'Conc'])

        if len(test_df) == 0:
            continue

        # 训练
        model = HybridEAVNA(k=30, power=-2, method='voronoi')
        model.fit(
            train_df[['Lon', 'Lat']].values,
            train_df['Conc'].values,
            train_df['CMAQ'].values
        )

        # 预测
        y_pred_grid = model.predict_with_params(X_grid_full, y_grid_model_full, best_beta)
        y_pred = model.extract_at_sites(
            X_grid_full, y_pred_grid,
            test_df['Lon'].values, test_df['Lat'].values,
            lon_cmaq, lat_cmaq
        )

        metrics = compute_metrics(test_df['Conc'].values, y_pred)
        results[fold_id] = {
            'y_true': test_df['Conc'].values,
            'y_pred': y_pred,
            **metrics
        }

        print(f"  Fold {fold_id}: R2={metrics['R2']:.4f}, MAE={metrics['MAE']:.2f}, RMSE={metrics['RMSE']:.2f}")

    # 汇总
    all_true = np.concatenate([results[f]['y_true'] for f in range(1, 11) if results[f]])
    all_pred = np.concatenate([results[f]['y_pred'] for f in range(1, 11) if results[f]])
    final_metrics = compute_metrics(all_true, all_pred)

    print("\n" + "="*60)
    print("Final Results (HybridEAVNA)")
    print("="*60)
    print(f"Optimal beta: {best_beta}")
    print(f"R2: {final_metrics['R2']:.4f}")
    print(f"MAE: {final_metrics['MAE']:.2f}")
    print(f"RMSE: {final_metrics['RMSE']:.2f}")
    print(f"MB: {final_metrics['MB']:.2f}")

    # 保存结果
    result_df = pd.DataFrame([final_metrics])
    result_df.to_csv(f'{output_dir}/HybridEAVNA_summary.csv', index=False)

    # 保存每折结果
    fold_results = []
    for fold_id in range(1, 11):
        if results[fold_id]:
            fold_results.append({
                'fold': fold_id,
                'R2': results[fold_id]['R2'],
                'MAE': results[fold_id]['MAE'],
                'RMSE': results[fold_id]['RMSE'],
                'MB': results[fold_id]['MB']
            })
    pd.DataFrame(fold_results).to_csv(f'{output_dir}/HybridEAVNA_fold_results.csv', index=False)

    print(f"\nResults saved to: {output_dir}/")

    return final_metrics, best_beta


if __name__ == '__main__':
    metrics, beta = run_hybrid_ten_fold('2020-01-01')
    print(f"\nFinal: R2={metrics['R2']:.4f}, Optimal beta={beta}")