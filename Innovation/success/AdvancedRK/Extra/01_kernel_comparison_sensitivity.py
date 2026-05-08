import os
from shared.paths import get_project_root, data_path
"""
核函数/方法敏感性实验：比较不同核函数和其他ML方法
"""
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel, RationalQuadratic, ExpSineSquared, DotProduct
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
import netCDF4 as nc
from datetime import datetime

root_dir = str(get_project_root())
cmaq_file = data_path('test_data/raw/CMAQ/2020_PM25.nc')
monitor_file = data_path('test_data/raw/Monitor/2020_DailyPM2.5Monitor.csv')
fold_file = data_path('test_data/fold_split_table_daily.csv')


def compute_metrics(y_true, y_pred):
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if len(y_true) == 0:
        return {'R2': np.nan, 'MAE': np.nan, 'RMSE': np.nan}
    return {
        'R2': r2_score(y_true, y_pred),
        'MAE': mean_absolute_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred))
    }


def get_cmaq_at_site(lon, lat, lon_grid, lat_grid, pm25_grid):
    dist = np.sqrt((lon_grid - lon)**2 + (lat_grid - lat)**2)
    idx = np.argmin(dist)
    ny, nx = lon_grid.shape
    return pm25_grid[idx // nx, idx % nx]


def get_cmaq_grid_coord(lon, lat, lon_grid, lat_grid):
    dist = np.sqrt((lon_grid - lon)**2 + (lat_grid - lat)**2)
    idx = np.argmin(dist)
    ny, nx = lon_grid.shape
    return lon_grid[idx // nx, idx % nx], lat_grid[idx // nx, idx % nx]


def run_comparison(selected_day, methods_dict):
    """比较不同核函数和ML方法的效果"""
    print(f"\n{'='*60}")
    print(f"Method Comparison - {selected_day}")
    print(f"{'='*60}")

    # 加载数据
    monitor_df = pd.read_csv(monitor_file)
    fold_df = pd.read_csv(fold_file)

    day_df = monitor_df[monitor_df['Date'] == selected_day].copy()
    day_df = day_df.merge(fold_df, on='Site', how='left').dropna(subset=['Lat', 'Lon', 'Conc'])

    if len(day_df) < 100:
        print("Insufficient data")
        return None

    ds = nc.Dataset(cmaq_file, 'r')
    lon_cmaq = ds.variables['lon'][:]
    lat_cmaq = ds.variables['lat'][:]
    pred_pm25 = ds.variables['pred_PM25'][:]
    ds.close()

    day_idx = (datetime.strptime(selected_day, '%Y-%m-%d') - datetime(2020, 1, 1)).days
    cmaq_day = pred_pm25[day_idx]

    day_df['CMAQ'] = [get_cmaq_at_site(row['Lon'], row['Lat'], lon_cmaq, lat_cmaq, cmaq_day) for _, row in day_df.iterrows()]

    # 存储每个方法的结果
    method_results = {name: [] for name in methods_dict.keys()}

    for fold_id in range(1, 11):
        train_df = day_df[day_df['fold'] != fold_id].dropna(subset=['Lon', 'Lat', 'CMAQ', 'Conc'])
        test_df = day_df[day_df['fold'] == fold_id].dropna(subset=['Lon', 'Lat', 'CMAQ', 'Conc'])

        if len(test_df) == 0 or len(train_df) == 0:
            continue

        y_train = train_df['Conc'].values
        m_train = train_df['CMAQ'].values
        m_test = test_df['CMAQ'].values
        y_test = test_df['Conc'].values

        # 多项式校正
        poly = PolynomialFeatures(degree=2, include_bias=False)
        m_train_poly = poly.fit_transform(m_train.reshape(-1, 1))
        m_test_poly = poly.transform(m_test.reshape(-1, 1))

        ols = LinearRegression()
        ols.fit(m_train_poly, y_train)
        residual = y_train - ols.predict(m_train_poly)
        pred_poly = ols.predict(m_test_poly)

        # 获取CMAQ网格坐标
        train_coords = [get_cmaq_grid_coord(row['Lon'], row['Lat'], lon_cmaq, lat_cmaq) for _, row in train_df.iterrows()]
        test_coords = [get_cmaq_grid_coord(row['Lon'], row['Lat'], lon_cmaq, lat_cmaq) for _, row in test_df.iterrows()]
        X_train = np.array(train_coords)
        X_test = np.array(test_coords)

        # 对每个方法测试
        for method_name, method in methods_dict.items():
            if method_name.startswith('GPR'):
                regressor = GaussianProcessRegressor(kernel=method, n_restarts_optimizer=0, alpha=0.1, normalize_y=True)
                regressor.fit(X_train, residual)
                pred, _ = regressor.predict(X_test, return_std=True)
            elif method_name == 'RF':
                regressor = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
                regressor.fit(X_train, residual)
                pred = regressor.predict(X_test)
            elif method_name == 'KNN':
                regressor = KNeighborsRegressor(n_neighbors=10, weights='distance')
                regressor.fit(X_train, residual)
                pred = regressor.predict(X_test)
            elif method_name == 'PolyOnly':
                # 仅多项式，无空间建模
                mean_m_test_poly = poly.transform([[np.mean(m_test)]])
                pred = np.full(len(y_test), ols.predict(mean_m_test_poly)[0])

            final_pred = pred_poly + pred if method_name != 'PolyOnly' else pred
            method_results[method_name].append((y_test, final_pred))

        print(f"  Fold {fold_id}: completed")

    # 计算每个方法的指标
    print(f"\n=== Results for {selected_day} ===")
    print(f"{'Method':<25} {'R2':>10} {'MAE':>10} {'RMSE':>10}")
    print("-" * 55)

    results = {}
    for method_name in methods_dict.keys():
        trues = np.concatenate([r[0] for r in method_results[method_name]])
        preds = np.concatenate([r[1] for r in method_results[method_name]])
        metrics = compute_metrics(trues, preds)
        results[method_name] = metrics
        print(f"  {method_name:<23} {metrics['R2']:>10.4f} {metrics['MAE']:>10.2f} {metrics['RMSE']:>10.2f}")

    return results


if __name__ == '__main__':
    # 定义要测试的方法
    methods = {
        # GPR with different kernels
        'GPR-RBF': ConstantKernel(10.0, (1e-2, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e1)),

        'GPR-Matern(0.5)': ConstantKernel(10.0, (1e-2, 1e3)) * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=0.5) + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e1)),

        'GPR-Matern(1.5)': ConstantKernel(10.0, (1e-2, 1e3)) * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=1.5) + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e1)),

        'GPR-Matern(2.5)': ConstantKernel(10.0, (1e-2, 1e3)) * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=2.5) + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e1)),

        'GPR-RationalQuad': ConstantKernel(10.0, (1e-2, 1e3)) * RationalQuadratic(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), alpha=1.0) + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e1)),

        # Other ML methods
        'RF': None,  # Random Forest不需要kernel
        'KNN': None,  # KNN不需要kernel

        # Baseline
        'PolyOnly': None,  # 仅多项式
    }

    all_days = {}

    # 测试四个阶段
    for day_name, selected_day in [('pre_exp', '2020-01-01'), ('stage1', '2020-01-15'), ('stage2', '2020-07-15'), ('stage3', '2020-12-15')]:
        all_days[day_name] = run_comparison(selected_day, methods)

    # 汇总表格
    print("\n" + "="*70)
    print("SUMMARY: Method Comparison Across All Stages")
    print("="*70)
    print(f"{'Method':<25} {'pre_exp':>10} {'stage1':>10} {'stage2':>10} {'stage3':>10} {'Avg':>10}")
    print("-" * 75)

    avg_results = {}
    for method_name in methods.keys():
        r2_values = [all_days[day][method_name]['R2'] for day in ['pre_exp', 'stage1', 'stage2', 'stage3']]
        avg_r2 = np.mean(r2_values)
        avg_results[method_name] = avg_r2
        print(f"  {method_name:<23} {r2_values[0]:>10.4f} {r2_values[1]:>10.4f} {r2_values[2]:>10.4f} {r2_values[3]:>10.4f} {avg_r2:>10.4f}")

    # 找出最佳方法
    best_method = max(avg_results, key=avg_results.get)
    print(f"\nBest Method: {best_method} (Avg R2={avg_results[best_method]:.4f})")
