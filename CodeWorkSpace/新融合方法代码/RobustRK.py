"""
RobustRK - Robust Polynomial Residual Kriging
=============================================
使用Huber回归进行稳健的多项式残差克里金
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.paths import get_project_root, data_path
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import HuberRegressor, LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
import netCDF4 as nc
from Code.VNAeVNAaVNA.nna_methods import NNA

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


def run_robust_rk_ten_fold(selected_day='2020-01-01'):
    """
    运行RobustRK十折交叉验证
    """
    print("="*60)
    print("RobustRK Ten-Fold Cross Validation")
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

    # 定义GPR核函数
    kernel = ConstantKernel(10.0, (1e-2, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e1))

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

        # 多项式特征
        poly = PolynomialFeatures(degree=2, include_bias=False)
        m_train_poly = poly.fit_transform(m_train.reshape(-1, 1))
        m_test_poly = poly.transform(m_test.reshape(-1, 1))

        # === 1. OLS多项式 ===
        ols_poly = LinearRegression()
        ols_poly.fit(m_train_poly, y_train)
        pred_ols = ols_poly.predict(m_test_poly)
        residual_ols = y_train - ols_poly.predict(m_train_poly)

        # === 2. Huber稳健多项式 ===
        huber_poly = HuberRegressor(epsilon=1.35, max_iter=1000)
        huber_poly.fit(m_train_poly, y_train)
        pred_huber = huber_poly.predict(m_test_poly)
        residual_huber = y_train - huber_poly.predict(m_train_poly)

        # GPR on residuals
        gpr_ols = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, alpha=0.1, normalize_y=True)
        gpr_ols.fit(X_train, residual_ols)
        gpr_ols_pred, _ = gpr_ols.predict(X_test, return_std=True)

        gpr_huber = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, alpha=0.1, normalize_y=True)
        gpr_huber.fit(X_train, residual_huber)
        gpr_huber_pred, _ = gpr_huber.predict(X_test, return_std=True)

        # 融合预测
        rk_ols_pred = pred_ols + gpr_ols_pred
        rk_huber_pred = pred_huber + gpr_huber_pred

        results[fold_id] = {
            'y_true': y_test,
            'rk_ols': rk_ols_pred,
            'rk_huber': rk_huber_pred
        }

        print(f"  Fold {fold_id}: completed")

    # 汇总
    rk_ols_all = np.concatenate([results[f]['rk_ols'] for f in range(1, 11) if results[f]])
    rk_huber_all = np.concatenate([results[f]['rk_huber'] for f in range(1, 11) if results[f]])
    true_all = np.concatenate([results[f]['y_true'] for f in range(1, 11) if results[f]])

    # 计算R2
    print("\n=== Results ===")
    ols_metrics = compute_metrics(true_all, rk_ols_all)
    huber_metrics = compute_metrics(true_all, rk_huber_all)

    print(f"  RK-OLS-Poly:  R2={ols_metrics['R2']:.4f}, MAE={ols_metrics['MAE']:.2f}, RMSE={ols_metrics['RMSE']:.2f}")
    print(f"  RK-Huber-Poly: R2={huber_metrics['R2']:.4f}, MAE={huber_metrics['MAE']:.2f}, RMSE={huber_metrics['RMSE']:.2f}")

    # 保存结果
    result_df = pd.DataFrame([{
        'method': 'RK_OLS_Poly',
        **ols_metrics
    }, {
        'method': 'RK_Huber_Poly',
        **huber_metrics
    }])
    result_df.to_csv(f'{output_dir}/RobustRK_summary.csv', index=False)

    print(f"\nResults saved to: {output_dir}/")

    return ols_metrics, huber_metrics


if __name__ == '__main__':
    ols_metrics, huber_metrics = run_robust_rk_ten_fold('2020-01-01')
    print(f"\nOLS: R2={ols_metrics['R2']:.4f}")
    print(f"Huber: R2={huber_metrics['R2']:.4f}")