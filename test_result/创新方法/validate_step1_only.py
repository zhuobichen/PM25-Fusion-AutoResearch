"""
验证：只用多项式校正（Step 1）vs 多项式+GPR（Step 2）
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.paths import get_project_root, data_path
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
import netCDF4 as nc

root_dir = str(get_project_root())
cmaq_file = data_path('test_data/raw/CMAQ/2020_PM25.nc')
monitor_file = data_path('test_data/raw/Monitor/2020_DailyPM2.5Monitor.csv')
fold_file = data_path('test_data/fold_split_table_daily.csv')


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


def run_experiment(selected_day='2020-01-01'):
    print("="*60)
    print("Step 1 Only vs Step 1+2 Comparison")
    print("="*60)

    # 加载数据
    monitor_df = pd.read_csv(monitor_file)
    fold_df = pd.read_csv(fold_file)

    day_df = monitor_df[monitor_df['Date'] == selected_day].copy()
    day_df = day_df.merge(fold_df, on='Site', how='left')
    day_df = day_df.dropna(subset=['Lat', 'Lon', 'Conc'])

    ds = nc.Dataset(cmaq_file, 'r')
    lon_cmaq = ds.variables['lon'][:]
    lat_cmaq = ds.variables['lat'][:]
    pred_pm25 = ds.variables['pred_PM25'][:]
    ds.close()

    from datetime import datetime
    date_obj = datetime.strptime(selected_day, '%Y-%m-%d')
    day_idx = (date_obj - datetime(2020, 1, 1)).days
    pred_day = pred_pm25[day_idx]

    cmaq_values = []
    for _, row in day_df.iterrows():
        val = get_cmaq_at_site(row['Lon'], row['Lat'], lon_cmaq, lat_cmaq, pred_day)
        cmaq_values.append(val)
    day_df['CMAQ'] = cmaq_values

    print(f"Data loaded: {len(day_df)} records")

    # GPR核函数
    kernel = ConstantKernel(10.0, (1e-2, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e1))

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

        # Step 1: 二次多项式 OLS
        poly = PolynomialFeatures(degree=2, include_bias=False)
        m_train_poly = poly.fit_transform(m_train.reshape(-1, 1))
        m_test_poly = poly.transform(m_test.reshape(-1, 1))

        ols_poly = LinearRegression()
        ols_poly.fit(m_train_poly, y_train)
        pred_poly_only = ols_poly.predict(m_test_poly)  # 仅第一步
        residual_poly = y_train - ols_poly.predict(m_train_poly)

        # Step 2: GPR残差建模
        gpr_poly = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, alpha=0.1, normalize_y=True)
        gpr_poly.fit(X_train, residual_poly)
        gpr_poly_pred, _ = gpr_poly.predict(X_test, return_std=True)

        # 完整方法：Step 1 + Step 2
        pred_full = pred_poly_only + gpr_poly_pred

        results[fold_id] = {
            'y_true': y_test,
            'pred_poly_only': pred_poly_only,  # 仅Step 1
            'pred_full': pred_full  # Step 1 + 2
        }

        print(f"  Fold {fold_id}: completed")

    # 汇总
    poly_only_all = np.concatenate([results[f]['pred_poly_only'] for f in range(1, 11) if results[f]])
    full_all = np.concatenate([results[f]['pred_full'] for f in range(1, 11) if results[f]])
    true_all = np.concatenate([results[f]['y_true'] for f in range(1, 11) if results[f]])

    print("\n=== Results ===")
    poly_only_metrics = compute_metrics(true_all, poly_only_all)
    full_metrics = compute_metrics(true_all, full_all)

    print(f"  Step 1 Only (Poly OLS):   R2={poly_only_metrics['R2']:.4f}, MAE={poly_only_metrics['MAE']:.2f}, RMSE={poly_only_metrics['RMSE']:.2f}, MB={poly_only_metrics['MB']:.2f}")
    print(f"  Step 1 + 2 (Poly+GPR):   R2={full_metrics['R2']:.4f}, MAE={full_metrics['MAE']:.2f}, RMSE={full_metrics['RMSE']:.2f}, MB={full_metrics['MB']:.2f}")
    print(f"  Improvement:              R2={full_metrics['R2']-poly_only_metrics['R2']:+.4f}")

    return poly_only_metrics, full_metrics


if __name__ == '__main__':
    print("\n" + "="*60)
    print("Day: 2020-01-01 (pre_exp)")
    print("="*60)
    run_experiment('2020-01-01')

    print("\n" + "="*60)
    print("Day: 2020-01-15 (stage1)")
    print("="*60)
    run_experiment('2020-01-15')
