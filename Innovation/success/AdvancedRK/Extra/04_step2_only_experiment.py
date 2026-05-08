"""
验证：跳过Step 1（多项式），直接从Step 2（GPR）开始
比较：
- Step 1+2（多项式+GPR）：原始AdvancedRK
- Step 2 Only（GPR Only）：跳过多项式，直接用GPR建模
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
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
import netCDF4 as nc

root_dir = str(get_project_root())
cmaq_file = data_path('test_data/raw/CMAQ/2020_PM25.nc')
monitor_file = data_path('test_data/raw/Monitor/2020_DailyPM2.5Monitor.csv')
fold_file = data_path('test_data/fold_split_table_daily.csv')
output_dir = f'{root_dir}/Innovation/success/AdvancedRK/Extra'
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


def run_experiment(selected_day='2020-01-01'):
    """比较 Step 1+2 vs Step 2 Only"""
    print(f"\n{'='*60}")
    print(f"Step 1+2 vs Step 2 Only Comparison - {selected_day}")
    print(f"{'='*60}")

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

    # GPR核函数（Matern 1.5）
    kernel = ConstantKernel(10.0, (1e-2, 1e3)) * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=1.5) + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e1))

    results = {fold_id: {} for fold_id in range(1, 11)}

    for fold_id in range(1, 11):
        train_df = day_df[day_df['fold'] != fold_id].copy()
        test_df = day_df[day_df['fold'] == fold_id].copy()

        train_df = train_df.dropna(subset=['Lon', 'Lat', 'CMAQ', 'Conc'])
        test_df = test_df.dropna(subset=['Lon', 'Lat', 'CMAQ', 'Conc'])

        if len(test_df) == 0:
            continue

        y_train = train_df['Conc'].values
        y_test = test_df['Conc'].values
        m_train = train_df['CMAQ'].values
        m_test = test_df['CMAQ'].values

        # ===== 方法1：Step 1 + Step 2（原始AdvancedRK）=====
        # Step 1: 多项式校正
        poly = PolynomialFeatures(degree=2, include_bias=False)
        m_train_poly = poly.fit_transform(m_train.reshape(-1, 1))
        m_test_poly = poly.transform(m_test.reshape(-1, 1))

        ols = LinearRegression()
        ols.fit(m_train_poly, y_train)
        pred_poly = ols.predict(m_test_poly)
        residual = y_train - ols.predict(m_train_poly)

        # Step 2: GPR残差建模
        X_train = train_df[['Lon', 'Lat']].values
        X_test = test_df[['Lon', 'Lat']].values

        gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0, alpha=0.1, normalize_y=True)
        gpr.fit(X_train, residual)
        gpr_pred, _ = gpr.predict(X_test, return_std=True)

        pred_step12 = pred_poly + gpr_pred

        # ===== 方法2：Step 2 Only（跳过多项式）=====
        # 直接用 (CMAQ, Lon, Lat) 作为特征
        X_train_full = np.column_stack([m_train, train_df['Lon'].values, train_df['Lat'].values])
        X_test_full = np.column_stack([m_test, test_df['Lon'].values, test_df['Lat'].values])

        gpr_direct = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0, alpha=0.1, normalize_y=True)
        gpr_direct.fit(X_train_full, y_train)
        pred_step2_only, _ = gpr_direct.predict(X_test_full, return_std=True)

        results[fold_id] = {
            'y_true': y_test,
            'pred_step12': pred_step12,
            'pred_step2_only': pred_step2_only
        }

        print(f"  Fold {fold_id}: completed")

    # 汇总
    print(f"\n=== Results for {selected_day} ===")
    print(f"{'Method':<25} {'R2':>10} {'MAE':>10} {'RMSE':>10}")
    print("-" * 50)

    step12_all = np.concatenate([results[f]['pred_step12'] for f in range(1, 11) if results[f]])
    step2_only_all = np.concatenate([results[f]['pred_step2_only'] for f in range(1, 11) if results[f]])
    true_all = np.concatenate([results[f]['y_true'] for f in range(1, 11) if results[f]])

    metrics_step12 = compute_metrics(true_all, step12_all)
    metrics_step2_only = compute_metrics(true_all, step2_only_all)

    print(f"  Step 1+2 (Poly+GPR):   {metrics_step12['R2']:>10.4f} {metrics_step12['MAE']:>10.2f} {metrics_step12['RMSE']:>10.2f}")
    print(f"  Step 2 Only (GPR):     {metrics_step2_only['R2']:>10.4f} {metrics_step2_only['MAE']:>10.2f} {metrics_step2_only['RMSE']:>10.2f}")
    print(f"  Difference:            {metrics_step12['R2'] - metrics_step2_only['R2']:>+10.4f}")

    return {
        'step12': metrics_step12,
        'step2_only': metrics_step2_only
    }


if __name__ == '__main__':
    all_days = {}

    for day_name, selected_day in [('pre_exp', '2020-01-01'), ('stage1', '2020-01-15'), ('stage2', '2020-07-15'), ('stage3', '2020-12-15')]:
        print(f"\n{'='*60}")
        print(f"Day: {selected_day} ({day_name})")
        print(f"{'='*60}")
        all_days[day_name] = run_experiment(selected_day)

    # 汇总表格
    print("\n" + "="*70)
    print("SUMMARY: Step 1+2 vs Step 2 Only")
    print("="*70)
    print(f"{'Day':<10} {'Step 1+2':>12} {'Step 2 Only':>12} {'Difference':>12}")
    print("-" * 50)
    for day_name in ['pre_exp', 'stage1', 'stage2', 'stage3']:
        r2_12 = all_days[day_name]['step12']['R2']
        r2_2 = all_days[day_name]['step2_only']['R2']
        diff = r2_12 - r2_2
        print(f"  {day_name:<8} {r2_12:>12.4f} {r2_2:>12.4f} {diff:>+12.4f}")

    print("\n结论：Step 1（多项式）的作用是捕捉CMAQ与监测值的非线性偏差关系，")
    print("如果跳过这一步，GPR需要同时学习：(1)CMAQ的非线性偏差 (2)空间残差")
    print("两步分离让GPR只专注于空间建模，降低学习难度")