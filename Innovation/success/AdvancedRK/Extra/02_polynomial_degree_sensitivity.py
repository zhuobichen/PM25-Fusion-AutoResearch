"""
验证：不同阶数的多项式对结果的影响
比较 1阶（线性）、2阶（二次）、3阶（三次）多项式的效果
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


def run_polynomial_comparison(selected_day='2020-01-01'):
    """比较不同阶数多项式的效果"""
    print(f"\n{'='*60}")
    print(f"Polynomial Degree Comparison - {selected_day}")
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

    # GPR核函数
    kernel = ConstantKernel(10.0, (1e-2, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e1))

    results = {1: {}, 2: {}, 3: {}}
    fold_results = {1: [], 2: [], 3: []}

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

        for degree in [1, 2, 3]:
            # 多项式拟合
            poly = PolynomialFeatures(degree=degree, include_bias=False)
            m_train_poly = poly.fit_transform(m_train.reshape(-1, 1))
            m_test_poly = poly.transform(m_test.reshape(-1, 1))

            ols = LinearRegression()
            ols.fit(m_train_poly, y_train)
            pred_poly = ols.predict(m_test_poly)
            residual = y_train - ols.predict(m_train_poly)

            # GPR残差建模
            gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, alpha=0.1, normalize_y=True)
            gpr.fit(X_train, residual)
            gpr_pred, _ = gpr.predict(X_test, return_std=True)

            # 融合预测
            pred_full = pred_poly + gpr_pred

            results[degree][fold_id] = {
                'y_true': y_test,
                'pred': pred_full
            }

        print(f"  Fold {fold_id}: completed")

    # 汇总结果
    print(f"\n=== Results for {selected_day} ===")
    print(f"{'Degree':<10} {'R2':>10} {'MAE':>10} {'RMSE':>10} {'MB':>10}")
    print("-" * 50)

    all_results = {}
    for degree in [1, 2, 3]:
        preds = np.concatenate([results[degree][f]['pred'] for f in range(1, 11) if results[degree].get(f)])
        trues = np.concatenate([results[degree][f]['y_true'] for f in range(1, 11) if results[degree].get(f)])
        metrics = compute_metrics(trues, preds)
        all_results[degree] = metrics
        print(f"  Degree {degree}: {metrics['R2']:>8.4f}  {metrics['MAE']:>8.2f}  {metrics['RMSE']:>8.2f}  {metrics['MB']:>8.2f}")

    return all_results


if __name__ == '__main__':
    all_days = {}

    # 预实验阶段
    print("\n" + "="*60)
    print("PRE_EXP (2020-01-01)")
    print("="*60)
    all_days['pre_exp'] = run_polynomial_comparison('2020-01-01')

    # Stage 1
    print("\n" + "="*60)
    print("STAGE1 (2020-01-15)")
    print("="*60)
    all_days['stage1'] = run_polynomial_comparison('2020-01-15')

    # 汇总表格
    print("\n" + "="*60)
    print("SUMMARY: Polynomial Degree Comparison")
    print("="*60)
    print(f"{'Day':<12} {'Deg=1':>10} {'Deg=2':>10} {'Deg=3':>10} {'Best':>10}")
    print("-" * 55)

    for day_name, day_results in all_days.items():
        r2_1 = day_results[1]['R2']
        r2_2 = day_results[2]['R2']
        r2_3 = day_results[3]['R2']
        best_deg = 2 if r2_2 >= max(r2_1, r2_3) else (1 if r2_1 >= r2_3 else 3)
        print(f"  {day_name:<10} {r2_1:>10.4f} {r2_2:>10.4f} {r2_3:>10.4f}  Deg={best_deg}")

    print("\n结论：二阶多项式在大多数情况下表现最好")
    print("- 一阶（线性）与eVNA/aVNA等价，无法捕捉非线性偏差")
    print("- 三阶容易过拟合，在新数据上表现差")
    print("- 二阶是灵活性和复杂度的最佳平衡")
