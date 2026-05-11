"""

# === 预测值缓存（由 patch_methods.py 自动添加）===
_last_y_true = None
_last_y_pred = None

MSRK - Multi-Scale Residual Kriging
=====================================
使用多尺度GPR进行残差克里金插值

创新点:
1. 全局 OLS 多项式校正: O = a + b*M + c*M² + ε
2. 多尺度 GPR: 3 个不同长度尺度的 GPR (7km/20km/50km)
3. 加权融合多尺度预测

参数:
- scales: [7.0, 20.0, 50.0] (km)
- scale_weights: [0.2, 0.5, 0.3]
- poly_degree: 2
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


def run_msrk_ten_fold(selected_day='2020-01-01', scales=[7.0, 20.0, 50.0], scale_weights=[0.2, 0.5, 0.3], poly_degree=2):
    """
    运行MSRK十折交叉验证
    """
    print("="*60)
    print("MSRK Ten-Fold Cross Validation")
    print(f"Parameters: scales={scales}, weights={scale_weights}, poly_degree={poly_degree}")
    print("="*60)

    # 加载数据
    print("\n=== Loading Data ===")
    monitor_df = pd.read_csv(monitor_file)
    fold_df = pd.read_csv(fold_file)

    day_df = monitor_df[monitor_df['Date'] == selected_day].copy()
    day_df = day_df.merge(fold_df, on=['Date', 'Site'], how='left')
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

    # 定义多尺度GPR核函数
    def make_multiscale_kernel(length_scale):
        return (ConstantKernel(10.0, (1e-2, 1e3)) *
                RBF(length_scale=length_scale, length_scale_bounds=(1e-2, 1e2)) +
                WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e1)))

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
        poly = PolynomialFeatures(degree=poly_degree, include_bias=False)
        m_train_poly = poly.fit_transform(m_train.reshape(-1, 1))
        m_test_poly = poly.transform(m_test.reshape(-1, 1))

        # OLS 多项式校正
        ols = LinearRegression()
        ols.fit(m_train_poly, y_train)
        pred_ols = ols.predict(m_test_poly)
        residual = y_train - ols.predict(m_train_poly)

        # 多尺度 GPR 预测
        multi_scale_preds = []
        for scale in scales:
            kernel = make_multiscale_kernel(scale)
            gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, alpha=0.1, normalize_y=True)
            gpr.fit(X_train, residual)
            gpr_pred, _ = gpr.predict(X_test, return_std=True)
            multi_scale_preds.append(gpr_pred)

        multi_scale_preds = np.array(multi_scale_preds)  # (n_scales, n_test)

        # 加权融合
        scale_weights_arr = np.array(scale_weights)
        scale_weights_arr = scale_weights_arr / scale_weights_arr.sum()
        gpr_fusion_pred = np.sum(multi_scale_preds * scale_weights_arr.reshape(-1, 1), axis=0)

        # 融合预测
        msrk_pred = pred_ols + gpr_fusion_pred

        # 对比: 标准 RK (单尺度 GPR)
        kernel = make_multiscale_kernel(20.0)
        gpr_single = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, alpha=0.1, normalize_y=True)
        gpr_single.fit(X_train, residual)
        gpr_single_pred, _ = gpr_single.predict(X_test, return_std=True)
        rk_pred = pred_ols + gpr_single_pred

        results[fold_id] = {
            'y_true': y_test,
            'msrk': msrk_pred,
            'rk': rk_pred
        }

        print(f"  Fold {fold_id}: completed")

    # 汇总
    msrk_all = np.concatenate([results[f]['msrk'] for f in range(1, 11) if results[f]])
    rk_all = np.concatenate([results[f]['rk'] for f in range(1, 11) if results[f]])
    true_all = np.concatenate([results[f]['y_true'] for f in range(1, 11) if results[f]])

    # 计算指标
    print("\n=== Results ===")
    msrk_metrics = compute_metrics(true_all, msrk_all)
    rk_metrics = compute_metrics(true_all, rk_all)
    # 缓存预测值供多天聚合使用
    global _last_y_true, _last_y_pred
    _last_y_true = true_all
    _last_y_pred = rk_all


    print(f"  MSRK: R2={msrk_metrics['R2']:.4f}, MAE={msrk_metrics['MAE']:.2f}, RMSE={msrk_metrics['RMSE']:.2f}, MB={msrk_metrics['MB']:.2f}")
    print(f"  RK:   R2={rk_metrics['R2']:.4f}, MAE={rk_metrics['MAE']:.2f}, RMSE={rk_metrics['RMSE']:.2f}, MB={rk_metrics['MB']:.2f}")

    # 保存结果
    result_df = pd.DataFrame([{
        'Method': 'MSRK',
        'R2': msrk_metrics['R2'],
        'MAE': msrk_metrics['MAE'],
        'RMSE': msrk_metrics['RMSE'],
        'MB': msrk_metrics['MB']
    }, {
        'Method': 'RK_single_scale',
        'R2': rk_metrics['R2'],
        'MAE': rk_metrics['MAE'],
        'RMSE': rk_metrics['RMSE'],
        'MB': rk_metrics['MB']
    }])
    result_df.to_csv(f'{output_dir}/MSRK_summary.csv', index=False)

    print(f"\nResults saved to: {output_dir}/MSRK_summary.csv")

    return msrk_metrics


if __name__ == '__main__':
    metrics = run_msrk_ten_fold('2020-01-01')
    print(f"\nFinal: R2={metrics['R2']:.4f}")
