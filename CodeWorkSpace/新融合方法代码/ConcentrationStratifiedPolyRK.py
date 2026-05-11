"""

# === 预测值缓存（由 patch_methods.py 自动添加）===
_last_y_true = None
_last_y_pred = None

ConcentrationStratifiedPolyRK - 浓度分层多项式残差克里金
=========================================================
把CMAQ数据按浓度分为高/中/低三层，分别拟合独立的多项式校正参数。

创新点:
1. 按浓度分层: M < T1 (低), T1 <= M < T2 (中), M >= T2 (高)
2. 每层独立做多项式OLS校正，捕捉不同浓度区的偏差特性
3. 合并残差做统一GPR克里金
4. 预测时按测试点浓度选择对应层的OLS

阈值: T1=35 ug/m3, T2=75 ug/m3
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

T1 = 35.0
T2 = 75.0


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


def get_concentration_layer(m_value):
    if m_value < T1:
        return 0
    elif m_value < T2:
        return 1
    else:
        return 2


def run_ConcentrationStratifiedPolyRK_ten_fold(selected_day='2020-01-01'):
    print("=" * 60)
    print("ConcentrationStratifiedPolyRK Ten-Fold Cross Validation")
    print("=" * 60)

    print("\n=== Loading Data ===")
    monitor_df = pd.read_csv(monitor_file)
    fold_df = pd.read_csv(fold_file)

    day_df = monitor_df[monitor_df['Date'] == selected_day].copy()
    day_df = day_df.merge(fold_df, on=['Date', 'Site'], how='left')
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

    print("=== Extracting CMAQ at Sites ===")
    cmaq_values = []
    for _, row in day_df.iterrows():
        val = get_cmaq_at_site(row['Lon'], row['Lat'], lon_cmaq, lat_cmaq, pred_day)
        cmaq_values.append(val)
    day_df['CMAQ'] = cmaq_values

    print(f"Data loaded: {len(day_df)} monitoring records")

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

        # Step 1: 浓度分层
        layers_train = np.array([get_concentration_layer(v) for v in m_train])
        layers_test = np.array([get_concentration_layer(v) for v in m_test])

        poly = PolynomialFeatures(degree=2, include_bias=False)

        # Step 2: 分层多项式拟合
        layer_models = {}
        residuals_list = []
        X_residuals_list = []

        for layer_id in [0, 1, 2]:
            mask = layers_train == layer_id
            if np.sum(mask) < 3:
                layer_models[layer_id] = None
                continue

            m_layer = m_train[mask]
            y_layer = y_train[mask]
            X_layer = X_train[mask]

            m_poly = poly.fit_transform(m_layer.reshape(-1, 1))
            ols = LinearRegression()
            ols.fit(m_poly, y_layer)
            residual_layer = y_layer - ols.predict(m_poly)

            layer_models[layer_id] = ols
            residuals_list.append(residual_layer)
            X_residuals_list.append(X_layer)

        # Step 3: 合并残差做GPR
        if len(residuals_list) > 0:
            residual_all = np.concatenate(residuals_list)
            X_all = np.vstack(X_residuals_list)

            residual_mean = np.mean(residual_all)
            residual_std = np.std(residual_all) + 1e-8
            residual_norm = (residual_all - residual_mean) / residual_std

            gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, alpha=0.1, normalize_y=True)
            gpr.fit(X_all, residual_norm)

            # Step 4: 预测
            pred_ols = np.zeros(len(m_test))
            for i, m_val in enumerate(m_test):
                layer_id = layers_test[i]
                m_poly_i = poly.transform([[m_val]])
                if layer_models.get(layer_id) is not None:
                    pred_ols[i] = layer_models[layer_id].predict(m_poly_i)[0]
                else:
                    # fallback: 使用所有有效层的平均
                    valid_preds = [layer_models[l].predict(m_poly_i)[0] for l in layer_models if layer_models[l] is not None]
                    pred_ols[i] = np.mean(valid_preds) if valid_preds else m_val

            gpr_pred_norm = gpr.predict(X_test)
            gpr_pred = gpr_pred_norm * residual_std + residual_mean
            csprk_pred = pred_ols + gpr_pred
        else:
            csprk_pred = m_test

        results[fold_id] = {
            'y_true': y_test,
            'csprk': csprk_pred
        }
        print(f"  Fold {fold_id}: completed")

    # 汇总
    csprk_all = np.concatenate([results[f]['csprk'] for f in range(1, 11) if results[f]])
    true_all = np.concatenate([results[f]['y_true'] for f in range(1, 11) if results[f]])

    print("\n=== Results ===")
    csprk_metrics = compute_metrics(true_all, csprk_all)
    # 缓存预测值供多天聚合使用
    global _last_y_true, _last_y_pred
    _last_y_true = true_all
    _last_y_pred = csprk_all

    print(f"  CSP-RK: R2={csprk_metrics['R2']:.4f}, MAE={csprk_metrics['MAE']:.2f}, RMSE={csprk_metrics['RMSE']:.2f}")

    result_df = pd.DataFrame([{'method': 'ConcentrationStratifiedPolyRK', **csprk_metrics}])
    result_df.to_csv(f'{output_dir}/ConcentrationStratifiedPolyRK_summary.csv', index=False)
    print(f"\nResults saved to: {output_dir}/ConcentrationStratifiedPolyRK_summary.csv")

    return csprk_metrics


if __name__ == '__main__':
    metrics = run_ConcentrationStratifiedPolyRK_ten_fold('2020-01-01')
    print(f"\nFinal: R2={metrics['R2']:.4f}")
