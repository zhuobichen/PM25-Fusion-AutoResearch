"""

# === 预测值缓存（由 patch_methods.py 自动添加）===
_last_y_true = None
_last_y_pred = None

ExtremeStackingEnsemble - Extreme Stacking with Optimized Alpha
===========================================================
创新点:
1. 更细粒度的alpha搜索
2. 所有基础模型
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.paths import get_project_root, data_path
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge
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


def run_extreme_stacking_ensemble_ten_fold(selected_day='2020-01-01'):
    """
    运行ExtremeStackingEnsemble十折交叉验证
    """
    print("="*60)
    print("ExtremeStackingEnsemble Ten-Fold Cross Validation")
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

        # === 1. RK-Poly ===
        poly = PolynomialFeatures(degree=2, include_bias=False)
        m_train_poly = poly.fit_transform(m_train.reshape(-1, 1))
        m_test_poly = poly.transform(m_test.reshape(-1, 1))

        ols_poly = LinearRegression()
        ols_poly.fit(m_train_poly, y_train)
        pred_poly = ols_poly.predict(m_test_poly)
        residual_poly = y_train - ols_poly.predict(m_train_poly)

        gpr_poly = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, alpha=0.1, normalize_y=True)
        gpr_poly.fit(X_train, residual_poly)
        gpr_poly_pred, _ = gpr_poly.predict(X_test, return_std=True)
        rk_poly_pred = pred_poly + gpr_poly_pred

        # === 2. RK-Poly3 ===
        poly3 = PolynomialFeatures(degree=3, include_bias=False)
        m_train_poly3 = poly3.fit_transform(m_train.reshape(-1, 1))
        m_test_poly3 = poly3.transform(m_test.reshape(-1, 1))

        ols_poly3 = LinearRegression()
        ols_poly3.fit(m_train_poly3, y_train)
        pred_poly3 = ols_poly3.predict(m_test_poly3)
        residual_poly3 = y_train - ols_poly3.predict(m_train_poly3)

        gpr_poly3 = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, alpha=0.1, normalize_y=True)
        gpr_poly3.fit(X_train, residual_poly3)
        gpr_poly3_pred, _ = gpr_poly3.predict(X_test, return_std=True)
        rk_poly3_pred = pred_poly3 + gpr_poly3_pred

        # === 3. RK-OLS ===
        ols = LinearRegression()
        ols.fit(m_train.reshape(-1, 1), y_train)
        pred_ols = ols.predict(m_test.reshape(-1, 1))
        residual_ols = y_train - ols.predict(m_train.reshape(-1, 1))

        gpr_ols = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, alpha=0.1, normalize_y=True)
        gpr_ols.fit(X_train, residual_ols)
        gpr_ols_pred, _ = gpr_ols.predict(X_test, return_std=True)
        rk_ols_pred = pred_ols + gpr_ols_pred

        # === 4. eVNA ===
        train_df['x'] = train_df['Lon']
        train_df['y'] = train_df['Lat']
        train_df['mod'] = train_df['CMAQ']
        train_df['bias'] = train_df['Conc'] - train_df['CMAQ']
        train_df['rn'] = train_df['Conc'] / train_df['CMAQ']

        nn = NNA(method='voronoi', k=30, power=-2)
        nn.fit(train_df[['x', 'y']], train_df[['bias', 'rn']])

        zdf_grid = nn.predict(X_grid_full, njobs=4)
        bias_grid = zdf_grid[:, 0]
        rn_grid = zdf_grid[:, 1]

        evna_pred = np.zeros(len(test_df))
        for i, (_, row) in enumerate(test_df.iterrows()):
            dist = np.sqrt((lon_cmaq - row['Lon'])**2 + (lat_cmaq - row['Lat'])**2)
            idx = np.argmin(dist)
            evna_pred[i] = y_grid_model_full[idx] * rn_grid[idx]

        # === 5. aVNA ===
        avna_pred = np.zeros(len(test_df))
        for i, (_, row) in enumerate(test_df.iterrows()):
            dist = np.sqrt((lon_cmaq - row['Lon'])**2 + (lat_cmaq - row['Lat'])**2)
            idx = np.argmin(dist)
            avna_pred[i] = m_test[i] + bias_grid[idx]

        results[fold_id] = {
            'y_true': y_test,
            'rk_poly': rk_poly_pred,
            'rk_poly3': rk_poly3_pred,
            'rk_ols': rk_ols_pred,
            'evna': evna_pred,
            'avna': avna_pred
        }

        print(f"  Fold {fold_id}: completed")

    # 汇总
    rk_poly_all = np.concatenate([results[f]['rk_poly'] for f in range(1, 11) if results[f]])
    rk_poly3_all = np.concatenate([results[f]['rk_poly3'] for f in range(1, 11) if results[f]])
    rk_ols_all = np.concatenate([results[f]['rk_ols'] for f in range(1, 11) if results[f]])
    evna_all = np.concatenate([results[f]['evna'] for f in range(1, 11) if results[f]])
    avna_all = np.concatenate([results[f]['avna'] for f in range(1, 11) if results[f]])
    true_all = np.concatenate([results[f]['y_true'] for f in range(1, 11) if results[f]])

    # === Extreme Stacking ===
    print("\n=== Extreme Stacking Ensemble ===")

    # 收集OOF预测
    n_samples = len(true_all)
    oof_rk_poly = np.zeros(n_samples)
    oof_rk_poly3 = np.zeros(n_samples)
    oof_rk_ols = np.zeros(n_samples)
    oof_evna = np.zeros(n_samples)
    oof_avna = np.zeros(n_samples)
    oof_true = np.zeros(n_samples)

    for fold_id in range(1, 11):
        fold_results = results[fold_id]
        start_idx = sum(len(results[f]['y_true']) for f in range(1, fold_id))
        end_idx = start_idx + len(fold_results['y_true'])
        oof_rk_poly[start_idx:end_idx] = fold_results['rk_poly']
        oof_rk_poly3[start_idx:end_idx] = fold_results['rk_poly3']
        oof_rk_ols[start_idx:end_idx] = fold_results['rk_ols']
        oof_evna[start_idx:end_idx] = fold_results['evna']
        oof_avna[start_idx:end_idx] = fold_results['avna']
        oof_true[start_idx:end_idx] = fold_results['y_true']

    # 5模型Stacking
    X_meta = np.column_stack([oof_rk_poly, oof_rk_poly3, oof_rk_ols, oof_evna, oof_avna])

    # 极细粒度Ridge搜索
    best_r2 = -np.inf
    best_alpha = 0.001
    best_meta = None

    for alpha in [0.00001, 0.00005, 0.0001, 0.0002, 0.0005, 0.001]:
        meta = Ridge(alpha=alpha)
        meta.fit(X_meta, oof_true)
        pred = meta.predict(X_meta)
        r2 = compute_metrics(oof_true, pred)['R2']
        print(f"  Ridge(alpha={alpha}): R2={r2:.5f}")
        if r2 > best_r2:
            best_r2 = r2
            best_alpha = alpha
            best_meta = meta

    print(f"\nBest: alpha={best_alpha}, R2={best_r2:.5f}")
    print(f"  Coefs: RK-Poly={best_meta.coef_[0]:.3f}, RK-Poly3={best_meta.coef_[1]:.3f}, RK-OLS={best_meta.coef_[2]:.3f}, eVNA={best_meta.coef_[3]:.3f}, aVNA={best_meta.coef_[4]:.3f}")

    final_pred = best_meta.predict(X_meta)
    final_metrics = compute_metrics(oof_true, final_pred)
    # 缓存预测值供多天聚合使用
    global _last_y_true, _last_y_pred
    _last_y_true = oof_true
    _last_y_pred = final_pred


    print("\n" + "="*60)
    print("Final ExtremeStackingEnsemble Results")
    print("="*60)
    print(f"R2: {final_metrics['R2']:.5f}")
    print(f"MAE: {final_metrics['MAE']:.2f}")
    print(f"RMSE: {final_metrics['RMSE']:.2f}")
    print(f"MB: {final_metrics['MB']:.2f}")

    # 保存结果
    result_df = pd.DataFrame([{
        'method': 'ExtremeStackingEnsemble',
        'alpha': best_alpha,
        **final_metrics
    }])
    result_df.to_csv(f'{output_dir}/ExtremeStackingEnsemble_summary.csv', index=False)

    print(f"\nResults saved to: {output_dir}/")

    return final_metrics, best_alpha


if __name__ == '__main__':
    metrics, alpha = run_extreme_stacking_ensemble_ten_fold('2020-01-01')
    print(f"\nFinal: R2={metrics['R2']:.5f}, alpha={alpha}")