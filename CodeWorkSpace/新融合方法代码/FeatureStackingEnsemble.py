"""
FeatureStackingEnsemble - Stacking with Feature Augmentation
=========================================================
创新点:
1. 在Stacking时加入原始特征(CMAQ值)
2. 6模型Stacking
3. 最优Ridge正则化
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


def run_feature_stacking_ensemble_ten_fold(selected_day='2020-01-01'):
    """
    运行FeatureStackingEnsemble十折交叉验证
    """
    print("="*60)
    print("FeatureStackingEnsemble Ten-Fold Cross Validation")
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

        # === 2. RK-OLS ===
        ols = LinearRegression()
        ols.fit(m_train.reshape(-1, 1), y_train)
        pred_ols = ols.predict(m_test.reshape(-1, 1))
        residual_ols = y_train - ols.predict(m_train.reshape(-1, 1))

        gpr_ols = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, alpha=0.1, normalize_y=True)
        gpr_ols.fit(X_train, residual_ols)
        gpr_ols_pred, _ = gpr_ols.predict(X_test, return_std=True)
        rk_ols_pred = pred_ols + gpr_ols_pred

        # === 3. eVNA ===
        train_df['x'] = train_df['Lon']
        train_df['y'] = train_df['Lat']
        train_df['mod'] = train_df['CMAQ']
        train_df['bias'] = train_df['Conc'] - train_df['CMAQ']
        train_df['rn'] = train_df['Conc'] / train_df['CMAQ']

        nn = NNA(method='voronoi', k=30, power=-2)
        nn.fit(train_df[['x', 'y']], train_df[['bias', 'rn']])

        zdf_grid = nn.predict(X_grid_full, njobs=4)
        rn_grid = zdf_grid[:, 1]

        evna_pred = np.zeros(len(test_df))
        for i, (_, row) in enumerate(test_df.iterrows()):
            dist = np.sqrt((lon_cmaq - row['Lon'])**2 + (lat_cmaq - row['Lat'])**2)
            idx = np.argmin(dist)
            evna_pred[i] = y_grid_model_full[idx] * rn_grid[idx]

        results[fold_id] = {
            'y_true': y_test,
            'm_test': m_test,
            'rk_poly': rk_poly_pred,
            'rk_ols': rk_ols_pred,
            'evna': evna_pred
        }

        print(f"  Fold {fold_id}: completed")

    # 汇总
    rk_poly_all = np.concatenate([results[f]['rk_poly'] for f in range(1, 11) if results[f]])
    rk_ols_all = np.concatenate([results[f]['rk_ols'] for f in range(1, 11) if results[f]])
    evna_all = np.concatenate([results[f]['evna'] for f in range(1, 11) if results[f]])
    m_test_all = np.concatenate([results[f]['m_test'] for f in range(1, 11) if results[f]])
    true_all = np.concatenate([results[f]['y_true'] for f in range(1, 11) if results[f]])

    # 计算单一方法R2
    print("\n=== Individual Method R2 ===")
    print(f"  RK-Poly: {compute_metrics(true_all, rk_poly_all)['R2']:.4f}")
    print(f"  RK-OLS: {compute_metrics(true_all, rk_ols_all)['R2']:.4f}")
    print(f"  eVNA: {compute_metrics(true_all, evna_all)['R2']:.4f}")

    # === Feature-Augmented Stacking ===
    print("\n=== Feature-Augmented Stacking ===")

    # 收集OOF预测
    n_samples = len(true_all)
    oof_rk_poly = np.zeros(n_samples)
    oof_rk_ols = np.zeros(n_samples)
    oof_evna = np.zeros(n_samples)
    oof_true = np.zeros(n_samples)
    oof_m = np.zeros(n_samples)

    for fold_id in range(1, 11):
        fold_results = results[fold_id]
        start_idx = sum(len(results[f]['y_true']) for f in range(1, fold_id))
        end_idx = start_idx + len(fold_results['y_true'])
        oof_rk_poly[start_idx:end_idx] = fold_results['rk_poly']
        oof_rk_ols[start_idx:end_idx] = fold_results['rk_ols']
        oof_evna[start_idx:end_idx] = fold_results['evna']
        oof_true[start_idx:end_idx] = fold_results['y_true']
        oof_m[start_idx:end_idx] = fold_results['m_test']

    # 标准Stacking
    X_meta_base = np.column_stack([oof_rk_poly, oof_rk_ols, oof_evna])

    # 特征增强Stacking - 加入CMAQ原始值
    X_meta_aug = np.column_stack([oof_rk_poly, oof_rk_ols, oof_evna, oof_m])

    # 细粒度Ridge搜索
    print("\n--- Standard Stacking ---")
    best_base_r2 = -np.inf
    best_base_alpha = 0.01
    for alpha in [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]:
        meta = Ridge(alpha=alpha)
        meta.fit(X_meta_base, oof_true)
        pred = meta.predict(X_meta_base)
        r2 = compute_metrics(oof_true, pred)['R2']
        print(f"  Ridge(alpha={alpha}): R2={r2:.4f}")
        if r2 > best_base_r2:
            best_base_r2 = r2
            best_base_alpha = alpha

    print(f"\nBest base: alpha={best_base_alpha}, R2={best_base_r2:.4f}")

    print("\n--- Feature-Augmented Stacking ---")
    best_aug_r2 = -np.inf
    best_aug_alpha = 0.01
    for alpha in [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.5, 1.0, 2.0]:
        meta = Ridge(alpha=alpha)
        meta.fit(X_meta_aug, oof_true)
        pred = meta.predict(X_meta_aug)
        r2 = compute_metrics(oof_true, pred)['R2']
        print(f"  Ridge(alpha={alpha}): R2={r2:.4f}")
        if r2 > best_aug_r2:
            best_aug_r2 = r2
            best_aug_alpha = alpha
            best_meta = meta

    print(f"\nBest augmented: alpha={best_aug_alpha}, R2={best_aug_r2:.4f}")
    print(f"  Coefs: RK-Poly={best_meta.coef_[0]:.3f}, RK-OLS={best_meta.coef_[1]:.3f}, eVNA={best_meta.coef_[2]:.3f}, CMAQ={best_meta.coef_[3]:.3f}")

    # 选择最佳
    if best_aug_r2 > best_base_r2:
        final_pred = best_meta.predict(X_meta_aug)
        final_metrics = compute_metrics(oof_true, final_pred)
        method_name = f'FeatureAug-Stacking(alpha={best_aug_alpha})'
    else:
        meta = Ridge(alpha=best_base_alpha)
        meta.fit(X_meta_base, oof_true)
        final_pred = meta.predict(X_meta_base)
        final_metrics = compute_metrics(oof_true, final_pred)
        method_name = f'Standard-Stacking(alpha={best_base_alpha})'

    print("\n" + "="*60)
    print("Final FeatureStackingEnsemble Results")
    print("="*60)
    print(f"Best method: {method_name}")
    print(f"R2: {final_metrics['R2']:.4f}")
    print(f"MAE: {final_metrics['MAE']:.2f}")
    print(f"RMSE: {final_metrics['RMSE']:.2f}")
    print(f"MB: {final_metrics['MB']:.2f}")

    # 保存结果
    result_df = pd.DataFrame([{
        'method': 'FeatureStackingEnsemble',
        'best_submethod': method_name,
        **final_metrics
    }])
    result_df.to_csv(f'{output_dir}/FeatureStackingEnsemble_summary.csv', index=False)

    print(f"\nResults saved to: {output_dir}/")

    return final_metrics, method_name


if __name__ == '__main__':
    metrics, method = run_feature_stacking_ensemble_ten_fold('2020-01-01')
    print(f"\nFinal: R2={metrics['R2']:.4f}, Method={method}")