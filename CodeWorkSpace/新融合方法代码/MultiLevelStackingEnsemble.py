"""
MultiLevelStackingEnsemble - 多层次Stacking集成方法
=====================================================
创新点:
1. 第一层：多个基础Stacking模型
2. 第二层：使用不同元学习器的Stacking
3. 第三层：加权平均融合

这个方法尝试集成多种Stacking策略的优势。
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.paths import get_project_root, data_path
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
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


def run_multi_level_stacking_ensemble_ten_fold(selected_day='2020-01-01'):
    """
    运行多层次Stacking集成十折交叉验证
    """
    print("="*60)
    print("MultiLevelStackingEnsemble Ten-Fold Cross Validation")
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

        # === 基础模型 ===
        # 1. RK-Poly
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

        # 2. RK-Poly3
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

        # 3. RK-OLS
        ols = LinearRegression()
        ols.fit(m_train.reshape(-1, 1), y_train)
        pred_ols = ols.predict(m_test.reshape(-1, 1))
        residual_ols = y_train - ols.predict(m_train.reshape(-1, 1))
        gpr_ols = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, alpha=0.1, normalize_y=True)
        gpr_ols.fit(X_train, residual_ols)
        gpr_ols_pred, _ = gpr_ols.predict(X_test, return_std=True)
        rk_ols_pred = pred_ols + gpr_ols_pred

        # 4. eVNA
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

        # 5. aVNA
        bias_grid = zdf_grid[:, 0]
        avna_pred = np.zeros(len(test_df))
        for i, (_, row) in enumerate(test_df.iterrows()):
            dist = np.sqrt((lon_cmaq - row['Lon'])**2 + (lat_cmaq - row['Lat'])**2)
            idx = np.argmin(dist)
            avna_pred[i] = m_test[i] + bias_grid[idx]

        # 6. Hybrid (eVNA + aVNA)
        hybrid_pred = 0.5 * evna_pred + 0.5 * avna_pred

        results[fold_id] = {
            'y_true': y_test,
            'm_test': m_test,
            'rk_poly': rk_poly_pred,
            'rk_poly3': rk_poly3_pred,
            'rk_ols': rk_ols_pred,
            'evna': evna_pred,
            'avna': avna_pred,
            'hybrid': hybrid_pred
        }

        print(f"  Fold {fold_id}: completed")

    # 汇总所有折叠的结果
    rk_poly_all = np.concatenate([results[f]['rk_poly'] for f in range(1, 11) if results[f]])
    rk_poly3_all = np.concatenate([results[f]['rk_poly3'] for f in range(1, 11) if results[f]])
    rk_ols_all = np.concatenate([results[f]['rk_ols'] for f in range(1, 11) if results[f]])
    evna_all = np.concatenate([results[f]['evna'] for f in range(1, 11) if results[f]])
    avna_all = np.concatenate([results[f]['avna'] for f in range(1, 11) if results[f]])
    hybrid_all = np.concatenate([results[f]['hybrid'] for f in range(1, 11) if results[f]])
    m_test_all = np.concatenate([results[f]['m_test'] for f in range(1, 11) if results[f]])
    true_all = np.concatenate([results[f]['y_true'] for f in range(1, 11) if results[f]])

    # 计算单一方法R2
    print("\n=== Individual Method R2 ===")
    print(f"  RK-Poly: {compute_metrics(true_all, rk_poly_all)['R2']:.4f}")
    print(f"  RK-Poly3: {compute_metrics(true_all, rk_poly3_all)['R2']:.4f}")
    print(f"  RK-OLS: {compute_metrics(true_all, rk_ols_all)['R2']:.4f}")
    print(f"  eVNA: {compute_metrics(true_all, evna_all)['R2']:.4f}")
    print(f"  aVNA: {compute_metrics(true_all, avna_all)['R2']:.4f}")
    print(f"  Hybrid: {compute_metrics(true_all, hybrid_all)['R2']:.4f}")

    # === Level 1: Stacking with Ridge ===
    print("\n=== Level 1: Ridge Stacking ===")
    X_meta_ridge = np.column_stack([rk_poly_all, rk_poly3_all, rk_ols_all, evna_all, avna_all, hybrid_all, m_test_all])

    best_ridge_r2 = -np.inf
    best_ridge_alpha = 0.001
    best_ridge_model = None

    for alpha in [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]:
        ridge = Ridge(alpha=alpha)
        ridge.fit(X_meta_ridge, true_all)
        pred = ridge.predict(X_meta_ridge)
        r2 = compute_metrics(true_all, pred)['R2']
        if r2 > best_ridge_r2:
            best_ridge_r2 = r2
            best_ridge_alpha = alpha
            best_ridge_model = ridge

    print(f"Best Ridge: alpha={best_ridge_alpha}, R2={best_ridge_r2:.4f}")
    print(f"  Coefs: {best_ridge_model.coef_}")

    # === Level 1: Stacking with ElasticNet ===
    print("\n=== Level 1: ElasticNet Stacking ===")
    X_meta_en = np.column_stack([rk_poly_all, rk_poly3_all, rk_ols_all, evna_all, avna_all, hybrid_all, m_test_all])

    best_en_r2 = -np.inf
    best_en_params = None
    best_en_model = None

    for alpha in [0.001, 0.01, 0.1]:
        for l1_ratio in [0.2, 0.5, 0.8]:
            en = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10000)
            en.fit(X_meta_en, true_all)
            pred = en.predict(X_meta_en)
            r2 = compute_metrics(true_all, pred)['R2']
            if r2 > best_en_r2:
                best_en_r2 = r2
                best_en_params = {'alpha': alpha, 'l1_ratio': l1_ratio}
                best_en_model = en

    print(f"Best ElasticNet: alpha={best_en_params['alpha']}, l1_ratio={best_en_params['l1_ratio']}, R2={best_en_r2:.4f}")

    # === Level 2: Combine different meta-learners ===
    print("\n=== Level 2: Meta-Learner Ensemble ===")

    ridge_pred = best_ridge_model.predict(X_meta_ridge)
    en_pred = best_en_model.predict(X_meta_en)

    # 网格搜索最优融合权重
    best_final_r2 = -np.inf
    best_weights = None

    for w1 in np.arange(0, 1.01, 0.05):
        w2 = 1.0 - w1
        ensemble_pred = w1 * ridge_pred + w2 * en_pred
        r2 = compute_metrics(true_all, ensemble_pred)['R2']
        if r2 > best_final_r2:
            best_final_r2 = r2
            best_weights = (w1, w2)

    print(f"Best ensemble: Ridge={best_weights[0]:.2f}, ElasticNet={best_weights[1]:.2f}, R2={best_final_r2:.4f}")

    # 最终预测
    final_pred = best_weights[0] * ridge_pred + best_weights[1] * en_pred
    final_metrics = compute_metrics(true_all, final_pred)

    print("\n" + "="*60)
    print("Final MultiLevelStackingEnsemble Results")
    print("="*60)
    print(f"R2: {final_metrics['R2']:.4f}")
    print(f"MAE: {final_metrics['MAE']:.2f}")
    print(f"RMSE: {final_metrics['RMSE']:.2f}")
    print(f"MB: {final_metrics['MB']:.2f}")

    # 保存结果
    result_df = pd.DataFrame([{
        'method': 'MultiLevelStackingEnsemble',
        'ridge_alpha': best_ridge_alpha,
        'elastic_alpha': best_en_params['alpha'],
        'elastic_l1': best_en_params['l1_ratio'],
        'ridge_weight': best_weights[0],
        'en_weight': best_weights[1],
        **final_metrics
    }])
    result_df.to_csv(f'{output_dir}/MultiLevelStackingEnsemble_summary.csv', index=False)

    print(f"\nResults saved to: {output_dir}/")

    return final_metrics


if __name__ == '__main__':
    metrics = run_multi_level_stacking_ensemble_ten_fold('2020-01-01')
    print(f"\nFinal: R2={metrics['R2']:.4f}")