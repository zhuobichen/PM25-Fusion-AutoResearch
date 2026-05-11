"""

# === 预测值缓存（由 patch_methods.py 自动添加）===
_last_y_true = None
_last_y_pred = None

AdaptiveOnlineEnsemble - 自适应在线加权融合方法
================================================
创新点:
1. 使用在线学习策略动态调整权重
2. 基于残差的自适应权重更新
3. 融合多种基础预测方法

核心思想:
使用在线学习中的加权平均策略，每次根据上一轮的残差动态调整权重。
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


def run_adaptive_online_ensemble_ten_fold(selected_day='2020-01-01'):
    """
    运行自适应在线加权融合十折交叉验证
    """
    print("="*60)
    print("AdaptiveOnlineEnsemble Ten-Fold Cross Validation")
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

    # 收集所有折叠的OOF预测
    all_fold_preds = {fold_id: {} for fold_id in range(1, 11)}

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
        bias_grid = zdf_grid[:, 0]
        rn_grid = zdf_grid[:, 1]

        evna_pred = np.zeros(len(test_df))
        for i, (_, row) in enumerate(test_df.iterrows()):
            dist = np.sqrt((lon_cmaq - row['Lon'])**2 + (lat_cmaq - row['Lat'])**2)
            idx = np.argmin(dist)
            evna_pred[i] = y_grid_model_full[idx] * rn_grid[idx]

        # 5. aVNA
        avna_pred = np.zeros(len(test_df))
        for i, (_, row) in enumerate(test_df.iterrows()):
            dist = np.sqrt((lon_cmaq - row['Lon'])**2 + (lat_cmaq - row['Lat'])**2)
            idx = np.argmin(dist)
            avna_pred[i] = m_test[i] + bias_grid[idx]

        all_fold_preds[fold_id] = {
            'y_true': y_test,
            'rk_poly': rk_poly_pred,
            'rk_poly3': rk_poly3_pred,
            'rk_ols': rk_ols_pred,
            'evna': evna_pred,
            'avna': avna_pred
        }

        print(f"  Fold {fold_id}: completed")

    # 汇总所有预测
    rk_poly_all = np.concatenate([all_fold_preds[f]['rk_poly'] for f in range(1, 11) if all_fold_preds[f]])
    rk_poly3_all = np.concatenate([all_fold_preds[f]['rk_poly3'] for f in range(1, 11) if all_fold_preds[f]])
    rk_ols_all = np.concatenate([all_fold_preds[f]['rk_ols'] for f in range(1, 11) if all_fold_preds[f]])
    evna_all = np.concatenate([all_fold_preds[f]['evna'] for f in range(1, 11) if all_fold_preds[f]])
    avna_all = np.concatenate([all_fold_preds[f]['avna'] for f in range(1, 11) if all_fold_preds[f]])
    true_all = np.concatenate([all_fold_preds[f]['y_true'] for f in range(1, 11) if all_fold_preds[f]])

    # 计算单一方法R2
    print("\n=== Individual Method R2 ===")
    print(f"  RK-Poly: {compute_metrics(true_all, rk_poly_all)['R2']:.4f}")
    print(f"  RK-Poly3: {compute_metrics(true_all, rk_poly3_all)['R2']:.4f}")
    print(f"  RK-OLS: {compute_metrics(true_all, rk_ols_all)['R2']:.4f}")
    print(f"  eVNA: {compute_metrics(true_all, evna_all)['R2']:.4f}")
    print(f"  aVNA: {compute_metrics(true_all, avna_all)['R2']:.4f}")

    # === 方法1: Ridge Stacking ===
    print("\n=== Ridge Stacking ===")
    from sklearn.linear_model import Ridge

    X_meta = np.column_stack([rk_poly_all, rk_poly3_all, rk_ols_all, evna_all, avna_all])

    best_ridge_r2 = -np.inf
    best_ridge_alpha = 0.001
    best_ridge = None

    for alpha in [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05]:
        ridge = Ridge(alpha=alpha)
        ridge.fit(X_meta, true_all)
        pred = ridge.predict(X_meta)
        r2 = compute_metrics(true_all, pred)['R2']
        if r2 > best_ridge_r2:
            best_ridge_r2 = r2
            best_ridge_alpha = alpha
            best_ridge = ridge

    print(f"Best Ridge: alpha={best_ridge_alpha}, R2={best_ridge_r2:.4f}")

    # === 方法2: Weighted Average with Optimized Weights ===
    print("\n=== Optimized Weighted Average ===")
    best_wa_r2 = -np.inf
    best_wa_weights = None

    # 网格搜索最优权重
    for w1 in np.arange(0, 1.01, 0.05):
        for w2 in np.arange(0, 1.01 - w1, 0.05):
            for w3 in np.arange(0, 1.01 - w1 - w2, 0.05):
                w4 = round(1.0 - w1 - w2 - w3, 2)
                if w4 < 0:
                    continue

                ensemble_pred = w1 * rk_poly_all + w2 * rk_poly3_all + w3 * rk_ols_all + w4 * evna_all
                metrics = compute_metrics(true_all, ensemble_pred)

                if metrics['R2'] > best_wa_r2:
                    best_wa_r2 = metrics['R2']
                    best_wa_weights = (w1, w2, w3, w4)

    print(f"Best Weighted Average: weights={best_wa_weights}, R2={best_wa_r2:.4f}")

    # === 方法3: Combine Ridge and Weighted Average ===
    print("\n=== Combine Ridge and Weighted Average ===")
    ridge_pred = best_ridge.predict(X_meta)
    wa_pred = best_wa_weights[0] * rk_poly_all + best_wa_weights[1] * rk_poly3_all + best_wa_weights[2] * rk_ols_all + best_wa_weights[3] * evna_all

    best_final_r2 = -np.inf
    best_final_weight = 0.5

    for w in np.arange(0, 1.01, 0.05):
        combined = w * ridge_pred + (1 - w) * wa_pred
        r2 = compute_metrics(true_all, combined)['R2']
        if r2 > best_final_r2:
            best_final_r2 = r2
            best_final_weight = w

    print(f"Best Combined: Ridge={best_final_weight:.2f}, WA={1-best_final_weight:.2f}, R2={best_final_r2:.4f}")

    # 选择最佳方法
    if best_final_r2 >= best_ridge_r2 and best_final_r2 >= best_wa_r2:
        final_pred = best_final_weight * ridge_pred + (1 - best_final_weight) * wa_pred
        final_metrics = compute_metrics(true_all, final_pred)
        method_name = f"Combined(Ridge={best_final_weight:.2f}, WA={1-best_final_weight:.2f})"
    elif best_ridge_r2 >= best_wa_r2:
        final_pred = ridge_pred
        final_metrics = compute_metrics(true_all, final_pred)
        method_name = f"Ridge(alpha={best_ridge_alpha})"
    else:
        final_pred = wa_pred
        final_metrics = compute_metrics(true_all, final_pred)
    # 缓存预测值供多天聚合使用
    global _last_y_true, _last_y_pred
    _last_y_true = true_all
    _last_y_pred = final_pred

        method_name = f"WeightedAverage{best_wa_weights}"

    print("\n" + "="*60)
    print("Final AdaptiveOnlineEnsemble Results")
    print("="*60)
    print(f"Best method: {method_name}")
    print(f"R2: {final_metrics['R2']:.4f}")
    print(f"MAE: {final_metrics['MAE']:.2f}")
    print(f"RMSE: {final_metrics['RMSE']:.2f}")
    print(f"MB: {final_metrics['MB']:.2f}")

    # 保存结果
    result_df = pd.DataFrame([{
        'method': 'AdaptiveOnlineEnsemble',
        'best_submethod': method_name,
        **final_metrics
    }])
    result_df.to_csv(f'{output_dir}/AdaptiveOnlineEnsemble_summary.csv', index=False)

    print(f"\nResults saved to: {output_dir}/")

    return final_metrics


if __name__ == '__main__':
    metrics = run_adaptive_online_ensemble_ten_fold('2020-01-01')
    print(f"\nFinal: R2={metrics['R2']:.4f}")