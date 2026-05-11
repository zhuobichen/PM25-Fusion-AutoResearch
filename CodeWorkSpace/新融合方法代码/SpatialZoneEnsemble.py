"""

# === 预测值缓存（由 patch_methods.py 自动添加）===
_last_y_true = None
_last_y_pred = None

SpatialZoneEnsemble - Spatial Zone Adaptive Ensemble
=====================================================
创新点:
1. 将空间分成多个区域(基于K-means聚类)
2. 每个区域独立建模，选择最优方法
3. 最终融合各区域的预测
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
from sklearn.cluster import KMeans
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


def run_spatial_zone_ensemble_ten_fold(selected_day='2020-01-01', n_zones=4):
    """
    运行SpatialZoneEnsemble十折交叉验证
    """
    print("="*60)
    print(f"SpatialZoneEnsemble Ten-Fold Cross Validation (n_zones={n_zones})")
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

        # === 空间聚类分区 ===
        kmeans = KMeans(n_clusters=n_zones, random_state=42, n_init=10)
        train_zones = kmeans.fit_predict(X_train)
        test_zones = kmeans.predict(X_test)

        # === Zone-wise RK-Poly ===
        zone_preds = {z: np.zeros(len(test_df)) for z in range(n_zones)}
        for z in range(n_zones):
            z_mask = train_zones == z
            if z_mask.sum() < 10:
                # 样本太少，直接用全局预测
                poly = PolynomialFeatures(degree=2, include_bias=False)
                m_train_poly = poly.fit_transform(m_train.reshape(-1, 1))
                ols_poly = LinearRegression()
                ols_poly.fit(m_train_poly, y_train)
                zone_preds[z][:] = ols_poly.predict(poly.transform(m_test.reshape(-1, 1)))
                continue

            X_train_z = X_train[z_mask]
            y_train_z = y_train[z_mask]
            m_train_z = m_train[z_mask]

            # Polynomial OLS
            poly = PolynomialFeatures(degree=2, include_bias=False)
            m_train_z_poly = poly.fit_transform(m_train_z.reshape(-1, 1))
            ols_poly = LinearRegression()
            ols_poly.fit(m_train_z_poly, y_train_z)
            pred_poly_z = ols_poly.predict(poly.transform(m_test.reshape(-1, 1)))

            # GPR残差
            residual_z = y_train_z - ols_poly.predict(m_train_z_poly)
            gpr_z = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, alpha=0.1, normalize_y=True)
            gpr_z.fit(X_train_z, residual_z)
            gpr_pred_z, _ = gpr_z.predict(X_test, return_std=True)

            zone_preds[z] = pred_poly_z + gpr_pred_z

        # 组合zone预测
        rk_zone_pred = np.zeros(len(test_df))
        for i, z in enumerate(test_zones):
            rk_zone_pred[i] = zone_preds[z][i]

        # === 全局RK-Poly (作为参考) ===
        poly = PolynomialFeatures(degree=2, include_bias=False)
        m_train_poly = poly.fit_transform(m_train.reshape(-1, 1))
        m_test_poly = poly.transform(m_test.reshape(-1, 1))

        ols_poly = LinearRegression()
        ols_poly.fit(m_train_poly, y_train)
        pred_poly = ols_poly.predict(m_test_poly)
        residual_poly = y_train - ols_poly.predict(m_train_poly)

        gpr_poly = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, alpha=0.1, normalize_y=True)
        gpr_poly.fit(X_train, residual_poly)
        gpr_pred, _ = gpr_poly.predict(X_test, return_std=True)
        rk_poly_pred = pred_poly + gpr_pred

        # === eVNA ===
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
            'rk_zone': rk_zone_pred,
            'rk_poly': rk_poly_pred,
            'evna': evna_pred
        }

        print(f"  Fold {fold_id}: completed")

    # 汇总
    rk_zone_all = np.concatenate([results[f]['rk_zone'] for f in range(1, 11) if results[f]])
    rk_poly_all = np.concatenate([results[f]['rk_poly'] for f in range(1, 11) if results[f]])
    evna_all = np.concatenate([results[f]['evna'] for f in range(1, 11) if results[f]])
    true_all = np.concatenate([results[f]['y_true'] for f in range(1, 11) if results[f]])

    # 计算单一方法R2
    print("\n=== Individual Method R2 ===")
    print(f"  RK-Zone: {compute_metrics(true_all, rk_zone_all)['R2']:.4f}")
    print(f"  RK-Poly: {compute_metrics(true_all, rk_poly_all)['R2']:.4f}")
    print(f"  eVNA: {compute_metrics(true_all, evna_all)['R2']:.4f}")

    # 优化Ensemble权重
    print("\n=== Optimizing Ensemble (RK-Zone + RK-Poly + eVNA) ===")
    best_r2 = -np.inf
    best_weights = None
    weight_results = []

    for w1 in np.arange(0, 1.05, 0.1):
        for w2 in np.arange(0, 1.05 - w1, 0.1):
            w3 = round(1.0 - w1 - w2, 2)
            if w3 < 0:
                continue

            ensemble_pred = w1 * rk_zone_all + w2 * rk_poly_all + w3 * evna_all
            metrics = compute_metrics(true_all, ensemble_pred)

            weight_results.append({
                'w1_rk_zone': w1, 'w2_rk_poly': w2, 'w3_evna': w3,
                'R2': metrics['R2'], 'MAE': metrics['MAE'], 'RMSE': metrics['RMSE']
            })

            if metrics['R2'] > best_r2:
                best_r2 = metrics['R2']
                best_weights = (w1, w2, w3)

    print(f"\nBest weights: RK-Zone={best_weights[0]:.2f}, RK-Poly={best_weights[1]:.2f}, eVNA={best_weights[2]:.2f}")
    print(f"Best R2: {best_r2:.4f}")

    # 最终评估
    final_pred = best_weights[0] * rk_zone_all + best_weights[1] * rk_poly_all + best_weights[2] * evna_all
    final_metrics = compute_metrics(true_all, final_pred)
    # 缓存预测值供多天聚合使用
    global _last_y_true, _last_y_pred
    _last_y_true = true_all
    _last_y_pred = final_pred


    print("\n" + "="*60)
    print("Final SpatialZoneEnsemble Results")
    print("="*60)
    print(f"Optimal weights: RK-Zone={best_weights[0]:.2f}, RK-Poly={best_weights[1]:.2f}, eVNA={best_weights[2]:.2f}")
    print(f"R2: {final_metrics['R2']:.4f}")
    print(f"MAE: {final_metrics['MAE']:.2f}")
    print(f"RMSE: {final_metrics['RMSE']:.2f}")
    print(f"MB: {final_metrics['MB']:.2f}")

    # 保存结果
    result_df = pd.DataFrame([{
        'method': 'SpatialZoneEnsemble',
        'n_zones': n_zones,
        'w1_rk_zone': best_weights[0],
        'w2_rk_poly': best_weights[1],
        'w3_evna': best_weights[2],
        **final_metrics
    }])
    result_df.to_csv(f'{output_dir}/SpatialZoneEnsemble_summary.csv', index=False)

    weight_df = pd.DataFrame(weight_results)
    weight_df.to_csv(f'{output_dir}/SpatialZoneEnsemble_weight_search.csv', index=False)

    print(f"\nResults saved to: {output_dir}/")

    return final_metrics, best_weights


if __name__ == '__main__':
    metrics, weights = run_spatial_zone_ensemble_ten_fold('2020-01-01', n_zones=4)
    print(f"\nFinal: R2={metrics['R2']:.4f}, Weights={weights}")