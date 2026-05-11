"""

# === 预测值缓存（由 patch_methods.py 自动添加）===
_last_y_true = None
_last_y_pred = None

BMA_Fusion - 贝叶斯模型平均融合法
===================================
Bayesian Model Averaging for PM2.5 Fusion

创新点:
1. 使用贝叶斯后验模型概率(非Ridge/Lasso回归权重)组合多个基础方法
2. BIC近似模型证据，计算后验模型概率
3. 最终预测 = 各方法预测的后验概率加权平均
4. 同时提供预测不确定性(后验方差)

基础方法: VNA, eVNA, aVNA, RK-Poly (或更多)
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


def predict_vna(lon_train, lat_train, conc_train, lon_test, lat_test):
    """VNA: Voronoi Neighbor Average"""
    nna = NNA(lon_train, lat_train, conc_train)
    return nna.predict(lon_test, lat_test)


def predict_evna(lon_train, lat_train, conc_train, cmaq_train, lon_test, lat_test, cmaq_test):
    """eVNA: enhanced VNA with multiplicative bias correction"""
    nna = NNA(lon_train, lat_train, conc_train)
    vna_train = nna.predict(lon_train, lat_train)
    bias_ratio = np.mean(conc_train / (vna_train + 1e-8))
    vna_test = nna.predict(lon_test, lat_test)
    return vna_test * bias_ratio


def predict_avna(lon_train, lat_train, conc_train, cmaq_train, lon_test, lat_test, cmaq_test):
    """aVNA: additive VNA with additive bias correction"""
    nna = NNA(lon_train, lat_train, conc_train)
    vna_train = nna.predict(lon_train, lat_train)
    bias_add = np.mean(conc_train - vna_train)
    vna_test = nna.predict(lon_test, lat_test)
    return vna_test + bias_add


def predict_rk_poly(X_train, y_train, m_train, X_test, m_test):
    """RK-Poly: Polynomial OLS + GPR residual kriging"""
    poly = PolynomialFeatures(degree=2, include_bias=False)
    m_train_poly = poly.fit_transform(m_train.reshape(-1, 1))
    m_test_poly = poly.transform(m_test.reshape(-1, 1))

    ols = LinearRegression()
    ols.fit(m_train_poly, y_train)
    pred_ols = ols.predict(m_test_poly)
    residual = y_train - ols.predict(m_train_poly)

    kernel = ConstantKernel(10.0, (1e-2, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e1))
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, alpha=0.1, normalize_y=True)
    gpr.fit(X_train, residual)
    gpr_pred = gpr.predict(X_test)

    return pred_ols + gpr_pred


def compute_bic(residuals, n_params):
    """计算BIC: BIC = n*log(MSE) + d*log(n)"""
    n = len(residuals)
    mse = np.mean(residuals**2)
    if mse <= 0:
        mse = 1e-10
    bic = n * np.log(mse) + n_params * np.log(n)
    return bic


def compute_posterior_weights(bic_values):
    """从BIC值计算后验模型概率"""
    # log p(D|M_k) approx -0.5 * BIC_k
    log_evidence = -0.5 * bic_values
    # 数值稳定的softmax
    log_evidence_shifted = log_evidence - np.max(log_evidence)
    evidence = np.exp(log_evidence_shifted)
    weights = evidence / np.sum(evidence)
    return weights


def run_BMA_Fusion_ten_fold(selected_day='2020-01-01'):
    print("=" * 60)
    print("BMA_Fusion Ten-Fold Cross Validation")
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

    # 基础方法列表
    method_names = ['VNA', 'eVNA', 'aVNA', 'RK_Poly']
    n_methods = len(method_names)
    # 参数数量 (用于BIC): VNA=1, eVNA=2, aVNA=2, RK_Poly=4
    n_params_list = [1, 2, 2, 4]

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

        # 各基础方法在训练集上的交叉验证预测 (leave-one-out近似)
        # 为了计算BIC，使用训练集上的拟合残差
        method_train_preds = {}
        method_test_preds = {}

        # VNA
        vna_train_pred = predict_vna(X_train[:, 0], X_train[:, 1], y_train, X_train[:, 0], X_train[:, 1])
        vna_test_pred = predict_vna(X_train[:, 0], X_train[:, 1], y_train, X_test[:, 0], X_test[:, 1])
        method_train_preds['VNA'] = vna_train_pred
        method_test_preds['VNA'] = vna_test_pred

        # eVNA
        evna_train_pred = predict_evna(X_train[:, 0], X_train[:, 1], y_train, m_train, X_train[:, 0], X_train[:, 1], m_train)
        evna_test_pred = predict_evna(X_train[:, 0], X_train[:, 1], y_train, m_train, X_test[:, 0], X_test[:, 1], m_test)
        method_train_preds['eVNA'] = evna_train_pred
        method_test_preds['eVNA'] = evna_test_pred

        # aVNA
        avna_train_pred = predict_avna(X_train[:, 0], X_train[:, 1], y_train, m_train, X_train[:, 0], X_train[:, 1], m_train)
        avna_test_pred = predict_avna(X_train[:, 0], X_train[:, 1], y_train, m_train, X_test[:, 0], X_test[:, 1], m_test)
        method_train_preds['aVNA'] = avna_train_pred
        method_test_preds['aVNA'] = avna_test_pred

        # RK-Poly
        rk_train_pred = predict_rk_poly(X_train, y_train, m_train, X_train, m_train)
        rk_test_pred = predict_rk_poly(X_train, y_train, m_train, X_test, m_test)
        method_train_preds['RK_Poly'] = rk_train_pred
        method_test_preds['RK_Poly'] = rk_test_pred

        # 计算每个方法的BIC
        bic_values = np.zeros(n_methods)
        for i, name in enumerate(method_names):
            residuals = y_train - method_train_preds[name]
            bic_values[i] = compute_bic(residuals, n_params_list[i])

        # 计算后验模型概率
        weights = compute_posterior_weights(bic_values)

        # BMA预测
        test_preds_matrix = np.array([method_test_preds[name] for name in method_names])
        bma_pred = np.dot(weights, test_preds_matrix)

        # 不确定性估计
        bma_var = np.zeros(len(y_test))
        for i in range(n_methods):
            bma_var += weights[i] * (method_test_preds[method_names[i]] - bma_pred)**2
        bma_std = np.sqrt(bma_var)

        results[fold_id] = {
            'y_true': y_test,
            'bma_pred': bma_pred,
            'weights': weights
        }
        print(f"  Fold {fold_id}: weights={dict(zip(method_names, [f'{w:.3f}' for w in weights]))}")

    # 汇总
    bma_all = np.concatenate([results[f]['bma_pred'] for f in range(1, 11) if results[f]])
    true_all = np.concatenate([results[f]['y_true'] for f in range(1, 11) if results[f]])

    # 平均权重
    all_weights = np.array([results[f]['weights'] for f in range(1, 11) if results[f]])
    avg_weights = np.mean(all_weights, axis=0)

    print("\n=== Results ===")
    bma_metrics = compute_metrics(true_all, bma_all)
    # 缓存预测值供多天聚合使用
    global _last_y_true, _last_y_pred
    _last_y_true = true_all
    _last_y_pred = bma_all

    print(f"  BMA-Fusion: R2={bma_metrics['R2']:.4f}, MAE={bma_metrics['MAE']:.2f}, RMSE={bma_metrics['RMSE']:.2f}")
    print(f"  Average weights: {dict(zip(method_names, [f'{w:.3f}' for w in avg_weights]))}")

    result_df = pd.DataFrame([{'method': 'BMA_Fusion', **bma_metrics}])
    result_df.to_csv(f'{output_dir}/BMA_Fusion_summary.csv', index=False)
    print(f"\nResults saved to: {output_dir}/BMA_Fusion_summary.csv")

    return bma_metrics


if __name__ == '__main__':
    metrics = run_BMA_Fusion_ten_fold('2020-01-01')
    print(f"\nFinal: R2={metrics['R2']:.4f}")
