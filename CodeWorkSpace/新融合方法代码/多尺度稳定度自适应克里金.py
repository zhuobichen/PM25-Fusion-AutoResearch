"""

# === 预测值缓存（由 patch_methods.py 自动添加）===
_last_y_true = None
_last_y_pred = None

多尺度稳定度自适应克里金 (Multi-Scale Stability-Adaptive Kriging, MSAK)
========================================================================
大气稳定度分类驱动相关长度自适应的多尺度克里金融合方法。

核心创新:
1. 大气稳定度分类 (Pasquill A-F) 驱动相关长度自适应
2. 多尺度克里金分解：区域背景 + 城市尺度 + 局地扩散
3. 稳定度感知权重融合：不稳定->短相关(监测主导)，稳定->长相关(CMAQ主导)

数学框架:
1. 多项式CMAQ校正: O = a + b*M + c*M^2 + ε
2. 稳定度自适应相关长度: λ(PG) = λ0 * [1 + β*exp(-|PG-PG_cr|/σ_PG)]
3. 多尺度残差克里金: Z_s(x) = μ_CMAQ,s(x) + ε_s(x)
4. 稳定度融合: Z_final = α(PG)*Z_CMAQ + [1-α(PG)]*Z_GPR

参数:
- lambda0: 基础相关长度 15.0 km
- beta: 稳定度响应强度 0.8
- n_scales: 多尺度层数 3
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.paths import get_project_root, data_path
import numpy as np
import pandas as pd
from datetime import datetime
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


def estimate_stability_from_cmaq(cmaq_values, residual_values):
    """
    从CMAQ数据和残差估计局地大气稳定度等级(PG 1-6)

    使用残差的空间变异系数作为稳定度代理指标：
    - 高变异 -> 不稳定大气 (PG低) -> 短相关长度
    - 低变异 -> 稳定大气 (PG高) -> 长相关长度

    Parameters:
    -----------
    cmaq_values: array
        CMAQ浓度值
    residual_values: array
        残差值

    Returns:
    --------
    pg: int
        Pasquill-Gifford稳定度等级 (1-6)
    """
    # 使用残差的变异系数估计稳定度
    cv = np.std(residual_values) / (np.mean(np.abs(cmaq_values)) + 1e-6)

    # 映射到PG等级
    if cv > 0.5:
        return 1  # A-B: 极度不稳定
    elif cv > 0.3:
        return 2  # B-C: 不稳定
    elif cv > 0.2:
        return 3  # C-D: 轻度不稳定/中性
    elif cv > 0.15:
        return 4  # D: 中性
    elif cv > 0.1:
        return 5  # E: 轻度稳定
    else:
        return 6  # F: 稳定


def compute_stability_correlation_length(pg, lambda0=15.0, beta=0.8, sigma_pg=1.2, pg_crit=3.0):
    """
    计算稳定度自适应的相关长度

    λ(PG) = λ0 * [1 + β * exp(-|PG - PG_crit|/σ_PG)]
    """
    lambda_pg = lambda0 * (1.0 + beta * np.exp(-abs(pg - pg_crit) / sigma_pg))
    return lambda_pg


def compute_stability_weight(pg, gamma=1.5, pg_mid=2.5):
    """
    计算稳定度权重 α(PG)

    α(PG) = 1 / [1 + exp(-γ * (PG - PG_mid))]

    稳定大气(PG高) -> 高α -> CMAQ权重高
    不稳定大气(PG低) -> 低α -> 监测数据权重高
    """
    alpha = 1.0 / (1.0 + np.exp(-gamma * (pg - pg_mid)))
    return alpha


def run_多尺度稳定度自适应克里金_ten_fold(selected_day='2020-01-01'):
    """
    运行多尺度稳定度自适应克里金(MSAK)十折交叉验证

    创新点:
    - 大气稳定度驱动相关长度自适应
    - 多尺度克里金分解
    - 稳定度感知权重融合

    Parameters:
    -----------
    selected_day: str
        选择验证的日期 (默认: '2020-01-01')

    Returns:
    --------
    metrics: dict
        包含R2, MAE, RMSE, MB的评估指标
    """
    # MSAK参数
    lambda0 = 15.0
    beta = 0.8
    sigma_pg = 1.2
    pg_crit = 3.0
    pg_mid = 2.5
    gamma = 1.5
    n_scales = 3

    print("=" * 60)
    print("多尺度稳定度自适应克里金 (MSAK) Ten-Fold Cross Validation")
    print(f"Parameters: lambda0={lambda0}, beta={beta}, n_scales={n_scales}")
    print("=" * 60)

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

        # Step 1: 多项式CMAQ偏差校正
        poly = PolynomialFeatures(degree=2, include_bias=False)
        m_train_poly = poly.fit_transform(m_train.reshape(-1, 1))
        m_test_poly = poly.transform(m_test.reshape(-1, 1))

        ols = LinearRegression()
        ols.fit(m_train_poly, y_train)
        pred_ols = ols.predict(m_test_poly)
        residual = y_train - ols.predict(m_train_poly)

        # Step 2: 稳定度估计
        pg = estimate_stability_from_cmaq(m_train, residual)
        lambda_pg = compute_stability_correlation_length(pg, lambda0, beta, sigma_pg, pg_crit)

        # Step 3: 多尺度相关长度
        scale_lengths = [lambda_pg / (2.0 ** s) for s in range(n_scales)]
        scale_weights = [1.0 / l for l in scale_lengths]
        w_total = sum(scale_weights)
        scale_weights = [w / w_total for w in scale_weights]

        # Step 4: 多尺度GPR残差克里金
        multi_scale_preds = []
        for s, lambda_s in enumerate(scale_lengths):
            kernel = (ConstantKernel(1.0, (1e-2, 1e3)) *
                      RBF(length_scale=lambda_s, length_scale_bounds=(1e-2, 1e2)) +
                      WhiteKernel(noise_level=0.05, noise_level_bounds=(1e-5, 1e1)))

            gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2,
                                           alpha=0.1, normalize_y=True)
            gpr.fit(X_train, residual)
            gpr_pred, _ = gpr.predict(X_test, return_std=True)
            multi_scale_preds.append(gpr_pred)

        multi_scale_preds = np.array(multi_scale_preds)

        # Step 5: 多尺度加权融合
        gpr_fusion = np.sum(multi_scale_preds * np.array(scale_weights).reshape(-1, 1), axis=0)

        # Step 6: 稳定度自适应融合
        alpha = compute_stability_weight(pg, gamma, pg_mid)

        # Z_final = α * Z_CMAQ + (1-α) * Z_GPR
        # Z_GPR = pred_ols + gpr_fusion (多项式校正 + 残差克里金)
        gpr_pred_final = pred_ols + gpr_fusion
        msak_pred = alpha * m_test + (1.0 - alpha) * gpr_pred_final

        results[fold_id] = {
            'y_true': y_test,
            'msak': msak_pred
        }

        print(f"  Fold {fold_id}: PG={pg}, alpha={alpha:.3f}, lambda_pg={lambda_pg:.1f}km")

    # 汇总
    msak_all = np.concatenate([results[f]['msak'] for f in range(1, 11) if results[f]])
    true_all = np.concatenate([results[f]['y_true'] for f in range(1, 11) if results[f]])

    # 计算指标
    metrics = compute_metrics(true_all, msak_all)
    # 缓存预测值供多天聚合使用
    global _last_y_true, _last_y_pred
    _last_y_true = true_all
    _last_y_pred = msak_all


    print(f"\n=== 多尺度稳定度自适应克里金 Results ===")
    print(f"  R2={metrics['R2']:.4f}, MAE={metrics['MAE']:.2f}, RMSE={metrics['RMSE']:.2f}, MB={metrics['MB']:.2f}")

    # 保存结果
    result_df = pd.DataFrame([{
        'Method': '多尺度稳定度自适应克里金',
        'R2': metrics['R2'],
        'MAE': metrics['MAE'],
        'RMSE': metrics['RMSE'],
        'MB': metrics['MB']
    }])
    result_df.to_csv(f'{output_dir}/多尺度稳定度自适应克里金_summary.csv', index=False)

    print(f"\nResults saved to: {output_dir}/多尺度稳定度自适应克里金_summary.csv")
    return metrics


if __name__ == '__main__':
    metrics = run_多尺度稳定度自适应克里金_ten_fold('2020-01-01')
    print(f"\nFinal: R2={metrics['R2']:.4f}")
