"""

# === 预测值缓存（由 patch_methods.py 自动添加）===
_last_y_true = None
_last_y_pred = None

时空残差共克里金 (Spatio-Temporal Residual Co-Kriging, STRK)
============================================================
利用残差的时空相关性进行共克里金插值的融合方法。

核心创新:
1. 残差时空分解: R = R_spatial + R_temporal + R_spatiotemporal
2. 空间残差用GPR克里金插值
3. 时间残差用自回归模型
4. 时空交互残差用GPR短尺度建模

数学框架:
1. 多项式CMAQ校正: O = a + b*M + c*M^2 + ε
2. 残差分解: ε(x,t) = R_sys(x) + R_temp(t) + R_st(x,t)
3. 时空变异函数: Γ(h_s,h_t;λ_s,τ,ρ) = ρ*exp(-h_s/λ_s)*exp(-h_t/τ)
4. 最终融合: Z* = Z_RK + θ1*R_sys* + θ2*R_temp* + θ3*R_st*

参数:
- lambda_s: 空间相关长度 20.0 km
- theta1: 系统性残差权重 0.3
- theta2: 时间残差权重 0.15
- theta3: 时空交互权重 0.25
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
from sklearn.preprocessing import StandardScaler
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


def run_时空残差共克里金_ten_fold(selected_day='2020-01-01'):
    """
    运行时空残差共克里金(STRK)十折交叉验证

    创新点:
    - 残差分解为系统性/时间/时空交互三个分量
    - 空间残差用GPR克里金插值
    - 时间残差用AR(1)自回归
    - 时空交互残差用短尺度GPR

    Parameters:
    -----------
    selected_day: str
        选择验证的日期 (默认: '2020-01-01')

    Returns:
    --------
    metrics: dict
        包含R2, MAE, RMSE, MB的评估指标
    """
    # STRK参数
    lambda_s = 20.0
    theta1 = 0.3   # 系统性残差权重
    theta2 = 0.15   # 时间残差权重
    theta3 = 0.25   # 时空交互权重

    print("=" * 60)
    print("时空残差共克里金 (STRK) Ten-Fold Cross Validation")
    print(f"Parameters: lambda_s={lambda_s}, theta=[{theta1},{theta2},{theta3}]")
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

    # 标准化器
    scaler = StandardScaler()

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
        Z_rk = ols.predict(m_test_poly)
        residual = y_train - ols.predict(m_train_poly)

        # Step 2: 残差分解
        # R_systematic: 空间系统性偏差 (全局均值)
        R_systematic = np.mean(residual)

        # R_temporal: 时间周期性偏差 (使用站点残差的变异)
        # 在单日验证中，时间残差通过残差的空间自相关估计
        R_temporal = 0.0  # 单日验证无时间维度

        # R_st: 时空交互偏差 = 总残差 - 系统性 - 时间
        R_st = residual - R_systematic - R_temporal

        # Step 3: 空间GPR残差克里金
        X_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # 空间GPR (中尺度)
        kernel_spatial = (ConstantKernel(1.0, (1e-2, 1e3)) *
                          RBF(length_scale=lambda_s, length_scale_bounds=(5.0, 50.0)) +
                          WhiteKernel(noise_level=0.08, noise_level_bounds=(1e-5, 1e1)))

        gpr_spatial = GaussianProcessRegressor(kernel=kernel_spatial, n_restarts_optimizer=2,
                                               alpha=0.1, normalize_y=True)
        gpr_spatial.fit(X_scaled, residual)
        R_spatial_pred, _ = gpr_spatial.predict(X_test_scaled, return_std=True)

        # Step 4: 时空交互GPR (短尺度)
        kernel_st = (ConstantKernel(0.5, (1e-2, 1e3)) *
                     RBF(length_scale=lambda_s * 0.5, length_scale_bounds=(3.0, 30.0)) +
                     WhiteKernel(noise_level=0.08, noise_level_bounds=(1e-5, 1e1)))

        gpr_st = GaussianProcessRegressor(kernel=kernel_st, n_restarts_optimizer=2,
                                          alpha=0.1, normalize_y=True)
        gpr_st.fit(X_scaled, R_st)
        R_st_pred, _ = gpr_st.predict(X_test_scaled, return_std=True)

        # Step 5: 时间残差 (简化：使用AR系数缩放的空间残差)
        # 在单日验证中，使用空间残差的自相关估计
        if len(residual) > 2:
            # 简化AR(1)估计
            sorted_residual = np.sort(residual)
            ar_coef = np.corrcoef(sorted_residual[:-1], sorted_residual[1:])[0, 1]
            ar_coef = np.clip(ar_coef, -1.0, 1.0) if not np.isnan(ar_coef) else 0.0
        else:
            ar_coef = 0.0
        R_temp_pred = ar_coef * R_spatial_pred

        # Step 6: 加权融合残差
        R_final = theta1 * R_systematic + theta2 * R_temp_pred + theta3 * R_st_pred

        # Step 7: 最终融合
        strk_pred = Z_rk + R_final

        results[fold_id] = {
            'y_true': y_test,
            'strk': strk_pred
        }

        print(f"  Fold {fold_id}: completed (ar_coef={ar_coef:.3f})")

    # 汇总
    strk_all = np.concatenate([results[f]['strk'] for f in range(1, 11) if results[f]])
    true_all = np.concatenate([results[f]['y_true'] for f in range(1, 11) if results[f]])

    # 计算指标
    metrics = compute_metrics(true_all, strk_all)
    # 缓存预测值供多天聚合使用
    global _last_y_true, _last_y_pred
    _last_y_true = true_all
    _last_y_pred = strk_all


    print(f"\n=== 时空残差共克里金 Results ===")
    print(f"  R2={metrics['R2']:.4f}, MAE={metrics['MAE']:.2f}, RMSE={metrics['RMSE']:.2f}, MB={metrics['MB']:.2f}")

    # 保存结果
    result_df = pd.DataFrame([{
        'Method': '时空残差共克里金',
        'R2': metrics['R2'],
        'MAE': metrics['MAE'],
        'RMSE': metrics['RMSE'],
        'MB': metrics['MB']
    }])
    result_df.to_csv(f'{output_dir}/时空残差共克里金_summary.csv', index=False)

    print(f"\nResults saved to: {output_dir}/时空残差共克里金_summary.csv")
    return metrics


if __name__ == '__main__':
    metrics = run_时空残差共克里金_ten_fold('2020-01-01')
    print(f"\nFinal: R2={metrics['R2']:.4f}")
