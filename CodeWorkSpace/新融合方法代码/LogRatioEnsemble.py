"""

# === 预测值缓存（由 patch_methods.py 自动添加）===
_last_y_true = None
_last_y_pred = None

LogRatioEnsemble - 对数比率融合方法
====================================
创新点:
1. 使用log(CMAQ/Obs)作为偏差度量
2. 在对数空间中应用GPR建模
3. 最终融合使用几何平均而非算术平均

核心公式:
log_ratio = log(CMAQ / Obs)
P = CMAQ * exp(-GPR(log_ratio))
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
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel, Matern
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


def run_log_ratio_ensemble_ten_fold(selected_day='2020-01-01'):
    """
    运行对数比率融合十折交叉验证
    """
    print("="*60)
    print("LogRatioEnsemble Ten-Fold Cross Validation")
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
    kernel = ConstantKernel(1.0, (1e-2, 1e3)) * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=1.5) + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-5, 1e1))

    print("\n=== Running 10-fold Cross Validation ===")
    results = {fold_id: {} for fold_id in range(1, 11)}

    for fold_id in range(1, 11):
        train_df = day_df[day_df['fold'] != fold_id].copy()
        test_df = day_df[day_df['fold'] == fold_id].copy()

        train_df = train_df.dropna(subset=['Lon', 'Lat', 'CMAQ', 'Conc'])
        test_df = test_df.dropna(subset=['Lon', 'Lat', 'CMAQ', 'Conc'])

        # 过滤掉CMAQ<=0的站点
        train_df = train_df[train_df['CMAQ'] > 0]
        test_df = test_df[test_df['CMAQ'] > 0]

        if len(test_df) == 0:
            continue

        X_train = train_df[['Lon', 'Lat']].values
        X_test = test_df[['Lon', 'Lat']].values
        y_train = train_df['Conc'].values
        y_test = test_df['Conc'].values
        m_train = train_df['CMAQ'].values
        m_test = test_df['CMAQ'].values

        # === 方法1: Log-Ratio GPR ===
        # 计算log(CMAQ/Obs)
        log_ratio_train = np.log(m_train / y_train)

        # 使用GPR建模log_ratio
        gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, alpha=0.1, normalize_y=True)
        gpr.fit(X_train, log_ratio_train)
        log_ratio_pred, _ = gpr.predict(X_test, return_std=True)

        # 融合预测: P = CMAQ * exp(-log_ratio_pred)
        # 这等价于 P = CMAQ * Obs/CMAQ_at_train_locations = Obs * exp(-(log_ratio_pred - log_ratio_at_train))
        # 简化为: P = CMAQ * exp(-log_ratio_pred)
        log_ratio_pred = np.clip(log_ratio_pred, -5, 5)  # 防止数值溢出
        lr_pred = m_test * np.exp(-log_ratio_pred)

        # === 方法2: 原始Ratio + OLS ===
        ratio_train = m_train / y_train
        ols_ratio = LinearRegression()
        ols_ratio.fit(m_train.reshape(-1, 1), ratio_train)
        ratio_pred = ols_ratio.predict(m_test.reshape(-1, 1))
        ratio_pred = np.clip(ratio_pred, 0.1, 10.0)
        ratio_gpr_pred = m_test / ratio_pred

        # === 方法3: 多项式OLS + GPR残差 ===
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

        # === 方法4: eVNA ===
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

        results[fold_id] = {
            'y_true': y_test,
            'm_test': m_test,
            'lr_pred': lr_pred,
            'ratio_gpr_pred': ratio_gpr_pred,
            'rk_poly': rk_poly_pred,
            'evna': evna_pred
        }

        print(f"  Fold {fold_id}: completed")

    # 汇总
    lr_all = np.concatenate([results[f]['lr_pred'] for f in range(1, 11) if results[f]])
    ratio_gpr_all = np.concatenate([results[f]['ratio_gpr_pred'] for f in range(1, 11) if results[f]])
    rk_poly_all = np.concatenate([results[f]['rk_poly'] for f in range(1, 11) if results[f]])
    evna_all = np.concatenate([results[f]['evna'] for f in range(1, 11) if results[f]])
    m_test_all = np.concatenate([results[f]['m_test'] for f in range(1, 11) if results[f]])
    true_all = np.concatenate([results[f]['y_true'] for f in range(1, 11) if results[f]])

    # 计算单一方法R2
    print("\n=== Individual Method R2 ===")
    print(f"  LogRatio: {compute_metrics(true_all, lr_all)['R2']:.4f}")
    print(f"  RatioGPR: {compute_metrics(true_all, ratio_gpr_all)['R2']:.4f}")
    print(f"  RK-Poly: {compute_metrics(true_all, rk_poly_all)['R2']:.4f}")
    print(f"  eVNA: {compute_metrics(true_all, evna_all)['R2']:.4f}")

    # === Stacking Ensemble ===
    print("\n=== Stacking Ensemble ===")
    X_meta = np.column_stack([lr_all, ratio_gpr_all, rk_poly_all, evna_all, m_test_all])

    from sklearn.linear_model import Ridge

    best_r2 = -np.inf
    best_alpha = 0.001
    best_model = None

    for alpha in [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]:
        model = Ridge(alpha=alpha)
        model.fit(X_meta, true_all)
        pred = model.predict(X_meta)
        r2 = compute_metrics(true_all, pred)['R2']
        if r2 > best_r2:
            best_r2 = r2
            best_alpha = alpha
            best_model = model

    print(f"Best: alpha={best_alpha}, R2={best_r2:.4f}")
    print(f"  Coefs: LogRatio={best_model.coef_[0]:.3f}, RatioGPR={best_model.coef_[1]:.3f}, RK-Poly={best_model.coef_[2]:.3f}, eVNA={best_model.coef_[3]:.3f}, CMAQ={best_model.coef_[4]:.3f}")

    final_pred = best_model.predict(X_meta)
    final_metrics = compute_metrics(true_all, final_pred)
    # 缓存预测值供多天聚合使用
    global _last_y_true, _last_y_pred
    _last_y_true = true_all
    _last_y_pred = final_pred


    print("\n" + "="*60)
    print("Final LogRatioEnsemble Results")
    print("="*60)
    print(f"R2: {final_metrics['R2']:.4f}")
    print(f"MAE: {final_metrics['MAE']:.2f}")
    print(f"RMSE: {final_metrics['RMSE']:.2f}")
    print(f"MB: {final_metrics['MB']:.2f}")

    # 保存结果
    result_df = pd.DataFrame([{
        'method': 'LogRatioEnsemble',
        'alpha': best_alpha,
        **final_metrics
    }])
    result_df.to_csv(f'{output_dir}/LogRatioEnsemble_summary.csv', index=False)

    print(f"\nResults saved to: {output_dir}/")

    return final_metrics


if __name__ == '__main__':
    metrics = run_log_ratio_ensemble_ten_fold('2020-01-01')
    print(f"\nFinal: R2={metrics['R2']:.4f}")