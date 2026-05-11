"""

# === 预测值缓存（由 patch_methods.py 自动添加）===
_last_y_true = None
_last_y_pred = None

ResidualDistMatchKriging (RDMK) - 残差分布匹配克里金
====================================================
针对PM2.5残差的非高斯特性（右偏、重尾），使用参数化Gamma分布进行残差分布匹配，
将非高斯残差变换为高斯变量后进行克里金插值，再反变换回原始尺度。

核心创新:
1. PM2.5浓度呈右偏分布（Gamma/对数正态），残差同样具有偏态
2. 现有克里金/GPR方法假设残差服从高斯分布，高浓度区建模不准确
3. 参数化分布匹配比经验分位数映射更稳健（参数少、外推能力强）

三步法:
1. 多项式校正CMAQ系统偏差
2. Gamma分布匹配将残差变换为高斯变量
3. 在高斯空间进行克里金插值后反变换

核心公式:
1. 多项式校正: Y_hat(s) = b0 + b1*CMAQ(s) + b2*CMAQ(s)^2
2. 残差计算: r(s) = Y(s) - Y_hat(s)
3. Gamma拟合: r(s) + offset ~ Gamma(alpha, beta)
4. 正态变换: z(s) = Phi^{-1}(F_Gamma(r(s) + offset; alpha, beta))
5. 高斯空间克里金: z_hat(s0) = k^T * K^{-1} * z
6. 反变换: r_hat(s0) = F_Gamma^{-1}(Phi(z_hat(s0)); alpha, beta) - offset
7. 最终融合: Y_fused(s0) = Y_hat(s0) + r_hat(s0)

参数:
- poly_degree: 2
- matern_nu: 1.5
- offset_factor: 1e-6
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
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from scipy import stats
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


def gamma_transform(residuals, offset_factor=1e-6):
    """
    Gamma分布匹配变换

    将非高斯残差变换为近似高斯变量:
    1. 加偏移使所有值为正
    2. 拟合Gamma分布
    3. 正态得分变换

    Parameters:
    -----------
    residuals: array
        残差值
    offset_factor: float
        偏移量系数

    Returns:
    --------
    z: array
        变换后的高斯变量
    alpha, beta: float
        Gamma分布参数
    offset: float
        偏移量
    """
    # 偏移使所有值为正
    r_min = np.min(residuals)
    offset = abs(r_min) + offset_factor * abs(r_min) if r_min < 0 else offset_factor
    r_shifted = residuals + offset

    # 确保所有值严格为正
    r_shifted = np.maximum(r_shifted, 1e-10)

    # 拟合Gamma分布 (使用MLE)
    try:
        # scipy.stats.gamma.fit 返回 (a, loc, scale)
        # 其中 a = shape (alpha), scale = beta
        a_fit, loc_fit, scale_fit = stats.gamma.fit(r_shifted, floc=0)

        # 正态得分变换: z = Phi^{-1}(F_Gamma(r))
        cdf_vals = stats.gamma.cdf(r_shifted, a_fit, loc=loc_fit, scale=scale_fit)

        # 避免0和1的极端值
        cdf_vals = np.clip(cdf_vals, 1e-6, 1.0 - 1e-6)

        z = stats.norm.ppf(cdf_vals)

        return z, a_fit, scale_fit, offset

    except Exception as e:
        # 如果Gamma拟合失败，使用简单的标准化
        print(f"  Gamma fit failed: {e}, using standard normalization")
        z = (residuals - np.mean(residuals)) / (np.std(residuals) + 1e-6)
        return z, 1.0, 1.0, 0.0


def gamma_inverse_transform(z_pred, alpha, beta, offset):
    """
    反变换: 将高斯空间的预测值变换回原始残差尺度

    r_hat = F_Gamma^{-1}(Phi(z_hat)) - offset
    """
    # 正态CDF
    cdf_vals = stats.norm.cdf(z_pred)

    # 避免极端值
    cdf_vals = np.clip(cdf_vals, 1e-6, 1.0 - 1e-6)

    # Gamma逆CDF (分位数函数)
    r_pred = stats.gamma.ppf(cdf_vals, alpha, scale=beta) - offset

    return r_pred


def run_ResidualDistMatchKriging_ten_fold(selected_day='2020-01-01'):
    """
    运行ResidualDistMatchKriging十折交叉验证

    创新点:
    - Gamma分布匹配处理非高斯残差
    - 正态得分变换后在高斯空间克里金
    - 反变换回原始尺度
    - Matérn核函数（比RBF更灵活）

    Parameters:
    -----------
    selected_day: str
        选择验证的日期 (默认: '2020-01-01')

    Returns:
    --------
    metrics: dict
        包含R2, MAE, RMSE, MB的评估指标
    """
    # RDMK参数
    poly_degree = 2
    matern_nu = 1.5
    offset_factor = 1e-6

    print("=" * 60)
    print("ResidualDistMatchKriging (RDMK) Ten-Fold Cross Validation")
    print(f"Parameters: poly_degree={poly_degree}, matern_nu={matern_nu}")
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

    print(f"Data loaded: {len(day_df)} monitoring records")

    # 定义Matérn核函数
    kernel = (ConstantKernel(1.0, (1e-2, 1e3)) *
              Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=matern_nu) +
              WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-5, 1e1)))

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

        # Step 1: 多项式校正
        poly = PolynomialFeatures(degree=poly_degree, include_bias=False)
        m_train_poly = poly.fit_transform(m_train.reshape(-1, 1))
        m_test_poly = poly.transform(m_test.reshape(-1, 1))

        ols = LinearRegression()
        ols.fit(m_train_poly, y_train)
        pred_ols = ols.predict(m_test_poly)
        residual = y_train - ols.predict(m_train_poly)

        # Step 2: Gamma分布匹配变换
        z_train, alpha_fit, beta_fit, offset = gamma_transform(residual, offset_factor)

        # 检查变换后的高斯性
        if np.any(np.isnan(z_train)):
            print(f"  Fold {fold_id}: NaN in transformed values, using raw residual")
            z_train = (residual - np.mean(residual)) / (np.std(residual) + 1e-6)
            alpha_fit, beta_fit, offset = 1.0, 1.0, 0.0

        # Step 3: 高斯空间GPR克里金
        gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2,
                                       alpha=0.1, normalize_y=True)
        gpr.fit(X_train, z_train)
        z_pred, _ = gpr.predict(X_test, return_std=True)

        # Step 4: 反变换回原始残差尺度
        if alpha_fit > 0 and beta_fit > 0:
            residual_pred = gamma_inverse_transform(z_pred, alpha_fit, beta_fit, offset)
        else:
            # 如果Gamma参数无效，直接使用高斯空间预测
            residual_pred = z_pred * (np.std(residual) + 1e-6) + np.mean(residual)

        # Step 5: 最终融合
        rdmk_pred = pred_ols + residual_pred

        results[fold_id] = {
            'y_true': y_test,
            'rdmk': rdmk_pred
        }

        print(f"  Fold {fold_id}: alpha={alpha_fit:.3f}, beta={beta_fit:.3f}, offset={offset:.3f}")

    # 汇总
    rdmk_all = np.concatenate([results[f]['rdmk'] for f in range(1, 11) if results[f]])
    true_all = np.concatenate([results[f]['y_true'] for f in range(1, 11) if results[f]])

    # 计算指标
    metrics = compute_metrics(true_all, rdmk_all)
    # 缓存预测值供多天聚合使用
    global _last_y_true, _last_y_pred
    _last_y_true = true_all
    _last_y_pred = rdmk_all


    print(f"\n=== ResidualDistMatchKriging Results ===")
    print(f"  R2={metrics['R2']:.4f}, MAE={metrics['MAE']:.2f}, RMSE={metrics['RMSE']:.2f}, MB={metrics['MB']:.2f}")

    # 保存结果
    result_df = pd.DataFrame([{
        'Method': 'ResidualDistMatchKriging',
        'R2': metrics['R2'],
        'MAE': metrics['MAE'],
        'RMSE': metrics['RMSE'],
        'MB': metrics['MB']
    }])
    result_df.to_csv(f'{output_dir}/ResidualDistMatchKriging_summary.csv', index=False)

    print(f"\nResults saved to: {output_dir}/ResidualDistMatchKriging_summary.csv")
    return metrics


if __name__ == '__main__':
    metrics = run_ResidualDistMatchKriging_ten_fold('2020-01-01')
    print(f"\nFinal: R2={metrics['R2']:.4f}")
