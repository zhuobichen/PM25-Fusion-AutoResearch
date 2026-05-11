"""

# === 预测值缓存（由 patch_methods.py 自动添加）===
_last_y_true = None
_last_y_pred = None

多项式样条克里金 (Polynomial Spline Kriging, PSK)
=================================================
在PolyRK架构基础上，用三次样条（cubic spline）替代二次多项式进行全局趋势建模。
多项式在边界区域容易出现剧烈振荡（Runge现象），而三次样条在保持局部灵活性的同时
保证全局光滑性，能更准确地捕捉CMAQ与监测值之间的非线性偏差关系。

核心公式:
1. 全局三次样条校正: M_cal(s) = spline(CMAQ(s))
2. 残差计算: R(s) = O(s) - M_cal(s)
3. GPR残差克里金: R*(s) ~ GP(0, k_RBF)
4. 最终融合: P(s) = M_cal(s) + R*(s)

参数:
- spline_knots: 5 (节点数)
- spline_degree: 3 (三次样条)
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
from scipy.interpolate import UnivariateSpline
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


def run_多项式样条克里金_ten_fold(selected_day='2020-01-01'):
    """
    运行多项式样条克里金(PSK)十折交叉验证

    创新点:
    - 三次样条替代二次多项式进行全局趋势建模
    - 分段多项式拼接，允许不同CMAQ浓度区间有不同的偏差斜率
    - 无Runge振荡，全局光滑

    Parameters:
    -----------
    selected_day: str
        选择验证的日期 (默认: '2020-01-01')

    Returns:
    --------
    metrics: dict
        包含R2, MAE, RMSE, MB的评估指标
    """
    spline_knots = 5
    spline_degree = 3

    print("=" * 60)
    print("多项式样条克里金 (PSK) Ten-Fold Cross Validation")
    print(f"Parameters: spline_knots={spline_knots}, spline_degree={spline_degree}")
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

    # 定义GPR核函数
    kernel = (ConstantKernel(10.0, (1e-2, 1e3)) *
              RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) +
              WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e1)))

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

        # === Step 1: 三次样条校正 ===
        try:
            # 按CMAQ值排序以确保样条拟合单调性
            sort_idx = np.argsort(m_train)
            m_sorted = m_train[sort_idx]
            y_sorted = y_train[sort_idx]

            # 节点数自适应
            n = len(m_train)
            actual_knots = min(spline_knots, max(1, n - 1))

            # 使用UnivariateSpline进行三次样条拟合
            spline = UnivariateSpline(m_sorted, y_sorted, k=spline_degree, s=None)
            pred_spline = spline(m_test)
            residual_spline = y_train - spline(m_train)

            # 检查NaN并处理外推问题
            if np.any(np.isnan(pred_spline)) or np.any(np.isnan(residual_spline)):
                raise ValueError("Spline prediction contains NaN")
        except Exception as e:
            print(f"  Fold {fold_id}: spline fitting failed ({e}), using polynomial fallback")
            # 回退到多项式
            poly = PolynomialFeatures(degree=2, include_bias=False)
            m_train_poly = poly.fit_transform(m_train.reshape(-1, 1))
            m_test_poly = poly.transform(m_test.reshape(-1, 1))
            ols = LinearRegression()
            ols.fit(m_train_poly, y_train)
            pred_spline = ols.predict(m_test_poly)
            residual_spline = y_train - ols.predict(m_train_poly)

        # === Step 2: GPR残差克里金 ===
        if np.any(np.isnan(residual_spline)):
            # 如果残差有NaN，使用零残差
            gpr_pred = np.zeros(len(X_test))
        else:
            gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2,
                                           alpha=0.1, normalize_y=True)
            gpr.fit(X_train, residual_spline)
            gpr_pred, _ = gpr.predict(X_test, return_std=True)

        # === Step 3: 融合输出 ===
        psk_pred = pred_spline + gpr_pred

        results[fold_id] = {
            'y_true': y_test,
            'psk': psk_pred
        }

        print(f"  Fold {fold_id}: completed")

    # 汇总
    psk_all = np.concatenate([results[f]['psk'] for f in range(1, 11) if results[f]])
    true_all = np.concatenate([results[f]['y_true'] for f in range(1, 11) if results[f]])

    # 计算指标
    metrics = compute_metrics(true_all, psk_all)
    # 缓存预测值供多天聚合使用
    global _last_y_true, _last_y_pred
    _last_y_true = true_all
    _last_y_pred = psk_all


    print(f"\n=== 多项式样条克里金 Results ===")
    print(f"  R2={metrics['R2']:.4f}, MAE={metrics['MAE']:.2f}, RMSE={metrics['RMSE']:.2f}, MB={metrics['MB']:.2f}")

    # 保存结果
    result_df = pd.DataFrame([{
        'Method': '多项式样条克里金',
        'R2': metrics['R2'],
        'MAE': metrics['MAE'],
        'RMSE': metrics['RMSE'],
        'MB': metrics['MB']
    }])
    result_df.to_csv(f'{output_dir}/多项式样条克里金_summary.csv', index=False)

    print(f"\nResults saved to: {output_dir}/多项式样条克里金_summary.csv")
    return metrics


if __name__ == '__main__':
    metrics = run_多项式样条克里金_ten_fold('2020-01-01')
    print(f"\nFinal: R2={metrics['R2']:.4f}")
