"""

# === 预测值缓存（由 patch_methods.py 自动添加）===
_last_y_true = None
_last_y_pred = None

鲁棒残差克里金 (Robust Residual Kriging, RRK)
==============================================
在PolyRK架构基础上，用鲁棒回归（Huber回归）替代普通OLS进行多项式偏差校正。
OLS对异常值极度敏感，而实际监测数据中常存在设备故障、极端污染事件等异常值，
Huber回归通过软化损失函数对异常值更具鲁棒性。

核心公式:
1. 全局多项式校正(Huber): M_cal(s) = a0 + a1*CMAQ(s) + a2*CMAQ(s)^2
   参数通过Huber回归求解（迭代加权最小二乘）
2. 残差计算: R(s) = O(s) - M_cal(s)
3. GPR残差克里金: R*(s) ~ GP(0, k_RBF)
4. 最终融合: P(s) = M_cal(s) + R*(s)

参数:
- poly_degree: 2
- huber_epsilon: 1.35 (统计效率与鲁棒性平衡)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.paths import get_project_root, data_path
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import HuberRegressor, LinearRegression
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


def run_鲁棒残差克里金_ten_fold(selected_day='2020-01-01'):
    """
    运行鲁棒残差克里金(RRK)十折交叉验证

    创新点:
    - Huber回归替代OLS进行二次多项式校正
    - Huber损失函数对大残差（异常值）使用线性惩罚
    - 使全局趋势建模更稳健，不受极端污染事件或设备故障影响

    Parameters:
    -----------
    selected_day: str
        选择验证的日期 (默认: '2020-01-01')

    Returns:
    --------
    metrics: dict
        包含R2, MAE, RMSE, MB的评估指标
    """
    poly_degree = 2
    huber_epsilon = 1.35

    print("=" * 60)
    print("鲁棒残差克里金 (RRK) Ten-Fold Cross Validation")
    print(f"Parameters: poly_degree={poly_degree}, huber_epsilon={huber_epsilon}")
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

        # Step 1: 多项式特征
        poly = PolynomialFeatures(degree=poly_degree, include_bias=False)
        m_train_poly = poly.fit_transform(m_train.reshape(-1, 1))
        m_test_poly = poly.transform(m_test.reshape(-1, 1))

        # Step 2: Huber稳健多项式校正
        huber = HuberRegressor(epsilon=huber_epsilon, max_iter=1000)
        huber.fit(m_train_poly, y_train)
        pred_huber = huber.predict(m_test_poly)
        residual_huber = y_train - huber.predict(m_train_poly)

        # Step 3: GPR残差克里金
        gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2,
                                       alpha=0.1, normalize_y=True)
        gpr.fit(X_train, residual_huber)
        gpr_pred, _ = gpr.predict(X_test, return_std=True)

        # Step 4: 融合输出
        rrk_pred = pred_huber + gpr_pred

        results[fold_id] = {
            'y_true': y_test,
            'rrk': rrk_pred
        }

        print(f"  Fold {fold_id}: completed")

    # 汇总
    rrk_all = np.concatenate([results[f]['rrk'] for f in range(1, 11) if results[f]])
    true_all = np.concatenate([results[f]['y_true'] for f in range(1, 11) if results[f]])

    # 计算指标
    metrics = compute_metrics(true_all, rrk_all)
    # 缓存预测值供多天聚合使用
    global _last_y_true, _last_y_pred
    _last_y_true = true_all
    _last_y_pred = rrk_all


    print(f"\n=== 鲁棒残差克里金 Results ===")
    print(f"  R2={metrics['R2']:.4f}, MAE={metrics['MAE']:.2f}, RMSE={metrics['RMSE']:.2f}, MB={metrics['MB']:.2f}")

    # 保存结果
    result_df = pd.DataFrame([{
        'Method': '鲁棒残差克里金',
        'R2': metrics['R2'],
        'MAE': metrics['MAE'],
        'RMSE': metrics['RMSE'],
        'MB': metrics['MB']
    }])
    result_df.to_csv(f'{output_dir}/鲁棒残差克里金_summary.csv', index=False)

    print(f"\nResults saved to: {output_dir}/鲁棒残差克里金_summary.csv")
    return metrics


if __name__ == '__main__':
    metrics = run_鲁棒残差克里金_ten_fold('2020-01-01')
    print(f"\nFinal: R2={metrics['R2']:.4f}")
