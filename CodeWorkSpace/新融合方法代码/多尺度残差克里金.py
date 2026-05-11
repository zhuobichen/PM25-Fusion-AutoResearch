"""

# === 预测值缓存（由 patch_methods.py 自动添加）===
_last_y_true = None
_last_y_pred = None

多尺度残差克里金 (Multi-Scale Residual Kriging, MSRK)
=====================================================
在PolyRK的二次多项式OLS全局校正+局部GPR克里金混合架构基础上，
引入多尺度GPR克里金融合。使用3个不同长度尺度的GPR（短/中/长）
分别对残差进行建模，加权融合多尺度预测以同时捕捉局地和区域尺度的空间相关性。

核心公式:
1. 全局多项式校正: M_cal(s) = a0 + a1*CMAQ(s) + a2*CMAQ(s)^2
2. 残差计算: R(s) = O(s) - M_cal(s)
3. 多尺度GPR: R*(s) = w_S*R_S(s) + w_M*R_M(s) + w_L*R_L(s)
4. 最终融合: P(s) = M_cal(s) + R*(s)

参数:
- 短尺度长度: 7.0 km, 权重: 0.5
- 中尺度长度: 20.0 km, 权重: 0.3
- 长尺度长度: 50.0 km, 权重: 0.2
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


def make_multiscale_kernel(length_scale):
    """构建指定长度尺度的GPR核函数"""
    return (ConstantKernel(10.0, (1e-2, 1e3)) *
            RBF(length_scale=length_scale, length_scale_bounds=(1e-2, 1e2)) +
            WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e1)))


def run_多尺度残差克里金_ten_fold(selected_day='2020-01-01'):
    """
    运行多尺度残差克里金(MSRK)十折交叉验证

    创新点:
    - 3个不同长度尺度的GPR: 短(7km), 中(20km), 长(50km)
    - 物理直觉权重融合: w_S=0.5, w_M=0.3, w_L=0.2
    - 同时捕捉局地微结构和区域大尺度趋势

    Parameters:
    -----------
    selected_day: str
        选择验证的日期 (默认: '2020-01-01')

    Returns:
    --------
    metrics: dict
        包含R2, MAE, RMSE, MB的评估指标
    """
    # 多尺度参数
    scales = [7.0, 20.0, 50.0]
    scale_weights = [0.5, 0.3, 0.2]

    print("=" * 60)
    print("多尺度残差克里金 (MSRK) Ten-Fold Cross Validation")
    print(f"Parameters: scales={scales}, weights={scale_weights}")
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

    # 归一化权重
    w_total = sum(scale_weights)
    scale_weights_norm = [w / w_total for w in scale_weights]

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

        # Step 1: 二次多项式OLS校正
        poly = PolynomialFeatures(degree=2, include_bias=False)
        m_train_poly = poly.fit_transform(m_train.reshape(-1, 1))
        m_test_poly = poly.transform(m_test.reshape(-1, 1))

        ols = LinearRegression()
        ols.fit(m_train_poly, y_train)
        pred_ols = ols.predict(m_test_poly)
        residual = y_train - ols.predict(m_train_poly)

        # Step 2: 多尺度GPR拟合与预测
        multi_scale_preds = []
        for scale in scales:
            kernel = make_multiscale_kernel(scale)
            gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2,
                                           alpha=0.1, normalize_y=True)
            gpr.fit(X_train, residual)
            gpr_pred, _ = gpr.predict(X_test, return_std=True)
            multi_scale_preds.append(gpr_pred)

        multi_scale_preds = np.array(multi_scale_preds)  # (n_scales, n_test)

        # Step 3: 加权融合多尺度预测
        gpr_fusion_pred = np.sum(
            multi_scale_preds * np.array(scale_weights_norm).reshape(-1, 1), axis=0
        )

        # Step 4: 最终融合
        msrk_pred = pred_ols + gpr_fusion_pred

        results[fold_id] = {
            'y_true': y_test,
            'msrk': msrk_pred
        }

        print(f"  Fold {fold_id}: completed")

    # 汇总
    msrk_all = np.concatenate([results[f]['msrk'] for f in range(1, 11) if results[f]])
    true_all = np.concatenate([results[f]['y_true'] for f in range(1, 11) if results[f]])

    # 计算指标
    metrics = compute_metrics(true_all, msrk_all)
    # 缓存预测值供多天聚合使用
    global _last_y_true, _last_y_pred
    _last_y_true = true_all
    _last_y_pred = msrk_all


    print(f"\n=== 多尺度残差克里金 Results ===")
    print(f"  R2={metrics['R2']:.4f}, MAE={metrics['MAE']:.2f}, RMSE={metrics['RMSE']:.2f}, MB={metrics['MB']:.2f}")

    # 保存结果
    result_df = pd.DataFrame([{
        'Method': '多尺度残差克里金',
        'R2': metrics['R2'],
        'MAE': metrics['MAE'],
        'RMSE': metrics['RMSE'],
        'MB': metrics['MB']
    }])
    result_df.to_csv(f'{output_dir}/多尺度残差克里金_summary.csv', index=False)

    print(f"\nResults saved to: {output_dir}/多尺度残差克里金_summary.csv")
    return metrics


if __name__ == '__main__':
    metrics = run_多尺度残差克里金_ten_fold('2020-01-01')
    print(f"\nFinal: R2={metrics['R2']:.4f}")
