# -*- coding: utf-8 -*-
"""
RK_OLS_Poly v11标准 pre_exp 验证
===============================
每天十折验证，合并所有天预测计算整体指标

基准对比（VNA pre_exp）:
  R² = 0.8941, RMSE = 16.42, MB = 0.76

创新阈值（需同时满足）:
  R² >= 0.8941 + 0.01 = 0.9041
  RMSE <= 16.42
  |MB| <= 0.76
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from shared.paths import get_project_root, data_path
import json
import numpy as np
import pandas as pd
import netCDF4 as nc
from datetime import datetime, timedelta
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import HuberRegressor, LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from joblib import Parallel, delayed

# 路径配置
ROOT_DIR = str(get_project_root())
CMAQ_FILE = data_path('test_data/raw/CMAQ/2020_PM25.nc')
MONITOR_FILE = data_path('test_data/raw/Monitor/2020_DailyPM2.5Monitor.csv')
FOLD_FILE = data_path('test_data/fold_split_table_daily.csv')
OUTPUT_FILE = f'{ROOT_DIR}/test_result/创新方法/RRK_v11_preexp.json'

# 基准（VNA pre_exp）
BASELINE = {
    'VNA': {'R2': 0.8941, 'RMSE': 16.42, 'MB': 0.76}
}
THRESHOLD_R2 = BASELINE['VNA']['R2'] + 0.01  # 0.9041


def compute_metrics(y_true, y_pred):
    """计算评估指标"""
    mask = ~(np.isnan(y_true) | np.isnan(y_pred) | np.isinf(y_true) | np.isinf(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if len(y_true) == 0:
        return {'R2': np.nan, 'MAE': np.nan, 'RMSE': np.nan, 'MB': np.nan}
    return {
        'R2': float(r2_score(y_true, y_pred)),
        'MAE': float(mean_absolute_error(y_true, y_pred)),
        'RMSE': float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'MB': float(np.mean(y_pred - y_true))
    }


def get_cmaq_at_site(lon, lat, lon_grid, lat_grid, pm25_grid):
    """获取站点位置的CMAQ值（最近邻）"""
    dist = np.sqrt((lon_grid - lon)**2 + (lat_grid - lat)**2)
    idx = np.argmin(dist)
    ny, nx = lon_grid.shape
    row, col = idx // nx, idx % nx
    return pm25_grid[row, col]


def ten_fold_for_day_rrk(selected_day, poly_degree=2):
    """
    RK_OLS_Poly 的十折交叉验证（单天）
    """
    monitor_df = pd.read_csv(MONITOR_FILE)
    fold_df = pd.read_csv(FOLD_FILE)

    day_df = monitor_df[monitor_df['Date'] == selected_day].copy()
    day_df = day_df.merge(fold_df, on='Site', how='left')
    day_df = day_df.dropna(subset=['Lat', 'Lon', 'Conc'])

    if len(day_df) < 100:
        return np.array([]), np.array([])

    ds = nc.Dataset(CMAQ_FILE, 'r')
    lon_cmaq = ds.variables['lon'][:]
    lat_cmaq = ds.variables['lat'][:]
    pred_pm25 = ds.variables['pred_PM25'][:]
    ds.close()

    date_obj = datetime.strptime(selected_day, '%Y-%m-%d')
    day_idx = (date_obj - datetime(2020, 1, 1)).days
    if day_idx >= pred_pm25.shape[0]:
        return np.array([]), np.array([])
    cmaq_day = pred_pm25[day_idx]

    # 提取站点CMAQ值
    cmaq_values = []
    for _, row in day_df.iterrows():
        val = get_cmaq_at_site(row['Lon'], row['Lat'], lon_cmaq, lat_cmaq, cmaq_day)
        cmaq_values.append(val)
    day_df['CMAQ'] = cmaq_values

    # GPR核函数
    kernel = (ConstantKernel(10.0, (1e-2, 1e3)) *
              RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) +
              WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e1)))

    all_y_true = []
    all_y_pred = []

    for fold_id in range(1, 11):
        train_df = day_df[day_df['fold'] != fold_id].copy()
        test_df = day_df[day_df['fold'] == fold_id].copy()

        train_df = train_df.dropna(subset=['Lon', 'Lat', 'CMAQ', 'Conc'])
        test_df = test_df.dropna(subset=['Lon', 'Lat', 'CMAQ', 'Conc'])

        if len(test_df) == 0 or len(train_df) == 0:
            continue

        X_train = train_df[['Lon', 'Lat']].values
        X_test = test_df[['Lon', 'Lat']].values
        y_train = train_df['Conc'].values
        y_test = test_df['Conc'].values
        m_train = train_df['CMAQ'].values
        m_test = test_df['CMAQ'].values

        # 多项式特征
        poly = PolynomialFeatures(degree=poly_degree, include_bias=False)
        m_train_poly = poly.fit_transform(m_train.reshape(-1, 1))
        m_test_poly = poly.transform(m_test.reshape(-1, 1))

        # OLS 多项式校正
        ols = LinearRegression()
        ols.fit(m_train_poly, y_train)
        pred_ols = ols.predict(m_test_poly)
        residual_ols = y_train - ols.predict(m_train_poly)

        # GPR on residuals
        gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, alpha=0.1, normalize_y=True)
        gpr.fit(X_train, residual_ols)
        gpr_pred, _ = gpr.predict(X_test, return_std=True)

        # 融合预测
        rk_pred = pred_ols + gpr_pred

        all_y_true.extend(y_test)
        all_y_pred.extend(rk_pred)

    return np.array(all_y_true), np.array(all_y_pred)


def run_pre_exp_validation():
    """运行 pre_exp 验证（5天，每天十折）"""
    print("="*70)
    print("RK_OLS_Poly v11 pre_exp Validation")
    print("="*70)
    print(f"\nBaseline (VNA pre_exp): R2={BASELINE['VNA']['R2']:.4f}, RMSE={BASELINE['VNA']['RMSE']:.2f}, MB={BASELINE['VNA']['MB']:.2f}")
    print(f"Threshold: R2>={THRESHOLD_R2:.4f}, RMSE<={BASELINE['VNA']['RMSE']}, |MB|<={abs(BASELINE['VNA']['MB'])}")

    # 生成日期列表
    start_date = '2020-01-01'
    end_date = '2020-01-05'
    date_list = []
    current_date = datetime.strptime(start_date, '%Y-%m-%d')
    end_date_obj = datetime.strptime(end_date, '%Y-%m-%d')
    while current_date <= end_date_obj:
        date_list.append(current_date.strftime('%Y-%m-%d'))
        current_date += timedelta(days=1)

    print(f"\n日期范围: {start_date} ~ {end_date} ({len(date_list)}天)")

    # 并行处理多天
    print("\n开始验证...")
    results = Parallel(n_jobs=4)(
        delayed(ten_fold_for_day_rrk)(date_str)
        for date_str in date_list
    )

    # 合并所有天的结果
    all_y_true = []
    all_y_pred = []
    day_count = 0
    for y_true, y_pred in results:
        if len(y_true) > 0:
            all_y_true.extend(y_true)
            all_y_pred.extend(y_pred)
            day_count += 1

    print(f"成功处理 {day_count} 天，共 {len(all_y_true)} 个预测点")

    if len(all_y_true) == 0:
        return {'R2': np.nan, 'MAE': np.nan, 'RMSE': np.nan, 'MB': np.nan}

    metrics = compute_metrics(np.array(all_y_true), np.array(all_y_pred))

    # 判定创新
    r2_pass = metrics['R2'] >= THRESHOLD_R2
    rmse_pass = metrics['RMSE'] <= BASELINE['VNA']['RMSE']
    mb_pass = abs(metrics['MB']) <= abs(BASELINE['VNA']['MB'])
    innovation_pass = r2_pass and rmse_pass and mb_pass

    print(f"\n{'='*70}")
    print(f"Result: R2={metrics['R2']:.4f}, RMSE={metrics['RMSE']:.2f}, MAE={metrics['MAE']:.2f}, MB={metrics['MB']:.2f}")
    print(f"判定: R2>={THRESHOLD_R2:.4f}? {'PASS' if r2_pass else 'FAIL'} | RMSE<={BASELINE['VNA']['RMSE']}? {'PASS' if rmse_pass else 'FAIL'} | |MB|<={abs(BASELINE['VNA']['MB'])}? {'PASS' if mb_pass else 'FAIL'}")
    print(f"Innovation: {'VERIFIED' if innovation_pass else 'NOT VERIFIED'}")
    print(f"{'='*70}")

    return metrics, innovation_pass


def main():
    metrics, innovation_pass = run_pre_exp_validation()

    # 保存结果
    result = {
        'method': 'RK_OLS_Poly',
        'stage': 'pre_exp',
        'stage_name': '预实验(5天)',
        'baseline': BASELINE,
        'threshold': {
            'R2': THRESHOLD_R2,
            'RMSE': BASELINE['VNA']['RMSE'],
            'MB': abs(BASELINE['VNA']['MB'])
        },
        'metrics': metrics,
        '判定': {
            'R2_pass': metrics['R2'] >= THRESHOLD_R2,
            'RMSE_pass': metrics['RMSE'] <= BASELINE['VNA']['RMSE'],
            'MB_pass': abs(metrics['MB']) <= abs(BASELINE['VNA']['MB']),
            'innovation_verified': innovation_pass
        }
    }

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\n结果已保存: {OUTPUT_FILE}")

    return result


if __name__ == '__main__':
    main()