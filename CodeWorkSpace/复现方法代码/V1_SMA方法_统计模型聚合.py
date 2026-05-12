"""
V1_SMA方法_统计模型聚合 - Statistical Model Aggregation
=======================================================
Reproduction of Shao et al. Section 2.2.2

核心公式:
  O(si) = a + b * M(si) + epsilon(si)
  P_SMA(s0) = a_hat + b_hat * M(s0)

OLS估计:
  b_hat = sum((M_i - M_bar)(O_i - O_bar)) / sum((M_i - M_bar)^2)
  a_hat = O_bar - b_hat * M_bar

支持线性和多项式回归变体。
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.paths import get_project_root, data_path
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
import netCDF4 as nc

root_dir = str(get_project_root())
cmaq_file = data_path('test_data/raw/CMAQ/2020_PM25.nc')
monitor_file = data_path('test_data/raw/Monitor/2020_DailyPM2.5Monitor.csv')
fold_file = data_path('test_data/fold_split_table_daily.csv')
output_dir = f'{root_dir}/test_result/复现方法'
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


class SMA:
    """
    Statistical Model Aggregation (统计模型聚合)

    使用OLS回归建立 O = a + b*M 的统计关系，
    然后对网格点预测 P_SMA = a + b*M_grid

    支持:
    - 线性回归: O = a + b*M
    - 多项式回归: O = a0 + a1*M + a2*M^2 + ...
    """

    def __init__(self, regression_type='linear', poly_degree=1):
        """
        Parameters:
        -----------
        regression_type : str
            'linear' 或 'polynomial'
        poly_degree : int
            多项式阶数（仅当regression_type='polynomial'时）
        """
        self.regression_type = regression_type
        self.poly_degree = poly_degree
        self.model = None
        self.a = None
        self.b = None

    def fit(self, train_obs, train_cmaq):
        """
        拟合OLS回归: O = a + b*M

        Parameters:
        -----------
        train_obs : array (n,) - 监测站观测值
        train_cmaq : array (n,) - CMAQ模型值
        """
        if self.regression_type == 'polynomial' and self.poly_degree > 1:
            # 多项式回归
            M = train_cmaq.reshape(-1, 1)
            M_poly = np.column_stack([M**k for k in range(1, self.poly_degree + 1)])
            self.model = LinearRegression()
            self.model.fit(M_poly, train_obs)
        else:
            # 线性回归: O = a + b*M
            M = train_cmaq.reshape(-1, 1)
            self.model = LinearRegression()
            self.model.fit(M, train_obs)
            self.a = self.model.intercept_
            self.b = self.model.coef_[0]

        return self

    def predict(self, pred_cmaq):
        """
        预测: P_SMA = a + b*M

        Parameters:
        -----------
        pred_cmaq : array (m,) - 预测点CMAQ值

        Returns:
        --------
        pred : array (m,) - 融合预测值
        """
        M = pred_cmaq.reshape(-1, 1)

        if self.regression_type == 'polynomial' and self.poly_degree > 1:
            M_poly = np.column_stack([M**k for k in range(1, self.poly_degree + 1)])
            pred = self.model.predict(M_poly)
        else:
            pred = self.model.predict(M)

        pred = np.maximum(pred, 0)
        return pred

    def get_params(self):
        """返回回归参数"""
        if self.regression_type == 'polynomial' and self.poly_degree > 1:
            return {
                'intercept': self.model.intercept_,
                'coefficients': self.model.coef_.tolist()
            }
        else:
            return {'a': self.a, 'b': self.b}


def run_SMA方法_统计模型聚合_ten_fold(selected_day='2020-01-01'):
    """
    运行SMA十折交叉验证
    """
    print("=" * 60)
    print("SMA (Statistical Model Aggregation) Ten-Fold Cross Validation")
    print("=" * 60)

    # 加载数据
    print("\n=== Loading Data ===")
    monitor_df = pd.read_csv(monitor_file)
    fold_df = pd.read_csv(fold_file)

    day_df = monitor_df[monitor_df['Date'] == selected_day].copy()
    day_df = day_df.merge(fold_df, on='Site', how='left')
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

    # 运行十折验证
    print("\n=== Running 10-fold Cross Validation ===")
    results = {fold_id: {} for fold_id in range(1, 11)}

    for fold_id in range(1, 11):
        train_df = day_df[day_df['fold'] != fold_id].copy()
        test_df = day_df[day_df['fold'] == fold_id].copy()

        train_df = train_df.dropna(subset=['Lon', 'Lat', 'CMAQ', 'Conc'])
        test_df = test_df.dropna(subset=['Lon', 'Lat', 'CMAQ', 'Conc'])

        if len(test_df) == 0:
            continue

        train_obs = train_df['Conc'].values
        train_cmaq = train_df['CMAQ'].values
        test_cmaq = test_df['CMAQ'].values
        test_obs = test_df['Conc'].values

        # 训练SMA
        model = SMA(regression_type='linear')
        model.fit(train_obs, train_cmaq)

        # 预测
        y_pred = model.predict(test_cmaq)

        results[fold_id] = {
            'y_true': test_obs,
            'y_pred': y_pred
        }

        params = model.get_params()
        print(f"  Fold {fold_id}: a={params['a']:.4f}, b={params['b']:.4f}")

    # 汇总
    y_pred_all = np.concatenate([results[f]['y_pred'] for f in range(1, 11) if results[f]])
    y_true_all = np.concatenate([results[f]['y_true'] for f in range(1, 11) if results[f]])

    metrics = compute_metrics(y_true_all, y_pred_all)

    print("\n=== Results ===")
    print(f"  SMA (linear): R2={metrics['R2']:.4f}, MAE={metrics['MAE']:.2f}, "
          f"RMSE={metrics['RMSE']:.2f}, MB={metrics['MB']:.4f}")

    # 保存结果
    result_df = pd.DataFrame([{
        'method': 'SMA_linear',
        **metrics
    }])
    result_df.to_csv(f'{output_dir}/SMA_summary.csv', index=False)
    print(f"\nResults saved to: {output_dir}/SMA_summary.csv")

    return metrics


if __name__ == '__main__':
    metrics = run_SMA方法_统计模型聚合_ten_fold('2020-01-01')
    print(f"\nSMA: R2={metrics['R2']:.4f}")
