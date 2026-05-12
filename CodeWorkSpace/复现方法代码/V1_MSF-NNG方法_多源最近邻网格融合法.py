"""
V1_MSF-NNG方法_多源最近邻网格融合法 - 多源时空最近邻网格融合法
================================================================
Reproduction of MSF-NNG (Multi-source Spatiotemporal Nearest Neighbor Grid Fusion)

核心思想：
  1. Cressman插值：用距离权重从稀疏监测站创建平滑观测场
  2. 最近邻网格匹配：找到与目标点最相似的CMAQ网格点
  3. 多源加权融合：Z_fused = alpha * Z_Cressman + (1-alpha) * Z_CMAQ

文献来源：
  An Improved Multi-source Spatiotemporal Data Fusion Model Based on
  the Nearest Neighbor Grids for PM2.5 Concentration Interpolation and Prediction
  Springer, 2023
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.paths import get_project_root, data_path
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.spatial.distance import cdist
import netCDF4 as nc

root_dir = str(get_project_root())
cmaq_file = data_path('test_data/raw/CMAQ/2020_PM25.nc')
monitor_file = data_path('test_data/raw/Monitor/2020_DailyPM2.5Monitor.csv')
fold_file = data_path('test_data/fold_split_table_daily.csv')
output_dir = f'{root_dir}/test_result/复现方法'
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


class MSF_NNG:
    """
    多源时空最近邻网格融合法

    步骤:
    1. Cressman插值：用训练站构建观测场
    2. 最近邻匹配：找CMAQ网格中最匹配的点
    3. 加权融合：Z_fused = alpha * Z_Cressman + (1-alpha) * Z_CMAQ
    """

    def __init__(self, radius_R=50.0, n_neighbors=5, alpha=0.7):
        """
        Parameters:
        -----------
        radius_R : float
            Cressman影响半径 (km)
        n_neighbors : int
            最近邻数量
        alpha : float
            观测数据权重 (固定，非学习)
        """
        self.radius_R = radius_R
        self.n_neighbors = n_neighbors
        self.alpha = alpha

    def _haversine_distance(self, lon1, lat1, lon2, lat2):
        """计算两点间的Haversine距离 (km)"""
        R = 6371.0  # 地球半径 km
        lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        return R * c

    def _cressman_interpolate(self, target_lon, target_lat, station_lons, station_lats, station_values):
        """
        Cressman插值

        公式: Z_hat(s0) = sum(w_i * Z(s_i))
        w_i = (R^2 - d_i^2) / (R^2 + d_i^2)

        Parameters:
        -----------
        target_lon, target_lat : float - 目标点坐标
        station_lons, station_lats : array - 站点坐标
        station_values : array - 站点观测值

        Returns:
        --------
        interpolated_value : float
        """
        # 计算目标点到各站点的距离 (km)
        distances = np.array([
            self._haversine_distance(target_lon, target_lat, slon, slat)
            for slon, slat in zip(station_lons, station_lats)
        ])

        # 搜索影响半径内的站点
        mask = distances < self.radius_R
        if not np.any(mask):
            # 半径内无站点，使用最近邻
            nearest_idx = np.argmin(distances)
            return station_values[nearest_idx]

        d_in = distances[mask]
        v_in = station_values[mask]

        # Cressman权重
        R2 = self.radius_R ** 2
        weights = (R2 - d_in**2) / (R2 + d_in**2)

        # 归一化
        w_sum = np.sum(weights)
        if w_sum > 0:
            weights = weights / w_sum
        else:
            weights = np.ones_like(weights) / len(weights)

        return np.sum(weights * v_in)

    def fit_predict(self, train_coords, train_obs, train_cmaq,
                    test_coords, test_cmaq):
        """
        训练并预测

        Parameters:
        -----------
        train_coords : array (n_train, 2) - 训练站坐标 [lon, lat]
        train_obs : array (n_train,) - 训练站观测值
        train_cmaq : array (n_train,) - 训练站CMAQ值
        test_coords : array (n_test, 2) - 测试站坐标 [lon, lat]
        test_cmaq : array (n_test,) - 测试站CMAQ值

        Returns:
        --------
        predictions : array (n_test,) - 融合预测值
        """
        n_test = len(test_coords)
        predictions = np.zeros(n_test)

        for i in range(n_test):
            target_lon = test_coords[i, 0]
            target_lat = test_coords[i, 1]

            # 步骤1: Cressman插值
            z_cressman = self._cressman_interpolate(
                target_lon, target_lat,
                train_coords[:, 0], train_coords[:, 1],
                train_obs
            )

            # 步骤2: 获取CMAQ值 (最近邻网格)
            z_cmaq = test_cmaq[i]

            # 步骤3: 加权融合
            predictions[i] = self.alpha * z_cressman + (1 - self.alpha) * z_cmaq

        # 非负约束
        predictions = np.maximum(predictions, 0)

        return predictions


def run_MSF_NNG_ten_fold(selected_day='2020-01-01'):
    """
    运行MSF-NNG十折交叉验证

    Parameters:
    -----------
    selected_day : str
        验证日期，格式 'YYYY-MM-DD'
    """
    print("=" * 60)
    print("MSF-NNG Ten-Fold Cross Validation")
    print(f"Date: {selected_day}")
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

    # 十折交叉验证
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

        # 训练模型
        model = MSF_NNG(radius_R=50.0, n_neighbors=5, alpha=0.7)
        y_pred = model.fit_predict(X_train, y_train, m_train, X_test, m_test)

        results[fold_id] = {
            'y_true': y_test,
            'y_pred': y_pred
        }

        fold_metrics = compute_metrics(y_test, y_pred)
        print(f"  Fold {fold_id}: R2={fold_metrics['R2']:.4f}, "
              f"RMSE={fold_metrics['RMSE']:.2f}, N={len(test_df)}")

    # 汇总
    y_pred_all = np.concatenate([results[f]['y_pred'] for f in range(1, 11) if results[f]])
    y_true_all = np.concatenate([results[f]['y_true'] for f in range(1, 11) if results[f]])

    metrics = compute_metrics(y_true_all, y_pred_all)

    print("\n=== Results ===")
    print(f"  MSF-NNG: R2={metrics['R2']:.4f}, MAE={metrics['MAE']:.2f}, "
          f"RMSE={metrics['RMSE']:.2f}, MB={metrics['MB']:.4f}")

    # 保存结果
    result_df = pd.DataFrame([{
        'method': 'MSF-NNG',
        **metrics
    }])
    result_df.to_csv(f'{output_dir}/MSF-NNG_folds.csv', index=False)

    print(f"\nResults saved to: {output_dir}/")

    return metrics


if __name__ == '__main__':
    metrics = run_MSF_NNG_ten_fold('2020-01-01')
    print(f"\nMSF-NNG: R2={metrics['R2']:.4f}")
