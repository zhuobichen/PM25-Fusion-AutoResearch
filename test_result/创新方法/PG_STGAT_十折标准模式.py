# -*- coding: utf-8 -*-
"""
PG-STGAT 十折交叉验证 - 标准模式
=================================
按照十折交叉验证架构文档：
- 训练：9折监测站(Lon, Lat) + CMAQ值
- 预测：对1折站点所在的CMAQ网格坐标预测
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
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
from joblib import Parallel, delayed

ROOT_DIR = str(get_project_root())
CMAQ_FILE = data_path('test_data/raw/CMAQ/2020_PM25.nc')
MONITOR_FILE = data_path('test_data/raw/Monitor/2020_DailyPM2.5Monitor.csv')
FOLD_FILE = data_path('test_data/fold_split_table_daily.csv')
OUTPUT_FILE = f'{ROOT_DIR}/Innovation/success/PG-STGAT/PG_STGAT_all_stages.json'

# VNA baseline (当前最优基准)
BASELINE = {
    'pre_exp': {'R2': 0.8941, 'RMSE': 16.42, 'MB': 0.76},
    'stage1':  {'R2': 0.9057, 'RMSE': 16.28, 'MB': 0.50},
    'stage2':  {'R2': 0.8458, 'RMSE': 4.97, 'MB': 0.04},
    'stage3':  {'R2': 0.9078, 'RMSE': 11.90, 'MB': 0.36},
}


def compute_metrics(y_true, y_pred):
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
    dist = np.sqrt((lon_grid - lon)**2 + (lat_grid - lat)**2)
    idx = np.argmin(dist)
    ny, nx = lon_grid.shape
    row, col = idx // nx, idx % nx
    return pm25_grid[row, col]


def get_cmaq_grid_coord(lon, lat, lon_grid, lat_grid):
    """获取站点对应的CMAQ网格坐标"""
    dist = np.sqrt((lon_grid - lon)**2 + (lat_grid - lat)**2)
    idx = np.argmin(dist)
    ny, nx = lon_grid.shape
    row, col = idx // nx, idx % nx
    return lon_grid[row, col], lat_grid[row, col]


def haversine_distance(lon1, lat1, lon2, lat2):
    R = 6371.0
    lon1_r, lat1_r, lon2_r, lat2_r = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2_r - lon1_r
    dlat = lat2_r - lat1_r
    a = np.sin(dlat/2)**2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))


def build_station_graph_features(lon, lat, y_cmaq, residual, k_neighbors=10, wind_dir=0.0):
    """为每个训练站点构建图注意力特征"""
    n = len(lon)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        dist_matrix[i, :] = haversine_distance(lon[i], lat[i], lon, lat)

    features = np.zeros((n, 24))

    for i in range(n):
        distances = dist_matrix[i, :]
        sorted_idx = np.argsort(distances)
        knn_idx = sorted_idx[1:k_neighbors+1]
        knn_dist = distances[knn_idx]
        knn_lon = lon[knn_idx]
        knn_lat = lat[knn_idx]
        knn_residual = residual[knn_idx]
        knn_cmaq = y_cmaq[knn_idx]

        # 头0: 空间距离衰减注意力
        sigma_space = 1.0
        w_space = np.exp(-knn_dist / sigma_space)
        w_space = w_space / (w_space.sum() + 1e-10)
        features[i, 0] = np.sum(w_space * knn_residual)
        features[i, 1] = np.sum(w_space * knn_cmaq)
        features[i, 2] = np.mean(knn_dist)
        features[i, 3] = np.var(knn_residual)
        features[i, 4] = np.max(knn_dist)
        features[i, 5] = np.sum(w_space > 0.1)

        # 头1: CMAQ相似性注意力
        cmaq_diff = np.abs(knn_cmaq - y_cmaq[i]) + 1.0
        w_cmaq = 1.0 / cmaq_diff
        w_cmaq = w_cmaq / (w_cmaq.sum() + 1e-10)
        features[i, 6] = np.sum(w_cmaq * knn_residual)
        features[i, 7] = np.sum(w_cmaq * knn_cmaq)
        features[i, 8] = np.mean(np.abs(knn_residual))
        features[i, 9] = np.sum(w_cmaq * np.abs(knn_residual))
        features[i, 10] = np.max(np.abs(knn_residual))
        features[i, 11] = np.sum(w_cmaq * np.sign(knn_residual))

        # 头2: 风场传输注意力
        dlon = knn_lon - lon[i]
        dlat = knn_lat - lat[i]
        theta_ij = np.arctan2(dlat, dlon)
        theta_wind = np.radians(wind_dir)
        cos_diff = np.cos(theta_wind - theta_ij)
        w_wind = np.maximum(cos_diff, 0) * np.exp(-knn_dist / 2.0)
        w_wind = w_wind / (w_wind.sum() + 1e-10)
        features[i, 12] = np.sum(w_wind * knn_residual)
        features[i, 13] = np.sum(w_wind * knn_cmaq)
        features[i, 14] = np.sum(w_wind * np.sign(knn_residual))
        features[i, 15] = np.sum(np.maximum(cos_diff, 0))
        features[i, 16] = np.sum(w_wind * np.abs(knn_residual))
        features[i, 17] = np.mean(knn_residual) * np.sum(np.maximum(cos_diff, 0))

        # 头3: 综合注意力
        w_combined = (w_space + w_cmaq + w_wind) / 3.0
        w_combined = w_combined / (w_combined.sum() + 1e-10)
        features[i, 18] = np.sum(w_combined * knn_residual)
        features[i, 19] = np.sum(w_combined * knn_cmaq)
        features[i, 20] = np.mean(knn_residual)
        features[i, 21] = np.sum(w_combined * np.abs(knn_residual))
        features[i, 22] = np.std(knn_residual)
        features[i, 23] = np.sum(w_space * np.abs(knn_residual))

    return features


class PGSTGATModel:
    """PG-STGAT: 物理引导时空图注意力网络"""

    def __init__(self, k_neighbors=20, wind_dir=180.0):
        self.k_neighbors = k_neighbors
        self.wind_dir = wind_dir

    def fit(self, X_train, y_train, y_cmaq_train):
        """训练：X_train是监测站(Lon, Lat)坐标"""
        lon_train = X_train[:, 0]
        lat_train = X_train[:, 1]
        self.lon_train = lon_train
        self.lat_train = lat_train
        self.n_train = len(lon_train)

        # 残差
        residual = y_train - y_cmaq_train
        self.residual_bias = np.mean(residual)
        self.residual_std = np.std(residual) + 1e-6

        # 构建图注意力特征
        self.gat_features = build_station_graph_features(
            lon_train, lat_train, y_cmaq_train, residual,
            k_neighbors=self.k_neighbors, wind_dir=self.wind_dir
        )

        # 归一化
        self.scaler = StandardScaler()
        gat_scaled = self.scaler.fit_transform(self.gat_features)

        # 训练: 预测归一化残差
        residual_norm = (residual - self.residual_bias) / self.residual_std
        self.model = Ridge(alpha=1.0)
        self.model.fit(gat_scaled, residual_norm)

        self.residual_train = residual

        return self

    def predict(self, X_query, y_cmaq_query):
        """预测：X_query是CMAQ网格坐标"""
        lon_query = X_query[:, 0]
        lat_query = X_query[:, 1]
        n_query = len(lon_query)

        # 使用训练时的监测站坐标来构建KNN（因为GAT特征是基于监测站的）
        train_coords = np.column_stack([self.lon_train, self.lat_train])
        query_coords = np.column_stack([lon_query, lat_query])
        dist = cdist(query_coords, train_coords)

        k = min(self.k_neighbors, self.n_train)
        knn = NearestNeighbors(n_neighbors=k)
        knn.fit(train_coords)
        dist_k, idx_k = knn.kneighbors(query_coords)

        # GAT特征IDW插值 + 模型预测
        gat_query = np.zeros((n_query, 24))
        for i in range(n_query):
            w_norm = 1.0 / (dist_k[i] + 1e-6)
            w_norm = w_norm / w_norm.sum()
            gat_query[i] = w_norm @ self.gat_features[idx_k[i]]

        gat_scaled = self.scaler.transform(gat_query)
        residual_pred_norm = self.model.predict(gat_scaled)
        residual_pred = residual_pred_norm * self.residual_std + self.residual_bias

        y_pred = y_cmaq_query + residual_pred

        return y_pred


def ten_fold_pg_stgat(selected_day, wind_dir=180.0):
    """标准模式十折验证"""
    monitor_df = pd.read_csv(MONITOR_FILE)
    fold_df = pd.read_csv(FOLD_FILE)

    day_df = monitor_df[monitor_df['Date'] == selected_day].copy()
    day_df = day_df.merge(fold_df, on=['Date', 'Site'], how='left')
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

    # 获取每个站点的CMAQ值
    cmaq_values = []
    for _, row in day_df.iterrows():
        val = get_cmaq_at_site(row['Lon'], row['Lat'], lon_cmaq, lat_cmaq, cmaq_day)
        cmaq_values.append(val)
    day_df['CMAQ'] = cmaq_values

    params = {'k_neighbors': 20, 'wind_dir': wind_dir}

    all_y_true = []
    all_y_pred = []

    for fold_id in range(1, 11):
        train_df = day_df[day_df['fold'] != fold_id].copy()
        test_df = day_df[day_df['fold'] == fold_id].copy()

        train_df = train_df.dropna(subset=['Lon', 'Lat', 'CMAQ', 'Conc'])
        test_df = test_df.dropna(subset=['Lon', 'Lat', 'CMAQ', 'Conc'])

        if len(test_df) == 0 or len(train_df) == 0:
            continue

        # 训练：使用监测站坐标
        X_train = train_df[['Lon', 'Lat']].values
        y_train = train_df['Conc'].values
        y_cmaq_train = train_df['CMAQ'].values

        # 测试：获取验证站点所在的CMAQ网格坐标
        test_cmaq_coords = []
        for _, row in test_df.iterrows():
            cmaq_lon, cmaq_lat = get_cmaq_grid_coord(row['Lon'], row['Lat'], lon_cmaq, lat_cmaq)
            test_cmaq_coords.append([cmaq_lon, cmaq_lat])
        X_test = np.array(test_cmaq_coords)

        y_test = test_df['Conc'].values
        y_cmaq_test = test_df['CMAQ'].values

        model = PGSTGATModel(**params)
        model.fit(X_train, y_train, y_cmaq_train)
        y_pred = model.predict(X_test, y_cmaq_test)

        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)

    return np.array(all_y_true), np.array(all_y_pred)


def run_stage_validation(stage_name, start_date, end_date, wind_dir=180.0):
    sep = "=" * 70
    print(sep)
    print(f"PG-STGAT Stage: {stage_name} ({start_date} ~ {end_date})")
    print(sep)

    base = BASELINE[stage_name]
    threshold_r2 = base['R2']
    print(f"VNA Baseline: R2={base['R2']:.4f}, RMSE={base['RMSE']:.2f}, MB={base['MB']:.2f}")
    print(f"Threshold: R2>{threshold_r2:.4f}, RMSE<={base['RMSE']}, |MB|<={abs(base['MB'])}")

    date_list = []
    current_date = datetime.strptime(start_date, '%Y-%m-%d')
    end_date_obj = datetime.strptime(end_date, '%Y-%m-%d')
    while current_date <= end_date_obj:
        date_list.append(current_date.strftime('%Y-%m-%d'))
        current_date += timedelta(days=1)

    print(f"Days: {len(date_list)}")

    n_jobs = min(8, len(date_list))
    results = Parallel(n_jobs=n_jobs)(
        delayed(ten_fold_pg_stgat)(date_str, wind_dir)
        for date_str in date_list
    )

    all_y_true = []
    all_y_pred = []
    day_count = 0
    for y_true, y_pred in results:
        if len(y_true) > 0:
            all_y_true.extend(y_true)
            all_y_pred.extend(y_pred)
            day_count += 1

    print(f"Processed: {day_count} days, {len(all_y_true)} predictions")

    if len(all_y_true) == 0:
        return {'R2': np.nan, 'MAE': np.nan, 'RMSE': np.nan, 'MB': np.nan}, False

    metrics = compute_metrics(np.array(all_y_true), np.array(all_y_pred))

    r2_pass = metrics['R2'] > threshold_r2
    rmse_pass = metrics['RMSE'] <= base['RMSE']
    mb_pass = abs(metrics['MB']) <= abs(base['MB'])
    innovation_pass = r2_pass and rmse_pass and mb_pass

    print(f"Result: R2={metrics['R2']:.4f}, RMSE={metrics['RMSE']:.2f}, MAE={metrics['MAE']:.2f}, MB={metrics['MB']:.2f}")
    r2_str = "PASS" if r2_pass else "FAIL"
    rmse_str = "PASS" if rmse_pass else "FAIL"
    mb_str = "PASS" if mb_pass else "FAIL"
    innov_str = "VERIFIED" if innovation_pass else "NOT VERIFIED"
    print(f"Check: R2>{threshold_r2:.4f}? {r2_str} | RMSE<={base['RMSE']}? {rmse_str} | |MB|<={abs(base['MB'])}? {mb_str}")
    print(f"Innovation: {innov_str}")

    return metrics, innovation_pass


def main():
    sep = "=" * 70
    print(sep)
    print("PG-STGAT All Stages - 标准模式")
    print("训练: 9折监测站(Lon,Lat) | 预测: 1折站点CMAQ网格坐标")
    print(sep)

    stages = {
        'pre_exp':  ('2020-01-01', '2020-01-05'),
        'stage1':   ('2020-01-01', '2020-01-31'),
        'stage2':   ('2020-07-01', '2020-07-31'),
        'stage3':   ('2020-12-01', '2020-12-31'),
    }

    results = {}
    all_pass = True

    for stage_name, (start, end) in stages.items():
        metrics, innovation_pass = run_stage_validation(stage_name, start, end)
        results[stage_name] = {'metrics': metrics, '判定': {'innovation_verified': innovation_pass}}
        if not innovation_pass:
            all_pass = False

    # 保存结果
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(sep)
    print("SUMMARY")
    print(sep)
    for stage, data in results.items():
        m = data['metrics']
        status = 'VERIFIED' if data['判定']['innovation_verified'] else 'NOT VERIFIED'
        print(f"{stage}: R2={m['R2']:.4f}, RMSE={m['RMSE']:.2f}, MB={m['MB']:.2f} -> {status}")

    print(f"\nAll stages passed: {all_pass}")
    print(f"Results saved: {OUTPUT_FILE}")

    return results


if __name__ == '__main__':
    main()
