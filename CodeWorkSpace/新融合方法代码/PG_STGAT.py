"""
PG-STGAT - 物理引导时空图注意力网络法
=======================================
Physical-Guided Spatio-Temporal Graph Attention Network

核心创新点:
1. CMAQ物理约束残差: R = Y_obs - Y_CMAQ
2. 空间图注意力: 多头邻居特征聚合
3. 风场传输约束: 顺风增强/逆风衰减
4. 距离-CMAQ双核残差插值

方法指纹: MD5: `pg_stgat_v3_spatial_attention`
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.paths import get_project_root, data_path
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
import netCDF4 as nc

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


def haversine_distance(lon1, lat1, lon2, lat2):
    R = 6371.0
    lon1_r, lat1_r, lon2_r, lat2_r = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2_r - lon1_r
    dlat = lat2_r - lat1_r
    a = np.sin(dlat/2)**2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))


def build_station_graph_features(lon, lat, y_cmaq, residual, k_neighbors=10, wind_dir=0.0):
    """
    为每个训练站点构建图注意力特征
    使用邻居的残差信息来构建特征
    """
    n = len(lon)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        dist_matrix[i, :] = haversine_distance(lon[i], lat[i], lon, lat)

    # 多头注意力特征 (4头 x 6特征 = 24)
    features = np.zeros((n, 24))

    for i in range(n):
        distances = dist_matrix[i, :]
        sorted_idx = np.argsort(distances)
        knn_idx = sorted_idx[1:k_neighbors+1]  # 排除自身
        knn_dist = distances[knn_idx]
        knn_lon = lon[knn_idx]
        knn_lat = lat[knn_idx]
        knn_residual = residual[knn_idx]
        knn_cmaq = y_cmaq[knn_idx]

        # --- 头0: 空间距离衰减注意力 ---
        sigma_space = 1.0
        w_space = np.exp(-knn_dist / sigma_space)
        w_space = w_space / (w_space.sum() + 1e-10)
        features[i, 0] = np.sum(w_space * knn_residual)  # 加权残差
        features[i, 1] = np.sum(w_space * knn_cmaq)   # 加权CMAQ
        features[i, 2] = np.mean(knn_dist)             # 平均距离
        features[i, 3] = np.var(knn_residual)          # 残差方差
        features[i, 4] = np.max(knn_dist)              # 最远邻居距离
        features[i, 5] = np.sum(w_space > 0.1)          # 有效邻居数

        # --- 头1: CMAQ相似性注意力 ---
        cmaq_diff = np.abs(knn_cmaq - y_cmaq[i]) + 1.0
        w_cmaq = 1.0 / cmaq_diff
        w_cmaq = w_cmaq / (w_cmaq.sum() + 1e-10)
        features[i, 6] = np.sum(w_cmaq * knn_residual)
        features[i, 7] = np.sum(w_cmaq * knn_cmaq)
        features[i, 8] = np.mean(np.abs(knn_residual))
        features[i, 9] = np.sum(w_cmaq * np.abs(knn_residual))
        features[i, 10] = np.max(np.abs(knn_residual))
        features[i, 11] = np.sum(w_cmaq * np.sign(knn_residual))

        # --- 头2: 风场传输注意力 ---
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

        # --- 头3: 综合注意力 ---
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
        lon_train = X_train[:, 0]
        lat_train = X_train[:, 1]
        self.lon_train = lon_train
        self.lat_train = lat_train
        self.n_train = len(lon_train)

        # 残差
        residual = y_train - y_cmaq_train
        self.residual_bias = np.mean(residual)
        self.residual_std = np.std(residual) + 1e-6

        # 构建图注意力特征(使用残差信息)
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

        # 存储训练残差用于备选
        self.residual_train = residual

        return self

    def predict(self, X_query, y_cmaq_query):
        lon_query = X_query[:, 0]
        lat_query = X_query[:, 1]
        n_query = len(lon_query)

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


class PGSTGATWrapper:
    def __init__(self, **kwargs):
        self.model = PGSTGATModel(**kwargs)

    def fit(self, X_train, y_train, y_model_train):
        self.model.fit(X_train, y_train, y_model_train)
        return self

    def predict(self, X_query, y_model_query):
        return self.model.predict(X_query, y_model_query)


def run_pg_stgat_ten_fold(selected_day='2020-01-01', wind_dir=180.0):
    print("="*60)
    print(f"PG-STGAT Ten-Fold CV (wind_dir={wind_dir})")
    print("="*60)

    print("\n=== Loading Data ===")
    monitor_df = pd.read_csv(monitor_file)
    fold_df = pd.read_csv(fold_file)

    day_df = monitor_df[monitor_df['Date'] == selected_day].copy()
    day_df = day_df.merge(fold_df, on='Site', how='left')
    day_df = day_df.dropna(subset=['Lat', 'Lon', 'Conc'])

    ds = nc.Dataset(cmaq_file, 'r')
    lon_cmaq = ds.variables['lon'][:]
    lat_cmaq = ds.variables['lat'][:]
    pred_pm25 = ds.variables['pred_PM25'][:]
    ds.close()

    from datetime import datetime
    date_obj = datetime.strptime(selected_day, '%Y-%m-%d')
    day_idx = (date_obj - datetime(2020, 1, 1)).days
    pred_day = pred_pm25[day_idx]

    cmaq_values = []
    for _, row in day_df.iterrows():
        val = get_cmaq_at_site(row['Lon'], row['Lat'], lon_cmaq, lat_cmaq, pred_day)
        cmaq_values.append(val)
    day_df['CMAQ'] = cmaq_values

    print(f"Data loaded: {len(day_df)} records")

    params = {'k_neighbors': 20, 'wind_dir': wind_dir}

    print("\n=== Ten-Fold Cross Validation ===")
    results = {fold_id: {} for fold_id in range(1, 11)}

    for fold_id in range(1, 11):
        train_df = day_df[day_df['fold'] != fold_id].copy()
        test_df = day_df[day_df['fold'] == fold_id].copy()

        train_df = train_df.dropna(subset=['Lon', 'Lat', 'CMAQ', 'Conc'])
        test_df = test_df.dropna(subset=['Lon', 'Lat', 'CMAQ', 'Conc'])

        if len(test_df) == 0:
            continue

        X_train = train_df[['Lon', 'Lat']].values
        y_train = train_df['Conc'].values
        y_cmaq_train = train_df['CMAQ'].values

        X_test = test_df[['Lon', 'Lat']].values
        y_test = test_df['Conc'].values
        y_cmaq_test = test_df['CMAQ'].values

        model = PGSTGATModel(**params)
        model.fit(X_train, y_train, y_cmaq_train)
        y_pred = model.predict(X_test, y_cmaq_test)

        results[fold_id] = {'y_true': y_test, 'y_pred': y_pred}
        metrics = compute_metrics(y_test, y_pred)
        print(f"  Fold {fold_id}: R2={metrics['R2']:.4f}, RMSE={metrics['RMSE']:.2f}, MB={metrics['MB']:.2f}")

    y_true_all = np.concatenate([results[f]['y_true'] for f in range(1, 11) if results[f]])
    y_pred_all = np.concatenate([results[f]['y_pred'] for f in range(1, 11) if results[f]])
    final_metrics = compute_metrics(y_true_all, y_pred_all)

    print("\n" + "="*60)
    print("Final PG-STGAT Results")
    print("="*60)
    print(f"R2: {final_metrics['R2']:.4f}")
    print(f"MAE: {final_metrics['MAE']:.2f}")
    print(f"RMSE: {final_metrics['RMSE']:.2f}")
    print(f"MB: {final_metrics['MB']:.2f}")

    result_df = pd.DataFrame([{'method': 'PG-STGAT', **params, **final_metrics}])
    result_df.to_csv(f'{output_dir}/PG_STGAT_summary.csv', index=False)

    return final_metrics


if __name__ == '__main__':
    # 使用优化后的参数: k=20, wind_dir=180
    metrics = run_pg_stgat_ten_fold('2020-01-01', wind_dir=180.0)
    print(f"\nFinal: R2={metrics['R2']:.4f}, RMSE={metrics['RMSE']:.2f}, MAE={metrics['MAE']:.2f}, MB={metrics['MB']:.2f}")
