# -*- coding: utf-8 -*-
"""
VCFFM 十折交叉验证 - 标准模式
=============================
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
OUTPUT_FILE = f'{ROOT_DIR}/Innovation/success/VCFFM/VCFFM_all_stages.json'

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


def compute_variogram_binned(lon, lat, values, n_bins=8, max_dist=None):
    """计算分箱变异函数"""
    n = len(values)
    distances = []
    semivariances = []

    for i in range(n):
        for j in range(i+1, min(n, i+50)):
            d = haversine_distance(lon[i], lat[i], lon[j], lat[j])
            gamma = 0.5 * (values[i] - values[j])**2
            distances.append(d)
            semivariances.append(gamma)

    distances = np.array(distances)
    semivariances = np.array(semivariances)

    if max_dist is None:
        max_dist = np.percentile(distances, 80)

    bin_edges = np.linspace(0, max_dist, n_bins + 1)
    gamma_bins = []
    h_bins = []
    count_bins = []

    for k in range(n_bins):
        mask = (distances >= bin_edges[k]) & (distances < bin_edges[k+1])
        if mask.sum() > 10:
            gamma_bins.append(semivariances[mask].mean())
            h_bins.append((bin_edges[k] + bin_edges[k+1]) / 2)
            count_bins.append(mask.sum())

    return np.array(h_bins), np.array(gamma_bins), np.array(count_bins)


def fit_exponential_variogram(h, gamma, nugget=0):
    """拟合指数变异函数"""
    if len(h) < 3:
        return {'nugget': 0, 'sill': np.var(gamma), 'range': 100.0}

    sill = np.max(gamma) - nugget
    range_est = h[np.argmax(gamma > 0.9 * (nugget + sill))] if any(gamma > 0.9 * (nugget + sill)) else h[-1]

    def objective(params):
        n, s, r = params
        gamma_pred = n + s * (1 - np.exp(-h / (r + 1e-6)))
        return np.sum((gamma - gamma_pred)**2)

    from scipy.optimize import minimize
    result = minimize(objective, [nugget, sill, range_est],
                     bounds=[(0, np.max(gamma)), (0, 2*np.max(gamma)), (1, 500)])
    return {'nugget': result.x[0], 'sill': result.x[1], 'range': result.x[2]}


def build_multiscale_features(lon, lat, y_cmaq, residual, k_neighbors=10):
    """构建多尺度特征"""
    n = len(lon)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        dist_matrix[i, :] = haversine_distance(lon[i], lat[i], lon, lat)

    features = np.zeros((n, 18))

    for i in range(n):
        distances = dist_matrix[i, :]
        sorted_idx = np.argsort(distances)

        # 大尺度(k=5, 宽核)
        k_large = min(5, n-1)
        knn_idx = sorted_idx[1:k_large+1]
        knn_dist = distances[knn_idx]
        knn_res = residual[knn_idx]
        knn_cmaq = y_cmaq[knn_idx]

        sigma_large = 5.0
        w_large = np.exp(-knn_dist / sigma_large)
        w_large = w_large / (w_large.sum() + 1e-10)
        features[i, 0] = np.sum(w_large * knn_res)
        features[i, 1] = np.sum(w_large * knn_cmaq)
        features[i, 2] = np.mean(knn_res)
        features[i, 3] = np.std(knn_res)
        features[i, 4] = np.mean(knn_dist)
        features[i, 5] = np.sum(w_large * np.abs(knn_res))

        # 中尺度(k=10, 中核)
        k_mid = min(10, n-1)
        knn_idx = sorted_idx[1:k_mid+1]
        knn_dist = distances[knn_idx]
        knn_res = residual[knn_idx]
        knn_cmaq = y_cmaq[knn_idx]

        sigma_mid = 2.0
        w_mid = np.exp(-knn_dist / sigma_mid)
        w_mid = w_mid / (w_mid.sum() + 1e-10)
        features[i, 6] = np.sum(w_mid * knn_res)
        features[i, 7] = np.sum(w_mid * knn_cmaq)
        features[i, 8] = np.mean(knn_res)
        features[i, 9] = np.std(knn_res)
        features[i, 10] = np.mean(knn_dist)
        features[i, 11] = np.sum(w_mid * np.abs(knn_res))

        # 小尺度(k=20, 窄核)
        k_small = min(20, n-1)
        knn_idx = sorted_idx[1:k_small+1]
        knn_dist = distances[knn_idx]
        knn_res = residual[knn_idx]
        knn_cmaq = y_cmaq[knn_idx]

        sigma_small = 0.5
        w_small = np.exp(-knn_dist / sigma_small)
        w_small = w_small / (w_small.sum() + 1e-10)
        features[i, 12] = np.sum(w_small * knn_res)
        features[i, 13] = np.sum(w_small * knn_cmaq)
        features[i, 14] = np.mean(knn_res)
        features[i, 15] = np.std(knn_res)
        features[i, 16] = np.mean(knn_dist)
        features[i, 17] = np.sum(w_small * np.abs(knn_res))

    return features


class VCFFMModel:
    """变分协方差场融合模型"""

    def __init__(self, latent_dim=32, scale_levels=3, k_neighbors=20):
        self.latent_dim = latent_dim
        self.scale_levels = scale_levels
        self.k_neighbors = k_neighbors

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

        # 协方差场建模
        h, gamma, counts = compute_variogram_binned(lon_train, lat_train, residual, n_bins=8)
        if len(h) >= 3:
            vario_params = fit_exponential_variogram(h, gamma)
        else:
            vario_params = {'nugget': 0, 'sill': self.residual_std**2, 'range': 50.0}

        self.vario_params = vario_params

        # 多尺度特征
        self.ms_features = build_multiscale_features(
            lon_train, lat_train, y_cmaq_train, residual,
            k_neighbors=self.k_neighbors
        )

        # 潜在表示
        cmaq_features = np.column_stack([
            lon_train, lat_train,
            y_cmaq_train,
            y_cmaq_train**2,
            y_cmaq_train**3
        ])

        np.random.seed(42)
        self.W_enc = np.random.randn(cmaq_features.shape[1], self.latent_dim) * 0.01
        self.b_enc = np.zeros(self.latent_dim)
        z_mean = cmaq_features @ self.W_enc + self.b_enc
        z_log_var = np.log(np.var(z_mean, axis=0) + 1e-6)
        self.z_mean_train = z_mean
        self.z_log_var = z_log_var

        # 融合解码器
        combined_features = np.hstack([self.ms_features, z_mean])
        self.scaler = StandardScaler()
        combined_scaled = self.scaler.fit_transform(combined_features)

        residual_norm = (residual - self.residual_bias) / self.residual_std
        self.decoder = Ridge(alpha=1.0)
        self.decoder.fit(combined_scaled, residual_norm)

        return self

    def predict(self, X_query, y_cmaq_query):
        """预测：X_query是CMAQ网格坐标"""
        lon_query = X_query[:, 0]
        lat_query = X_query[:, 1]
        n_query = len(lon_query)

        # 使用训练时的监测站坐标来构建KNN
        train_coords = np.column_stack([self.lon_train, self.lat_train])
        query_coords = np.column_stack([lon_query, lat_query])

        dist = cdist(query_coords, train_coords)
        k = min(self.k_neighbors, self.n_train)
        knn = NearestNeighbors(n_neighbors=k)
        knn.fit(train_coords)
        dist_k, idx_k = knn.kneighbors(query_coords)

        # 多尺度特征IDW插值
        ms_query = np.zeros((n_query, 18))
        for i in range(n_query):
            w_norm = 1.0 / (dist_k[i] + 1e-6)
            w_norm = w_norm / w_norm.sum()
            ms_query[i] = w_norm @ self.ms_features[idx_k[i]]

        # 潜在表示IDW插值
        z_query = np.zeros((n_query, self.latent_dim))
        for i in range(n_query):
            w_norm = 1.0 / (dist_k[i] + 1e-6)
            w_norm = w_norm / w_norm.sum()
            z_query[i] = w_norm @ self.z_mean_train[idx_k[i]]

        # 组合特征
        combined_query = np.hstack([ms_query, z_query])
        combined_scaled = self.scaler.transform(combined_query)

        # 预测
        residual_pred_norm = self.decoder.predict(combined_scaled)
        residual_pred = residual_pred_norm * self.residual_std + self.residual_bias

        y_pred = y_cmaq_query + residual_pred

        return y_pred


def ten_fold_vcffm(selected_day):
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

    params = {'latent_dim': 32, 'scale_levels': 3, 'k_neighbors': 20}

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

        model = VCFFMModel(**params)
        model.fit(X_train, y_train, y_cmaq_train)
        y_pred = model.predict(X_test, y_cmaq_test)

        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)

    return np.array(all_y_true), np.array(all_y_pred)


def run_stage_validation(stage_name, start_date, end_date):
    sep = "=" * 70
    print(sep)
    print(f"VCFFM Stage: {stage_name} ({start_date} ~ {end_date})")
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
        delayed(ten_fold_vcffm)(date_str)
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
    print("VCFFM All Stages - 标准模式")
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
