"""
PM2.5 CMAQ融合方法 - 快速多阶段验证
====================================
验证流程: 预实验(5天) -> 1月整月验证
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
import json
from datetime import datetime, timedelta

root_dir = str(get_project_root())
cmaq_file = data_path('test_data/raw/CMAQ/2020_PM25.nc')
monitor_file = data_path('test_data/raw/Monitor/2020_DailyPM2.5Monitor.csv')
fold_file = data_path('test_data/fold_split_table_daily.csv')
output_dir = f'{root_dir}/test_result/创新方法'
os.makedirs(output_dir, exist_ok=True)

BENCHMARK_R2 = 0.8200
BENCHMARK_RMSE = 12.52
BENCHMARK_MB = 0.08

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

        sigma_space = 1.0
        w_space = np.exp(-knn_dist / sigma_space)
        w_space = w_space / (w_space.sum() + 1e-10)
        features[i, 0] = np.sum(w_space * knn_residual)
        features[i, 1] = np.sum(w_space * knn_cmaq)
        features[i, 2] = np.mean(knn_dist)
        features[i, 3] = np.var(knn_residual)
        features[i, 4] = np.max(knn_dist)
        features[i, 5] = np.sum(w_space > 0.1)

        cmaq_diff = np.abs(knn_cmaq - y_cmaq[i]) + 1.0
        w_cmaq = 1.0 / cmaq_diff
        w_cmaq = w_cmaq / (w_cmaq.sum() + 1e-10)
        features[i, 6] = np.sum(w_cmaq * knn_residual)
        features[i, 7] = np.sum(w_cmaq * knn_cmaq)
        features[i, 8] = np.mean(np.abs(knn_residual))
        features[i, 9] = np.sum(w_cmaq * np.abs(knn_residual))
        features[i, 10] = np.max(np.abs(knn_residual))
        features[i, 11] = np.sum(w_cmaq * np.sign(knn_residual))

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

        w_combined = (w_space + w_cmaq + w_wind) / 3.0
        w_combined = w_combined / (w_combined.sum() + 1e-10)
        features[i, 18] = np.sum(w_combined * knn_residual)
        features[i, 19] = np.sum(w_combined * knn_cmaq)
        features[i, 20] = np.mean(knn_residual)
        features[i, 21] = np.sum(w_combined * np.abs(knn_residual))
        features[i, 22] = np.std(knn_residual)
        features[i, 23] = np.sum(w_space * np.abs(knn_residual))

    return features

def compute_variogram_binned(lon, lat, values, n_bins=8, max_dist=None):
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
    n = len(lon)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        dist_matrix[i, :] = haversine_distance(lon[i], lat[i], lon, lat)

    features = np.zeros((n, 18))

    for i in range(n):
        distances = dist_matrix[i, :]
        sorted_idx = np.argsort(distances)

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


class PGSTGATModel:
    def __init__(self, k_neighbors=20, wind_dir=180.0):
        self.k_neighbors = k_neighbors
        self.wind_dir = wind_dir

    def fit(self, X_train, y_train, y_cmaq_train):
        lon_train = X_train[:, 0]
        lat_train = X_train[:, 1]
        self.lon_train = lon_train
        self.lat_train = lat_train
        self.n_train = len(lon_train)

        residual = y_train - y_cmaq_train
        self.residual_bias = np.mean(residual)
        self.residual_std = np.std(residual) + 1e-6

        self.gat_features = build_station_graph_features(
            lon_train, lat_train, y_cmaq_train, residual,
            k_neighbors=self.k_neighbors, wind_dir=self.wind_dir
        )

        self.scaler = StandardScaler()
        gat_scaled = self.scaler.fit_transform(self.gat_features)

        residual_norm = (residual - self.residual_bias) / self.residual_std
        self.model = Ridge(alpha=1.0)
        self.model.fit(gat_scaled, residual_norm)

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


class VCFFMModel:
    def __init__(self, latent_dim=32, scale_levels=3, k_neighbors=20):
        self.latent_dim = latent_dim
        self.scale_levels = scale_levels
        self.k_neighbors = k_neighbors

    def fit(self, X_train, y_train, y_cmaq_train):
        lon_train = X_train[:, 0]
        lat_train = X_train[:, 1]
        self.lon_train = lon_train
        self.lat_train = lat_train
        self.n_train = len(lon_train)

        residual = y_train - y_cmaq_train
        self.residual_bias = np.mean(residual)
        self.residual_std = np.std(residual) + 1e-6

        h, gamma, counts = compute_variogram_binned(lon_train, lat_train, residual, n_bins=8)
        if len(h) >= 3:
            vario_params = fit_exponential_variogram(h, gamma)
        else:
            vario_params = {'nugget': 0, 'sill': self.residual_std**2, 'range': 50.0}

        self.vario_params = vario_params

        self.ms_features = build_multiscale_features(
            lon_train, lat_train, y_cmaq_train, residual,
            k_neighbors=self.k_neighbors
        )

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

        combined_features = np.hstack([self.ms_features, z_mean])
        self.scaler = StandardScaler()
        combined_scaled = self.scaler.fit_transform(combined_features)

        residual_norm = (residual - self.residual_bias) / self.residual_std
        self.decoder = Ridge(alpha=1.0)
        self.decoder.fit(combined_scaled, residual_norm)

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

        ms_query = np.zeros((n_query, 18))
        for i in range(n_query):
            w_norm = 1.0 / (dist_k[i] + 1e-6)
            w_norm = w_norm / w_norm.sum()
            ms_query[i] = w_norm @ self.ms_features[idx_k[i]]

        z_query = np.zeros((n_query, self.latent_dim))
        for i in range(n_query):
            w_norm = 1.0 / (dist_k[i] + 1e-6)
            w_norm = w_norm / w_norm.sum()
            z_query[i] = w_norm @ self.z_mean_train[idx_k[i]]

        combined_query = np.hstack([ms_query, z_query])
        combined_scaled = self.scaler.transform(combined_query)

        residual_pred_norm = self.decoder.predict(combined_scaled)
        residual_pred = residual_pred_norm * self.residual_std + self.residual_bias

        y_pred = y_cmaq_query + residual_pred
        return y_pred


def run_ten_fold_for_day(model_class, selected_day, **model_params):
    """执行单日十折交叉验证"""
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

    date_obj = datetime.strptime(selected_day, '%Y-%m-%d')
    day_idx = (date_obj - datetime(2020, 1, 1)).days
    pred_day = pred_pm25[day_idx]

    cmaq_values = []
    for _, row in day_df.iterrows():
        val = get_cmaq_at_site(row['Lon'], row['Lat'], lon_cmaq, lat_cmaq, pred_day)
        cmaq_values.append(val)
    day_df['CMAQ'] = cmaq_values

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

        model = model_class(**model_params)
        model.fit(X_train, y_train, y_cmaq_train)
        y_pred = model.predict(X_test, y_cmaq_test)

        results[fold_id] = {'y_true': y_test, 'y_pred': y_pred}

    y_true_all = np.concatenate([results[f]['y_true'] for f in range(1, 11) if results[f]])
    y_pred_all = np.concatenate([results[f]['y_pred'] for f in range(1, 11) if results[f]])
    final_metrics = compute_metrics(y_true_all, y_pred_all)

    return final_metrics


def run_month(model_class, year_month, **model_params):
    """运行整月验证"""
    year, month = year_month.split('-')
    start_date = datetime(int(year), int(month), 1)
    if int(month) == 12:
        end_date = datetime(int(year) + 1, 1, 1) - timedelta(days=1)
    else:
        end_date = datetime(int(year), int(month) + 1, 1) - timedelta(days=1)

    all_y_true = []
    all_y_pred = []

    current_date = start_date
    while current_date <= end_date:
        date_str = current_date.strftime('%Y-%m-%d')
        try:
            metrics = run_ten_fold_for_day(model_class, date_str, **model_params)
            if not np.isnan(metrics['R2']):
                # 获取详细数据
                monitor_df = pd.read_csv(monitor_file)
                fold_df = pd.read_csv(fold_file)

                day_df = monitor_df[monitor_df['Date'] == date_str].copy()
                day_df = day_df.merge(fold_df, on='Site', how='left')
                day_df = day_df.dropna(subset=['Lat', 'Lon', 'Conc'])

                ds = nc.Dataset(cmaq_file, 'r')
                lon_cmaq = ds.variables['lon'][:]
                lat_cmaq = ds.variables['lat'][:]
                pred_pm25 = ds.variables['pred_PM25'][:]
                ds.close()

                date_obj = datetime.strptime(date_str, '%Y-%m-%d')
                day_idx = (date_obj - datetime(2020, 1, 1)).days
                pred_day = pred_pm25[day_idx]

                cmaq_values = []
                for _, row in day_df.iterrows():
                    val = get_cmaq_at_site(row['Lon'], row['Lat'], lon_cmaq, lat_cmaq, pred_day)
                    cmaq_values.append(val)
                day_df['CMAQ'] = cmaq_values

                for fold_id in range(1, 11):
                    test_df = day_df[day_df['fold'] == fold_id].copy()
                    test_df = test_df.dropna(subset=['Lon', 'Lat', 'CMAQ', 'Conc'])

                    if len(test_df) == 0:
                        continue

                    train_df = day_df[day_df['fold'] != fold_id].copy()
                    train_df = train_df.dropna(subset=['Lon', 'Lat', 'CMAQ', 'Conc'])

                    if len(train_df) == 0:
                        continue

                    X_train = train_df[['Lon', 'Lat']].values
                    y_train = train_df['Conc'].values
                    y_cmaq_train = train_df['CMAQ'].values

                    X_test = test_df[['Lon', 'Lat']].values
                    y_test = test_df['Conc'].values
                    y_cmaq_test = test_df['CMAQ'].values

                    model = model_class(**model_params)
                    model.fit(X_train, y_train, y_cmaq_train)
                    y_pred = model.predict(X_test, y_cmaq_test)

                    all_y_true.extend(y_test)
                    all_y_pred.extend(y_pred)

        except Exception as e:
            print(f"        Warning: Failed for {date_str}: {e}")

        current_date += timedelta(days=1)

    if len(all_y_true) == 0:
        return {'R2': np.nan, 'MAE': np.nan, 'RMSE': np.nan, 'MB': np.nan}

    return compute_metrics(np.array(all_y_true), np.array(all_y_pred))


def main():
    print("="*70)
    print("PM2.5 CMAQ Fusion Methods - Multi-Stage Validation")
    print("="*70)
    print(f"\nBaseline: R2>={BENCHMARK_R2}, RMSE<={BENCHMARK_RMSE}, |MB|<={BENCHMARK_MB}")

    pg_stgat_params = {'k_neighbors': 20, 'wind_dir': 180.0}
    vcffm_params = {'latent_dim': 32, 'scale_levels': 3, 'k_neighbors': 20}

    results = {}

    # ============================================================
    # Stage 1: Pre-experiment (5 days: 2020-01-01 ~ 2020-01-05)
    # ============================================================
    print("\n" + "="*70)
    print("Stage 1: Pre-experiment (5 days: 2020-01-01 ~ 2020-01-05)")
    print("="*70)

    pre_exp_days = ['2020-01-01', '2020-01-02', '2020-01-03', '2020-01-04', '2020-01-05']

    for method_name, model_class, params in [
        ('PG-STGAT', PGSTGATModel, pg_stgat_params),
        ('VCFFM', VCFFMModel, vcffm_params)
    ]:
        print(f"\n--- {method_name} ---")
        daily_results = []

        for day in pre_exp_days:
            metrics = run_ten_fold_for_day(model_class, day, **params)
            daily_results.append(metrics)
            print(f"  {day}: R2={metrics['R2']:.4f}, RMSE={metrics['RMSE']:.2f}, MB={metrics['MB']:.2f}")

        avg_r2 = np.mean([r['R2'] for r in daily_results if not np.isnan(r['R2'])])
        avg_rmse = np.mean([r['RMSE'] for r in daily_results if not np.isnan(r['RMSE'])])
        avg_mb = np.mean([r['MB'] for r in daily_results if not np.isnan(r['MB'])])

        print(f"\n  {method_name} Pre-exp Avg: R2={avg_r2:.4f}, RMSE={avg_rmse:.2f}, MB={avg_mb:.2f}")

        stage_pass = avg_r2 >= BENCHMARK_R2
        print(f"  Pass: {stage_pass}")

        results[f'{method_name}_stage1'] = {
            'stage': 'Pre-exp (5 days)',
            'daily_results': [{k: float(v) for k, v in r.items()} for r in daily_results],
            'avg_r2': float(avg_r2),
            'avg_rmse': float(avg_rmse),
            'avg_mb': float(avg_mb),
            'pass': bool(stage_pass)
        }

    # Check if pre-experiment passed
    stage1_pass = all(results[f'{m}_stage1']['pass'] for m in ['PG-STGAT', 'VCFFM'])

    if not stage1_pass:
        print("\n*** Pre-experiment FAILED. Innovation validation terminates. ***")
        return results

    print("\n*** Pre-experiment PASSED. Continuing to Stage 2 (January)... ***")

    # ============================================================
    # Stage 2: January Full Month
    # ============================================================
    print("\n" + "="*70)
    print("Stage 2: January 2020 Full Month")
    print("="*70)

    for method_name, model_class, params in [
        ('PG-STGAT', PGSTGATModel, pg_stgat_params),
        ('VCFFM', VCFFMModel, vcffm_params)
    ]:
        print(f"\n--- {method_name} ---")
        print(f"  Running 31 days...")
        metrics = run_month(model_class, '2020-01', **params)
        print(f"  January: R2={metrics['R2']:.4f}, RMSE={metrics['RMSE']:.2f}, MB={metrics['MB']:.2f}")

        stage_pass = metrics['R2'] >= BENCHMARK_R2
        print(f"  Pass: {stage_pass}")

        results[f'{method_name}_stage2'] = {
            'stage': 'January 2020',
            'R2': float(metrics['R2']),
            'MAE': float(metrics['MAE']),
            'RMSE': float(metrics['RMSE']),
            'MB': float(metrics['MB']),
            'pass': bool(stage_pass)
        }

    # Check if Stage 2 passed
    stage2_pass = all(results[f'{m}_stage2']['pass'] for m in ['PG-STGAT', 'VCFFM'])

    if not stage2_pass:
        print("\n*** January validation FAILED. ***")
    else:
        print("\n*** January validation PASSED. ***")

    # ============================================================
    # Stage 3: July Full Month
    # ============================================================
    print("\n" + "="*70)
    print("Stage 3: July 2020 Full Month")
    print("="*70)

    for method_name, model_class, params in [
        ('PG-STGAT', PGSTGATModel, pg_stgat_params),
        ('VCFFM', VCFFMModel, vcffm_params)
    ]:
        print(f"\n--- {method_name} ---")
        print(f"  Running 31 days...")
        metrics = run_month(model_class, '2020-07', **params)
        print(f"  July: R2={metrics['R2']:.4f}, RMSE={metrics['RMSE']:.2f}, MB={metrics['MB']:.2f}")

        stage_pass = metrics['R2'] >= BENCHMARK_R2
        print(f"  Pass: {stage_pass}")

        results[f'{method_name}_stage3'] = {
            'stage': 'July 2020',
            'R2': float(metrics['R2']),
            'MAE': float(metrics['MAE']),
            'RMSE': float(metrics['RMSE']),
            'MB': float(metrics['MB']),
            'pass': bool(stage_pass)
        }

    stage3_pass = all(results[f'{m}_stage3']['pass'] for m in ['PG-STGAT', 'VCFFM'])

    if not stage3_pass:
        print("\n*** July validation FAILED. ***")
    else:
        print("\n*** July validation PASSED. ***")

    # ============================================================
    # Stage 4: December Full Month
    # ============================================================
    print("\n" + "="*70)
    print("Stage 4: December 2020 Full Month")
    print("="*70)

    for method_name, model_class, params in [
        ('PG-STGAT', PGSTGATModel, pg_stgat_params),
        ('VCFFM', VCFFMModel, vcffm_params)
    ]:
        print(f"\n--- {method_name} ---")
        print(f"  Running 31 days...")
        metrics = run_month(model_class, '2020-12', **params)
        print(f"  December: R2={metrics['R2']:.4f}, RMSE={metrics['RMSE']:.2f}, MB={metrics['MB']:.2f}")

        stage_pass = metrics['R2'] >= BENCHMARK_R2
        print(f"  Pass: {stage_pass}")

        results[f'{method_name}_stage4'] = {
            'stage': 'December 2020',
            'R2': float(metrics['R2']),
            'MAE': float(metrics['MAE']),
            'RMSE': float(metrics['RMSE']),
            'MB': float(metrics['MB']),
            'pass': bool(stage_pass)
        }

    stage4_pass = all(results[f'{m}_stage4']['pass'] for m in ['PG-STGAT', 'VCFFM'])

    # ============================================================
    # Final Summary
    # ============================================================
    print("\n" + "="*70)
    print("MULTI-STAGE VALIDATION COMPLETE - FINAL SUMMARY")
    print("="*70)

    for method_name in ['PG-STGAT', 'VCFFM']:
        print(f"\n{method_name}:")
        for stage_id in [1, 2, 3, 4]:
            stage_key = f'{method_name}_stage{stage_id}'
            if stage_key in results:
                r = results[stage_key]
                status = "PASS" if r['pass'] else "FAIL"
                r2_val = r.get('R2', r.get('avg_r2', 'N/A'))
                print(f"  Stage {stage_id}: R2={r2_val:.4f} [{status}]")

    innovation_pass = stage1_pass and stage2_pass and stage3_pass and stage4_pass

    print(f"\n{'='*70}")
    print(f"Innovation Validated: {'YES' if innovation_pass else 'NO'}")
    print(f"{'='*70}")

    # Save results
    with open(f'{output_dir}/multi_stage_validation_summary.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Save CSV summary
    summary_data = []
    for method_name in ['PG-STGAT', 'VCFFM']:
        for stage_id in [1, 2, 3, 4]:
            stage_key = f'{method_name}_stage{stage_id}'
            if stage_key in results:
                r = results[stage_key]
                r2_val = r.get('R2', r.get('avg_r2', np.nan))
                rmse_val = r.get('RMSE', r.get('avg_rmse', np.nan))
                mb_val = r.get('MB', r.get('avg_mb', np.nan))
                summary_data.append({
                    'method': method_name,
                    'stage': stage_id,
                    'R2': float(r2_val) if not pd.isna(r2_val) else None,
                    'RMSE': float(rmse_val) if not pd.isna(rmse_val) else None,
                    'MB': float(mb_val) if not pd.isna(mb_val) else None,
                    'pass': bool(r['pass'])
                })

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(f'{output_dir}/summary.csv', index=False)

    print(f"\nResults saved to {output_dir}/")

    return results


if __name__ == '__main__':
    results = main()