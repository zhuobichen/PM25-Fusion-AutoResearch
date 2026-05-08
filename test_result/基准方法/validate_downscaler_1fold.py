# -*- coding: utf-8 -*-
"""
Downscaler 单日单折快速验证
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from shared.paths import get_project_root, data_path
import json
import numpy as np
import pandas as pd
import netCDF4 as nc
from datetime import datetime
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from Code.Downscaler.pm25_downscaler import PM25Downscaler
from Code.Downscaler.common_setting import CommonSetting

ROOT_DIR = str(get_project_root())
CMAQ_FILE = data_path('test_data/raw/CMAQ/2020_PM25.nc')
MONITOR_FILE = data_path('test_data/raw/Monitor/2020_DailyPM2.5Monitor.csv')
FOLD_FILE = data_path('test_data/fold_split_table_daily.csv')

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

print("Loading data...")
monitor_df = pd.read_csv(MONITOR_FILE)
fold_df = pd.read_csv(FOLD_FILE)
selected_day = '2020-01-01'

day_df = monitor_df[monitor_df['Date'] == selected_day].copy()
day_df = day_df.merge(fold_df, on='Site', how='left')
day_df = day_df.dropna(subset=['Lat', 'Lon', 'Conc'])

print(f"Sites: {len(day_df)}")

ds = nc.Dataset(CMAQ_FILE, 'r')
lon_cmaq = ds.variables['lon'][:]
lat_cmaq = ds.variables['lat'][:]
pred_pm25 = ds.variables['pred_PM25'][:]
ds.close()

cmaq_day = pred_pm25[0]

lon_flat = lon_cmaq.flatten()
lat_flat = lat_cmaq.flatten()
cmaq_flat = cmaq_day.flatten()

valid_mask = ~np.isnan(cmaq_flat)
buffer = 15.0
lon_min, lon_max = day_df['Lon'].min() - buffer, day_df['Lon'].max() + buffer
lat_min, lat_max = day_df['Lat'].min() - buffer, day_df['Lat'].max() + buffer

spatial_mask = valid_mask & (
    (lon_flat >= lon_min) & (lon_flat <= lon_max) &
    (lat_flat >= lat_min) & (lat_flat <= lat_max)
)

matrix_latlon_model = np.column_stack([lon_flat[spatial_mask], lat_flat[spatial_mask]])
matrix_model = cmaq_flat[spatial_mask].reshape(-1, 1)

print(f"Model grid points: {len(matrix_latlon_model)}")

# 只用 fold 1
fold_id = 1
train_df = day_df[day_df['fold'] != fold_id].copy()
test_df = day_df[day_df['fold'] == fold_id].copy()

train_df = train_df.dropna(subset=['Lon', 'Lat', 'Conc'])
test_df = test_df.dropna(subset=['Lon', 'Lat', 'Conc'])

print(f"Train: {len(train_df)}, Test: {len(test_df)}")

matrix_latlon_monitor = train_df[['Lon', 'Lat']].values
matrix_monitor = train_df['Conc'].values.reshape(-1, 1)

setting = CommonSetting()
setting.numit = 100
setting.burn = 50

print(f"Running Downscaler (numit={setting.numit}, burn={setting.burn})...")
import time
t0 = time.time()

downscaler = PM25Downscaler(setting)
result = downscaler.run(
    matrix_latlon_model.astype(float),
    matrix_latlon_monitor.astype(float),
    matrix_model.astype(float),
    matrix_monitor.astype(float),
    seed=42
)

elapsed = time.time() - t0
print(f"Done in {elapsed:.1f}s")

if result is None:
    print(f"FAILED: {downscaler.error_msg}")
else:
    pred_grid, _ = result

    test_lon = test_df['Lon'].values
    test_lat = test_df['Lat'].values
    y_true = test_df['Conc'].values

    y_pred = []
    for lon, lat in zip(test_lon, test_lat):
        dist = np.sqrt((matrix_latlon_model[:, 0] - lon)**2 +
                      (matrix_latlon_model[:, 1] - lat)**2)
        nearest_idx = np.argmin(dist)
        y_pred.append(pred_grid[nearest_idx])

    y_pred = np.array(y_pred)

    metrics = compute_metrics(y_true, y_pred)
    print(f"Single fold result: R2={metrics['R2']:.4f}, RMSE={metrics['RMSE']:.2f}, MAE={metrics['MAE']:.2f}, MB={metrics['MB']:.2f}")

    # 保存
    OUTPUT_FILE = f'{ROOT_DIR}/test_result/基准方法/downscaler_1fold.json'
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"Saved to {OUTPUT_FILE}")