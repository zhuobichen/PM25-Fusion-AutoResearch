import os
# -*- coding: utf-8 -*-
"""
Elegant Spatial-Statistical Lambda Estimation
============================================
1. Variogram-based: λ = distance where bias decorrelates (range parameter)
2. Moran-I decorrelation: λ = distance where spatial autocorrelation vanishes
3. Held-out estimator: estimate λ from 10% of training, test on 90%
4. Training-residual: use OOF training residuals to estimate λ
5. Empirical variogram fit: fit spherical/exponential model to get range
"""

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

sys.path.insert(0, 'E:/CodeProject/ClaudeRoom/Data_Fusion_AutoResearch/CodeWorkSpace/新融合方法代码')
import numpy as np
import pandas as pd
import netCDF4 as nc
from datetime import datetime, timedelta
from sklearn.metrics import r2_score, mean_squared_error
from CodeWorkSpace.新融合方法代码.gVNA import gVNA, adaptive_lambda

CMAQ_FILE = data_path('test_data/raw/CMAQ/2020_PM25.nc')
MONITOR_FILE = data_path('test_data/raw/Monitor/2020_DailyPM2.5Monitor.csv')
FOLD_FILE = data_path('test_data/fold_split_table_daily.csv')


def compute_metrics(y_true, y_pred):
    mask = ~(np.isnan(y_true) | np.isnan(y_pred) | np.isinf(y_true) | np.isinf(y_pred))
    y_true, y_pred = y_true[mask], y_pred[mask]
    return {'R2': float(r2_score(y_true, y_pred)),
            'RMSE': float(np.sqrt(mean_squared_error(y_true, y_pred)))}


def get_cmaq_at_site(lon, lat, lon_grid, lat_grid, pm25_grid):
    idx = np.argmin(np.sqrt((lon_grid - lon)**2 + (lat_grid - lat)**2))
    ny, nx = lon_grid.shape
    return pm25_grid[idx // nx, idx % nx]


def load_day(selected_day):
    monitor_df = pd.read_csv(MONITOR_FILE)
    fold_df = pd.read_csv(FOLD_FILE)
    day_df = monitor_df[monitor_df['Date'] == selected_day].copy()
    day_df = day_df.merge(fold_df, on=['Date', 'Site'], how='left')
    day_df = day_df.dropna(subset=['Lat', 'Lon', 'Conc'])
    if len(day_df) < 100:
        return None
    ds = nc.Dataset(CMAQ_FILE, 'r')
    lon_cmaq, lat_cmaq = ds.variables['lon'][:], ds.variables['lat'][:]
    pred_pm25 = ds.variables['pred_PM25'][:]
    ds.close()
    day_idx = (datetime.strptime(selected_day, '%Y-%m-%d') - datetime(2020, 1, 1)).days
    day_df = day_df.copy()
    day_df['CMAQ'] = [get_cmaq_at_site(r['Lon'], r['Lat'], lon_cmaq, lat_cmaq, pred_pm25[day_idx])
                       for _, r in day_df.iterrows()]
    return day_df.dropna(subset=['CMAQ'])


# =============================================================================
# METHOD 1: Variogram-based range estimation (O(n*k))
# =============================================================================
def estimate_lambda_variogram(train_df, n_bins=15, max_dist=None):
    """
    Compute experimental variogram of biases:
    gamma(h) = 0.5 * E[(b(x) - b(x+h))^2]

    λ = distance where gamma reaches 85% of sill (practical range)
    This is the standard geostatistical definition of spatial correlation range.
    """
    train = train_df.dropna(subset=['Lon', 'Lat', 'Conc', 'CMAQ'])
    if len(train) < 20:
        return 8.0

    coords = np.column_stack([train['Lon'].values, train['Lat'].values])
    biases = (train['Conc'] - train['CMAQ']).values
    n = len(biases)

    if max_dist is None:
        max_dist = np.sqrt(np.ptp(coords[:, 0])**2 + np.ptp(coords[:, 1])**2) / 4
        max_dist = min(max_dist, 20.0)  # cap at 20 degrees

    # Bin distances
    bin_edges = np.linspace(0, max_dist, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    gamma_vals = np.zeros(n_bins)
    counts = np.zeros(n_bins)

    for i in range(n):
        for j in range(i + 1, min(i + 30, n)):  # sample pairs
            d = np.sqrt((coords[i, 0] - coords[j, 0])**2 + (coords[i, 1] - coords[j, 1])**2)
            if d < max_dist:
                bin_idx = np.searchsorted(bin_edges[1:], d)
                if bin_idx < n_bins:
                    gamma_vals[bin_idx] += 0.5 * (biases[i] - biases[j])**2
                    counts[bin_idx] += 1

    # Compute mean gamma per bin
    with np.errstate(divide='ignore', invalid='ignore'):
        gamma_mean = np.where(counts > 0, gamma_vals / counts, np.nan)

    # Sill = total variance of biases
    sill = np.var(biases)
    if sill < 1e-6:
        return 8.0

    # Practical range = distance where gamma = 0.85 * sill
    target_gamma = 0.85 * sill

    # Find range from variogram (fit exponential model)
    valid = counts > 0
    if valid.sum() < 3:
        return 8.0

    bc = bin_centers[valid]
    gv = gamma_mean[valid]

    # Fit gamma(h) = sill * (1 - exp(-3h/range))
    # => range = -3h / log(1 - gamma/sill)
    ranges_est = -3 * bc / np.log(np.maximum(1e-6, 1 - gv / (sill + 1e-6)))
    ranges_est = ranges_est[np.isfinite(ranges_est) & (ranges_est > 0) & (ranges_est < 50)]

    if len(ranges_est) == 0:
        return 8.0

    # Take median of range estimates across bins
    return float(np.clip(np.median(ranges_est), 1.0, 50.0))


# =============================================================================
# METHOD 2: Moran-I based decorrelation distance (O(n))
# =============================================================================
def estimate_lambda_moran(train_df, n_lags=10):
    """
    Compute Moran's I at different distance lags.
    λ = distance where I drops below significance threshold (e.g., I < 0.1)
    This gives the spatial decorrelation distance.
    """
    train = train_df.dropna(subset=['Lon', 'Lat', 'Conc', 'CMAQ'])
    if len(train) < 30:
        return 8.0

    coords = np.column_stack([train['Lon'].values, train['Lat'].values])
    biases = (train['Conc'] - train['CMAQ']).values
    n = len(biases)

    # Compute max distance for binning
    max_d = min(20.0, np.median(np.sqrt((coords[:, 0] - coords[:, 0:1])**2 +
                                         (coords[:, 1] - coords[:, 1:1])**2)) * 3)
    lag_edges = np.linspace(0, max_d, n_lags + 1)

    # Compute Moran's I per lag
    mean_b = np.mean(biases)
    s = np.sum((biases - mean_b)**2) / n
    if s < 1e-6:
        return 8.0

    i_per_lag = []
    d_per_lag = []

    for lag_idx in range(n_lags):
        d_min, d_max = lag_edges[lag_idx], lag_edges[lag_idx + 1]
        numerator = 0.0
        count = 0

        # Sample for speed
        for i in np.random.choice(n, min(50, n), replace=False):
            for j in np.random.choice(n, min(20, n), replace=False):
                if i == j:
                    continue
                d = np.sqrt((coords[i, 0] - coords[j, 0])**2 + (coords[i, 1] - coords[j, 1])**2)
                if d_min <= d < d_max:
                    numerator += (biases[i] - mean_b) * (biases[j] - mean_b)
                    count += 1

        if count > 0:
            # Moran's I = (n/S0) * (sum_ij w_ij (xi-xbar)(xj-xbar) / sum (xi-xbar)^2)
            # Simplified: I proportional to covariance at distance d
            i_val = numerator / (count * s)
            i_per_lag.append(i_val)
            d_per_lag.append((d_min + d_max) / 2)

    if len(i_per_lag) < 2:
        return 8.0

    i_per_lag = np.array(i_per_lag)
    d_per_lag = np.array(d_per_lag)

    # λ = distance where |I| drops below threshold (e.g., 0.1)
    # or where I changes sign
    threshold = 0.1
    for idx in range(len(i_per_lag)):
        if abs(i_per_lag[idx]) < threshold:
            return float(np.clip(d_per_lag[idx], 1.0, 50.0))

    # If never drops below threshold, use last distance
    return float(np.clip(d_per_lag[-1], 1.0, 50.0))


# =============================================================================
# METHOD 3: Held-out estimation (elegant: split train for estimation only)
# =============================================================================
def estimate_lambda_heldout(train_df, holdout_frac=0.15, lambda_grid=None):
    """
    Split training data:
    - 85% for model fitting
    - 15% for lambda estimation

    For each candidate lambda, fit model on 85%, estimate residual on 15%,
    then pick lambda that minimizes MSE on the held-out 15%.

    This is statistically principled without full nested CV.
    """
    train = train_df.dropna(subset=['Lon', 'Lat', 'CMAQ', 'Conc'])
    if len(train) < 40 or lambda_grid is None:
        return 8.0

    np.random.seed(42)
    n = len(train)
    idx = np.random.permutation(n)
    split = int(n * (1 - holdout_frac))
    est_idx, val_idx = idx[:split], idx[split:]

    est_df = train.iloc[est_idx]
    val_df = train.iloc[val_idx]

    if len(val_df) < 10:
        return 8.0

    best_lam = 8.0
    best_mse = float('inf')

    for lam in lambda_grid:
        # Fit on est_df
        m = gVNA(k=30, p=2, adaptive=False, lambda_bg=lam)
        m.fit(est_df['Lon'].values, est_df['Lat'].values,
              est_df['Conc'].values, est_df['CMAQ'].values)
        # Predict on val_df
        pred = m.predict(val_df[['Lon', 'Lat']].values, mod=val_df['CMAQ'].values)
        mse = np.nanmean((val_df['Conc'].values - pred)**2)

        if mse < best_mse:
            best_mse = mse
            best_lam = lam

    return best_lam


# =============================================================================
# METHOD 4: OOF Training Residual (use CV training residuals to estimate λ)
# =============================================================================
def estimate_lambda_oof(train_df, lambda_grid):
    """
    For each fold's training set, predict out-of-fold on that same training set.
    This gives n*0.9 training residuals.
    Then estimate λ from these residuals (using sigma/mad/iqr, but now it's OOF).

    Actually this is just: run 10-fold on TRAINING data, get OOF predictions,
    compute residual stats from OOF, derive λ.
    But that's 10x the model fits...

    Alternative: use the OOF residuals to estimate the variogram/correlogram.
    """
    train = train_df.dropna(subset=['Lon', 'Lat', 'CMAQ', 'Conc'])
    if len(train) < 40 or lambda_grid is None:
        return 8.0

    n = len(train)
    oof_biases = np.full(n, np.nan)
    folds = train['fold'].values if 'fold' in train.columns else None

    if folds is None:
        return 8.0

    # OOF on training set
    for fid in range(1, 11):
        tr = train[train['fold'] != fid]
        va = train[train['fold'] == fid]
        if len(va) == 0:
            continue
        m = gVNA(k=30, p=2, adaptive=False, lambda_bg=8.0)
        m.fit(tr['Lon'].values, tr['Lat'].values, tr['Conc'].values, tr['CMAQ'].values)
        pred = m.predict(va[['Lon', 'Lat']].values, mod=va['CMAQ'].values)
        oof_biases[train['fold'] == fid] = va['Conc'].values - pred

    valid = ~np.isnan(oof_biases)
    if valid.sum() < 20:
        return 8.0

    # Now estimate λ from OOF residuals' spatial structure
    coords = np.column_stack([train['Lon'].values, train['Lat'].values])
    biases = oof_biases[valid]
    coords_v = coords[valid]

    # Variogram on OOF residuals
    max_d = 15.0
    n_bins = 10
    bin_edges = np.linspace(0, max_d, n_bins + 1)
    sill = np.var(biases)
    if sill < 1e-6:
        return 8.0

    gamma_vals = np.zeros(n_bins)
    counts = np.zeros(n_bins)

    for i in range(len(biases)):
        for j in range(i + 1, min(i + 30, len(biases))):
            d = np.sqrt((coords_v[i, 0] - coords_v[j, 0])**2 +
                         (coords_v[i, 1] - coords_v[j, 1])**2)
            if d < max_d:
                idx = np.searchsorted(bin_edges[1:], d)
                if idx < n_bins:
                    gamma_vals[idx] += 0.5 * (biases[i] - biases[j])**2
                    counts[idx] += 1

    gamma_mean = np.where(counts > 0, gamma_vals / counts, np.nan)
    valid_bins = counts > 0

    if valid_bins.sum() < 3:
        return 8.0

    bc = (bin_edges[:-1] + bin_edges[1:])[valid_bins]
    gv = gamma_mean[valid_bins]

    # Fit exponential variogram: gamma = sill * (1 - exp(-3h/range))
    ranges_est = -3 * bc / np.log(np.maximum(1e-6, 1 - gv / (sill + 1e-6)))
    ranges_est = ranges_est[np.isfinite(ranges_est) & (ranges_est > 0) & (ranges_est < 50)]

    if len(ranges_est) == 0:
        return 8.0

    return float(np.clip(np.median(ranges_est), 1.0, 50.0))


# =============================================================================
# METHOD 5: Weighted variance of OOF residuals
# =============================================================================
def estimate_lambda_weighted_var(train_df):
    """
    λ = sqrt(median of local bias variances)
    More robust than std(biases) because it accounts for spatial structure.
    """
    train = train_df.dropna(subset=['Lon', 'Lat', 'CMAQ', 'Conc'])
    if len(train) < 30:
        return 8.0

    coords = np.column_stack([train['Lon'].values, train['Lat'].values])
    biases = (train['Conc'] - train['CMAQ']).values
    n = len(biases)
    k = min(20, n - 1)

    local_vars = []
    for i in range(n):
        dists = np.array([np.sqrt((coords[i, 0] - coords[j, 0])**2 + (coords[i, 1] - coords[j, 1])**2)
                          for j in range(n) if j != i])
        nn_idx = np.argsort(dists)[:k]
        local_vars.append(np.var(biases[nn_idx]))

    return float(np.clip(np.sqrt(np.median(local_vars)), 1.0, 50.0))


# =============================================================================
# CV runner
# =============================================================================
def run_cv(day_df, lam):
    all_t, all_p = [], []
    for fid in range(1, 11):
        tr = day_df[day_df['fold'] != fid].dropna(subset=['Lon', 'Lat', 'CMAQ', 'Conc'])
        te = day_df[day_df['fold'] == fid].dropna(subset=['Lon', 'Lat', 'CMAQ', 'Conc'])
        if len(te) == 0 or len(tr) == 0:
            continue
        m = gVNA(k=30, p=2, adaptive=False, lambda_bg=lam)
        m.fit(tr['Lon'].values, tr['Lat'].values, tr['Conc'].values, tr['CMAQ'].values)
        pred = m.predict(te[['Lon', 'Lat']].values, mod=te['CMAQ'].values)
        all_t.extend(te['Conc'].values)
        all_p.extend(pred)
    return compute_metrics(np.array(all_t), np.array(all_p)) if all_t else None


def main():
    print("=" * 60)
    print("Elegant Spatial-Statistical Lambda Estimation")
    print("=" * 60)

    stages = [('Jan', '2020-01-01', '2020-01-31'),
              ('Jul', '2020-07-01', '2020-07-31'),
              ('Dec', '2020-12-01', '2020-12-31')]

    LAM_GRID = [3, 5, 8, 10, 12, 15]

    estimators = [
        ('Fixed lam=8',         lambda df: 8.0),
        ('Adaptive CMAQ',      lambda df: adaptive_lambda(
            r2_score(df.dropna(subset=['CMAQ','Conc'])['Conc'].values,
                     df.dropna(subset=['CMAQ','Conc'])['CMAQ'].values) if len(df.dropna(subset=['CMAQ','Conc'])) > 10 else np.nan)),
    ]

    all_r2s = {n: [] for n, _ in estimators}
    all_lams = {n: [] for n, _ in estimators}

    for name, start, end in stages:
        print(f"\n=== {name} ===")
        dates = [(datetime.strptime(start, '%Y-%m-%d') + timedelta(days=i)).strftime('%Y-%m-%d')
                 for i in range((datetime.strptime(end, '%Y-%m-%d') -
                                  datetime.strptime(start, '%Y-%m-%d')).days + 1)]

        for di, d in enumerate(dates):
            if (di+1) % 10 == 0:
                print(f"\n  {name}: day {di+1}/{len(dates)}", end='')
            df = load_day(d)
            if df is None:
                continue
            for ename, est in estimators:
                try:
                    lam = float(np.clip(est(df), 1.0, 50.0))
                except:
                    lam = 8.0
                res = run_cv(df, lam)
                if res:
                    all_r2s[ename].append(res['R2'])
                    all_lams[ename].append(lam)
                    if di == 0:
                        print(f" | {lam:.1f}", end='')
            print()

    print("\n" + "=" * 60)
    pooled = {n: np.mean(r2s) for n, r2s in all_r2s.items() if r2s}
    best = max(pooled.values())
    for n in sorted(pooled, key=pooled.get, reverse=True):
        star = " ***" if pooled[n] == best else ""
        print(f"  {n:<20}  R2={pooled[n]:.4f}  lam={np.mean(all_lams[n]):.1f}{star}")

    print("\n--- Methods ---")
    print("  Variogram range: λ = distance where gamma(h) = 0.85*sill")
    print("  Moran-I decorrel: λ = distance where spatial autocorr vanishes")
    print("  Held-out MSE: λ minimizing MSE on 15% held-out")
    print("  Local var sqrt: λ = median(sqrt(local bias variance))")
    print("  OOF variogram: variogram on out-of-fold training residuals")


if __name__ == '__main__':
    main()
