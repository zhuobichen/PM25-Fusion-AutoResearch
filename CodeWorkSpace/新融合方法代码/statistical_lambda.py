import os
# -*- coding: utf-8 -*-
"""
Statistical methods to determine lambda for gVNA
Compare 5 approaches:
1. Fixed lambda=15 (baseline)
2. CMAQ_R2-based adaptive (threshold rules)
3. Nested CV (train-fold inner CV to find lambda)
4. Bayesian model averaging (BMA over lambda grid)
5. Residual variance weighting (no fixed lambda needed)
"""

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

sys.path.insert(0, 'E:/CodeProject/ClaudeRoom/Data_Fusion_AutoResearch/CodeWorkSpace/新融合方法代码')
import numpy as np
import pandas as pd
import netCDF4 as nc
from datetime import datetime, timedelta
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from CodeWorkSpace.新融合方法代码.gVNA import gVNA, adaptive_lambda

CMAQ_FILE = data_path('test_data/raw/CMAQ/2020_PM25.nc')
MONITOR_FILE = data_path('test_data/raw/Monitor/2020_DailyPM2.5Monitor.csv')
FOLD_FILE = data_path('test_data/fold_split_table_daily.csv')
LAMBDA_GRID = [3, 5, 8, 10, 12, 15]


def compute_metrics(y_true, y_pred):
    mask = ~(np.isnan(y_true) | np.isnan(y_pred) | np.isinf(y_true) | np.isinf(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if len(y_true) == 0:
        return {'R2': np.nan, 'RMSE': np.nan, 'MB': np.nan}
    return {
        'R2': float(r2_score(y_true, y_pred)),
        'RMSE': float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'MB': float(np.mean(y_pred - y_true))
    }


def get_cmaq_at_site(lon, lat, lon_grid, lat_grid, pm25_grid):
    dist = np.sqrt((lon_grid - lon)**2 + (lat_grid - lat)**2)
    idx = np.argmin(dist)
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
    lon_cmaq = ds.variables['lon'][:]
    lat_cmaq = ds.variables['lat'][:]
    pred_pm25 = ds.variables['pred_PM25'][:]
    ds.close()
    date_obj = datetime.strptime(selected_day, '%Y-%m-%d')
    day_idx = (date_obj - datetime(2020, 1, 1)).days
    cmaq_day = pred_pm25[day_idx]
    day_df = day_df.copy()
    day_df['CMAQ'] = [get_cmaq_at_site(row['Lon'], row['Lat'], lon_cmaq, lat_cmaq, cmaq_day)
                       for _, row in day_df.iterrows()]
    return day_df.dropna(subset=['CMAQ'])


# =========================================================================
# METHOD 1: Fixed lambda=15 (baseline)
# =========================================================================
def run_cv_fixed(day_df, lam):
    all_y_true, all_y_pred = [], []
    for fold_id in range(1, 11):
        train = day_df[day_df['fold'] != fold_id].dropna(subset=['Lon', 'Lat', 'CMAQ', 'Conc'])
        test = day_df[day_df['fold'] == fold_id].dropna(subset=['Lon', 'Lat', 'CMAQ', 'Conc'])
        if len(test) == 0 or len(train) == 0:
            continue
        model = gVNA(k=30, p=2, adaptive=False, lambda_bg=lam)
        model.fit(train['Lon'].values, train['Lat'].values, train['Conc'].values, train['CMAQ'].values)
        y_pred = model.predict(test[['Lon', 'Lat']].values, mod=test['CMAQ'].values)
        all_y_true.extend(test['Conc'].values)
        all_y_pred.extend(y_pred)
    if not all_y_true:
        return np.nan
    return compute_metrics(np.array(all_y_true), np.array(all_y_pred))


# =========================================================================
# METHOD 2: CMAQ_R2 adaptive (threshold rules)
# =========================================================================
def run_cv_adaptive(day_df):
    all_y_true, all_y_pred = [], []
    for fold_id in range(1, 11):
        train = day_df[day_df['fold'] != fold_id].dropna(subset=['Lon', 'Lat', 'CMAQ', 'Conc'])
        test = day_df[day_df['fold'] == fold_id].dropna(subset=['Lon', 'Lat', 'CMAQ', 'Conc'])
        if len(test) == 0 or len(train) == 0:
            continue
        valid = ~np.isnan(train['CMAQ']) & ~np.isnan(train['Conc'])
        cmaq_r2 = r2_score(train.loc[valid, 'Conc'], train.loc[valid, 'CMAQ']) if valid.sum() > 10 else np.nan
        model = gVNA(k=30, p=2, adaptive=True)
        model.fit(train['Lon'].values, train['Lat'].values, train['Conc'].values, train['CMAQ'].values, cmaq_r2=cmaq_r2)
        y_pred = model.predict(test[['Lon', 'Lat']].values, mod=test['CMAQ'].values)
        all_y_true.extend(test['Conc'].values)
        all_y_pred.extend(y_pred)
    if not all_y_true:
        return np.nan
    return compute_metrics(np.array(all_y_true), np.array(all_y_pred))


# =========================================================================
# METHOD 3: Nested CV - inner 3-fold on training to find best lambda
# =========================================================================
def find_best_lambda_nested(train_df, lambda_grid):
    """Use 3-fold CV on training data to find best lambda"""
    n_train = len(train_df)
    if n_train < 30:
        return 8  # fallback

    # Create 3 folds from training data
    np.random.seed(42)
    idx = np.random.permutation(n_train)
    fold_size = n_train // 3
    inner_folds = []
    for i in range(3):
        start = i * fold_size
        end = start + fold_size if i < 2 else n_train
        inner_folds.append(idx[start:end])

    inner_results = {lam: [] for lam in lambda_grid}

    for inner_id in range(3):
        inner_train_idx = np.concatenate([inner_folds[i] for i in range(3) if i != inner_id])
        inner_val_idx = inner_folds[inner_id]

        inner_train = train_df.iloc[inner_train_idx]
        inner_val = train_df.iloc[inner_val_idx]

        inner_train = inner_train.dropna(subset=['Lon', 'Lat', 'CMAQ', 'Conc'])
        inner_val = inner_val.dropna(subset=['Lon', 'Lat', 'CMAQ', 'Conc'])

        if len(inner_val) < 10:
            continue

        X_val = inner_val[['Lon', 'Lat']].values
        mod_val = inner_val['CMAQ'].values
        y_val = inner_val['Conc'].values

        for lam in lambda_grid:
            model = gVNA(k=30, p=2, adaptive=False, lambda_bg=lam)
            model.fit(inner_train['Lon'].values, inner_train['Lat'].values,
                      inner_train['Conc'].values, inner_train['CMAQ'].values)
            y_pred = model.predict(X_val, mod=mod_val)
            inner_results[lam].append(r2_score(y_val, y_pred))

    # Pick lambda with highest mean inner R2
    inner_avg = {lam: np.mean(v) for lam, v in inner_results.items() if v}
    if not inner_avg:
        return 8
    return max(inner_avg, key=lambda x: inner_avg[x])


def run_cv_nested(day_df, lambda_grid):
    all_y_true, all_y_pred = [], []
    for fold_id in range(1, 11):
        train = day_df[day_df['fold'] != fold_id].dropna(subset=['Lon', 'Lat', 'CMAQ', 'Conc'])
        test = day_df[day_df['fold'] == fold_id].dropna(subset=['Lon', 'Lat', 'CMAQ', 'Conc'])
        if len(test) == 0 or len(train) == 0:
            continue

        # Find best lambda via inner CV on training data
        best_lam = find_best_lambda_nested(train, lambda_grid)

        model = gVNA(k=30, p=2, adaptive=False, lambda_bg=best_lam)
        model.fit(train['Lon'].values, train['Lat'].values, train['Conc'].values, train['CMAQ'].values)
        y_pred = model.predict(test[['Lon', 'Lat']].values, mod=test['CMAQ'].values)
        all_y_true.extend(test['Conc'].values)
        all_y_pred.extend(y_pred)
    if not all_y_true:
        return np.nan
    return compute_metrics(np.array(all_y_true), np.array(all_y_pred))


# =========================================================================
# METHOD 4: Bayesian Model Averaging over lambda grid
# =========================================================================
def run_cv_bma(day_df, lambda_grid):
    """
    BMA: weight each lambda's prediction by its inner-CV likelihood
    For each fold, compute inner-CV R2 for each lambda -> use as weights
    Then weight-average the predictions across lambdas
    """
    all_y_true, all_y_pred = [], []
    for fold_id in range(1, 11):
        train = day_df[day_df['fold'] != fold_id].dropna(subset=['Lon', 'Lat', 'CMAQ', 'Conc'])
        test = day_df[day_df['fold'] == fold_id].dropna(subset=['Lon', 'Lat', 'CMAQ', 'Conc'])
        if len(test) == 0 or len(train) == 0:
            continue

        test = test.dropna(subset=['Lon', 'Lat', 'CMAQ', 'Conc'])
        if len(test) == 0:
            continue

        X_test = test[['Lon', 'Lat']].values
        mod_test = test['CMAQ'].values
        y_true_test = test['Conc'].values

        # Inner 3-fold CV on training to get weights
        n_train = len(train)
        np.random.seed(42)
        idx = np.random.permutation(n_train)
        fold_size = n_train // 3
        inner_folds = []
        for i in range(3):
            start = i * fold_size
            end = start + fold_size if i < 2 else n_train
            inner_folds.append(idx[start:end])

        inner_r2 = {lam: [] for lam in lambda_grid}

        for inner_id in range(3):
            inner_train_idx = np.concatenate([inner_folds[i] for i in range(3) if i != inner_id])
            inner_val_idx = inner_folds[inner_id]
            inner_train = train.iloc[inner_train_idx].dropna(subset=['Lon', 'Lat', 'CMAQ', 'Conc'])
            inner_val = train.iloc[inner_val_idx].dropna(subset=['Lon', 'Lat', 'CMAQ', 'Conc'])
            if len(inner_val) < 10:
                continue
            X_val = inner_val[['Lon', 'Lat']].values
            mod_val = inner_val['CMAQ'].values
            y_val = inner_val['Conc'].values
            for lam in lambda_grid:
                m = gVNA(k=30, p=2, adaptive=False, lambda_bg=lam)
                m.fit(inner_train['Lon'].values, inner_train['Lat'].values,
                      inner_train['Conc'].values, inner_train['CMAQ'].values)
                y_p = m.predict(X_val, mod=mod_val)
                inner_r2[lam].append(r2_score(y_val, y_p))

        # Compute weights from inner R2 (softmax-like)
        avg_r2 = {lam: np.mean(v) for lam, v in inner_r2.items() if v}
        if not avg_r2:
            best_lam = 8
        else:
            # Convert R2 to weights: higher R2 -> higher weight
            # Use exponential weighting based on R2 rank
            r2_vals = np.array([avg_r2.get(lam, 0) for lam in lambda_grid])
            # Shift to positive, then softmax
            r2_shift = r2_vals - r2_vals.min() + 0.001
            weights = r2_shift / r2_shift.sum()
            best_lam = lambda_grid[np.argmax(r2_vals)]  # also track best single

        # Get predictions from all lambda models
        preds_by_lam = {}
        for lam in lambda_grid:
            m = gVNA(k=30, p=2, adaptive=False, lambda_bg=lam)
            m.fit(train['Lon'].values, train['Lat'].values, train['Conc'].values, train['CMAQ'].values)
            preds_by_lam[lam] = m.predict(X_test, mod=mod_test)

        # BMA: weight-average
        y_pred_bma = np.zeros(len(y_true_test))
        total_weight = 0
        for i, lam in enumerate(lambda_grid):
            y_pred_bma += weights[i] * preds_by_lam[lam]
            total_weight += weights[i]

        y_pred_bma /= total_weight

        all_y_true.extend(y_true_test)
        all_y_pred.extend(y_pred_bma)

    if not all_y_true:
        return np.nan
    return compute_metrics(np.array(all_y_true), np.array(all_y_pred))


# =========================================================================
# METHOD 5: Residual variance weighted (no fixed lambda)
# =========================================================================
class gVNA_varWeighted:
    """
    Instead of fixed lambda, use per-station residual variance as similarity measure.
    Stations with lower residual variance (CMAQ more reliable) get higher weight.
    """
    def __init__(self, k=30, p=2, clip_nonnegative=True):
        self.k = k
        self.p = p
        self.clip_nonnegative = clip_nonnegative

    def _compute_distance(self, coord1, coord2):
        return np.sqrt((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2)

    def _find_k_nearest(self, target_coord, coords_array, k):
        n = len(coords_array)
        distances = np.array([self._compute_distance(target_coord, coords_array[i]) for i in range(n)])
        indices = np.argsort(distances)[:k]
        return indices, distances[indices]

    def fit(self, train_lon, train_lat, train_Conc, train_mod):
        train_lon = np.asarray(train_lon, dtype=np.float64)
        train_lat = np.asarray(train_lat, dtype=np.float64)
        train_Conc = np.asarray(train_Conc, dtype=np.float64)
        train_mod = np.asarray(train_mod, dtype=np.float64)
        mask = ~(np.isnan(train_lon) | np.isnan(train_lat) |
                 np.isnan(train_Conc) | np.isnan(train_mod) |
                 np.isinf(train_lon) | np.isinf(train_lat) |
                 np.isinf(train_Conc) | np.isinf(train_mod))
        self.train_lon = train_lon[mask]
        self.train_lat = train_lat[mask]
        self.train_Conc = train_Conc[mask]
        self.train_mod = train_mod[mask]

        # Compute residual variance per station (using k-NN on training to estimate local variance)
        n = len(self.train_mod)
        residuals = self.train_Conc - self.train_mod

        # Estimate local residual variance for each station using k-NN
        self.residuals = residuals
        coords = np.column_stack([self.train_lon, self.train_lat])
        # Use a small neighborhood to estimate variance
        k_var = min(20, n)
        self.local_var = np.zeros(n)
        for i in range(n):
            dists = np.array([self._compute_distance(coords[i], coords[j]) for j in range(n)])
            nn_idx = np.argsort(dists)[:k_var]
            self.local_var[i] = np.var(residuals[nn_idx]) + 1e-6  # add small epsilon

        return self

    def predict_single(self, lon, lat, mod):
        coords = np.column_stack([self.train_lon, self.train_lat])
        target_coord = np.array([lon, lat])
        k_eff = min(self.k, len(coords))
        indices, dists = self._find_k_nearest(target_coord, coords, k_eff)
        if len(indices) == 0:
            return np.nan

        # Distance weight
        dist_weights = np.power(dists + 1e-10, -self.p)

        # Variance-based weight: lower variance -> higher weight
        # weight_i = 1 / variance_i (normalized)
        var_weights = 1.0 / self.local_var[indices]
        var_weights = var_weights / (var_weights.sum() + 1e-10)

        # Combined
        weights = dist_weights * var_weights
        weights = weights / (weights.sum() + 1e-10)

        # Additive bias correction
        biases = self.residuals
        bias_interp = np.sum(weights * biases[indices])
        y_pred = mod + bias_interp

        if self.clip_nonnegative:
            y_pred = max(0.0, y_pred)
        return y_pred

    def predict(self, X, mod):
        X = np.asarray(X)
        mod = np.asarray(mod)
        if X.ndim != 2 or X.shape[1] != 2:
            raise ValueError("X must have shape (n, 2)")
        return np.array([self.predict_single(X[i, 0], X[i, 1], mod[i]) for i in range(len(X))])


def run_cv_varweighted(day_df):
    all_y_true, all_y_pred = [], []
    for fold_id in range(1, 11):
        train = day_df[day_df['fold'] != fold_id].dropna(subset=['Lon', 'Lat', 'CMAQ', 'Conc'])
        test = day_df[day_df['fold'] == fold_id].dropna(subset=['Lon', 'Lat', 'CMAQ', 'Conc'])
        if len(test) == 0 or len(train) == 0:
            continue
        model = gVNA_varWeighted(k=30, p=2)
        model.fit(train['Lon'].values, train['Lat'].values, train['Conc'].values, train['CMAQ'].values)
        y_pred = model.predict(test[['Lon', 'Lat']].values, mod=test['CMAQ'].values)
        all_y_true.extend(test['Conc'].values)
        all_y_pred.extend(y_pred)
    if not all_y_true:
        return np.nan
    return compute_metrics(np.array(all_y_true), np.array(all_y_pred))


# =========================================================================
# MAIN
# =========================================================================
def main():
    print("=" * 70)
    print("Statistical Lambda Methods Comparison")
    print("=" * 70)

    stages = [
        ('stage1 (Jan)', '2020-01-01', '2020-01-10'),
        ('stage2 (Jul)', '2020-07-01', '2020-07-10'),
        ('stage3 (Dec)', '2020-12-01', '2020-12-10'),
    ]

    methods = {
        'M1: Fixed lam=15': lambda df: run_cv_fixed(df, 15),
        'M2: Adaptive (CMAQ_R2)': lambda df: run_cv_adaptive(df),
        'M3: Nested CV (inner 3-fold)': lambda df: run_cv_nested(df, LAMBDA_GRID),
        'M4: BMA over lambda grid': lambda df: run_cv_bma(df, LAMBDA_GRID),
        'M5: Var-Weighted (no lambda)': lambda df: run_cv_varweighted(df),
    }

    all_results = {name: [] for name in methods}

    for stage_name, start, end in stages:
        print(f"\n{'='*70}")
        print(f"Stage: {stage_name}")
        print(f"{'='*70}")

        date_list = []
        cur = datetime.strptime(start, '%Y-%m-%d')
        while cur <= datetime.strptime(end, '%Y-%m-%d'):
            date_list.append(cur.strftime('%Y-%m-%d'))
            cur += timedelta(days=1)

        day_datas = [load_day(d) for d in date_list]
        day_datas = [d for d in day_datas if d is not None]
        print(f"Valid days: {len(day_datas)}")

        # Run all methods
        method_r2s = {name: [] for name in methods}
        for i, day_df in enumerate(day_datas):
            if (i+1) % 5 == 0 or i == 0:
                print(f"  Day {i+1}/{len(day_datas)}: {date_list[i]}")
            for name, method in methods.items():
                m = method(day_df)
                if m is not None and not np.isnan(m['R2']):
                    method_r2s[name].append(m['R2'])

        # Summary
        pooled = {name: np.mean(r2s) for name, r2s in method_r2s.items() if r2s}
        best_method = max(pooled, key=lambda x: pooled[x])

        print(f"\n  Pooled R2 (mean across days):")
        print(f"  {'Method':<30}  {'R2':>8}  {'vs_best':>8}")
        print(f"  {'-'*50}")
        for name in sorted(pooled, key=lambda x: pooled[x], reverse=True):
            diff = pooled[name] - pooled[best_method]
            marker = " <-- BEST" if name == best_method else ""
            print(f"  {name:<30}  {pooled[name]:>8.4f}  {diff:>+8.4f}{marker}")

        for name, r2s in method_r2s.items():
            all_results[name].extend(r2s)

    # Cross-stage summary
    print(f"\n{'='*70}")
    print("CROSS-STAGE SUMMARY")
    print(f"{'='*70}")
    pooled_all = {name: np.mean(r2s) for name, r2s in all_results.items() if r2s}
    best_all = max(pooled_all, key=lambda x: pooled_all[x])
    print(f"  {'Method':<30}  {'R2':>8}  {'vs_best':>8}")
    print(f"  {'-'*50}")
    for name in sorted(pooled_all, key=lambda x: pooled_all[x], reverse=True):
        diff = pooled_all[name] - pooled_all[best_all]
        marker = " <-- BEST" if name == best_all else ""
        print(f"  {name:<30}  {pooled_all[name]:>8.4f}  {diff:>+8.4f}{marker}")


if __name__ == '__main__':
    main()
