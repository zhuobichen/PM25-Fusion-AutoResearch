import os
# -*- coding: utf-8 -*-
"""
Robust Variogram Lambda Estimation
================================
对比三种变异函数方法：
1. 标准变异函数 (mean-based)
2. 稳健变异函数 (median-based)
3. 对数稳健变异函数 ( Cressie-Hawkings estimator)

原理:
- 标准: γ(h) = 0.5 * mean[(b(x) - b(x+h))²]
- 稳健: γ(h) = median(|b(x) - b(x+h)|)² / (2 * 0.455)
- Cressie-Hawkings: 更复杂的稳健估计
"""

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

sys.path.insert(0, 'E:/CodeProject/ClaudeRoom/Data_Fusion_AutoResearch/CodeWorkSpace/新融合方法代码')
import numpy as np
import pandas as pd
import netCDF4 as nc
from datetime import datetime, timedelta
from sklearn.metrics import r2_score
from CodeWorkSpace.新融合方法代码.gVNA import gVNA, adaptive_lambda

CMAQ_FILE = data_path('test_data/raw/CMAQ/2020_PM25.nc')
MONITOR_FILE = data_path('test_data/raw/Monitor/2020_DailyPM2.5Monitor.csv')
FOLD_FILE = data_path('test_data/fold_split_table_daily.csv')


def load_day(d):
    monitor_df = pd.read_csv(MONITOR_FILE)
    fold_df = pd.read_csv(FOLD_FILE)
    day_df = monitor_df[monitor_df['Date'] == d].copy()
    day_df = day_df.merge(fold_df, on=['Date', 'Site'], how='left')
    day_df = day_df.dropna(subset=['Lat', 'Lon', 'Conc'])
    if len(day_df) < 100:
        return None
    ds = nc.Dataset(CMAQ_FILE, 'r')
    lon_cmaq, lat_cmaq = ds.variables['lon'][:], ds.variables['lat'][:]
    pred_pm25 = ds.variables['pred_PM25'][:]
    ds.close()
    day_idx = (datetime.strptime(d, '%Y-%m-%d') - datetime(2020, 1, 1)).days
    if day_idx >= pred_pm25.shape[0]:
        return None
    day_df = day_df.copy()
    day_df['CMAQ'] = [np.nan] * len(day_df)
    for i, (_, r) in enumerate(day_df.iterrows()):
        idx = np.argmin(np.sqrt((lon_cmaq - r['Lon'])**2 + (lat_cmaq - r['Lat'])**2))
        ny, nx = lon_cmaq.shape
        day_df.iloc[i, day_df.columns.get_loc('CMAQ')] = pred_pm25[day_idx, idx // nx, idx % nx]
    return day_df.dropna(subset=['CMAQ'])


# =============================================================================
# 标准变异函数 (Standard Variogram)
# =============================================================================
def estimate_lambda_standard_variogram(train_df, n_bins=10, max_dist=None):
    """
    标准变异函数:
    gamma(h) = 0.5 * mean[(b(x) - b(x+h))^2]

    拟合指数模型: gamma(h) = sill * (1 - exp(-3h/range))
    range 就是 lambda
    """
    train = train_df.dropna(subset=['Lon', 'Lat', 'CMAQ', 'Conc'])
    if len(train) < 30:
        return 8.0

    coords = np.column_stack([train['Lon'].values, train['Lat'].values])
    biases = (train['Conc'] - train['CMAQ']).values
    n = len(biases)

    if max_dist is None:
        max_dist = min(20.0, np.median(np.sqrt((coords[:, 0] - coords[:, 0:1])**2 +
                                               (coords[:, 1] - coords[:, 1:1])**2)) * 2)

    bin_edges = np.linspace(0, max_dist, n_bins + 1)
    gamma_vals = np.zeros(n_bins)
    counts = np.zeros(n_bins)

    # 计算实验变异函数
    for i in range(n):
        for j in range(i + 1, min(i + 30, n)):
            d = np.sqrt((coords[i, 0] - coords[j, 0])**2 + (coords[i, 1] - coords[j, 1])**2)
            if d < max_dist:
                bin_idx = np.searchsorted(bin_edges[1:], d)
                if bin_idx < n_bins:
                    gamma_vals[bin_idx] += 0.5 * (biases[i] - biases[j])**2
                    counts[bin_idx] += 1

    gamma_mean = np.where(counts > 0, gamma_vals / counts, np.nan)
    sill = np.var(biases)
    if sill < 1e-6:
        return 8.0

    valid = counts > 0
    if valid.sum() < 3:
        return 8.0

    bc = (bin_edges[:-1] + bin_edges[1:])[valid]
    gv = gamma_mean[valid]

    # 指数模型: gamma = sill * (1 - exp(-3h/range))
    # => range = -3h / log(1 - gamma/sill)
    ranges_est = -3 * bc / np.log(np.maximum(1e-6, 1 - gv / (sill + 1e-6)))
    ranges_est = ranges_est[np.isfinite(ranges_est) & (ranges_est > 0) & (ranges_est < 50)]

    return float(np.clip(np.median(ranges_est), 1.0, 50.0)) if len(ranges_est) > 0 else 8.0


# =============================================================================
# 稳健变异函数 (Median-based Robust Variogram)
# =============================================================================
def estimate_lambda_robust_variogram(train_df, n_bins=10, max_dist=None):
    """
    稳健变异函数 (Median of Absolute Deviations):
    gamma_r(h) = (median |b(x) - b(x+h)|) ^ 2 / (2 * 0.455)

    因子 0.455 = 1/(2 * CDF^{-1}(0.75)) 使得正态分布下稳健估计 = 标准差

    更稳健的做法：用多个 bin 的中位数直接估计 range
    原理：gamma_r(h) / sill = 1 - exp(-3h/range)
    当 h = range 时，gamma_r/sill = 1 - e^{-3} ≈ 0.95
    即：range 是半方差达到 95% sill 时的距离

    直接找: h使得 median(|diff|) / MAD(biases) = 0.95
    这等价于：|diff| 在 range 距离外趋于常数
    """
    train = train_df.dropna(subset=['Lon', 'Lat', 'CMAQ', 'Conc'])
    if len(train) < 30:
        return 8.0

    coords = np.column_stack([train['Lon'].values, train['Lat'].values])
    biases = (train['Conc'] - train['CMAQ']).values
    n = len(biases)

    if max_dist is None:
        max_dist = min(20.0, np.median(np.sqrt((coords[:, 0] - coords[:, 0:1])**2 +
                                               (coords[:, 1] - coords[:, 1:1])**2)) * 2)

    bin_edges = np.linspace(0, max_dist, n_bins + 1)
    med_diffs = []
    h_vals = []

    for b in range(n_bins):
        d_min, d_max = bin_edges[b], bin_edges[b + 1]
        diffs = []
        for i in range(n):
            for j in range(i + 1, min(i + 30, n)):
                d = np.sqrt((coords[i, 0] - coords[j, 0])**2 + (coords[i, 1] - coords[j, 1])**2)
                if d_min <= d < d_max:
                    diffs.append(abs(biases[i] - biases[j]))
        if len(diffs) >= 5:
            med_diffs.append(np.median(diffs))
            h_vals.append((d_min + d_max) / 2)

    if len(med_diffs) < 3:
        return 8.0

    med_diffs = np.array(med_diffs)
    h_vals = np.array(h_vals)

    # MAD -> sigma 转换: sigma = MAD / 0.455
    mad = np.median(np.abs(biases - np.median(biases)))
    sigma_est = mad / 0.455 if mad > 0 else np.std(biases)

    # 目标: 找 h 使得 med_diffs / sigma_est = 1 - exp(-3h/lambda)
    # 即: exp(-3h/lambda) = 1 - med_diff/sigma
    # lambda = -3h / log(1 - med_diff/sigma)
    lambda_cands = []
    for h, md in zip(h_vals, med_diffs):
        if md < sigma_est * 0.99 and h > 0.1:  # 确保在合理范围内
            lam = -3 * h / np.log(max(1e-6, 1 - md / sigma_est))
            if 1 < lam < 50:
                lambda_cands.append(lam)

    if not lambda_cands:
        # Fallback: 找半方差达到 0.95*sill 的 h
        sill = np.std(biases)
        target = 0.95 * sill
        for h, md in zip(h_vals, med_diffs):
            if md >= target * 0.5 and h > 0.1:
                lam = -3 * h / np.log(max(1e-6, 1 - target / (sigma_est**2 + 1e-6)))
                if 1 < lam < 50:
                    lambda_cands.append(lam)

    return float(np.clip(np.median(lambda_cands), 1.0, 50.0)) if lambda_cands else 8.0


# =============================================================================
# Cressie-Hawkings 稳健估计
# =============================================================================
def estimate_lambda_cressie_hawkings(train_df, n_bins=8, max_dist=None):
    """
    Cressie-Hawkings 稳健变异函数估计:

    gamma_CH(h) = [1/(2 * N(h))] *
                  sum[ |d|^0.5 * (d)^2 ]^2 /
                  [sum |d|^0.5]^2

    其中 d = b(x) - b(x+h), |d| = 绝对值

    这个估计对异常值更稳健，且渐近效率高。

    或者更简单的简化版:
    gamma_r(h) = median(|b(x) - b(x+h)|) * MAD(biases)

    实现简化版 Cressie-Hawkings:
    用 |d|^0.5 加权的中位数
    """
    train = train_df.dropna(subset=['Lon', 'Lat', 'CMAQ', 'Conc'])
    if len(train) < 30:
        return 8.0

    coords = np.column_stack([train['Lon'].values, train['Lat'].values])
    biases = (train['Conc'] - train['CMAQ']).values
    n = len(biases)

    if max_dist is None:
        max_dist = min(15.0, np.median(np.sqrt((coords[:, 0] - coords[:, 0:1])**2 +
                                               (coords[:, 1] - coords[:, 1:1])**2)) * 2)

    bin_edges = np.linspace(0, max_dist, n_bins + 1)
    ch_gammas = []
    h_vals = []

    for b in range(n_bins):
        d_min, d_max = bin_edges[b], bin_edges[b + 1]
        abs_diffs = []
        for i in range(n):
            for j in range(i + 1, min(i + 30, n)):
                d = np.sqrt((coords[i, 0] - coords[j, 0])**2 + (coords[i, 1] - coords[j, 1])**2)
                if d_min <= d < d_max:
                    diff = abs(biases[i] - biases[j])
                    abs_diffs.append(diff ** 0.5 * diff ** 2)  # |d|^0.5 * d^2
        if len(abs_diffs) >= 5:
            # Cressie-Hawkings 简化: median of |d|^0.5 * d^2
            ch_gammas.append(np.median(abs_diffs))
            h_vals.append((d_min + d_max) / 2)

    if len(ch_gammas) < 3:
        return 8.0

    ch_gammas = np.array(ch_gammas)
    h_vals = np.array(h_vals)
    sill = np.var(biases)

    if sill < 1e-6:
        return 8.0

    # 指数模型拟合
    # gamma = sill * (1 - exp(-3h/lambda))
    lambda_cands = []
    for h, g in zip(h_vals, ch_gammas):
        if g < sill * 0.99 and h > 0.1:
            lam = -3 * h / np.log(max(1e-6, 1 - g / (sill + 1e-6)))
            if 1 < lam < 50:
                lambda_cands.append(lam)

    return float(np.clip(np.median(lambda_cands), 1.0, 50.0)) if lambda_cands else 8.0


# =============================================================================
# Huber 变异函数 (抗异常值)
# =============================================================================
def estimate_lambda_huber_variogram(train_df, n_bins=8, max_dist=None):
    """
    Huber 变异函数:
    用 Huber 权重函数减少异常值影响
    weight(d) = 1, if |d| < k
              = k/|d|, if |d| >= k
    k = 1.345 * MAD (Huber 常数)

    实现: 用加权中位数
    """
    train = train_df.dropna(subset=['Lon', 'Lat', 'CMAQ', 'Conc'])
    if len(train) < 30:
        return 8.0

    coords = np.column_stack([train['Lon'].values, train['Lat'].values])
    biases = (train['Conc'] - train['CMAQ']).values
    n = len(biases)

    # MAD 和 Huber 阈值
    mad = np.median(np.abs(biases - np.median(biases)))
    k_huber = 1.345 * mad

    if max_dist is None:
        max_dist = min(15.0, np.median(np.sqrt((coords[:, 0] - coords[:, 0:1])**2 +
                                               (coords[:, 1] - coords[:, 1:1])**2)) * 2)

    bin_edges = np.linspace(0, max_dist, n_bins + 1)
    huber_gammas = []
    h_vals = []

    for b in range(n_bins):
        d_min, d_max = bin_edges[b], bin_edges[b + 1]
        weighted_diffs = []
        for i in range(n):
            for j in range(i + 1, min(i + 30, n)):
                d = np.sqrt((coords[i, 0] - coords[j, 0])**2 + (coords[i, 1] - coords[j, 1])**2)
                if d_min <= d < d_max:
                    diff = biases[i] - biases[j]
                    ad = abs(diff)
                    # Huber 权重
                    w = min(1.0, k_huber / (ad + 1e-10))
                    weighted_diffs.append(w * ad ** 2)
        if len(weighted_diffs) >= 5:
            huber_gammas.append(np.median(weighted_diffs))
            h_vals.append((d_min + d_max) / 2)

    if len(huber_gammas) < 3:
        return 8.0

    huber_gammas = np.array(huber_gammas)
    h_vals = np.array(h_vals)
    sill = np.var(biases)

    if sill < 1e-6:
        return 8.0

    lambda_cands = []
    for h, g in zip(h_vals, huber_gammas):
        if g < sill * 0.99 and h > 0.1:
            lam = -3 * h / np.log(max(1e-6, 1 - g / (sill + 1e-6)))
            if 1 < lam < 50:
                lambda_cands.append(lam)

    return float(np.clip(np.median(lambda_cands), 1.0, 50.0)) if lambda_cands else 8.0


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
    if not all_t:
        return np.nan
    mask = ~(np.isnan(all_t) | np.isnan(all_p))
    return r2_score(np.array(all_t)[mask], np.array(all_p)[mask])


def main():
    print("=" * 65, flush=True)
    print("Robust Variogram Lambda Estimation - 3 Methods Comparison", flush=True)
    print("=" * 65, flush=True)

    stages = [('Jan', '2020-01-01', '2020-01-31'),
              ('Jul', '2020-07-01', '2020-07-31'),
              ('Dec', '2020-12-01', '2020-12-31')]

    methods = [
        ('Fixed lam=8',             lambda df: 8.0),
        ('Adaptive CMAQ',           lambda df: adaptive_lambda(
            r2_score(df.dropna()['Conc'].values, df.dropna()['CMAQ'].values)
            if len(df.dropna()) > 10 else np.nan)),
        ('Std Variogram range',     estimate_lambda_standard_variogram),
        ('Robust Variogram (med)',  estimate_lambda_robust_variogram),
        ('Cressie-Hawkings',        estimate_lambda_cressie_hawkings),
        ('Huber Variogram',         estimate_lambda_huber_variogram),
    ]

    all_r2s = {n: [] for n, _ in methods}
    all_lams = {n: [] for n, _ in methods}

    for name, start, end in stages:
        dates = [(datetime.strptime(start, '%Y-%m-%d') + timedelta(days=i)).strftime('%Y-%m-%d')
                 for i in range((datetime.strptime(end, '%Y-%m-%d') -
                                  datetime.strptime(start, '%Y-%m-%d')).days + 1)]
        print(f"\n=== {name} ===", flush=True)
        for di, d in enumerate(dates):
            if (di+1) % 10 == 0:
                print(f"  day {di+1}/{len(dates)}", flush=True)
            df = load_day(d)
            if df is None:
                continue
            for mname, est in methods:
                try:
                    lam = float(np.clip(est(df), 1.0, 50.0))
                except:
                    lam = 8.0
                r2 = run_cv(df, lam)
                if not np.isnan(r2):
                    all_r2s[mname].append(r2)
                    all_lams[mname].append(lam)

        print(f"\n  {name} Results:", flush=True)
        for mname, _ in methods:
            if all_r2s[mname]:
                print(f"    {mname:<25} R2={np.mean(all_r2s[mname]):.4f}  "
                      f"lam={np.mean(all_lams[mname]):.1f}", flush=True)

    # Cross-stage summary
    print("\n" + "=" * 65, flush=True)
    print("CROSS-STAGE SUMMARY", flush=True)
    print("=" * 65, flush=True)
    pooled = {n: np.mean(r2s) for n, r2s in all_r2s.items() if r2s}
    best = max(pooled.values())
    for n in sorted(pooled, key=pooled.get, reverse=True):
        diff = pooled[n] - best
        star = " ***" if pooled[n] == best else ""
        print(f"  {n:<25} R2={pooled[n]:.4f}  "
              f"lam={np.mean(all_lams[n]):.1f}  "
              f"delta={diff:+.4f}{star}", flush=True)

    print("\n--- Methods ---", flush=True)
    print("  Standard:     gamma(h) = 0.5 * mean[(b(x)-b(x+h))^2]", flush=True)
    print("  Robust:       gamma(h) = median(|b(x)-b(x+h)|)^2 / (2*0.455)", flush=True)
    print("  Cressie-Hawk: gamma(h) = median(|d|^0.5 * d^2)", flush=True)
    print("  Huber:        gamma(h) = median(w(d) * d^2), w = Huber weight", flush=True)


if __name__ == '__main__':
    main()
