"""
HGP-RK - Heteroscedastic GPR PolyRK
=====================================
GPR 残差建模使用异方差（不同浓度区域不同方差权重）

核心创新：
1. 全局多项式 OLS 校正
2. 分层计算残差方差：σ²_1, σ²_2, σ²_3
3. 构建异方差权重：w_i = σ²_min / σ²_layer_i
4. 用 sample_weight 传入 GPR 拟合

阈值：T1=35 μg/m³, T2=75 μg/m³
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.paths import get_project_root, data_path
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
import netCDF4 as nc

root_dir = str(get_project_root())
cmaq_file = data_path('test_data/raw/CMAQ/2020_PM25.nc')
monitor_file = data_path('test_data/raw/Monitor/2020_DailyPM2.5Monitor.csv')
fold_file = data_path('test_data/fold_split_table_daily.csv')
output_dir = f'{root_dir}/test_result/创新方法'
os.makedirs(output_dir, exist_ok=True)

# 浓度分层阈值
T1 = 35.0  # 低/中分界
T2 = 75.0  # 中/高分界


def compute_metrics(y_true, y_pred):
    """计算评估指标"""
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
    """获取站点位置的 CMAQ 值"""
    dist = np.sqrt((lon_grid - lon)**2 + (lat_grid - lat)**2)
    idx = np.argmin(dist)
    ny, nx = lon_grid.shape
    row, col = idx // nx, idx % nx
    return pm25_grid[row, col]


def get_concentration_layer(m_value):
    """
    根据 CMAQ 浓度值确定所属层次
    返回：0=低层, 1=中层, 2=高层
    """
    if m_value < T1:
        return 0
    elif m_value < T2:
        return 1
    else:
        return 2


def compute_heteroscedastic_weights(m_values, residual_values):
    """
    计算异方差权重

    参数：
    m_values: CMAQ 模型值 (n,)
    residual_values: OLS 残差 (n,)

    返回：
    weights: 异方差权重 (n,)
    """
    layers = np.array([get_concentration_layer(v) for v in m_values])

    # 计算每层的残差方差
    layer_variances = {}
    for layer_id in [0, 1, 2]:
        mask = layers == layer_id
        if np.sum(mask) > 0:
            layer_variances[layer_id] = np.var(residual_values[mask])
        else:
            layer_variances[layer_id] = np.var(residual_values)

    # 找到最小方差
    sigma_min_sq = min(layer_variances.values())

    # 计算权重：w_i = σ²_min / σ²_layer_i
    weights = np.zeros_like(m_values, dtype=float)
    for i, m_val in enumerate(m_values):
        layer_id = layers[i]
        weights[i] = sigma_min_sq / (layer_variances[layer_id] + 1e-8)

    return weights


class HGPRK:
    """
    HGP-RK: Heteroscedastic GPR PolyRK

    异方差高斯过程残差克里金
    使用分层 GPR：每层使用不同的 alpha（噪声方差）来反映异方差特性
    """

    def __init__(self, poly_degree=2):
        self.poly_degree = poly_degree
        self.poly = PolynomialFeatures(degree=poly_degree, include_bias=False)
        self.ols = None
        self.layer_gprs = {}  # 每层的 GPR 模型
        self.layer_alphas = {}  # 每层的 alpha 值
        self.X_train = None
        self.residual_mean = None
        self.residual_std = None
        self.layer_threshold = {'T1': T1, 'T2': T2}

    def fit(self, X, y, m):
        """
        训练 HGP-RK 模型

        参数：
        X: 位置坐标 (n, 2) - [Lon, Lat]
        y: 真实浓度值 (n,)
        m: CMAQ 模型值 (n,)
        """
        X = np.asarray(X)
        y = np.asarray(y)
        m = np.asarray(m)

        # 多项式 OLS 拟合
        m_poly = self.poly.fit_transform(m.reshape(-1, 1))
        self.ols = LinearRegression()
        self.ols.fit(m_poly, y)

        # 预测并计算残差
        residual = y - self.ols.predict(m_poly)

        # 残差标准化
        self.residual_mean = np.mean(residual)
        self.residual_std = np.std(residual) + 1e-8
        residual_normalized = (residual - self.residual_mean) / self.residual_std

        # 按浓度分层
        layers = np.array([get_concentration_layer(v) for v in m])

        # 计算每层的残差方差
        layer_variances = {}
        for layer_id in [0, 1, 2]:
            mask = layers == layer_id
            if np.sum(mask) > 0:
                layer_variances[layer_id] = np.var(residual[mask])
            else:
                layer_variances[layer_id] = np.var(residual)

        # 找到最小方差用于归一化
        sigma_min_sq = min(layer_variances.values())

        # 每层训练一个 GPR，使用不同的 alpha
        # alpha = sigma_layer^2 / sigma_min^2，即权重越高 alpha 越低
        kernel = ConstantKernel(10.0, (1e-2, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e1))

        self.layer_gprs = {}
        self.layer_alphas = {}

        for layer_id in [0, 1, 2]:
            mask = layers == layer_id
            if np.sum(mask) < 3:  # 需要至少3个样本
                self.layer_gprs[layer_id] = None
                self.layer_alphas[layer_id] = 1.0
                continue

            X_layer = X[mask]
            residual_layer = residual_normalized[mask]

            # 计算该层的 alpha（异方差权重）
            # 方差大的层 alpha 大（噪声大，权重小）
            alpha_layer = layer_variances[layer_id] / sigma_min_sq
            self.layer_alphas[layer_id] = alpha_layer

            # 使用该层的 alpha 拟合 GPR
            gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, alpha=alpha_layer, normalize_y=True)
            gpr.fit(X_layer, residual_layer)
            self.layer_gprs[layer_id] = gpr

        self.X_train = X

    def predict(self, X, m):
        """
        预测

        参数：
        X: 位置坐标 (n, 2) - [Lon, Lat]
        m: CMAQ 模型值 (n,)

        返回：
        预测浓度值 (n,)
        """
        X = np.asarray(X)
        m = np.asarray(m)
        n = len(m)

        # OLS 预测
        m_poly = self.poly.transform(m.reshape(-1, 1))
        pred_ols = self.ols.predict(m_poly)

        # 按层预测残差
        residual_pred = np.zeros(n)
        layers = np.array([get_concentration_layer(v) for v in m])

        for i in range(n):
            layer_id = layers[i]
            if self.layer_gprs.get(layer_id) is not None:
                gpr_pred, _ = self.layer_gprs[layer_id].predict([X[i]], return_std=True)
                residual_pred[i] = gpr_pred[0]
            else:
                # 如果该层没有 GPR，使用全局平均
                residual_pred[i] = 0.0

        # 反标准化
        residual_pred = residual_pred * self.residual_std + self.residual_mean
        pred = pred_ols + residual_pred

        return pred


def run_hgprk_ten_fold(selected_day='2020-01-01'):
    """运行 HGP-RK 十折交叉验证"""
    print("=" * 60)
    print("HGP-RK Ten-Fold Cross Validation")
    print("=" * 60)

    # 加载数据
    print("\n=== Loading Data ===")
    monitor_df = pd.read_csv(monitor_file)
    fold_df = pd.read_csv(fold_file)

    day_df = monitor_df[monitor_df['Date'] == selected_day].copy()
    day_df = day_df.merge(fold_df, on='Site', how='left')
    day_df = day_df.dropna(subset=['Lat', 'Lon', 'Conc'])

    # 加载 CMAQ 数据
    ds = nc.Dataset(cmaq_file, 'r')
    lon_cmaq = ds.variables['lon'][:]
    lat_cmaq = ds.variables['lat'][:]
    pred_pm25 = ds.variables['pred_PM25'][:]
    ds.close()

    from datetime import datetime
    date_obj = datetime.strptime(selected_day, '%Y-%m-%d')
    day_idx = (date_obj - datetime(2020, 1, 1)).days
    pred_day = pred_pm25[day_idx]

    # 提取站点 CMAQ 值
    print("=== Extracting CMAQ at Sites ===")
    cmaq_values = []
    for _, row in day_df.iterrows():
        val = get_cmaq_at_site(row['Lon'], row['Lat'], lon_cmaq, lat_cmaq, pred_day)
        cmaq_values.append(val)
    day_df['CMAQ'] = cmaq_values

    ny, nx = lon_cmaq.shape
    X_grid_full = np.column_stack([lon_cmaq.ravel(), lat_cmaq.ravel()])
    y_grid_model_full = pred_day.ravel()

    print(f"Data loaded: {len(day_df)} monitoring records")

    # 定义 GPR 核函数
    kernel = ConstantKernel(10.0, (1e-2, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e1))

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

        # === 二次多项式 OLS 校正 ===
        poly = PolynomialFeatures(degree=2, include_bias=False)
        m_train_poly = poly.fit_transform(m_train.reshape(-1, 1))
        m_test_poly = poly.transform(m_test.reshape(-1, 1))

        ols_poly = LinearRegression()
        ols_poly.fit(m_train_poly, y_train)
        pred_poly = ols_poly.predict(m_test_poly)
        residual_poly = y_train - ols_poly.predict(m_train_poly)

        # === 标准 GPR (同方差) ===
        residual_mean = np.mean(residual_poly)
        residual_std = np.std(residual_poly) + 1e-8
        residual_norm = (residual_poly - residual_mean) / residual_std

        gpr_homo = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, alpha=0.1, normalize_y=True)
        gpr_homo.fit(X_train, residual_norm)
        gpr_homo_pred, _ = gpr_homo.predict(X_test, return_std=True)
        gpr_homo_pred = gpr_homo_pred * residual_std + residual_mean

        rk_homo_pred = pred_poly + gpr_homo_pred

        # === HGP-RK (异方差) ===
        hgprk = HGPRK(poly_degree=2)
        hgprk.fit(X_train, y_train, m_train)
        hgprk_pred = hgprk.predict(X_test, m_test)

        results[fold_id] = {
            'y_true': y_test,
            'rk_homo': rk_homo_pred,
            'hgprk': hgprk_pred
        }

        print(f"  Fold {fold_id}: completed")

    # 汇总
    rk_homo_all = np.concatenate([results[f]['rk_homo'] for f in range(1, 11) if results[f]])
    hgprk_all = np.concatenate([results[f]['hgprk'] for f in range(1, 11) if results[f]])
    true_all = np.concatenate([results[f]['y_true'] for f in range(1, 11) if results[f]])

    # 计算 R2
    print("\n=== Results ===")
    rk_homo_metrics = compute_metrics(true_all, rk_homo_all)
    hgprk_metrics = compute_metrics(true_all, hgprk_all)

    print(f"  RK-Homo:  R2={rk_homo_metrics['R2']:.4f}, MAE={rk_homo_metrics['MAE']:.2f}, RMSE={rk_homo_metrics['RMSE']:.2f}")
    print(f"  HGP-RK:   R2={hgprk_metrics['R2']:.4f}, MAE={hgprk_metrics['MAE']:.2f}, RMSE={hgprk_metrics['RMSE']:.2f}")

    # 保存结果
    result_df = pd.DataFrame([{
        'method': 'RK_Homo',
        **rk_homo_metrics
    }, {
        'method': 'HGP_RK',
        **hgprk_metrics
    }])
    result_df.to_csv(f'{output_dir}/HGPRK_summary.csv', index=False)

    print(f"\nResults saved to: {output_dir}/HGPRK_summary.csv")

    return rk_homo_metrics, hgprk_metrics


if __name__ == '__main__':
    rk_homo_metrics, hgprk_metrics = run_hgprk_ten_fold('2020-01-01')
    print(f"\nRK-Homo: R2={rk_homo_metrics['R2']:.4f}")
    print(f"HGP-RK:  R2={hgprk_metrics['R2']:.4f}")