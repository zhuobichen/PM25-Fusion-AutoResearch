"""

# === 预测值缓存（由 patch_methods.py 自动添加）===
_last_y_true = None
_last_y_pred = None

CSP-RK-ATO - Adaptive Threshold Optimization
=============================================
CSP-RK with grid-search optimized thresholds and softmax smooth transition

核心创新：
1. 阈值 T1, T2 从固定值改为网格搜索优化
2. 硬边界改为 softmax 风格平滑过渡权重
3. 每层独立做多项式 OLS 校正，权重平滑过渡

关键公式：
w_low = 1 / (1 + np.exp(kappa * (m - T1)))
w_mid = 1 - w_low - w_high
w_high = 1 / (1 + np.exp(-kappa * (m - T2)))
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

# 默认阈值（网格搜索前）
DEFAULT_T1 = 35.0
DEFAULT_T2 = 75.0
DEFAULT_KAPPA = 0.1


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
    根据 CMAQ 浓度值确定所属层次（使用全局 DEFAULT_T1, DEFAULT_T2）
    返回：0=低层, 1=中层, 2=高层
    """
    if m_value < DEFAULT_T1:
        return 0
    elif m_value < DEFAULT_T2:
        return 1
    else:
        return 2


def compute_softmax_weights(m, T1, T2, kappa):
    """
    计算 softmax 风格的三层权重
    w_low = 1 / (1 + exp(kappa * (m - T1)))
    w_high = 1 / (1 + exp(-kappa * (m - T2)))
    w_mid = 1 - w_low - w_high
    """
    w_low = 1.0 / (1.0 + np.exp(kappa * (m - T1)))
    w_high = 1.0 / (1.0 + np.exp(-kappa * (m - T2)))
    w_mid = 1.0 - w_low - w_high
    # 确保权重非负
    w_low = np.maximum(w_low, 0)
    w_mid = np.maximum(w_mid, 0)
    w_high = np.maximum(w_high, 0)
    # 归一化
    w_sum = w_low + w_mid + w_high + 1e-8
    w_low = w_low / w_sum
    w_mid = w_mid / w_sum
    w_high = w_high / w_sum
    return w_low, w_mid, w_high


class CSPRKATO:
    """
    CSP-RK-ATO: Adaptive Threshold Optimization

    使用网格搜索优化阈值，softmax 平滑过渡
    """

    def __init__(self, poly_degree=2, T1=35.0, T2=75.0, kappa=0.1):
        self.poly_degree = poly_degree
        self.T1 = T1
        self.T2 = T2
        self.kappa = kappa
        self.poly = PolynomialFeatures(degree=poly_degree, include_bias=False)
        self.layer_models = {}
        self.gpr = None
        self.X_train = None
        self.residual_mean = None
        self.residual_std = None

    def fit(self, X, y, m):
        """
        训练 CSP-RK-ATO 模型

        参数：
        X: 位置坐标 (n, 2) - [Lon, Lat]
        y: 真实浓度值 (n,)
        m: CMAQ 模型值 (n,)
        """
        X = np.asarray(X)
        y = np.asarray(y)
        m = np.asarray(m)

        # 计算每层权重
        w_low, w_mid, w_high = compute_softmax_weights(m, self.T1, self.T2, self.kappa)

        # 存储每层的 OLS 模型和残差
        self.layer_models = {}

        for layer_id, w_layer in zip([0, 1, 2], [w_low, w_mid, w_high]):
            mask = w_layer > 0.01  # 权重大于阈值才训练
            if np.sum(mask) < 3:
                self.layer_models[layer_id] = None
                continue

            X_layer = X[mask]
            y_layer = y[mask]
            m_layer = m[mask]

            # 多项式 OLS 拟合
            m_poly = self.poly.fit_transform(m_layer.reshape(-1, 1))
            ols = LinearRegression()
            ols.fit(m_poly, y_layer)

            # 预测并计算残差
            residual_layer = y_layer - ols.predict(m_poly)

            self.layer_models[layer_id] = {
                'ols': ols,
                'X': X_layer,
                'residual': residual_layer
            }

        # 合并所有残差用于 GPR
        all_residuals = []
        all_X = []
        for layer_id, model_info in self.layer_models.items():
            if model_info is not None:
                all_residuals.append(model_info['residual'])
                all_X.append(model_info['X'])

        if len(all_residuals) > 0:
            residual_all = np.concatenate(all_residuals)
            X_all = np.vstack(all_X)

            self.residual_mean = np.mean(residual_all)
            self.residual_std = np.std(residual_all)
            residual_normalized = (residual_all - self.residual_mean) / (self.residual_std + 1e-8)

            # GPR 核函数
            kernel = ConstantKernel(10.0, (1e-2, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e1))
            self.gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, alpha=0.1, normalize_y=True)
            self.gpr.fit(X_all, residual_normalized)
            self.X_train = X_all

    def predict(self, X, m):
        """
        预测（使用 softmax 权重平滑过渡）
        """
        X = np.asarray(X)
        m = np.asarray(m)
        n = len(m)

        # 计算权重
        w_low, w_mid, w_high = compute_softmax_weights(m, self.T1, self.T2, self.kappa)

        predictions = np.zeros(n)

        # 获取每层预测
        pred_low = np.zeros(n)
        pred_mid = np.zeros(n)
        pred_high = np.zeros(n)

        for layer_id, w_layer in zip([0, 1, 2], [w_low, w_mid, w_high]):
            model_info = self.layer_models.get(layer_id)
            if model_info is not None:
                m_poly = self.poly.transform(m.reshape(-1, 1))
                pred_layer = model_info['ols'].predict(m_poly)
                if layer_id == 0:
                    pred_low = pred_layer
                elif layer_id == 1:
                    pred_mid = pred_layer
                else:
                    pred_high = pred_layer

        # 加权组合
        predictions = w_low * pred_low + w_mid * pred_mid + w_high * pred_high

        # GPR 克里金校正
        if self.gpr is not None:
            residual_pred_normalized, _ = self.gpr.predict(X, return_std=True)
            residual_pred = residual_pred_normalized * (self.residual_std + 1e-8) + self.residual_mean
            predictions = predictions + residual_pred

        return predictions


def grid_search_thresholds(X_train, y_train, m_train, T1_candidates, T2_candidates, kappa=0.1):
    """网格搜索最优阈值（仅用 OLS，不含 GPR，加速）"""
    best_r2 = -np.inf
    best_T1 = DEFAULT_T1
    best_T2 = DEFAULT_T2

    for T1 in T1_candidates:
        for T2 in T2_candidates:
            if T2 <= T1:
                continue
            # 简化的分层评估
            layers = np.array([get_concentration_layer(v) for v in m_train])
            y_pred = np.zeros(len(m_train))

            for layer_id in [0, 1, 2]:
                mask = layers == layer_id
                if np.sum(mask) < 3:
                    continue
                m_layer = m_train[mask]
                y_layer = y_train[mask]
                ols = LinearRegression()
                ols.fit(m_layer.reshape(-1, 1), y_layer)
                y_pred[mask] = ols.predict(m_layer.reshape(-1, 1))

            r2 = r2_score(y_train, y_pred)
            if r2 > best_r2:
                best_r2 = r2
                best_T1 = T1
                best_T2 = T2

    return best_T1, best_T2, best_r2


def run_csprkato_ten_fold(selected_day='2020-01-01'):
    """运行 CSP-RK-ATO 十折交叉验证"""
    print("=" * 60)
    print("CSP-RK-ATO Ten-Fold Cross Validation")
    print("=" * 60)

    # 加载数据
    print("\n=== Loading Data ===")
    monitor_df = pd.read_csv(monitor_file)
    fold_df = pd.read_csv(fold_file)

    day_df = monitor_df[monitor_df['Date'] == selected_day].copy()
    day_df = day_df.merge(fold_df, on=['Date', 'Site'], how='left')
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

        # 网格搜索最优阈值（使用训练集）
        print(f"  Fold {fold_id}: Grid searching thresholds...")
        T1_candidates = [25, 30, 35, 40]
        T2_candidates = [55, 65, 75, 85]
        best_T1, best_T2, _ = grid_search_thresholds(X_train, y_train, m_train, T1_candidates, T2_candidates)

        # CSP-RK-ATO 预测
        csprkato = CSPRKATO(poly_degree=2, T1=best_T1, T2=best_T2, kappa=DEFAULT_KAPPA)
        csprkato.fit(X_train, y_train, m_train)
        csprkato_pred = csprkato.predict(X_test, m_test)

        # 对比：固定阈值的 CSP-RK-ATO
        csprkato_fixed = CSPRKATO(poly_degree=2, T1=DEFAULT_T1, T2=DEFAULT_T2, kappa=DEFAULT_KAPPA)
        csprkato_fixed.fit(X_train, y_train, m_train)
        csprkato_fixed_pred = csprkato_fixed.predict(X_test, m_test)

        results[fold_id] = {
            'y_true': y_test,
            'csprkato': csprkato_pred,
            'csprkato_fixed': csprkato_fixed_pred,
            'best_T1': best_T1,
            'best_T2': best_T2
        }

        print(f"  Fold {fold_id}: T1={best_T1:.1f}, T2={best_T2:.1f} -> R2={compute_metrics(y_test, csprkato_pred)['R2']:.4f}")

    # 汇总
    csprkato_all = np.concatenate([results[f]['csprkato'] for f in range(1, 11) if results[f]])
    csprkato_fixed_all = np.concatenate([results[f]['csprkato_fixed'] for f in range(1, 11) if results[f]])
    true_all = np.concatenate([results[f]['y_true'] for f in range(1, 11) if results[f]])

    # 计算指标
    print("\n=== Results ===")
    csprkato_metrics = compute_metrics(true_all, csprkato_all)
    csprkato_fixed_metrics = compute_metrics(true_all, csprkato_fixed_all)
    # 缓存预测值供多天聚合使用
    global _last_y_true, _last_y_pred
    _last_y_true = true_all
    _last_y_pred = csprkato_fixed_all


    print(f"  CSP-RK-ATO (Adaptive): R2={csprkato_metrics['R2']:.4f}, MAE={csprkato_metrics['MAE']:.2f}, RMSE={csprkato_metrics['RMSE']:.2f}")
    print(f"  CSP-RK-ATO (Fixed):    R2={csprkato_fixed_metrics['R2']:.4f}, MAE={csprkato_fixed_metrics['MAE']:.2f}, RMSE={csprkato_fixed_metrics['RMSE']:.2f}")

    # 保存结果
    result_df = pd.DataFrame([{
        'method': 'CSP_RK_ATO_Adaptive',
        **csprkato_metrics
    }, {
        'method': 'CSP_RK_ATO_Fixed',
        **csprkato_fixed_metrics
    }])
    result_df.to_csv(f'{output_dir}/CSPRKATO_summary.csv', index=False)

    print(f"\nResults saved to: {output_dir}/CSPRKATO_summary.csv")

    return csprkato_metrics, csprkato_fixed_metrics


if __name__ == '__main__':
    csprkato_metrics, csprkato_fixed_metrics = run_csprkato_ten_fold('2020-01-01')
    print(f"\nCSP-RK-ATO (Adaptive): R2={csprkato_metrics['R2']:.4f}")
    print(f"CSP-RK-ATO (Fixed):    R2={csprkato_fixed_metrics['R2']:.4f}")
