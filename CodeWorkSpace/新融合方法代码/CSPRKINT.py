"""
CSP-RK-INT - Interaction Model
===============================
CSP-RK with CMAQ×Position interaction terms

核心创新：
1. 在 CSP-RK 基础上添加 CMAQ×Lat 和 CMAQ×Lon 交互项
2. 特征从 [1, M, M²] 扩展为 [1, M, M², M×Lat, M×Lon]

关键公式：
O = a + b*M + c*M² + d*(M×Lat) + e*(M×Lon) + ε
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
T1 = 35.0
T2 = 75.0


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


def create_interaction_features(m, X):
    """
    创建交互特征
    特征从 [1, M, M²] 扩展为 [1, M, M², M×Lat, M×Lon]

    参数：
    m: CMAQ 模型值 (n,) or scalar
    X: 位置坐标 (n, 2) - [Lon, Lat]

    返回：
    features: 扩展后的特征矩阵 (n, 5)
    """
    m = np.asarray(m).reshape(-1, 1)
    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    lon = X[:, 0].reshape(-1, 1)
    lat = X[:, 1].reshape(-1, 1)

    # 基础多项式特征
    m_poly = np.hstack([m, m**2])

    # 交互项
    interaction = np.hstack([m * lat, m * lon])

    # 合并
    features = np.hstack([m_poly, interaction])

    return features


class CSPRKINT:
    """
    CSP-RK-INT: Concentration-Stratified PolyRK with Interaction

    分层多项式残差克里金 + CMAQ×位置交互项
    """

    def __init__(self, poly_degree=2, use_interaction=True):
        self.poly_degree = poly_degree
        self.use_interaction = use_interaction
        self.poly = PolynomialFeatures(degree=poly_degree, include_bias=False)
        self.layer_models = {}
        self.gpr = None
        self.X_train = None
        self.residual_mean = None
        self.residual_std = None

    def fit(self, X, y, m):
        """
        训练 CSP-RK-INT 模型

        参数：
        X: 位置坐标 (n, 2) - [Lon, Lat]
        y: 真实浓度值 (n,)
        m: CMAQ 模型值 (n,)
        """
        X = np.asarray(X)
        y = np.asarray(y)
        m = np.asarray(m)

        # 按浓度分层
        layers = np.array([get_concentration_layer(v) for v in m])

        # 存储每层的 OLS 模型和残差
        self.layer_models = {}

        for layer_id in [0, 1, 2]:
            mask = layers == layer_id
            if np.sum(mask) < 3:
                self.layer_models[layer_id] = None
                continue

            X_layer = X[mask]
            y_layer = y[mask]
            m_layer = m[mask]

            if self.use_interaction:
                # 使用交互特征
                X_layer_feat = create_interaction_features(m_layer, X_layer)
                ols = LinearRegression()
                ols.fit(X_layer_feat, y_layer)
                residual_layer = y_layer - ols.predict(X_layer_feat)
                self.layer_models[layer_id] = {
                    'ols': ols,
                    'X': X_layer,
                    'residual': residual_layer,
                    'use_interaction': True
                }
            else:
                # 使用基础多项式特征
                m_poly = self.poly.fit_transform(m_layer.reshape(-1, 1))
                ols = LinearRegression()
                ols.fit(m_poly, y_layer)
                residual_layer = y_layer - ols.predict(m_poly)
                self.layer_models[layer_id] = {
                    'ols': ols,
                    'X': X_layer,
                    'residual': residual_layer,
                    'use_interaction': False,
                    'poly': self.poly
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

        predictions = np.zeros(n)
        layers = np.array([get_concentration_layer(v) for v in m])

        for i in range(n):
            layer_id = layers[i]
            model_info = self.layer_models.get(layer_id)

            if model_info is None:
                if self.layer_models:
                    valid_models = [info for info in self.layer_models.values() if info is not None]
                    if valid_models:
                        if valid_models[0]['use_interaction']:
                            X_feat = create_interaction_features([m[i]], [X[i]])
                            predictions[i] = valid_models[0]['ols'].predict(X_feat)[0]
                        else:
                            m_poly = valid_models[0]['poly'].transform([[m[i]]])
                            predictions[i] = valid_models[0]['ols'].predict(m_poly)[0]
                    else:
                        predictions[i] = m[i]
                else:
                    predictions[i] = m[i]
            else:
                if model_info['use_interaction']:
                    X_feat = create_interaction_features([m[i]], [X[i]])
                    predictions[i] = model_info['ols'].predict(X_feat)[0]
                else:
                    m_poly = model_info['poly'].transform([[m[i]]])
                    predictions[i] = model_info['ols'].predict(m_poly)[0]

        # GPR 克里金校正
        if self.gpr is not None:
            residual_pred_normalized, _ = self.gpr.predict(X, return_std=True)
            residual_pred = residual_pred_normalized * (self.residual_std + 1e-8) + self.residual_mean
            predictions = predictions + residual_pred

        return predictions


def run_csprkint_ten_fold(selected_day='2020-01-01'):
    """运行 CSP-RK-INT 十折交叉验证"""
    print("=" * 60)
    print("CSP-RK-INT Ten-Fold Cross Validation")
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

        # CSP-RK-INT (with interaction)
        csprkint = CSPRKINT(poly_degree=2, use_interaction=True)
        csprkint.fit(X_train, y_train, m_train)
        csprkint_pred = csprkint.predict(X_test, m_test)

        # CSP-RK (without interaction) for comparison
        csprk = CSPRKINT(poly_degree=2, use_interaction=False)
        csprk.fit(X_train, y_train, m_train)
        csprk_pred = csprk.predict(X_test, m_test)

        results[fold_id] = {
            'y_true': y_test,
            'csprkint': csprkint_pred,
            'csprk': csprk_pred
        }

        print(f"  Fold {fold_id}: INT R2={compute_metrics(y_test, csprkint_pred)['R2']:.4f}, Base R2={compute_metrics(y_test, csprk_pred)['R2']:.4f}")

    # 汇总
    csprkint_all = np.concatenate([results[f]['csprkint'] for f in range(1, 11) if results[f]])
    csprk_all = np.concatenate([results[f]['csprk'] for f in range(1, 11) if results[f]])
    true_all = np.concatenate([results[f]['y_true'] for f in range(1, 11) if results[f]])

    # 计算指标
    print("\n=== Results ===")
    csprkint_metrics = compute_metrics(true_all, csprkint_all)
    csprk_metrics = compute_metrics(true_all, csprk_all)

    print(f"  CSP-RK-INT (Interaction): R2={csprkint_metrics['R2']:.4f}, MAE={csprkint_metrics['MAE']:.2f}, RMSE={csprkint_metrics['RMSE']:.2f}")
    print(f"  CSP-RK (No Interaction):  R2={csprk_metrics['R2']:.4f}, MAE={csprk_metrics['MAE']:.2f}, RMSE={csprk_metrics['RMSE']:.2f}")

    # 保存结果
    result_df = pd.DataFrame([{
        'method': 'CSP_RK_INT',
        **csprkint_metrics
    }, {
        'method': 'CSP_RK_NoINT',
        **csprk_metrics
    }])
    result_df.to_csv(f'{output_dir}/CSPRKINT_summary.csv', index=False)

    print(f"\nResults saved to: {output_dir}/CSPRKINT_summary.csv")

    return csprkint_metrics, csprk_metrics


if __name__ == '__main__':
    csprkint_metrics, csprk_metrics = run_csprkint_ten_fold('2020-01-01')
    print(f"\nCSP-RK-INT: R2={csprkint_metrics['R2']:.4f}")
    print(f"CSP-RK:     R2={csprk_metrics['R2']:.4f}")
