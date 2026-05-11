"""

# === 预测值缓存（由 patch_methods.py 自动添加）===
_last_y_true = None
_last_y_pred = None

CSP-RK-HLG - Hybrid Layer GPR
===============================
CSP-RK with Matern kernel and hybrid global + hierarchical GPR strategy

核心创新：
1. 使用 Matern 核替代 RBF 核
2. 混合策略：全局 GPR + 分层微调

关键公式：
kernel = ConstantKernel(10.0) * Matern(length_scale=15.0, nu=2.5) + WhiteKernel(1.0)

Matern 核特点：
- nu=1.5: 一次可导，线性尾巴
- nu=2.5: 二次可导，常用于物理问题（推荐）
- nu=∞: 退化为 RBF 核
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
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel, RBF
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


class CSPRKHLG:
    """
    CSP-RK-HLG: Hybrid Layer GPR

    分层多项式残差克里金 + Matern核 + 混合全局/分层 GPR
    """

    def __init__(self, poly_degree=2, nu=2.5, hybrid=True):
        self.poly_degree = poly_degree
        self.nu = nu  # Matern 核参数
        self.hybrid = hybrid  # 是否使用混合策略
        self.poly = PolynomialFeatures(degree=poly_degree, include_bias=False)
        self.layer_models = {}
        self.gpr_global = None  # 全局 GPR
        self.gpr_layer = {}  # 分层 GPR
        self.X_train = None
        self.residual_mean = None
        self.residual_std = None

    def _create_matern_kernel(self):
        """创建 Matern 核"""
        return ConstantKernel(10.0, (1e-2, 1e3)) * Matern(
            length_scale=15.0,
            length_scale_bounds=(1e-2, 1e2),
            nu=self.nu
        ) + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e1))

    def _create_rbf_kernel(self):
        """创建 RBF 核（对比用）"""
        return ConstantKernel(10.0, (1e-2, 1e3)) * RBF(
            length_scale=1.0,
            length_scale_bounds=(1e-2, 1e2)
        ) + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e1))

    def fit(self, X, y, m):
        """
        训练 CSP-RK-HLG 模型

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

        # 合并所有残差用于全局 GPR
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

            # 全局 GPR（使用 Matern 核）
            kernel = self._create_matern_kernel()
            self.gpr_global = GaussianProcessRegressor(
                kernel=kernel,
                n_restarts_optimizer=2,
                alpha=0.1,
                normalize_y=True
            )
            self.gpr_global.fit(X_all, residual_normalized)
            self.X_train = X_all

            # 如果使用混合策略，为每层训练分层 GPR
            if self.hybrid:
                self.gpr_layer = {}
                for layer_id, model_info in self.layer_models.items():
                    if model_info is not None and len(model_info['residual']) >= 3:
                        residual_layer = model_info['residual']
                        X_layer = model_info['X']
                        residual_layer_norm = (residual_layer - self.residual_mean) / (self.residual_std + 1e-8)

                        kernel_layer = self._create_matern_kernel()
                        gpr_layer = GaussianProcessRegressor(
                            kernel=kernel_layer,
                            n_restarts_optimizer=2,
                            alpha=0.1,
                            normalize_y=True
                        )
                        gpr_layer.fit(X_layer, residual_layer_norm)
                        self.gpr_layer[layer_id] = gpr_layer

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
                # 如果该层没有模型，使用全局多项式
                if self.layer_models:
                    valid_models = [info for info in self.layer_models.values() if info is not None]
                    if valid_models:
                        m_poly = self.poly.transform([[m[i]]])
                        predictions[i] = valid_models[0]['ols'].predict(m_poly)[0]
                    else:
                        predictions[i] = m[i]
                else:
                    predictions[i] = m[i]
            else:
                m_poly = self.poly.transform([[m[i]]])
                predictions[i] = model_info['ols'].predict(m_poly)[0]

        # GPR 克里金校正
        if self.gpr_global is not None:
            # 全局 GPR 预测
            residual_pred_global, _ = self.gpr_global.predict(X, return_std=True)
            residual_global = residual_pred_global * (self.residual_std + 1e-8) + self.residual_mean

            if self.hybrid and self.gpr_layer:
                # 混合策略：全局 + 分层加权平均
                residual_hybrid = np.zeros(n)
                for i in range(n):
                    layer_id = layers[i]
                    if layer_id in self.gpr_layer:
                        # 使用分层 GPR 预测
                        residual_pred_layer, _ = self.gpr_layer[layer_id].predict([X[i]], return_std=True)
                        residual_layer = residual_pred_layer[0] * (self.residual_std + 1e-8) + self.residual_mean
                        # 混合权重：分层 0.7，全局 0.3
                        residual_hybrid[i] = 0.7 * residual_layer + 0.3 * residual_global[i]
                    else:
                        residual_hybrid[i] = residual_global[i]

                predictions = predictions + residual_hybrid
            else:
                # 仅使用全局 GPR
                predictions = predictions + residual_global

        return predictions


def run_csprkhlg_ten_fold(selected_day='2020-01-01'):
    """运行 CSP-RK-HLG 十折交叉验证"""
    print("=" * 60)
    print("CSP-RK-HLG Ten-Fold Cross Validation")
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

        # CSP-RK-HLG (Matern + Hybrid)
        csprkhlg = CSPRKHLG(poly_degree=2, nu=2.5, hybrid=True)
        csprkhlg.fit(X_train, y_train, m_train)
        csprkhlg_pred = csprkhlg.predict(X_test, m_test)

        # CSP-RK-HLG (Matern, no hybrid)
        csprkhlg_nh = CSPRKHLG(poly_degree=2, nu=2.5, hybrid=False)
        csprkhlg_nh.fit(X_train, y_train, m_train)
        csprkhlg_nh_pred = csprkhlg_nh.predict(X_test, m_test)

        # CSP-RK-HLG (RBF, for comparison)
        csprkhlg_rbf = CSPRKHLG(poly_degree=2, nu=float('inf'), hybrid=False)
        csprkhlg_rbf.fit(X_train, y_train, m_train)
        csprkhlg_rbf_pred = csprkhlg_rbf.predict(X_test, m_test)

        results[fold_id] = {
            'y_true': y_test,
            'csprkhlg': csprkhlg_pred,
            'csprkhlg_nh': csprkhlg_nh_pred,
            'csprkhlg_rbf': csprkhlg_rbf_pred
        }

        print(f"  Fold {fold_id}: HLG={compute_metrics(y_test, csprkhlg_pred)['R2']:.4f}, "
              f"HLG(noHybrid)={compute_metrics(y_test, csprkhlg_nh_pred)['R2']:.4f}, "
              f"RBF={compute_metrics(y_test, csprkhlg_rbf_pred)['R2']:.4f}")

    # 汇总
    csprkhlg_all = np.concatenate([results[f]['csprkhlg'] for f in range(1, 11) if results[f]])
    csprkhlg_nh_all = np.concatenate([results[f]['csprkhlg_nh'] for f in range(1, 11) if results[f]])
    csprkhlg_rbf_all = np.concatenate([results[f]['csprkhlg_rbf'] for f in range(1, 11) if results[f]])
    true_all = np.concatenate([results[f]['y_true'] for f in range(1, 11) if results[f]])

    # 计算指标
    print("\n=== Results ===")
    csprkhlg_metrics = compute_metrics(true_all, csprkhlg_all)
    csprkhlg_nh_metrics = compute_metrics(true_all, csprkhlg_nh_all)
    csprkhlg_rbf_metrics = compute_metrics(true_all, csprkhlg_rbf_all)
    # 缓存预测值供多天聚合使用
    global _last_y_true, _last_y_pred
    _last_y_true = true_all
    _last_y_pred = csprkhlg_rbf_all


    print(f"  CSP-RK-HLG (Matern+Hybrid): R2={csprkhlg_metrics['R2']:.4f}, MAE={csprkhlg_metrics['MAE']:.2f}, RMSE={csprkhlg_metrics['RMSE']:.2f}")
    print(f"  CSP-RK-HLG (Matern):        R2={csprkhlg_nh_metrics['R2']:.4f}, MAE={csprkhlg_nh_metrics['MAE']:.2f}, RMSE={csprkhlg_nh_metrics['RMSE']:.2f}")
    print(f"  CSP-RK-HLG (RBF):           R2={csprkhlg_rbf_metrics['R2']:.4f}, MAE={csprkhlg_rbf_metrics['MAE']:.2f}, RMSE={csprkhlg_rbf_metrics['RMSE']:.2f}")

    # 保存结果
    result_df = pd.DataFrame([{
        'method': 'CSP_RK_HLG_MaternHybrid',
        **csprkhlg_metrics
    }, {
        'method': 'CSP_RK_HLG_Matern',
        **csprkhlg_nh_metrics
    }, {
        'method': 'CSP_RK_HLG_RBF',
        **csprkhlg_rbf_metrics
    }])
    result_df.to_csv(f'{output_dir}/CSPRKHLG_summary.csv', index=False)

    print(f"\nResults saved to: {output_dir}/CSPRKHLG_summary.csv")

    return csprkhlg_metrics, csprkhlg_nh_metrics, csprkhlg_rbf_metrics


if __name__ == '__main__':
    csprkhlg_metrics, csprkhlg_nh_metrics, csprkhlg_rbf_metrics = run_csprkhlg_ten_fold('2020-01-01')
    print(f"\nCSP-RK-HLG (Matern+Hybrid): R2={csprkhlg_metrics['R2']:.4f}")
    print(f"CSP-RK-HLG (Matern):        R2={csprkhlg_nh_metrics['R2']:.4f}")
    print(f"CSP-RK-HLG (RBF):           R2={csprkhlg_rbf_metrics['R2']:.4f}")
