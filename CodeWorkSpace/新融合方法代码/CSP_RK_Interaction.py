"""

# === 预测值缓存（由 patch_methods.py 自动添加）===
_last_y_true = None
_last_y_pred = None

CSP_RK_Interaction - 交互效应增强的浓度分层多项式残差克里金
============================================================
CSP-RK with Interaction Terms (CSP-RK-INT)

核心创新：
1. 在 CSP-RK 基础上添加 CMAQ x Lat 和 CMAQ x Lon 交互项
2. 特征从 [M, M^2] 扩展为 [M, M^2, M*Lat, M*Lon]
3. 允许偏差校正系数随空间位置连续变化，捕捉城市-郊区梯度效应

关键公式：
  O_i(s) = a_i + b_i*M + c_i*M^2 + d_i*(M*Lat) + e_i*(M*Lon) + eps_i

  P(s) = M_cal(s) + R*(s)

其中 M*Lat 捕捉南北梯度（供暖排放），M*Lon 捕捉东西梯度（距海距离）。
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


# ============================================================================
# 工具函数
# ============================================================================
def compute_metrics(y_true, y_pred):
    """计算评估指标"""
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true, y_pred = y_true[mask], y_pred[mask]
    if len(y_true) == 0:
        return {'R2': np.nan, 'MAE': np.nan, 'RMSE': np.nan, 'MB': np.nan}
    return {
        'R2': r2_score(y_true, y_pred),
        'MAE': mean_absolute_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MB': np.mean(y_pred - y_true)
    }


def get_cmaq_at_site(lon, lat, lon_grid, lat_grid, pm25_grid):
    """获取站点位置的 CMAQ 值（最近格点）"""
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


def build_interaction_features(m, X):
    """
    构建带交互项的特征矩阵

    特征: [M, M^2, M*Lon, M*Lat]  (4维，不含截距)

    参数:
        m: CMAQ 浓度 (n,)
        X: 站点坐标 (n, 2) [Lon, Lat]

    返回:
        features: (n, 4) 扩展特征矩阵
    """
    m = np.asarray(m, dtype=float).reshape(-1, 1)
    X = np.asarray(X, dtype=float)
    lon = X[:, 0].reshape(-1, 1)
    lat = X[:, 1].reshape(-1, 1)

    # 基础多项式: [M, M^2]
    m_poly = np.hstack([m, m ** 2])

    # 交互项: [M*Lon, M*Lat]
    m_lon = m * lon
    m_lat = m * lat

    return np.hstack([m_poly, m_lon, m_lat])


# ============================================================================
# 核心模型类
# ============================================================================
class CSP_RK_Interaction:
    """
    CSP-RK-INT: 交互效应增强的浓度分层多项式残差克里金

    模型流程:
        1. 按浓度分层，每层用 [M, M^2, M*Lon, M*Lat] 特征拟合 OLS
        2. 合并所有层残差，训练全局 GPR 克里金
        3. 预测时按浓度选择对应层 OLS，叠加 GPR 残差校正
    """

    def __init__(self, poly_degree=2, use_interaction=True, T1=35.0, T2=75.0):
        """
        参数:
            poly_degree: 多项式阶数（仅在 use_interaction=False 时使用）
            use_interaction: 是否使用交互项
            T1, T2: 浓度分层阈值
        """
        self.poly_degree = poly_degree
        self.use_interaction = use_interaction
        self.T1 = T1
        self.T2 = T2
        self.poly = PolynomialFeatures(degree=poly_degree, include_bias=False)
        self.layer_models = {}
        self.gpr = None
        self.residual_mean = 0.0
        self.residual_std = 1.0

    def fit(self, X, y, m):
        """
        训练 CSP-RK-INT 模型

        参数:
            X: 站点坐标 (n, 2) [Lon, Lat]
            y: 监测浓度 (n,)
            m: CMAQ 浓度 (n,)
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        m = np.asarray(m, dtype=float)

        # 按浓度分层
        layers = np.where(m < self.T1, 0, np.where(m < self.T2, 1, 2))

        all_residuals = []
        all_X = []

        for layer_id in [0, 1, 2]:
            mask = layers == layer_id
            # 交互特征有 4 维，需要至少 5 个样本
            min_samples = 5 if self.use_interaction else 3
            if np.sum(mask) < min_samples:
                self.layer_models[layer_id] = None
                continue

            m_layer = m[mask]
            y_layer = y[mask]
            X_layer = X[mask]

            if self.use_interaction:
                # 交互特征: [M, M^2, M*Lon, M*Lat]
                X_feat = build_interaction_features(m_layer, X_layer)
                ols = LinearRegression()
                ols.fit(X_feat, y_layer)
                residual_layer = y_layer - ols.predict(X_feat)
                self.layer_models[layer_id] = {
                    'ols': ols,
                    'use_interaction': True,
                }
            else:
                # 基础多项式: [M, M^2]
                m_poly = self.poly.fit_transform(m_layer.reshape(-1, 1))
                ols = LinearRegression()
                ols.fit(m_poly, y_layer)
                residual_layer = y_layer - ols.predict(m_poly)
                # 存储 poly 以便 predict 时 transform
                poly_layer = PolynomialFeatures(degree=self.poly_degree, include_bias=False)
                poly_layer.fit(m_layer.reshape(-1, 1))
                self.layer_models[layer_id] = {
                    'ols': ols,
                    'use_interaction': False,
                    'poly': poly_layer,
                }

            all_residuals.append(residual_layer)
            all_X.append(X_layer)

        # 合并残差训练全局 GPR
        if len(all_residuals) == 0:
            return

        residual_all = np.concatenate(all_residuals)
        X_all = np.vstack(all_X)

        self.residual_mean = np.mean(residual_all)
        self.residual_std = np.std(residual_all) + 1e-8
        residual_norm = (residual_all - self.residual_mean) / self.residual_std

        kernel = (ConstantKernel(10.0, (1e-2, 1e3))
                  * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
                  + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e1)))
        self.gpr = GaussianProcessRegressor(
            kernel=kernel, n_restarts_optimizer=2, alpha=0.1,
            normalize_y=True
        )
        self.gpr.fit(X_all, residual_norm)

    def predict(self, X, m):
        """
        预测

        参数:
            X: 站点坐标 (n, 2)
            m: CMAQ 浓度 (n,)

        返回:
            融合预测浓度 (n,)
        """
        X = np.asarray(X, dtype=float)
        m = np.asarray(m, dtype=float)
        n = len(m)

        layers = np.where(m < self.T1, 0, np.where(m < self.T2, 1, 2))

        # 各层多项式预测
        M_cal = np.zeros(n)
        for layer_id in [0, 1, 2]:
            mask = layers == layer_id
            if np.sum(mask) == 0:
                continue

            model_info = self.layer_models.get(layer_id)
            if model_info is not None:
                if model_info['use_interaction']:
                    X_feat = build_interaction_features(m[mask], X[mask])
                    M_cal[mask] = model_info['ols'].predict(X_feat)
                else:
                    m_poly = model_info['poly'].transform(m[mask].reshape(-1, 1))
                    M_cal[mask] = model_info['ols'].predict(m_poly)
            else:
                # 回退到任意有效层
                for lid, minfo in self.layer_models.items():
                    if minfo is not None:
                        if minfo['use_interaction']:
                            X_feat = build_interaction_features(m[mask], X[mask])
                            M_cal[mask] = minfo['ols'].predict(X_feat)
                        else:
                            m_poly = minfo['poly'].transform(m[mask].reshape(-1, 1))
                            M_cal[mask] = minfo['ols'].predict(m_poly)
                        break

        # GPR 残差校正
        if self.gpr is not None:
            residual_norm, _ = self.gpr.predict(X, return_std=True)
            residual_pred = residual_norm * self.residual_std + self.residual_mean
            M_cal = M_cal + residual_pred

        return M_cal


# ============================================================================
# 十折交叉验证主函数
# ============================================================================
def run_CSP_RK_Interaction_ten_fold(selected_day='2020-01-01'):
    """
    CSP-RK-INT 十折交叉验证

    对比两种配置:
        1. CSP-RK-INT: 含交互项 [M, M^2, M*Lon, M*Lat]
        2. CSP-RK:     无交互项 [M, M^2]（对比基线）

    参数:
        selected_day: 验证日期

    返回:
        metrics_int, metrics_base
    """
    print("=" * 60)
    print("CSP-RK-INT (Interaction) Ten-Fold Cross Validation")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. 加载数据
    # ------------------------------------------------------------------
    print("\n=== Loading Data ===")
    monitor_df = pd.read_csv(monitor_file)
    fold_df = pd.read_csv(fold_file)

    day_df = monitor_df[monitor_df['Date'] == selected_day].copy()
    day_df = day_df.merge(fold_df, on=['Date', 'Site'], how='left')
    day_df = day_df.dropna(subset=['Lat', 'Lon', 'Conc'])

    # 加载 CMAQ
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

    print(f"Data loaded: {len(day_df)} monitoring records")

    # ------------------------------------------------------------------
    # 2. 十折交叉验证
    # ------------------------------------------------------------------
    print("\n=== Running 10-fold Cross Validation ===")
    results = {}

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

        # --- CSP-RK-INT (含交互项) ---
        model_int = CSP_RK_Interaction(
            poly_degree=2, use_interaction=True, T1=T1, T2=T2
        )
        model_int.fit(X_train, y_train, m_train)
        pred_int = model_int.predict(X_test, m_test)

        # --- CSP-RK (无交互项，对比基线) ---
        model_base = CSP_RK_Interaction(
            poly_degree=2, use_interaction=False, T1=T1, T2=T2
        )
        model_base.fit(X_train, y_train, m_train)
        pred_base = model_base.predict(X_test, m_test)

        results[fold_id] = {
            'y_true': y_test,
            'pred_int': pred_int,
            'pred_base': pred_base,
        }

        r2_i = compute_metrics(y_test, pred_int)['R2']
        r2_b = compute_metrics(y_test, pred_base)['R2']
        print(f"  Fold {fold_id:2d}: INT R2={r2_i:.4f}, Base R2={r2_b:.4f}")

    # ------------------------------------------------------------------
    # 3. 汇总
    # ------------------------------------------------------------------
    valid_folds = [f for f in range(1, 11) if f in results]
    if len(valid_folds) == 0:
        print("ERROR: No valid folds!")
        nan_m = {'R2': np.nan, 'MAE': np.nan, 'RMSE': np.nan, 'MB': np.nan}
        return nan_m, nan_m

    true_all = np.concatenate([results[f]['y_true'] for f in valid_folds])
    int_all = np.concatenate([results[f]['pred_int'] for f in valid_folds])
    base_all = np.concatenate([results[f]['pred_base'] for f in valid_folds])

    metrics_int = compute_metrics(true_all, int_all)
    metrics_base = compute_metrics(true_all, base_all)
    # 缓存预测值供多天聚合使用
    global _last_y_true, _last_y_pred
    _last_y_true = true_all
    _last_y_pred = base_all


    print("\n=== Results ===")
    print(f"  CSP-RK-INT (Interaction): R2={metrics_int['R2']:.4f}, "
          f"MAE={metrics_int['MAE']:.2f}, RMSE={metrics_int['RMSE']:.2f}")
    print(f"  CSP-RK (No Interaction):  R2={metrics_base['R2']:.4f}, "
          f"MAE={metrics_base['MAE']:.2f}, RMSE={metrics_base['RMSE']:.2f}")

    # 保存结果
    result_df = pd.DataFrame([
        {'method': 'CSP_RK_Interaction', **metrics_int},
        {'method': 'CSP_RK_NoInteraction', **metrics_base},
    ])
    out_path = f'{output_dir}/CSP_RK_Interaction_summary.csv'
    result_df.to_csv(out_path, index=False)
    print(f"\nResults saved to: {out_path}")

    return metrics_int, metrics_base


if __name__ == '__main__':
    m_int, m_base = run_CSP_RK_Interaction_ten_fold('2020-01-01')
    print(f"\nCSP-RK-INT: R2={m_int['R2']:.4f}")
    print(f"CSP-RK:     R2={m_base['R2']:.4f}")
