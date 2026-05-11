"""

# === 预测值缓存（由 patch_methods.py 自动添加）===
_last_y_true = None
_last_y_pred = None

CSP_RK_AdaptiveThreshold - 自适应阈值优化的浓度分层多项式残差克里金
==================================================================
CSP-RK with Adaptive Threshold Optimization (CSP-RK-ATO)

核心创新：
1. 阈值 T1, T2 从固定值改为网格搜索优化（数据驱动）
2. 硬边界改为 softmax 风格平滑过渡权重，消除边界不连续性
3. 每层独立做多项式 OLS 校正，加权平滑过渡融合

关键公式：
  w_low  = 1 / (1 + exp(kappa * (m - T1)))
  w_high = 1 / (1 + exp(-kappa * (m - T2)))
  w_mid  = 1 - w_low - w_high

  P(s) = w_low * O1(s) + w_mid * O2(s) + w_high * O3(s) + R*(s)

其中 O_i 为各层多项式校正，R* 为 GPR 残差克里金。
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


# ============================================================================
# 阈值候选空间
# ============================================================================
T1_CANDIDATES = [25, 30, 35, 40, 45]
T2_CANDIDATES = [60, 65, 70, 75, 80, 85, 90]
DEFAULT_T1 = 35.0
DEFAULT_T2 = 75.0
DEFAULT_KAPPA = 0.1


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


def compute_softmax_weights(m, T1, T2, kappa):
    """
    计算 softmax 风格的三层平滑权重

    w_low  = sigmoid(-kappa * (m - T1)) = 1 / (1 + exp(kappa * (m - T1)))
    w_high = sigmoid( kappa * (m - T2)) = 1 / (1 + exp(-kappa * (m - T2)))
    w_mid  = 1 - w_low - w_high

    参数:
        m: CMAQ 浓度值 (n,)
        T1: 低-中浓度阈值
        T2: 中-高浓度阈值
        kappa: 平滑过渡宽度参数

    返回:
        w_low, w_mid, w_high: 各 (n,) 权重数组
    """
    m = np.asarray(m, dtype=float)
    w_low = 1.0 / (1.0 + np.exp(kappa * (m - T1)))
    w_high = 1.0 / (1.0 + np.exp(-kappa * (m - T2)))
    w_mid = 1.0 - w_low - w_high
    # 确保非负并归一化
    w_low = np.maximum(w_low, 0.0)
    w_mid = np.maximum(w_mid, 0.0)
    w_high = np.maximum(w_high, 0.0)
    w_sum = w_low + w_mid + w_high + 1e-10
    w_low /= w_sum
    w_mid /= w_sum
    w_high /= w_sum
    return w_low, w_mid, w_high


# ============================================================================
# 核心模型类
# ============================================================================
class CSP_RK_AdaptiveThreshold:
    """
    CSP-RK-ATO: 自适应阈值优化的浓度分层多项式残差克里金

    模型流程:
        1. 网格搜索确定最优 (T1*, T2*) 阈值
        2. 按最优阈值分层，每层独立拟合二次多项式 OLS
        3. 预测时使用 softmax 平滑权重加权各层预测
        4. 合并残差做全局 GPR 克里金校正
    """

    def __init__(self, T1=35.0, T2=75.0, kappa=0.1, poly_degree=2):
        self.T1 = T1
        self.T2 = T2
        self.kappa = kappa
        self.poly_degree = poly_degree
        self.poly = PolynomialFeatures(degree=poly_degree, include_bias=False)
        self.layer_models = {}   # {layer_id: {'ols': ..., 'poly': ...}}
        self.gpr = None
        self.residual_mean = 0.0
        self.residual_std = 1.0

    def fit(self, X, y, m):
        """
        训练 CSP-RK-ATO 模型

        参数:
            X: 站点坐标 (n, 2) [Lon, Lat]
            y: 监测浓度 (n,)
            m: CMAQ 浓度 (n,)
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        m = np.asarray(m, dtype=float)

        # 按固定阈值分层（训练时用硬分层）
        layers = np.where(m < self.T1, 0, np.where(m < self.T2, 1, 2))

        all_residuals = []
        all_X = []

        for layer_id in [0, 1, 2]:
            mask = layers == layer_id
            if np.sum(mask) < 3:
                self.layer_models[layer_id] = None
                continue

            m_layer = m[mask]
            y_layer = y[mask]
            X_layer = X[mask]

            m_poly = self.poly.fit_transform(m_layer.reshape(-1, 1))
            ols = LinearRegression()
            ols.fit(m_poly, y_layer)
            residual_layer = y_layer - ols.predict(m_poly)

            self.layer_models[layer_id] = {
                'ols': ols,
                'poly': PolynomialFeatures(degree=self.poly_degree, include_bias=False)
            }
            # 重新 fit poly 以便 predict 时 transform
            self.layer_models[layer_id]['poly'].fit(m_layer.reshape(-1, 1))

            all_residuals.append(residual_layer)
            all_X.append(X_layer)

        # 全局 GPR 残差克里金
        if len(all_residuals) > 0:
            residual_all = np.concatenate(all_residuals)
            X_all = np.vstack(all_X)

            self.residual_mean = np.mean(residual_all)
            self.residual_std = np.std(residual_all) + 1e-8
            residual_norm = (residual_all - self.residual_mean) / self.residual_std

            kernel = (ConstantKernel(10.0, (1e-2, 1e3))
                      * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
                      + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e1)))
            self.gpr = GaussianProcessRegressor(
                kernel=kernel, n_restarts_optimizer=2, alpha=0.1, normalize_y=True
            )
            self.gpr.fit(X_all, residual_norm)

    def predict(self, X, m):
        """
        预测（使用 softmax 平滑权重加权）

        参数:
            X: 站点坐标 (n, 2)
            m: CMAQ 浓度 (n,)

        返回:
            融合预测浓度 (n,)
        """
        X = np.asarray(X, dtype=float)
        m = np.asarray(m, dtype=float)
        n = len(m)

        # 计算平滑权重
        w_low, w_mid, w_high = compute_softmax_weights(m, self.T1, self.T2, self.kappa)

        # 各层多项式预测
        pred_layers = np.zeros((3, n))
        for layer_id in [0, 1, 2]:
            model_info = self.layer_models.get(layer_id)
            if model_info is not None:
                m_poly = model_info['poly'].transform(m.reshape(-1, 1))
                pred_layers[layer_id] = model_info['ols'].predict(m_poly)

        # 加权融合
        O_weighted = w_low * pred_layers[0] + w_mid * pred_layers[1] + w_high * pred_layers[2]

        # GPR 残差校正
        if self.gpr is not None:
            residual_norm, _ = self.gpr.predict(X, return_std=True)
            residual_pred = residual_norm * self.residual_std + self.residual_mean
            O_weighted = O_weighted + residual_pred

        return O_weighted


# ============================================================================
# 网格搜索最优阈值
# ============================================================================
def grid_search_thresholds(X_train, y_train, m_train, fold_ids,
                           T1_cands, T2_cands, kappa=0.1):
    """
    在训练集上用内部折验证搜索最优阈值组合

    参数:
        X_train: (n, 2)
        y_train: (n,)
        m_train: (n,)
        fold_ids: (n,) 每个样本的折号
        T1_cands, T2_cands: 候选阈值列表
        kappa: 平滑参数

    返回:
        best_T1, best_T2, best_r2
    """
    best_r2 = -np.inf
    best_T1, best_T2 = DEFAULT_T1, DEFAULT_T2
    unique_folds = np.unique(fold_ids)

    for T1 in T1_cands:
        for T2 in T2_cands:
            if T1 >= T2:
                continue
            r2_scores = []
            for val_fold in unique_folds:
                val_mask = fold_ids == val_fold
                tr_mask = ~val_mask
                if np.sum(val_mask) < 2 or np.sum(tr_mask) < 5:
                    continue

                model = CSP_RK_AdaptiveThreshold(T1=T1, T2=T2, kappa=kappa)
                model.fit(X_train[tr_mask], y_train[tr_mask], m_train[tr_mask])
                y_pred = model.predict(X_train[val_mask], m_train[val_mask])
                r2_scores.append(r2_score(y_train[val_mask], y_pred))

            if len(r2_scores) > 0:
                mean_r2 = np.mean(r2_scores)
                if mean_r2 > best_r2:
                    best_r2 = mean_r2
                    best_T1, best_T2 = T1, T2

    return best_T1, best_T2, best_r2


# ============================================================================
# 十折交叉验证主函数
# ============================================================================
def run_CSP_RK_AdaptiveThreshold_ten_fold(selected_day='2020-01-01'):
    """
    CSP-RK-ATO 十折交叉验证

    流程:
        1. 加载监测数据、CMAQ 数据、折分配表
        2. 对每折: 网格搜索最优阈值 -> 训练 -> 预测
        3. 汇总全部折的预测结果，计算指标

    参数:
        selected_day: 验证日期，默认 '2020-01-01'

    返回:
        metrics_adaptive: 自适应阈值指标
        metrics_fixed: 固定阈值指标（对比基线）
    """
    print("=" * 60)
    print("CSP-RK-ATO (Adaptive Threshold) Ten-Fold Cross Validation")
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

        # --- 自适应阈值 ---
        # 在训练集内部用留折法搜索最优阈值
        train_fold_ids = train_df['fold'].values
        best_T1, best_T2, _ = grid_search_thresholds(
            X_train, y_train, m_train, train_fold_ids,
            T1_CANDIDATES, T2_CANDIDATES, kappa=DEFAULT_KAPPA
        )

        model_adaptive = CSP_RK_AdaptiveThreshold(
            T1=best_T1, T2=best_T2, kappa=DEFAULT_KAPPA
        )
        model_adaptive.fit(X_train, y_train, m_train)
        pred_adaptive = model_adaptive.predict(X_test, m_test)

        # --- 固定阈值（对比基线）---
        model_fixed = CSP_RK_AdaptiveThreshold(
            T1=DEFAULT_T1, T2=DEFAULT_T2, kappa=DEFAULT_KAPPA
        )
        model_fixed.fit(X_train, y_train, m_train)
        pred_fixed = model_fixed.predict(X_test, m_test)

        results[fold_id] = {
            'y_true': y_test,
            'pred_adaptive': pred_adaptive,
            'pred_fixed': pred_fixed,
            'best_T1': best_T1,
            'best_T2': best_T2,
        }

        r2_adp = compute_metrics(y_test, pred_adaptive)['R2']
        r2_fix = compute_metrics(y_test, pred_fixed)['R2']
        print(f"  Fold {fold_id:2d}: T1*={best_T1:.0f}, T2*={best_T2:.0f} | "
              f"Adaptive R2={r2_adp:.4f}, Fixed R2={r2_fix:.4f}")

    # ------------------------------------------------------------------
    # 3. 汇总
    # ------------------------------------------------------------------
    valid_folds = [f for f in range(1, 11) if f in results]
    if len(valid_folds) == 0:
        print("ERROR: No valid folds!")
        return {'R2': np.nan, 'MAE': np.nan, 'RMSE': np.nan, 'MB': np.nan}, \
               {'R2': np.nan, 'MAE': np.nan, 'RMSE': np.nan, 'MB': np.nan}

    true_all = np.concatenate([results[f]['y_true'] for f in valid_folds])
    adaptive_all = np.concatenate([results[f]['pred_adaptive'] for f in valid_folds])
    fixed_all = np.concatenate([results[f]['pred_fixed'] for f in valid_folds])

    metrics_adaptive = compute_metrics(true_all, adaptive_all)
    metrics_fixed = compute_metrics(true_all, fixed_all)
    # 缓存预测值供多天聚合使用
    global _last_y_true, _last_y_pred
    _last_y_true = true_all
    _last_y_pred = fixed_all


    print("\n=== Results ===")
    print(f"  CSP-RK-ATO (Adaptive): R2={metrics_adaptive['R2']:.4f}, "
          f"MAE={metrics_adaptive['MAE']:.2f}, RMSE={metrics_adaptive['RMSE']:.2f}")
    print(f"  CSP-RK-ATO (Fixed):    R2={metrics_fixed['R2']:.4f}, "
          f"MAE={metrics_fixed['MAE']:.2f}, RMSE={metrics_fixed['RMSE']:.2f}")

    # 保存结果
    result_df = pd.DataFrame([
        {'method': 'CSP_RK_AdaptiveThreshold_Adaptive', **metrics_adaptive},
        {'method': 'CSP_RK_AdaptiveThreshold_Fixed', **metrics_fixed},
    ])
    out_path = f'{output_dir}/CSP_RK_AdaptiveThreshold_summary.csv'
    result_df.to_csv(out_path, index=False)
    print(f"\nResults saved to: {out_path}")

    return metrics_adaptive, metrics_fixed


if __name__ == '__main__':
    m_adp, m_fix = run_CSP_RK_AdaptiveThreshold_ten_fold('2020-01-01')
    print(f"\nAdaptive: R2={m_adp['R2']:.4f}")
    print(f"Fixed:    R2={m_fix['R2']:.4f}")
