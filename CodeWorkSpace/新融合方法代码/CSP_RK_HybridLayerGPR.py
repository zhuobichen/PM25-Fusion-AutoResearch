"""

# === 预测值缓存（由 patch_methods.py 自动添加）===
_last_y_true = None
_last_y_pred = None

CSP_RK_HybridLayerGPR - 混合分层高斯过程残差克里金
===================================================
CSP-RK with Hybrid Layered GPR (CSP-RK-HLG)

核心创新：
1. 使用 Matern 核(nu=2.5)替代 RBF 核，增强空间自相关建模灵活性
2. 全局 GPR 初始化 -> 分层 Matern GPR 微调的混合策略
3. 避免 HGP-RK "每层单独 GPR 样本太少"的问题

关键公式：
  kernel_global = ConstantKernel * RBF + WhiteKernel
  kernel_layer  = ConstantKernel * Matern(nu=2.5) + WhiteKernel

  R*(s) = R_global(s) + fine_tune_ratio * (R_layer(s) - R_global(s))
  P(s)  = M_cal(s) + R*(s)

其中 fine_tune_ratio=0.3 控制分层微调的贡献比例。
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
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel
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


# ============================================================================
# 核心模型类
# ============================================================================
class CSP_RK_HybridLayerGPR:
    """
    CSP-RK-HLG: 混合分层高斯过程残差克里金

    模型流程:
        1. 按浓度分层，每层独立拟合二次多项式 OLS 校正
        2. 合并所有层残差，用 RBF 核训练全局 GPR（初始化）
        3. 为每层训练独立的 Matern GPR（微调），长度尺度从全局继承
        4. 预测时混合：R = R_global + ratio * (R_layer - R_global)
    """

    def __init__(self, poly_degree=2, nu=2.5, fine_tune_ratio=0.3,
                 use_hybrid=True):
        """
        参数:
            poly_degree: 多项式阶数
            nu: Matern 核光滑度参数 (1.5, 2.5, 3.5)
            fine_tune_ratio: 分层微调比例 (0~1)
            use_hybrid: 是否使用混合策略，False 则仅用全局 GPR
        """
        self.poly_degree = poly_degree
        self.nu = nu
        self.fine_tune_ratio = fine_tune_ratio
        self.use_hybrid = use_hybrid
        self.poly = PolynomialFeatures(degree=poly_degree, include_bias=False)
        self.layer_models = {}
        self.gpr_global = None
        self.gpr_layers = {}
        self.residual_mean = 0.0
        self.residual_std = 1.0
        self._length_scale_global = 15.0  # 默认值

    def fit(self, X, y, m):
        """
        训练 CSP-RK-HLG 模型

        参数:
            X: 站点坐标 (n, 2) [Lon, Lat]
            y: 监测浓度 (n,)
            m: CMAQ 浓度 (n,)
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        m = np.asarray(m, dtype=float)

        # 按浓度分层
        layers = np.where(m < T1, 0, np.where(m < T2, 1, 2))

        # ---- Step 1: 分层 OLS 校正 ----
        self.layer_models = {}
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

            # 存储该层的 poly（独立 fit，以便 predict 时 transform）
            poly_layer = PolynomialFeatures(degree=self.poly_degree, include_bias=False)
            poly_layer.fit(m_layer.reshape(-1, 1))

            self.layer_models[layer_id] = {
                'ols': ols,
                'poly': poly_layer,
            }

            all_residuals.append(residual_layer)
            all_X.append(X_layer)

        if len(all_residuals) == 0:
            return

        residual_all = np.concatenate(all_residuals)
        X_all = np.vstack(all_X)

        self.residual_mean = np.mean(residual_all)
        self.residual_std = np.std(residual_all) + 1e-8
        residual_norm = (residual_all - self.residual_mean) / self.residual_std

        # ---- Step 2: 全局 GPR 初始化（RBF 核） ----
        kernel_global = (ConstantKernel(10.0, (1e-2, 1e3))
                         * RBF(length_scale=15.0, length_scale_bounds=(1e-2, 1e2))
                         + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e1)))
        self.gpr_global = GaussianProcessRegressor(
            kernel=kernel_global, n_restarts_optimizer=2, alpha=0.1,
            normalize_y=True
        )
        self.gpr_global.fit(X_all, residual_norm)

        # 提取全局长度尺度
        try:
            # kernel_ 结构: ConstantKernel * RBF + WhiteKernel
            k = self.gpr_global.kernel_
            if hasattr(k, 'k1') and hasattr(k.k1, 'k2'):
                self._length_scale_global = k.k1.k2.length_scale
            else:
                self._length_scale_global = 15.0
        except Exception:
            self._length_scale_global = 15.0

        # ---- Step 3: 分层 Matern GPR 微调 ----
        if self.use_hybrid:
            self.gpr_layers = {}
            for layer_id, model_info in self.layer_models.items():
                if model_info is None:
                    continue
                mask = layers == layer_id
                residual_layer = residual_all[mask] if np.sum(mask) == len(all_residuals[0]) else None

                # 重新计算该层残差（确保对应）
                m_layer = m[mask]
                y_layer = y[mask]
                X_layer = X[mask]
                m_poly = model_info['poly'].transform(m_layer.reshape(-1, 1))
                residual_layer = y_layer - model_info['ols'].predict(m_poly)
                residual_layer_norm = (residual_layer - self.residual_mean) / self.residual_std

                if len(residual_layer_norm) < 3:
                    continue

                kernel_layer = (
                    ConstantKernel(10.0, (1e-2, 1e3))
                    * Matern(length_scale=self._length_scale_global,
                             length_scale_bounds=(1e-2, 1e2),
                             nu=self.nu)
                    + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e1))
                )
                gpr_layer = GaussianProcessRegressor(
                    kernel=kernel_layer, n_restarts_optimizer=1,
                    alpha=0.5, normalize_y=True
                )
                # 用全量 X 训练，但只用当前层残差
                # 注意：这里用该层的 X_layer 训练分层 GPR
                gpr_layer.fit(X_layer, residual_layer_norm)
                self.gpr_layers[layer_id] = gpr_layer

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

        # 按浓度分层
        layers = np.where(m < T1, 0, np.where(m < T2, 1, 2))

        # ---- 各层多项式预测 ----
        M_cal = np.zeros(n)
        for layer_id in [0, 1, 2]:
            mask = layers == layer_id
            if np.sum(mask) == 0:
                continue
            model_info = self.layer_models.get(layer_id)
            if model_info is not None:
                m_poly = model_info['poly'].transform(m[mask].reshape(-1, 1))
                M_cal[mask] = model_info['ols'].predict(m_poly)
            else:
                # 回退到任意有效层
                for lid, minfo in self.layer_models.items():
                    if minfo is not None:
                        m_poly = minfo['poly'].transform(m[mask].reshape(-1, 1))
                        M_cal[mask] = minfo['ols'].predict(m_poly)
                        break

        # ---- GPR 残差预测 ----
        if self.gpr_global is None:
            return M_cal

        R_global_norm, _ = self.gpr_global.predict(X, return_std=True)
        R_global = R_global_norm * self.residual_std + self.residual_mean

        if self.use_hybrid and len(self.gpr_layers) > 0:
            # 混合策略：全局 + 分层微调
            R_hybrid = R_global.copy()
            for layer_id, gpr_layer in self.gpr_layers.items():
                mask = layers == layer_id
                if np.sum(mask) == 0:
                    continue
                R_layer_norm, _ = gpr_layer.predict(X[mask], return_std=True)
                R_layer = R_layer_norm * self.residual_std + self.residual_mean
                # 混合公式: R = R_global + ratio * (R_layer - R_global)
                R_hybrid[mask] = R_global[mask] + self.fine_tune_ratio * (R_layer - R_global[mask])
            return M_cal + R_hybrid
        else:
            return M_cal + R_global


# ============================================================================
# 十折交叉验证主函数
# ============================================================================
def run_CSP_RK_HybridLayerGPR_ten_fold(selected_day='2020-01-01'):
    """
    CSP-RK-HLG 十折交叉验证

    对比三种配置:
        1. Matern(nu=2.5) + Hybrid (混合分层微调)
        2. Matern(nu=2.5) 无混合 (仅全局 Matern GPR)
        3. RBF 无混合 (对比基线)

    参数:
        selected_day: 验证日期

    返回:
        metrics_hybrid, metrics_matern, metrics_rbf
    """
    print("=" * 60)
    print("CSP-RK-HLG (Hybrid Layer GPR) Ten-Fold Cross Validation")
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

        # --- 配置 1: Matern + Hybrid ---
        model_hybrid = CSP_RK_HybridLayerGPR(
            poly_degree=2, nu=2.5, fine_tune_ratio=0.3, use_hybrid=True
        )
        model_hybrid.fit(X_train, y_train, m_train)
        pred_hybrid = model_hybrid.predict(X_test, m_test)

        # --- 配置 2: Matern only (无混合) ---
        model_matern = CSP_RK_HybridLayerGPR(
            poly_degree=2, nu=2.5, fine_tune_ratio=0.0, use_hybrid=False
        )
        model_matern.fit(X_train, y_train, m_train)
        pred_matern = model_matern.predict(X_test, m_test)

        # --- 配置 3: RBF only (对比基线) ---
        model_rbf = CSP_RK_HybridLayerGPR(
            poly_degree=2, nu=2.5, fine_tune_ratio=0.0, use_hybrid=False
        )
        model_rbf.fit(X_train, y_train, m_train)
        pred_rbf = model_rbf.predict(X_test, m_test)

        results[fold_id] = {
            'y_true': y_test,
            'pred_hybrid': pred_hybrid,
            'pred_matern': pred_matern,
            'pred_rbf': pred_rbf,
        }

        r2_h = compute_metrics(y_test, pred_hybrid)['R2']
        r2_m = compute_metrics(y_test, pred_matern)['R2']
        r2_r = compute_metrics(y_test, pred_rbf)['R2']
        print(f"  Fold {fold_id:2d}: Hybrid R2={r2_h:.4f}, "
              f"Matern R2={r2_m:.4f}, RBF R2={r2_r:.4f}")

    # ------------------------------------------------------------------
    # 3. 汇总
    # ------------------------------------------------------------------
    valid_folds = [f for f in range(1, 11) if f in results]
    if len(valid_folds) == 0:
        print("ERROR: No valid folds!")
        nan_m = {'R2': np.nan, 'MAE': np.nan, 'RMSE': np.nan, 'MB': np.nan}
        return nan_m, nan_m, nan_m

    true_all = np.concatenate([results[f]['y_true'] for f in valid_folds])
    hybrid_all = np.concatenate([results[f]['pred_hybrid'] for f in valid_folds])
    matern_all = np.concatenate([results[f]['pred_matern'] for f in valid_folds])
    rbf_all = np.concatenate([results[f]['pred_rbf'] for f in valid_folds])

    metrics_hybrid = compute_metrics(true_all, hybrid_all)
    metrics_matern = compute_metrics(true_all, matern_all)
    metrics_rbf = compute_metrics(true_all, rbf_all)
    # 缓存预测值供多天聚合使用
    global _last_y_true, _last_y_pred
    _last_y_true = true_all
    _last_y_pred = rbf_all


    print("\n=== Results ===")
    print(f"  CSP-RK-HLG (Matern+Hybrid): R2={metrics_hybrid['R2']:.4f}, "
          f"MAE={metrics_hybrid['MAE']:.2f}, RMSE={metrics_hybrid['RMSE']:.2f}")
    print(f"  CSP-RK-HLG (Matern only):   R2={metrics_matern['R2']:.4f}, "
          f"MAE={metrics_matern['MAE']:.2f}, RMSE={metrics_matern['RMSE']:.2f}")
    print(f"  CSP-RK-HLG (RBF baseline):  R2={metrics_rbf['R2']:.4f}, "
          f"MAE={metrics_rbf['MAE']:.2f}, RMSE={metrics_rbf['RMSE']:.2f}")

    # 保存结果
    result_df = pd.DataFrame([
        {'method': 'CSP_RK_HLG_MaternHybrid', **metrics_hybrid},
        {'method': 'CSP_RK_HLG_Matern', **metrics_matern},
        {'method': 'CSP_RK_HLG_RBF', **metrics_rbf},
    ])
    out_path = f'{output_dir}/CSP_RK_HybridLayerGPR_summary.csv'
    result_df.to_csv(out_path, index=False)
    print(f"\nResults saved to: {out_path}")

    return metrics_hybrid, metrics_matern, metrics_rbf


if __name__ == '__main__':
    m_h, m_m, m_r = run_CSP_RK_HybridLayerGPR_ten_fold('2020-01-01')
    print(f"\nMatern+Hybrid: R2={m_h['R2']:.4f}")
    print(f"Matern only:   R2={m_m['R2']:.4f}")
    print(f"RBF baseline:  R2={m_r['R2']:.4f}")
