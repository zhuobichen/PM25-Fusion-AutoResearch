"""
# === 预测值缓存（由 patch_methods.py 自动添加）===
_last_y_true = None
_last_y_pred = None

CSP-AdvancedRK - Adaptive Concentration-Stratified Advanced RK
==============================================================
结合 CSPRK 的自适应分层和 AdvancedRK 的 Matern 核函数的优势

核心创新：
1. 自适应分层：根据训练数据分位数自动确定分层阈值
2. 每层独立做多项式 OLS 校正
3. 使用 Matern(ν=1.5) 核函数进行 GPR 残差建模（来自 AdvancedRK）
4. 合并残差，做统一 GPR 克里金
5. 预测时按测试点浓度选择对应层的 OLS
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.paths import get_project_root, data_path
from shared.metrics import compute_metrics
from shared.data_utils import get_project_paths, get_cmaq_at_site, load_daily_data, get_cmaq_grid
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel

paths = get_project_paths()


def get_adaptive_thresholds(m_values, n_layers=3):
    """
    根据训练数据的分位数自适应确定分层阈值

    参数：
    m_values: CMAQ 浓度值数组
    n_layers: 分层数量，默认 3

    返回：
    thresholds: 阈值数组，长度为 n_layers-1
    """
    if n_layers < 2:
        return []
    
    # 计算分位数
    quantiles = np.linspace(0, 1, n_layers + 1)
    thresholds = np.quantile(m_values, quantiles[1:-1])
    
    # 确保阈值是严格递增的
    thresholds = np.sort(thresholds)
    thresholds = np.unique(thresholds)
    
    # 如果唯一值不够，补充一些中间值
    while len(thresholds) < n_layers - 1:
        if len(thresholds) == 0:
            # 如果没有阈值，添加两个固定值
            thresholds = np.array([np.percentile(m_values, 33), np.percentile(m_values, 66)])
        else:
            # 在现有阈值之间插入中间值
            new_thresholds = []
            for i in range(len(thresholds) - 1):
                new_thresholds.append(thresholds[i])
                new_thresholds.append((thresholds[i] + thresholds[i + 1]) / 2)
            new_thresholds.append(thresholds[-1])
            thresholds = np.array(new_thresholds)
    
    return thresholds


def get_concentration_layer_adaptive(m_value, thresholds):
    """
    根据浓度值和自适应阈值确定所属层次

    参数：
    m_value: CMAQ 浓度值
    thresholds: 分层阈值数组

    返回：
    layer_id: 层次编号，从 0 开始
    """
    for i, t in enumerate(thresholds):
        if m_value < t:
            return i
    return len(thresholds)


class CSP_AdvancedRK:
    """
    CSP-AdvancedRK: Adaptive Concentration-Stratified Advanced Residual Kriging

    自适应分层高级残差克里金，结合 CSPRK 和 AdvancedRK 的优势
    """

    def __init__(self, poly_degree=2, n_layers=3, matern_nu=1.5):
        """
        初始化 CSP-AdvancedRK

        参数：
        poly_degree: 多项式阶数，默认 2
        n_layers: 分层数量，默认 3
        matern_nu: Matern 核函数的 nu 参数，默认 1.5
        """
        self.poly_degree = poly_degree
        self.n_layers = n_layers
        self.matern_nu = matern_nu
        self.thresholds = None
        self.layer_models = {}
        self.poly = PolynomialFeatures(degree=poly_degree, include_bias=False)
        self.gpr = None
        self.X_train = None

    def fit(self, X, y, m):
        """
        训练 CSP-AdvancedRK 模型

        参数：
        X: 位置坐标 (n, 2) - [Lon, Lat]
        y: 真实浓度值 (n,)
        m: CMAQ 模型值 (n,)
        """
        X = np.asarray(X)
        y = np.asarray(y)
        m = np.asarray(m)

        # 自适应确定分层阈值
        self.thresholds = get_adaptive_thresholds(m, self.n_layers)
        n_layers_actual = len(self.thresholds) + 1
        
        print(f"Adaptive thresholds: {self.thresholds} (total {n_layers_actual} layers)")

        # 按浓度分层
        layers = np.array([get_concentration_layer_adaptive(v, self.thresholds) for v in m])

        # 存储每层的 OLS 模型和残差
        self.layer_models = {}

        for layer_id in range(n_layers_actual):
            mask = layers == layer_id
            if np.sum(mask) < 3:  # 需要至少3个样本
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
                'residual': residual_layer,
                'size': np.sum(mask)
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

            # 使用 Matern 核函数（来自 AdvancedRK）
            kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
                length_scale=1.0, 
                length_scale_bounds=(1e-3, 1e3),
                nu=self.matern_nu
            ) + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-10, 1e3))
            
            self.gpr = GaussianProcessRegressor(
                kernel=kernel, 
                n_restarts_optimizer=10,
                random_state=42
            )
            self.gpr.fit(X_all, residual_all)
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
        layers = np.array([get_concentration_layer_adaptive(v, self.thresholds) for v in m])

        for i in range(n):
            layer_id = layers[i]
            model_info = self.layer_models.get(layer_id)

            if model_info is None:
                # 如果该层没有模型，找最近的有模型的层
                valid_layers = [lid for lid, info in self.layer_models.items() if info is not None]
                if valid_layers:
                    # 找最近的层
                    closest_layer = min(valid_layers, key=lambda x: abs(x - layer_id))
                    model_info = self.layer_models[closest_layer]
                
                if model_info is not None:
                    m_poly = self.poly.transform([[m[i]]])
                    predictions[i] = model_info['ols'].predict(m_poly)[0]
                else:
                    # 如果所有层都没有模型，直接用 CMAQ
                    predictions[i] = m[i]
            else:
                m_poly = self.poly.transform([[m[i]]])
                predictions[i] = model_info['ols'].predict(m_poly)[0]

        # GPR 克里金校正（使用 Matern 核函数）
        if self.gpr is not None:
            residual_pred, _ = self.gpr.predict(X, return_std=True)
            predictions = predictions + residual_pred

        return predictions


def run_csp_advanced_rk_ten_fold(selected_day='2020-01-01', n_layers=3, matern_nu=1.5):
    """运行 CSP-AdvancedRK 十折交叉验证"""
    print("=" * 60)
    print(f"CSP-AdvancedRK Ten-Fold Cross Validation (n_layers={n_layers}, nu={matern_nu})")
    print("=" * 60)

    # 加载数据
    print("\n=== Loading Data ===")
    day_df, lon_cmaq, lat_cmaq, pred_day = load_daily_data(selected_day, paths)
    
    if day_df is None or len(day_df) == 0:
        print("No data available for selected day")
        return None

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

        # === 1. CSP-AdvancedRK: 自适应分层 + Matern 核 ===
        csp_advanced_rk = CSP_AdvancedRK(poly_degree=2, n_layers=n_layers, matern_nu=matern_nu)
        csp_advanced_rk.fit(X_train, y_train, m_train)
        csp_advanced_rk_pred = csp_advanced_rk.predict(X_test, m_test)

        # === 2. CSPRK: 固定阈值 + RBF 核对比 ===
        from CodeWorkSpace.新融合方法代码.CSPRK import CSPRK
        csprk_original = CSPRK(poly_degree=2)
        csprk_original.fit(X_train, y_train, m_train)
        csprk_original_pred = csprk_original.predict(X_test, m_test)

        # === 3. CSPRK Adaptive: 自适应阈值 + RBF 核对比 ===
        from CodeWorkSpace.新融合方法代码.CSPRK_Adaptive import CSPRK_Adaptive
        csprk_adaptive = CSPRK_Adaptive(poly_degree=2, n_layers=n_layers)
        csprk_adaptive.fit(X_train, y_train, m_train)
        csprk_adaptive_pred = csprk_adaptive.predict(X_test, m_test)

        # === 4. AdvancedRK: 不分层 + Matern 核对比 ===
        # 使用 AdvancedRK 的核心逻辑
        poly = PolynomialFeatures(degree=2, include_bias=False)
        m_train_poly = poly.fit_transform(m_train.reshape(-1, 1))
        m_test_poly = poly.transform(m_test.reshape(-1, 1))
        
        ols = LinearRegression()
        ols.fit(m_train_poly, y_train)
        pred_ols = ols.predict(m_test_poly)
        residual_train = y_train - ols.predict(m_train_poly)
        
        kernel_advanced = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
            length_scale=1.0, 
            length_scale_bounds=(1e-3, 1e3),
            nu=1.5
        ) + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-10, 1e3))
        
        gpr_advanced = GaussianProcessRegressor(
            kernel=kernel_advanced, 
            n_restarts_optimizer=10,
            random_state=42
        )
        gpr_advanced.fit(X_train, residual_train)
        gpr_advanced_pred, _ = gpr_advanced.predict(X_test, return_std=True)
        advanced_rk_pred = pred_ols + gpr_advanced_pred

        results[fold_id] = {
            'y_true': y_test,
            'csprk_original': csprk_original_pred,
            'csprk_adaptive': csprk_adaptive_pred,
            'advanced_rk': advanced_rk_pred,
            'csp_advanced_rk': csp_advanced_rk_pred
        }

        print(f"  Fold {fold_id}: completed")

    # 汇总
    true_all = np.concatenate([results[f]['y_true'] for f in range(1, 11) if results[f]])
    csprk_original_all = np.concatenate([results[f]['csprk_original'] for f in range(1, 11) if results[f]])
    csprk_adaptive_all = np.concatenate([results[f]['csprk_adaptive'] for f in range(1, 11) if results[f]])
    advanced_rk_all = np.concatenate([results[f]['advanced_rk'] for f in range(1, 11) if results[f]])
    csp_advanced_rk_all = np.concatenate([results[f]['csp_advanced_rk'] for f in range(1, 11) if results[f]])

    # 计算指标
    print("\n=== Results ===")
    csprk_original_metrics = compute_metrics(true_all, csprk_original_all)
    csprk_adaptive_metrics = compute_metrics(true_all, csprk_adaptive_all)
    advanced_rk_metrics = compute_metrics(true_all, advanced_rk_all)
    csp_advanced_rk_metrics = compute_metrics(true_all, csp_advanced_rk_all)
    # 缓存预测值供多天聚合使用
    global _last_y_true, _last_y_pred
    _last_y_true = true_all
    _last_y_pred = csp_advanced_rk_all


    print(f"  CSP-RK (fixed):   R2={csprk_original_metrics['R2']:.4f}, MAE={csprk_original_metrics['MAE']:.2f}, RMSE={csprk_original_metrics['RMSE']:.2f}")
    print(f"  CSP-RK (adap):    R2={csprk_adaptive_metrics['R2']:.4f}, MAE={csprk_adaptive_metrics['MAE']:.2f}, RMSE={csprk_adaptive_metrics['RMSE']:.2f}")
    print(f"  AdvancedRK:       R2={advanced_rk_metrics['R2']:.4f}, MAE={advanced_rk_metrics['MAE']:.2f}, RMSE={advanced_rk_metrics['RMSE']:.2f}")
    print(f"  CSP-AdvancedRK:   R2={csp_advanced_rk_metrics['R2']:.4f}, MAE={csp_advanced_rk_metrics['MAE']:.2f}, RMSE={csp_advanced_rk_metrics['RMSE']:.2f}")

    # 找出最佳方法
    methods_metrics = [
        ('CSP-RK (fixed)', csprk_original_metrics),
        ('CSP-RK (adap)', csprk_adaptive_metrics),
        ('AdvancedRK', advanced_rk_metrics),
        ('CSP-AdvancedRK', csp_advanced_rk_metrics)
    ]
    
    best_method = max(methods_metrics, key=lambda x: x[1]['R2'])
    print(f"\n🏆 Best method: {best_method[0]} with R2={best_method[1]['R2']:.4f}")

    # 保存结果
    result_df = pd.DataFrame([{
        'method': 'CSP_RK_Fixed',
        **csprk_original_metrics
    }, {
        'method': 'CSP_RK_Adaptive',
        **csprk_adaptive_metrics
    }, {
        'method': 'AdvancedRK',
        **advanced_rk_metrics
    }, {
        'method': 'CSP_AdvancedRK',
        **csp_advanced_rk_metrics
    }])
    result_df.to_csv(f'{paths["output_dir"]}/CSP_AdvancedRK_summary.csv', index=False)

    print(f"\nResults saved to: {paths['output_dir']}/CSP_AdvancedRK_summary.csv")

    return csprk_original_metrics, csprk_adaptive_metrics, advanced_rk_metrics, csp_advanced_rk_metrics


if __name__ == '__main__':
    results = run_csp_advanced_rk_ten_fold('2020-01-01', n_layers=3, matern_nu=1.5)
    if results:
        _, _, _, csp_advanced_rk_metrics = results
        print(f"\nCSP-AdvancedRK: R2={csp_advanced_rk_metrics['R2']:.4f}")
