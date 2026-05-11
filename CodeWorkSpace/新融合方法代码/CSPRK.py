"""

# === 预测值缓存（由 patch_methods.py 自动添加）===
_last_y_true = None
_last_y_pred = None

CSP-RK - Concentration-Stratified PolyRK
==========================================
把 CMAQ 数据按浓度分为高/中/低三层，分别拟合独立的多项式校正参数

核心创新：
1. 按浓度分层：M < T1 (低), T1 <= M < T2 (中), M >= T2 (高)
2. 每层独立做多项式 OLS 校正
3. 合并残差，做统一 GPR 克里金
4. 预测时按测试点浓度选择对应层的 OLS

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


class CSPRK:
    """
    CSP-RK: Concentration-Stratified PolyRK

    分层多项式残差克里金
    """

    def __init__(self, poly_degree=2):
        self.poly_degree = poly_degree
        self.layer_models = {}  # 存储每层的 OLS 模型
        self.poly = PolynomialFeatures(degree=poly_degree, include_bias=False)
        self.gpr = None
        self.X_train = None
        self.residual_mean = None
        self.residual_std = None

    def fit(self, X, y, m):
        """
        训练 CSP-RK 模型

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
                m_poly = self.poly.fit_transform([[m[i]]])
                # 使用所有层的平均系数
                if self.layer_models:
                    valid_models = [info for info in self.layer_models.values() if info is not None]
                    if valid_models:
                        # 取第一个有效模型的预测
                        predictions[i] = valid_models[0]['ols'].predict(m_poly)[0]
                    else:
                        predictions[i] = m[i]
                else:
                    predictions[i] = m[i]
            else:
                m_poly = self.poly.transform([[m[i]]])
                predictions[i] = model_info['ols'].predict(m_poly)[0]

        # GPR 克里金校正
        if self.gpr is not None:
            residual_pred_normalized, _ = self.gpr.predict(X, return_std=True)
            residual_pred = residual_pred_normalized * (self.residual_std + 1e-8) + self.residual_mean
            predictions = predictions + residual_pred

        return predictions


def cross_validate(day_df, lon_cmaq, lat_cmaq, pred_day, fold_list=None):
    """
    十折交叉验证

    参数：
    day_df: 包含 Lat, Lon, Conc, fold 的 DataFrame
    lon_cmaq, lat_cmaq: CMAQ 网格坐标
    pred_day: 当日的 CMAQ 预测值网格
    fold_list: 折数列表，默认 1-10

    返回：
    results: 字典，包含每折的预测结果
    """
    if fold_list is None:
        fold_list = list(range(1, 11))

    results = {fold_id: {'y_true': None, 'y_pred': None} for fold_id in fold_list}

    kernel = ConstantKernel(10.0, (1e-2, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e1))

    for fold_id in fold_list:
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

        # 使用全局多项式 OLS 校正
        poly = PolynomialFeatures(degree=2, include_bias=False)
        m_train_poly = poly.fit_transform(m_train.reshape(-1, 1))
        m_test_poly = poly.transform(m_test.reshape(-1, 1))

        ols = LinearRegression()
        ols.fit(m_train_poly, y_train)
        pred_ols = ols.predict(m_test_poly)
        residual_train = y_train - ols.predict(m_train_poly)

        # GPR 残差建模
        gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, alpha=0.1, normalize_y=True)
        gpr.fit(X_train, residual_train)
        gpr_pred, _ = gpr.predict(X_test, return_std=True)

        # CSP-RK: 使用分层多项式校正
        csprk = CSPRK(poly_degree=2)
        csprk.fit(X_train, y_train, m_train)
        csprk_pred = csprk.predict(X_test, m_test)

        # 融合预测：OLS + GPR
        rk_pred = pred_ols + gpr_pred

        # CSP-RK 也加上 GPR 校正
        # (CSPRK.predict 已经包含残差校正，这里不需要再加)

        results[fold_id] = {
            'y_true': y_test,
            'csprk': csprk_pred,
            'rk': rk_pred
        }

    return results


def run_csprk_ten_fold(selected_day='2020-01-01'):
    """运行 CSP-RK 十折交叉验证"""
    print("=" * 60)
    print("CSP-RK Ten-Fold Cross Validation")
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

        # === CSP-RK: 分层多项式校正 ===
        csprk = CSPRK(poly_degree=2)
        csprk.fit(X_train, y_train, m_train)
        csprk_pred = csprk.predict(X_test, m_test)

        # GPR on residuals
        gpr_poly = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, alpha=0.1, normalize_y=True)
        gpr_poly.fit(X_train, residual_poly)
        gpr_poly_pred, _ = gpr_poly.predict(X_test, return_std=True)

        # 融合预测
        rk_poly_pred = pred_poly + gpr_poly_pred

        results[fold_id] = {
            'y_true': y_test,
            'rk_poly': rk_poly_pred,
            'csprk': csprk_pred
        }

        print(f"  Fold {fold_id}: completed")

    # 汇总
    rk_poly_all = np.concatenate([results[f]['rk_poly'] for f in range(1, 11) if results[f]])
    csprk_all = np.concatenate([results[f]['csprk'] for f in range(1, 11) if results[f]])
    true_all = np.concatenate([results[f]['y_true'] for f in range(1, 11) if results[f]])

    # 计算 R2
    print("\n=== Results ===")
    rk_metrics = compute_metrics(true_all, rk_poly_all)
    csprk_metrics = compute_metrics(true_all, csprk_all)
    # 缓存预测值供多天聚合使用
    global _last_y_true, _last_y_pred
    _last_y_true = true_all
    _last_y_pred = csprk_all


    print(f"  RK-Poly:  R2={rk_metrics['R2']:.4f}, MAE={rk_metrics['MAE']:.2f}, RMSE={rk_metrics['RMSE']:.2f}")
    print(f"  CSP-RK:   R2={csprk_metrics['R2']:.4f}, MAE={csprk_metrics['MAE']:.2f}, RMSE={csprk_metrics['RMSE']:.2f}")

    # 保存结果
    result_df = pd.DataFrame([{
        'method': 'RK_Poly',
        **rk_metrics
    }, {
        'method': 'CSP_RK',
        **csprk_metrics
    }])
    result_df.to_csv(f'{output_dir}/CSPRK_summary.csv', index=False)

    print(f"\nResults saved to: {output_dir}/CSPRK_summary.csv")

    return rk_metrics, csprk_metrics


if __name__ == '__main__':
    rk_metrics, csprk_metrics = run_csprk_ten_fold('2020-01-01')
    print(f"\nRK-Poly: R2={rk_metrics['R2']:.4f}")
    print(f"CSP-RK:  R2={csprk_metrics['R2']:.4f}")