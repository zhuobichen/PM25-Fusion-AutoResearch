"""
MSAK - Multi-Scale Stability-Adaptive Kriging
===============================================
多尺度稳定度自适应克里金融合方法

核心创新:
1. 大气稳定度分类 (Pasquill A-F) 驱动相关长度自适应
2. 多尺度克里金分解：区域背景 + 城市尺度 + 局地扩散
3. 稳定度感知权重融合：不稳定→短相关(监测主导)，稳定→长相关(CMAQ主导)

数学框架:
1. 多项式CMAQ校正: O = a + b*M + c*M^2 + ε
2. 稳定度自适应相关长度: λ(PG) = λ₀ × [1 + β×exp(-|PG-PG_cr|/σ_PG)]
3. 多尺度残差克里金: Z_s(x) = μ_CMAQ,s(x) + ε_s(x)
4. 稳定度融合: Z_final = α(PG)×Z_CMAQ + [1-α(PG)]×Z_GPR

作者: Claude Agent
日期: 2026-04-09
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.paths import get_project_root, data_path
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
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
    """获取站点位置的CMAQ值（最近邻）"""
    dist = np.sqrt((lon_grid - lon)**2 + (lat_grid - lat)**2)
    idx = np.argmin(dist)
    ny, nx = lon_grid.shape
    row, col = idx // nx, idx % nx
    return pm25_grid[row, col]


def classify_pasquill_stability(ws, radiation=None, cloud_cover=None):
    """
    根据风速、辐射和云量计算Pasquill-Gifford稳定度等级

    参数:
        ws: 风速 (m/s)
        radiation: 辐射 (W/m^2), 白天用
        cloud_cover: 云量 (0-10)

    返回:
        PG等级 (1-6): 1=A, 2=B, 3=C, 4=D, 5=E, 6=F
    """
    # 简化版PG分类：基于风速和辐射/云量
    # A=极度不稳定, B=不稳定, C=轻度不稳定, D=中性, E=轻度稳定, F=稳定

    if radiation is not None:
        # 白天分类
        if radiation > 200:  # 强辐射
            if ws < 2.0:
                return 1  # A
            elif ws < 3.0:
                return 2  # B
            elif ws < 5.0:
                return 3  # C
            else:
                return 4  # D
        elif radiation > 100:  # 中等辐射
            if ws < 3.0:
                return 2  # B
            elif ws < 5.0:
                return 3  # C
            elif ws < 7.0:
                return 4  # D
            else:
                return 4  # D
        else:  # 弱辐射
            if ws < 3.0:
                return 3  # C
            elif ws < 5.0:
                return 4  # D
            elif ws < 8.0:
                return 5  # E
            else:
                return 5  # E
    elif cloud_cover is not None:
        # 夜间分类（基于云量）
        if cloud_cover >= 5:
            return 4  # D (厚云层接近中性)
        elif cloud_cover >= 3:
            if ws < 3.0:
                return 5  # E
            else:
                return 4  # D
        else:
            if ws < 3.0:
                return 6  # F
            elif ws < 5.0:
                return 5  # E
            else:
                return 4  # D
    else:
        # 仅用风速的简化分类
        if ws < 2.0:
            return 2  # B
        elif ws < 3.0:
            return 3  # C
        elif ws < 5.0:
            return 4  # D
        elif ws < 8.0:
            return 5  # E
        else:
            return 6  # F


def compute_stability_correlation_length(pg, lambda0=15.0, beta=0.8, sigma_pg=1.2, pg_crit=3.0):
    """
    计算稳定度自适应的相关长度

    λ(PG) = λ₀ × [1 + β × exp(-|PG - PG_cr|/σ_PG)]

    参数:
        pg: Pasquill-Gifford稳定度等级 (1-6)
        lambda0: 基础相关长度 (km)
        beta: 稳定度响应强度
        sigma_pg: 稳定度特征宽度
        pg_crit: 临界稳定度等级

    返回:
        相关长度 (km)
    """
    lambda_pg = lambda0 * (1.0 + beta * np.exp(-abs(pg - pg_crit) / sigma_pg))
    return lambda_pg


def compute_multi_scale_lengths(lambda_base, n_scales=3):
    """
    计算多尺度相关长度序列

    λ_s = λ(PG) / 2^(s-1)

    参数:
        lambda_base: 基础相关长度
        n_scales: 尺度层数

    返回:
        各尺度相关长度列表
    """
    return [lambda_base / (2.0 ** s) for s in range(n_scales)]


def compute_stability_weight(pg, gamma=1.5, pg_mid=2.5):
    """
    计算稳定度权重 α(PG)

    α(PG) = 1 / [1 + exp(-γ × (PG - PG_mid))]

    参数:
        pg: Pasquill-Gifford稳定度等级
        gamma: 平滑参数
        pg_mid: 稳定度阈值

    返回:
        α权重 (CMAQ在稳定大气中权重更高)
    """
    alpha = 1.0 / (1.0 + np.exp(-gamma * (pg - pg_mid)))
    return alpha


class MSAK:
    """
    Multi-Scale Stability-Adaptive Kriging (MSAK)

    多尺度稳定度自适应克里金融合方法

    核心参数:
        lambda0: 基础相关长度 (km), 默认15.0
        beta: 稳定度响应强度, 默认0.8
        sigma_pg: 稳定度特征宽度, 默认1.2
        pg_crit: 临界稳定度等级, 默认3.0
        pg_mid: 稳定度阈值, 默认2.5
        gamma: 稳定度权重平滑参数, 默认1.5
        n_scales: 多尺度层数, 默认3
        rho_nugget: 块金效应, 默认0.05
        rho_sill: 基台值, 默认1.0
    """

    def __init__(self,
                 lambda0=15.0, beta=0.8, sigma_pg=1.2,
                 pg_crit=3.0, pg_mid=2.5, gamma=1.5,
                 n_scales=3, rho_nugget=0.05, rho_sill=1.0):
        self.lambda0 = lambda0
        self.beta = beta
        self.sigma_pg = sigma_pg
        self.pg_crit = pg_crit
        self.pg_mid = pg_mid
        self.gamma = gamma
        self.n_scales = n_scales
        self.rho_nugget = rho_nugget
        self.rho_sill = rho_sill

        # 内部状态
        self.bias_model = None
        self.gpr_models = {}
        self.scaler = None

    def fit_bias_correction(self, train_df):
        """
        步骤1: 多项式CMAQ偏差校正
        O = a + b*M + c*M^2 + ε
        """
        m = train_df['CMAQ'].values
        o = train_df['Conc'].values

        self.poly = PolynomialFeatures(degree=2, include_bias=False)
        m_poly = self.poly.fit_transform(m.reshape(-1, 1))

        self.bias_model = LinearRegression()
        self.bias_model.fit(m_poly, o)

        # 计算残差
        residual = o - self.bias_model.predict(m_poly)
        train_df = train_df.copy()
        train_df['residual'] = residual

        return train_df

    def compute_stability(self, df, ws_col='WS', radiation_col=None, cloud_col=None):
        """
        计算稳定度等级

        参数:
            df: 包含气象数据的DataFrame
            ws_col: 风速列名
            radiation_col: 辐射列名（可选）
            cloud_col: 云量列名（可选）
        """
        if radiation_col and radiation_col in df.columns:
            radiation = df[radiation_col].values
        else:
            radiation = None

        if cloud_col and cloud_col in df.columns:
            cloud = df[cloud_col].values
        else:
            cloud = None

        ws = df[ws_col].values if ws_col in df.columns else np.full(len(df), 3.0)

        pg_values = []
        for i in range(len(df)):
            pg = classify_pasquill_stability(
                ws[i],
                radiation[i] if radiation is not None else None,
                cloud[i] if cloud is not None else None
            )
            pg_values.append(pg)

        df = df.copy()
        df['PG'] = pg_values
        return df

    def fit_spatial_model(self, train_df):
        """
        步骤2-3: 多尺度克里金空间建模
        """
        from sklearn.preprocessing import StandardScaler

        X_train = train_df[['Lon', 'Lat']].values
        y_train = train_df['residual'].values

        # 计算稳定度参数
        if 'PG' not in train_df.columns:
            train_df = self.compute_stability(train_df)

        pg_mean = train_df['PG'].mean()
        lambda_pg = compute_stability_correlation_length(
            pg_mean, self.lambda0, self.beta, self.sigma_pg, self.pg_crit
        )

        # 多尺度权重（反比于相关长度）
        scale_lengths = compute_multi_scale_lengths(lambda_pg, self.n_scales)
        self.scale_weights = [1.0 / l for l in scale_lengths]
        sum_weights = sum(self.scale_weights)
        self.scale_weights = [w / sum_weights for w in self.scale_weights]

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_train)

        # 为每个尺度构建GPR模型
        self.gpr_models = {}
        for s, (lambda_s, weight_s) in enumerate(zip(scale_lengths, self.scale_weights)):
            kernel = ConstantKernel(self.rho_sill, (1e-2, 1e3)) * RBF(
                length_scale=lambda_s, length_scale_bounds=(1e-2, 1e2)
            ) + WhiteKernel(noise_level=self.rho_nugget, noise_level_bounds=(1e-5, 1e1))

            gpr = GaussianProcessRegressor(
                kernel=kernel,
                n_restarts_optimizer=2,
                alpha=0.1,
                normalize_y=True
            )
            gpr.fit(X_scaled, y_train)
            self.gpr_models[s] = gpr

        self.lambda_pg = lambda_pg
        self.scale_lengths = scale_lengths

    def fit(self, train_df, ws_col='WS', radiation_col=None, cloud_col=None):
        """
        完整拟合流程

        参数:
            train_df: 训练数据
            ws_col: 风速列名
            radiation_col: 辐射列名（可选）
            cloud_col: 云量列名（可选）
        """
        # 步骤1: 多项式偏差校正
        train_df = self.fit_bias_correction(train_df)

        # 步骤2: 稳定度分类
        train_df = self.compute_stability(
            train_df, ws_col, radiation_col, cloud_col
        )

        # 步骤3: 多尺度克里金建模
        self.fit_spatial_model(train_df)

        return self

    def predict(self, test_df, ws_col='WS', radiation_col=None, cloud_col=None):
        """
        预测融合值

        参数:
            test_df: 测试数据

        返回:
            dict: 包含融合结果及各分量
        """
        # CMAQ偏差校正预测
        m_test = test_df['CMAQ'].values
        m_test_poly = self.poly.transform(m_test.reshape(-1, 1))
        bias_pred = self.bias_model.predict(m_test_poly)

        # 稳定度计算
        if 'PG' not in test_df.columns:
            test_df = self.compute_stability(
                test_df, ws_col, radiation_col, cloud_col
            )

        pg_test = test_df['PG'].values

        # 多尺度残差预测
        X_test = test_df[['Lon', 'Lat']].values
        X_test_scaled = self.scaler.transform(X_test)

        scale_preds = []
        for s, gpr in self.gpr_models.items():
            pred_s, _ = gpr.predict(X_test_scaled, return_std=True)
            scale_preds.append(pred_s)

        # 多尺度加权融合
        residual_pred = np.zeros(len(test_df))
        for s, pred_s in enumerate(scale_preds):
            residual_pred += self.scale_weights[s] * pred_s

        # 稳定度自适应权重
        alpha_values = []
        for pg in pg_test:
            alpha = compute_stability_weight(pg, self.gamma, self.pg_mid)
            alpha_values.append(alpha)
        alpha_values = np.array(alpha_values)

        # 最终融合: Z_final = α×Z_CMAQ + (1-α)×Z_GPR
        # 其中 Z_GPR = bias_pred + residual_pred
        cmaq_pred = m_test
        gpr_pred = bias_pred + residual_pred

        # 使用稳定度权重调整CMAQ和GPR的贡献
        # 稳定大气(PG高) → 高α → CMAQ权重高
        # 不稳定大气(PG低) → 低α → GPR/监测权重高
        fusion_pred = alpha_values * cmaq_pred + (1.0 - alpha_values) * gpr_pred

        return {
            'fusion': fusion_pred,
            'cmaq': cmaq_pred,
            'bias_corrected': bias_pred,
            'residual': residual_pred,
            'gpr_pred': gpr_pred,
            'alpha': alpha_values,
            'pg': pg_test
        }


def fuse_method(cmaq_data, station_data, station_coords, params):
    """
    MSAK PM2.5融合方法（对外接口）

    参数:
        cmaq_data: CMAQ格点数据 (dict with 'lon', 'lat', 'pm25')
        station_data: 监测站点数据 (DataFrame, columns: Site, Date, Conc, Lon, Lat, [WS])
        station_coords: 站点坐标 (DataFrame, columns: Site, Lon, Lat)
        params: 参数字典

    返回:
        fused_data: 融合结果DataFrame
    """
    # 解析CMAQ数据
    lon_cmaq = cmaq_data['lon']
    lat_cmaq = cmaq_data['lat']
    pm25_grid = cmaq_data['pm25']

    # 准备站点CMAQ值
    df = station_data.copy()

    if 'CMAQ' not in df.columns:
        cmaq_vals = []
        for _, row in df.iterrows():
            val = get_cmaq_at_site(row['Lon'], row['Lat'], lon_cmaq, lat_cmaq, pm25_grid)
            cmaq_vals.append(val)
        df['CMAQ'] = cmaq_vals

    # 获取参数
    lambda0 = params.get('lambda0', 15.0)
    beta = params.get('beta', 0.8)
    sigma_pg = params.get('sigma_pg', 1.2)
    pg_crit = params.get('pg_crit', 3.0)
    pg_mid = params.get('pg_mid', 2.5)
    gamma = params.get('gamma', 1.5)
    n_scales = params.get('n_scales', 3)

    # 创建并训练MSAK模型
    msak = MSAK(
        lambda0=lambda0, beta=beta, sigma_pg=sigma_pg,
        pg_crit=pg_crit, pg_mid=pg_mid, gamma=gamma, n_scales=n_scales
    )

    ws_col = 'WS' if 'WS' in df.columns else None

    if len(df) > 0:
        msak.fit(df, ws_col=ws_col)
        result = msak.predict(df, ws_col=ws_col)
        df['Fused'] = result['fusion']
    else:
        df['Fused'] = df['CMAQ'].values

    return df


def cross_validate(method_func, fold_split_table, selected_days):
    """
    十折交叉验证

    参数:
        method_func: 融合方法函数 (同fuse_method)
        fold_split_table: 折划分表 (DataFrame, columns: Site, fold)
        selected_days: 选择的天列表

    返回:
        dict: 包含R2, MAE, RMSE, MB
    """
    print("=" * 60)
    print("MSAK Ten-Fold Cross Validation")
    print("=" * 60)

    # 加载CMAQ数据
    ds = nc.Dataset(cmaq_file, 'r')
    lon_cmaq = ds.variables['lon'][:]
    lat_cmaq = ds.variables['lat'][:]
    pred_pm25 = ds.variables['pred_PM25'][:]
    ds.close()

    monitor_df = pd.read_csv(monitor_file)
    fold_df = pd.read_csv(fold_file)

    # 支持多天验证
    if isinstance(selected_days, str):
        selected_days = [selected_days]

    all_y_true = []
    all_y_pred = []

    for day in selected_days:
        print(f"\n--- Date: {day} ---")
        day_df = monitor_df[monitor_df['Date'] == day].copy()
        day_df = day_df.merge(fold_df, on='Site', how='left')
        day_df = day_df.dropna(subset=['Lat', 'Lon', 'Conc'])

        date_obj = datetime.strptime(day, '%Y-%m-%d')
        day_idx = (date_obj - datetime(2020, 1, 1)).days
        if 0 <= day_idx < pred_pm25.shape[0]:
            pm25_day = pred_pm25[day_idx]
        else:
            continue

        # 提取站点CMAQ值
        cmaq_vals = []
        for _, row in day_df.iterrows():
            val = get_cmaq_at_site(row['Lon'], row['Lat'], lon_cmaq, lat_cmaq, pm25_day)
            cmaq_vals.append(val)
        day_df['CMAQ'] = cmaq_vals

        # 十折验证
        for fold_id in range(1, 11):
            train_df = day_df[day_df['fold'] != fold_id].copy()
            test_df = day_df[day_df['fold'] == fold_id].copy()

            train_df = train_df.dropna(subset=['Lon', 'Lat', 'CMAQ', 'Conc'])
            test_df = test_df.dropna(subset=['Lon', 'Lat', 'CMAQ', 'Conc'])

            if len(test_df) == 0:
                continue

            # 训练
            msak = MSAK(lambda0=15.0, beta=0.8, sigma_pg=1.2,
                        pg_crit=3.0, pg_mid=2.5, gamma=1.5, n_scales=3)
            ws_col = 'WS' if 'WS' in train_df.columns else None
            msak.fit(train_df, ws_col=ws_col)

            # 预测
            result = msak.predict(test_df, ws_col=ws_col)

            all_y_true.extend(test_df['Conc'].values)
            all_y_pred.extend(result['fusion'])

        print(f"  Fold completed")

    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)

    metrics = compute_metrics(all_y_true, all_y_pred)
    print(f"\n=== MSAK Results ===")
    print(f"  R2={metrics['R2']:.4f}, MAE={metrics['MAE']:.2f}, RMSE={metrics['RMSE']:.2f}, MB={metrics['MB']:.2f}")

    return metrics


def run_msak_ten_fold(selected_day='2020-01-01'):
    """
    运行MSAK十折交叉验证（快捷入口）
    """
    return cross_validate(None, None, selected_day)


def calculate_metrics(y_true, y_pred):
    """兼容接口"""
    return compute_metrics(y_true, y_pred)


if __name__ == '__main__':
    print("\n>>> MSAK Ten-Fold Test <<<")
    metrics = run_msak_ten_fold('2020-01-01')
    print(f"\nFinal: R2={metrics['R2']:.4f}, MAE={metrics['MAE']:.2f}, RMSE={metrics['RMSE']:.2f}")
