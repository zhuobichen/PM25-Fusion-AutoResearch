"""
STRK - Spatio-Temporal Residual Co-Kriging
==========================================
时空残差共克里金融合方法

核心创新:
1. 残差时空分解: R = R_spatial + R_temporal + R_spatiotemporal
2. 空间残差用克里金插值 (捕捉排放源影响)
3. 时间残差用自回归模型 (捕捉日变化规律)
4. 时空交互残差用协克里金 (捕捉早高峰+稳定大气等组合效应)

数学框架:
1. 多项式CMAQ校正: O = a + b*M + c*M^2 + ε
2. 残差分解: ε(x,t) = R_sys(x) + R_temp(t) + R_st(x,t)
3. 时空变异函数: Γ(h_s,h_t;λ_s,τ,ρ) = ρ×exp(-h_s/λ_s)×exp(-h_t/τ)
4. 最终融合: Z* = Z_RK + θ₁×R_sys* + θ₂×R_temp* + θ₃×R_st*

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
from sklearn.preprocessing import StandardScaler
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


def haversine_distance(lon1, lat1, lon2, lat2):
    """计算两点间距离 (km)"""
    R = 6371.0  # 地球半径 km
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))


def compute_spatial_variogram(coords, values, max_dist=50.0, n_bins=15):
    """
    计算空间变异函数

    γ(h) = (1/2N) × Σ [z(x_i) - z(x_j)]²
    """
    n = len(values)
    distances = []
    gamma_vals = []

    bin_edges = np.linspace(0, max_dist, n_bins + 1)

    for i in range(n):
        for j in range(i + 1, n):
            d = haversine_distance(
                coords[i, 0], coords[i, 1],
                coords[j, 0], coords[j, 1]
            )
            if d <= max_dist:
                distances.append(d)
                gamma_vals.append((values[i] - values[j])**2)

    if len(distances) == 0:
        return None, None

    distances = np.array(distances)
    gamma_vals = np.array(gamma_vals)

    # 分箱统计
    gamma_binned = []
    dist_binned = []
    for k in range(n_bins):
        mask = (distances >= bin_edges[k]) & (distances < bin_edges[k + 1])
        if np.sum(mask) > 0:
            gamma_binned.append(np.mean(gamma_vals[mask]) / 2.0)
            dist_binned.append((bin_edges[k] + bin_edges[k + 1]) / 2.0)

    return np.array(dist_binned), np.array(gamma_binned)


def fit_variogram_model(dist_binned, gamma_binned, lambda_s=20.0, rho_s=1.0, nugget=0.08):
    """
    拟合变异函数模型参数

    Γ(h;λ,ρ) = nugget + ρ × [1 - exp(-h/λ)]
    """
    # 使用简单的加权最小二乘
    valid = ~(np.isnan(dist_binned) | np.isnan(gamma_binned))
    if np.sum(valid) < 3:
        return {'lambda_s': lambda_s, 'rho_s': rho_s, 'nugget': nugget}

    d = dist_binned[valid]
    g = gamma_binned[valid]

    # 权重：近距离点权重更高
    weights = 1.0 / (d + 1.0)

    # 拟合
    def model(params):
        lam, rho, nug = params
        pred = nug + rho * (1.0 - np.exp(-d / lam))
        return np.sum(weights * (g - pred)**2)

    from scipy.optimize import minimize
    result = minimize(model, [lambda_s, rho_s, nugget],
                      bounds=[(5.0, 50.0), (0.3, 2.0), (0.01, 0.2)])

    return {
        'lambda_s': result.x[0],
        'rho_s': result.x[1],
        'nugget': result.x[2]
    }


class STRK:
    """
    Spatio-Temporal Residual Co-Kriging (STRK)

    时空残差共克里金融合方法

    核心参数:
        lambda_s: 空间相关长度 (km), 默认20.0
        tau: 时间相关尺度 (h), 默认3.0
        rho_s: 空间方差权重, 默认0.5
        theta1: 系统性残差权重, 默认0.3
        theta2: 时间残差权重, 默认0.15
        theta3: 时空交互权重, 默认0.25
        rho_nugget: 块金效应, 默认0.08
        rho_sill: 基台值, 默认1.0
        t_window: 时间窗口 (h), 默认12
    """

    def __init__(self,
                 lambda_s=20.0, tau=3.0, rho_s=0.5,
                 theta1=0.3, theta2=0.15, theta3=0.25,
                 rho_nugget=0.08, rho_sill=1.0, t_window=12):
        self.lambda_s = lambda_s
        self.tau = tau
        self.rho_s = rho_s
        self.theta1 = theta1
        self.theta2 = theta2
        self.theta3 = theta3
        self.rho_nugget = rho_nugget
        self.rho_sill = rho_sill
        self.t_window = t_window

        # 内部状态
        self.bias_model = None
        self.gpr_spatial = None
        self.gpr_st = None
        self.ar_model = None
        self.scaler = StandardScaler()
        self.vario_params = {}

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

    def decompose_residuals(self, train_df, monitor_df=None, dates=None):
        """
        步骤2: 残差时空分解

        R(x,t) = R_systematic(x) + R_temporal(t) + R_spatiotemporal(x,t)

        参数:
            train_df: 训练数据（单日）
            monitor_df: 全部监测数据（可选，用于时间建模）
            dates: 日期列表
        """
        residuals = train_df['residual'].values
        coords = train_df[['Lon', 'Lat']].values

        # 1. 系统性空间残差（站点均值）
        R_systematic = residuals.mean()  # 标量：全局均值偏差

        # 2. 站点空间残差（每个站点的残差减去均值）
        site_residuals = {}
        if 'Site' in train_df.columns:
            for site in train_df['Site'].unique():
                site_data = train_df[train_df['Site'] == site]
                site_residuals[site] = {
                    'coords': site_data[['Lon', 'Lat']].values[0],
                    'R_sys': site_data['residual'].values - residuals.mean()
                }

        # 3. 时间残差（如果有连续数据）
        R_temporal = np.zeros(len(train_df))  # 默认无时间效应

        # 4. 时空交互残差 = 总残差 - 系统性 - 时间
        R_st = residuals - R_systematic - R_temporal

        self.R_systematic = R_systematic
        self.site_residuals = site_residuals
        self.R_temporal = R_temporal
        train_df = train_df.copy()
        train_df['R_st'] = R_st

        return train_df

    def fit_spatial_model(self, train_df):
        """
        步骤3: 空间残差克里金插值
        """
        X_train = train_df[['Lon', 'Lat']].values
        y_train = train_df['residual'].values  # 总残差用于空间建模

        # 标准化
        X_scaled = self.scaler.fit_transform(X_train)

        # 空间GPR
        kernel = ConstantKernel(self.rho_sill, (1e-2, 1e3)) * RBF(
            length_scale=self.lambda_s, length_scale_bounds=(5.0, 50.0)
        ) + WhiteKernel(noise_level=self.rho_nugget, noise_level_bounds=(1e-5, 1e1))

        self.gpr_spatial = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=2,
            alpha=0.1,
            normalize_y=True
        )
        self.gpr_spatial.fit(X_scaled, y_train)

    def fit_st_model(self, train_df):
        """
        步骤4: 时空交互残差建模
        """
        X_train = train_df[['Lon', 'Lat']].values
        y_train = train_df['R_st'].values if 'R_st' in train_df.columns else train_df['residual'].values

        X_scaled = self.scaler.transform(X_train)

        # 时空交互GPR（使用更短的相关长度）
        kernel_st = ConstantKernel(self.rho_sill * 0.5, (1e-2, 1e3)) * RBF(
            length_scale=self.lambda_s * 0.5, length_scale_bounds=(5.0, 30.0)
        ) + WhiteKernel(noise_level=self.rho_nugget, noise_level_bounds=(1e-5, 1e1))

        self.gpr_st = GaussianProcessRegressor(
            kernel=kernel_st,
            n_restarts_optimizer=2,
            alpha=0.1,
            normalize_y=True
        )
        self.gpr_st.fit(X_scaled, y_train)

    def fit_temporal_model(self, train_df, monitor_df, dates):
        """
        步骤5: 时间残差自回归建模

        使用连续多日数据建立时间自相关模型
        """
        if monitor_df is None or dates is None:
            self.ar_coef = 0.0
            return

        # 构建时间序列
        site = train_df['Site'].iloc[0] if 'Site' in train_df.columns else None
        if site is None:
            self.ar_coef = 0.0
            return

        # 获取该站点的时间序列
        residuals_ts = []
        for date_str in dates:
            day_data = monitor_df[(monitor_df['Site'] == site) &
                                   (monitor_df['Date'] == date_str)]
            if len(day_data) > 0:
                residuals_ts.append(day_data['residual'].values[0])
            else:
                residuals_ts.append(np.nan)

        residuals_ts = np.array(residuals_ts)
        valid_mask = ~np.isnan(residuals_ts)

        if np.sum(valid_mask) < 3:
            self.ar_coef = 0.0
            return

        # 一阶自回归 AR(1)
        valid_residuals = residuals_ts[valid_mask]
        n = len(valid_residuals)
        if n < 3:
            self.ar_coef = 0.0
            return

        # y(t) = a * y(t-1) + e
        y = valid_residuals[1:]
        x = valid_residuals[:-1]
        ar_coef = np.sum(x * y) / (np.sum(x * x) + 1e-6)

        self.ar_coef = float(np.clip(ar_coef, -1.0, 1.0))

    def fit(self, train_df, monitor_df=None, dates=None):
        """
        完整拟合流程

        参数:
            train_df: 训练数据（当日）
            monitor_df: 全部监测数据（可选，用于时间建模）
            dates: 日期列表（可选）
        """
        # 步骤1: 多项式偏差校正
        train_df = self.fit_bias_correction(train_df)

        # 步骤2: 残差时空分解
        train_df = self.decompose_residuals(train_df, monitor_df, dates)

        # 步骤3: 空间残差克里金
        self.fit_spatial_model(train_df)

        # 步骤4: 时空交互建模
        self.fit_st_model(train_df)

        # 步骤5: 时间自回归建模
        if monitor_df is not None and dates is not None:
            self.fit_temporal_model(train_df, monitor_df, dates)
        else:
            self.ar_coef = 0.0

        return self

    def predict(self, test_df):
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
        Z_rk = self.bias_model.predict(m_test_poly)

        # 空间克里金预测
        X_test = test_df[['Lon', 'Lat']].values
        X_test_scaled = self.scaler.transform(X_test)

        # 总残差空间预测
        R_spatial_pred, _ = self.gpr_spatial.predict(X_test_scaled, return_std=True)

        # 时空交互预测
        R_st_pred, _ = self.gpr_st.predict(X_test_scaled, return_std=True)

        # 时间残差预测（简化：使用AR系数缩放的空间残差）
        R_temp_pred = self.ar_coef * R_spatial_pred

        # 加权融合残差
        R_final = (self.theta1 * self.R_systematic +
                   self.theta2 * R_temp_pred +
                   self.theta3 * R_st_pred)

        # 最终融合: Z* = Z_RK + R_final
        fusion_pred = Z_rk + R_final

        return {
            'fusion': fusion_pred,
            'z_rk': Z_rk,
            'r_spatial': R_spatial_pred,
            'r_temp': R_temp_pred,
            'r_st': R_st_pred,
            'r_final': R_final,
            'r_systematic': self.R_systematic,
            'ar_coef': self.ar_coef
        }


def fuse_method(cmaq_data, station_data, station_coords, params):
    """
    STRK PM2.5融合方法（对外接口）

    参数:
        cmaq_data: CMAQ格点数据 (dict with 'lon', 'lat', 'pm25')
        station_data: 监测站点数据 (DataFrame)
        station_coords: 站点坐标 (DataFrame)
        params: 参数字典

    返回:
        fused_data: 融合结果DataFrame
    """
    lon_cmaq = cmaq_data['lon']
    lat_cmaq = cmaq_data['lat']
    pm25_grid = cmaq_data['pm25']

    df = station_data.copy()

    if 'CMAQ' not in df.columns:
        cmaq_vals = []
        for _, row in df.iterrows():
            val = get_cmaq_at_site(row['Lon'], row['Lat'], lon_cmaq, lat_cmaq, pm25_grid)
            cmaq_vals.append(val)
        df['CMAQ'] = cmaq_vals

    # 获取参数
    lambda_s = params.get('lambda_s', 20.0)
    tau = params.get('tau', 3.0)
    rho_s = params.get('rho_s', 0.5)
    theta1 = params.get('theta1', 0.3)
    theta2 = params.get('theta2', 0.15)
    theta3 = params.get('theta3', 0.25)

    # 创建并训练STRK模型
    strk = STRK(
        lambda_s=lambda_s, tau=tau, rho_s=rho_s,
        theta1=theta1, theta2=theta2, theta3=theta3
    )

    if len(df) > 0:
        strk.fit(df)
        result = strk.predict(df)
        df['Fused'] = result['fusion']
    else:
        df['Fused'] = df['CMAQ'].values

    return df


def cross_validate(method_func, fold_split_table, selected_days):
    """
    十折交叉验证

    参数:
        method_func: 融合方法函数
        fold_split_table: 折划分表
        selected_days: 选择的天列表

    返回:
        dict: 包含R2, MAE, RMSE, MB
    """
    print("=" * 60)
    print("STRK Ten-Fold Cross Validation")
    print("=" * 60)

    ds = nc.Dataset(cmaq_file, 'r')
    lon_cmaq = ds.variables['lon'][:]
    lat_cmaq = ds.variables['lat'][:]
    pred_pm25 = ds.variables['pred_PM25'][:]
    ds.close()

    monitor_df = pd.read_csv(monitor_file)
    fold_df = pd.read_csv(fold_file)

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

        cmaq_vals = []
        for _, row in day_df.iterrows():
            val = get_cmaq_at_site(row['Lon'], row['Lat'], lon_cmaq, lat_cmaq, pm25_day)
            cmaq_vals.append(val)
        day_df['CMAQ'] = cmaq_vals

        for fold_id in range(1, 11):
            train_df = day_df[day_df['fold'] != fold_id].copy()
            test_df = day_df[day_df['fold'] == fold_id].copy()

            train_df = train_df.dropna(subset=['Lon', 'Lat', 'CMAQ', 'Conc'])
            test_df = test_df.dropna(subset=['Lon', 'Lat', 'CMAQ', 'Conc'])

            if len(test_df) == 0:
                continue

            strk = STRK(lambda_s=20.0, tau=3.0, rho_s=0.5,
                        theta1=0.3, theta2=0.15, theta3=0.25)
            strk.fit(train_df)

            result = strk.predict(test_df)

            all_y_true.extend(test_df['Conc'].values)
            all_y_pred.extend(result['fusion'])

        print(f"  Fold completed")

    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)

    metrics = compute_metrics(all_y_true, all_y_pred)
    print(f"\n=== STRK Results ===")
    print(f"  R2={metrics['R2']:.4f}, MAE={metrics['MAE']:.2f}, RMSE={metrics['RMSE']:.2f}, MB={metrics['MB']:.2f}")

    return metrics


def run_strk_ten_fold(selected_day='2020-01-01'):
    """运行STRK十折交叉验证（快捷入口）"""
    return cross_validate(None, None, selected_day)


def calculate_metrics(y_true, y_pred):
    """兼容接口"""
    return compute_metrics(y_true, y_pred)


if __name__ == '__main__':
    print("\n>>> STRK Ten-Fold Test <<<")
    metrics = run_strk_ten_fold('2020-01-01')
    print(f"\nFinal: R2={metrics['R2']:.4f}, MAE={metrics['MAE']:.2f}, RMSE={metrics['RMSE']:.2f}")
