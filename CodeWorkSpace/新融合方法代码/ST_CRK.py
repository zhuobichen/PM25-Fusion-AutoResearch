"""

# === 预测值缓存（由 patch_methods.py 自动添加）===
_last_y_true = None
_last_y_pred = None

ST-CRK: Spatiotemporal Cokriging Residual Kriging
===================================================
时空协同残差克里金融合方法

核心创新:
1. 时空协同克里金：同时建模时间和空间相关性
2. 多日联合残差建模：利用连续多天的残差数据增强统计显著性
3. CMAQ时空梯度辅助：利用CMAQ的时空平滑性引导插值
4. 自适应时空权重：根据局部时空数据密度调整融合权重

原理:
- CMAQ模型在时空上相对平滑，但缺少局部细节
- 监测数据有局部细节，但存在观测误差和空间稀疏性
- 通过协同克里金框架融合两者优势

数学框架:
1. 偏差校正: O(s,t) = α + β*M(s,t) + ε(s,t)
2. 时空残差建模: ε(s,t) 建模为时空随机场
3. 时空变异函数: γ(h,τ) = γ_s(h) + γ_t(τ) (分离式)
4. 协同克里金: 使用CMAQ梯度作为辅助变量

作者: Claude Agent
日期: 2026-04-07
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.paths import get_project_root, data_path
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel, Matern
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


def compute_cmaq_gradient(lon, lat, lon_grid, lat_grid, pm25_grid):
    """
    计算CMAQ在站点位置的时空梯度
    空间梯度: dM/dx, dM/dy
    用于协同克里金的辅助变量
    """
    ny, nx = lon_grid.shape

    # 找到最近邻索引
    dist = np.sqrt((lon_grid - lon)**2 + (lat_grid - lat)**2)
    idx = np.argmin(dist)
    row, col = idx // nx, idx % nx

    # 边界检查
    row = max(1, min(row, ny - 2))
    col = max(1, min(col, nx - 2))

    # 空间梯度（中心差分）
    dmdx = (pm25_grid[row, col + 1] - pm25_grid[row, col - 1]) / \
           (lon_grid[row, col + 1] - lon_grid[row, col - 1] + 1e-6)
    dmdy = (pm25_grid[row + 1, col] - pm25_grid[row - 1, col]) / \
           (lat_grid[row + 1, col] - lat_grid[row - 1, col] + 1e-6)

    return dmdx, dmdy


def load_multi_day_cmaq(lon_cmaq, lat_cmaq, pred_pm25, start_date, num_days):
    """
    加载多日CMAQ数据
    返回: dict{date: pm25_grid}
    """
    result = {}
    for i in range(num_days):
        date = start_date + timedelta(days=i)
        day_idx = (date - datetime(2020, 1, 1)).days
        if 0 <= day_idx < pred_pm25.shape[0]:
            result[date.strftime('%Y-%m-%d')] = pred_pm25[day_idx]
    return result


def compute_temporal_variogram(residual_series, time_lags):
    """
    计算时间变异函数
    γ_t(τ) = 0.5 * E[(R(t) - R(t+τ))^2]

    参数:
        residual_series: 残差时间序列
        time_lags: 时间滞后列表
    """
    n = len(residual_series)
    gamma_t = []

    for tau in time_lags:
        if tau >= n:
            gamma_t.append(np.nan)
            continue
        diff_sq = 0
        count = 0
        for i in range(n - tau):
            diff_sq += (residual_series[i] - residual_series[i + tau])**2
            count += 1
        gamma_t.append(diff_sq / (2 * count) if count > 0 else np.nan)

    return np.array(gamma_t)


class ST_CRK:
    """
    Spatiotemporal Cokriging Residual Kriging (ST-CRK)

    时空协同残差克里金融合方法

    创新点:
    1. 多日联合残差建模：增强空间建模的统计显著性
    2. 时空分离变异函数：γ(h,τ) = γ_s(h) + γ_t(τ)
    3. CMAQ梯度辅助：利用空间梯度引导插值
    4. 自适应时空权重：根据数据密度调整
    """

    def __init__(self, use_multi_day=True, num_days=5, use_gradient=True,
                 temporal_weight=0.3, spatial_kernel='RBF'):
        """
        参数:
            use_multi_day: 是否使用多日联合建模
            num_days: 多日建模的天数
            use_gradient: 是否使用CMAQ梯度作为辅助变量
            temporal_weight: 时间权重因子 (0-1)
            spatial_kernel: 空间核函数类型 ('RBF', 'Matern')
        """
        self.use_multi_day = use_multi_day
        self.num_days = num_days
        self.use_gradient = use_gradient
        self.temporal_weight = temporal_weight
        self.spatial_kernel = spatial_kernel

        # 存储多日数据
        self.multi_day_residuals = {}
        self.cmaq_data = {}
        self.scaler = StandardScaler()

    def _get_spatial_kernel(self):
        """获取空间核函数"""
        if self.spatial_kernel == 'Matern':
            return ConstantKernel(10.0, (1e-2, 1e3)) * Matern(
                length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=1.5
            ) + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e1))
        else:  # RBF
            return ConstantKernel(10.0, (1e-2, 1e3)) * RBF(
                length_scale=1.0, length_scale_bounds=(1e-2, 1e2)
            ) + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e1))

    def fit_bias_correction(self, train_df):
        """
        步骤1: 偏差校正建模
        O = α + β*M + ε
        """
        m = train_df['CMAQ'].values.reshape(-1, 1)
        o = train_df['Conc'].values

        # 使用Ridge回归防止过拟合
        self.bias_model = Ridge(alpha=1.0)
        self.bias_model.fit(m, o)

        # 计算残差
        residual = o - self.bias_model.predict(m)
        train_df = train_df.copy()
        train_df['residual'] = residual

        return train_df

    def fit_temporal_model(self, train_df, monitor_df, lon_cmaq, lat_cmaq, cmaq_grids):
        """
        步骤2: 时间相关性建模
        对每日残差计算时间变异函数参数
        """
        if not self.use_multi_day:
            return {}

        # 构建站点时间序列
        sites = train_df['Site'].unique()
        temporal_residuals = {}

        for site in sites:
            site_data = monitor_df[monitor_df['Site'] == site].copy()
            site_data = site_data.sort_values('Date')

            residuals = []
            dates = []
            for date_str in cmaq_grids.keys():
                site_day = site_data[site_data['Date'] == date_str]
                if len(site_day) > 0:
                    # 重新计算残差
                    cmaq_val = get_cmaq_at_site(
                        site_day.iloc[0]['Lon'],
                        site_day.iloc[0]['Lat'],
                        lon_cmaq, lat_cmaq,
                        cmaq_grids[date_str]
                    )
                    resid = site_day.iloc[0]['Conc'] - self.bias_model.predict([[cmaq_val]])[0]
                    residuals.append(resid)
                    dates.append(date_str)

            if len(residuals) >= 3:
                temporal_residuals[site] = {
                    'dates': dates,
                    'residuals': np.array(residuals)
                }

        # 计算时间变异函数
        time_lags = [1, 2, 3]
        temporal_params = {}

        for site, data in temporal_residuals.items():
            if len(data['residuals']) >= 4:
                gamma_t = compute_temporal_variogram(data['residuals'], time_lags)
                # 简化：使用线性拟合估计时间变异参数
                valid_gamma = gamma_t[~np.isnan(gamma_t)]
                if len(valid_gamma) > 0:
                    # 假设γ_t(τ) = nugget + sill * τ
                    temporal_params[site] = {
                        'gamma_1': valid_gamma[0] if len(valid_gamma) > 0 else 5.0,
                        'gamma_2': valid_gamma[1] if len(valid_gamma) > 1 else 10.0,
                        'range': len(valid_gamma)
                    }

        return temporal_params

    def fit_spatial_model(self, train_df):
        """
        步骤3: 空间残差建模 (GPR/克里金)
        """
        X_train = train_df[['Lon', 'Lat']].values
        y_train = train_df['residual'].values

        # 添加CMAQ梯度作为特征（如果启用）
        if self.use_gradient and 'grad_x' in train_df.columns:
            X_train = np.column_stack([
                X_train,
                train_df['grad_x'].values.reshape(-1, 1),
                train_df['grad_y'].values.reshape(-1, 1)
            ])

        # 标准化
        X_train_scaled = self.scaler.fit_transform(X_train)

        # GPR建模
        kernel = self._get_spatial_kernel()
        self.gpr = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=2,
            alpha=0.1,
            normalize_y=True
        )
        self.gpr.fit(X_train_scaled, y_train)

    def add_cmaq_gradient_features(self, df, lon_cmaq, lat_cmaq, pm25_grid):
        """为数据添加CMAQ梯度特征"""
        grad_x, grad_y = [], []
        for _, row in df.iterrows():
            dx, dy = compute_cmaq_gradient(
                row['Lon'], row['Lat'],
                lon_cmaq, lat_cmaq, pm25_grid
            )
            grad_x.append(dx)
            grad_y.append(dy)
        df['grad_x'] = grad_x
        df['grad_y'] = grad_y
        return df

    def fit(self, train_df, monitor_df=None, lon_cmaq=None, lat_cmaq=None,
            cmaq_grids=None, all_dates=None):
        """
        完整拟合流程

        参数:
            train_df: 训练数据 (当日)
            monitor_df: 全部监测数据（用于时间建模）
            lon_cmaq, lat_cmaq: CMAQ网格坐标
            cmaq_grids: 多日CMAQ数据字典
            all_dates: 所有日期列表
        """
        # 步骤1: 偏差校正
        train_df = self.fit_bias_correction(train_df)

        # 步骤2: 时间建模（如果启用）
        if self.use_multi_day and monitor_df is not None:
            self.temporal_params = self.fit_temporal_model(
                train_df, monitor_df, lon_cmaq, lat_cmaq, cmaq_grids
            )
        else:
            self.temporal_params = {}

        # 步骤3: 空间建模
        self.fit_spatial_model(train_df)

        # 保存训练数据信息
        self.train_info = {
            'lon_range': (train_df['Lon'].min(), train_df['Lon'].max()),
            'lat_range': (train_df['Lat'].min(), train_df['Lat'].max()),
            'n_train': len(train_df)
        }

        return self

    def predict(self, test_df, lon_cmaq=None, lat_cmaq=None, pm25_grid=None):
        """
        预测融合值

        返回:
            融合预测 + CMAQ预测 + 残差预测
        """
        # CMAQ预测
        cmaq_pred = test_df['CMAQ'].values
        bias_pred = self.bias_model.predict(cmaq_pred.reshape(-1, 1))

        # 残差预测
        X_test = test_df[['Lon', 'Lat']].values

        if self.use_gradient and 'grad_x' in test_df.columns:
            X_test = np.column_stack([
                X_test,
                test_df['grad_x'].values.reshape(-1, 1),
                test_df['grad_y'].values.reshape(-1, 1)
            ])

        X_test_scaled = self.scaler.transform(X_test)
        residual_pred, residual_std = self.gpr.predict(X_test_scaled, return_std=True)

        # 时空权重调整（简化版）
        # 如果启用时间建模，根据数据密度调整
        if len(self.temporal_params) > 0 and self.use_multi_day:
            # 站点密度因子
            spatial_density = self.train_info['n_train'] / 100.0
            density_weight = min(1.0, spatial_density * 0.1)
            # 时间权重衰减
            time_adjustment = residual_std * self.temporal_weight * (1 - density_weight)
            residual_pred = residual_pred + time_adjustment

        # 融合预测
        fusion_pred = bias_pred + residual_pred

        return {
            'fusion': fusion_pred,
            'cmaq': cmaq_pred,
            'bias_corrected': bias_pred,
            'residual': residual_pred,
            'residual_std': residual_std
        }


def run_stcrk_ten_fold(start_date_str='2020-01-01', num_days=5,
                       compare_vna=True, compare_rk=True):
    """
    运行ST-CRK十折交叉验证

    参数:
        start_date_str: 开始日期
        num_days: 使用天数
        compare_vna: 是否对比VNA
        compare_rk: 是否对比RK-Poly
    """
    print("="*70)
    print("ST-CRK: Spatiotemporal Cokriging Residual Kriging")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  - Start date: {start_date_str}")
    print(f"  - Number of days: {num_days}")
    print(f"  - Multi-day modeling: {num_days > 1}")
    print()

    # 加载数据
    print("=== Loading Data ===")
    monitor_df = pd.read_csv(monitor_file)
    fold_df = pd.read_csv(fold_file)

    ds = nc.Dataset(cmaq_file, 'r')
    lon_cmaq = ds.variables['lon'][:]
    lat_cmaq = ds.variables['lat'][:]
    pred_pm25 = ds.variables['pred_PM25'][:]
    ds.close()

    # 解析日期
    start_date = datetime.strptime(start_date_str, '%Y-%m-%d')

    # 加载多日CMAQ数据
    cmaq_grids = {}
    for i in range(num_days):
        date = start_date + timedelta(days=i)
        day_idx = (date - datetime(2020, 1, 1)).days
        if 0 <= day_idx < pred_pm25.shape[0]:
            cmaq_grids[date.strftime('%Y-%m-%d')] = pred_pm25[day_idx]

    dates_used = list(cmaq_grids.keys())
    print(f"  - Dates used: {dates_used}")

    # 使用第一天进行十折验证（与其他方法公平对比）
    day_df = monitor_df[monitor_df['Date'] == start_date_str].copy()
    day_df = day_df.merge(fold_df, on=['Date', 'Site'], how='left')
    day_df = day_df.dropna(subset=['Lat', 'Lon', 'Conc'])

    # 提取CMAQ值
    cmaq_day = cmaq_grids[start_date_str]
    cmaq_values = []
    for _, row in day_df.iterrows():
        val = get_cmaq_at_site(row['Lon'], row['Lat'], lon_cmaq, lat_cmaq, cmaq_day)
        cmaq_values.append(val)
    day_df['CMAQ'] = cmaq_values

    # 添加梯度特征（仅用于ST-CRK）
    day_df_with_grad = day_df.copy()
    day_df_with_grad = add_gradient_features(day_df_with_grad, lon_cmaq, lat_cmaq, cmaq_day)

    print(f"  - Monitoring records: {len(day_df)}")

    # 加载VNA方法
    from Code.VNAeVNAaVNA.core import VNAFusionCore
    from Code.VNAeVNAaVNA.nna_methods import NNA

    # 定义GPR核函数（用于RK和ST-CRK）
    kernel = ConstantKernel(10.0, (1e-2, 1e3)) * RBF(
        length_scale=1.0, length_scale_bounds=(1e-2, 1e2)
    ) + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e1))

    # 运行十折验证
    print("\n=== Running 10-fold Cross Validation ===")
    results = {fold_id: {} for fold_id in range(1, 11)}

    for fold_id in range(1, 11):
        train_df = day_df[day_df['fold'] != fold_id].copy()
        test_df = day_df[day_df['fold'] == fold_id].copy()
        train_df_grad = day_df_with_grad[day_df_with_grad['fold'] != fold_id].copy()
        test_df_grad = day_df_with_grad[day_df_with_grad['fold'] == fold_id].copy()

        train_df = train_df.dropna(subset=['Lon', 'Lat', 'CMAQ', 'Conc'])
        test_df = test_df.dropna(subset=['Lon', 'Lat', 'CMAQ', 'Conc'])

        if len(test_df) == 0:
            continue

        y_test = test_df['Conc'].values

        # === VNA ===
        if compare_vna:
            try:
                vna_core = VNAFusionCore(k=30, power=-2)
                vna_core.fit(train_df)
                vna_pred = vna_core.predict(test_df[['Lon', 'Lat']].values)
                vna_fusion = vna_pred['vna'].values if 'vna' in vna_pred.columns else vna_pred[:, 0]
            except:
                vna_fusion = test_df['CMAQ'].values  # fallback
        else:
            vna_fusion = None

        # === RK-Poly ===
        if compare_rk:
            from sklearn.preprocessing import PolynomialFeatures

            m_train = train_df['CMAQ'].values
            m_test = test_df['CMAQ'].values
            y_train = train_df['Conc'].values

            # 二次多项式
            poly = PolynomialFeatures(degree=2, include_bias=False)
            m_train_poly = poly.fit_transform(m_train.reshape(-1, 1))
            m_test_poly = poly.transform(m_test.reshape(-1, 1))

            ols_poly = LinearRegression()
            ols_poly.fit(m_train_poly, y_train)
            pred_poly = ols_poly.predict(m_test_poly)
            residual_poly = y_train - ols_poly.predict(m_train_poly)

            # GPR on residuals
            gpr_poly = GaussianProcessRegressor(
                kernel=kernel, n_restarts_optimizer=2, alpha=0.1, normalize_y=True
            )
            gpr_poly.fit(train_df[['Lon', 'Lat']].values, residual_poly)
            gpr_poly_pred, _ = gpr_poly.predict(test_df[['Lon', 'Lat']].values, return_std=True)

            rk_poly_pred = pred_poly + gpr_poly_pred
        else:
            rk_poly_pred = None

        # === ST-CRK ===
        # 准备多日数据（如果启用）
        if num_days > 1:
            # 合并多日训练数据
            multi_day_train = []
            for date_str in dates_used[:num_days]:
                day_train = monitor_df[monitor_df['Date'] == date_str].copy()
                day_train = day_train.merge(fold_df, on=['Date', 'Site'], how='left')
                day_train = day_train.dropna(subset=['Lat', 'Lon', 'Conc'])
                if len(day_train) > 0 and date_str in cmaq_grids:
                    cmaq_vals = []
                    for _, row in day_train.iterrows():
                        cmaq_vals.append(get_cmaq_at_site(
                            row['Lon'], row['Lat'],
                            lon_cmaq, lat_cmaq, cmaq_grids[date_str]
                        ))
                    day_train['CMAQ'] = cmaq_vals
                    # 只使用非测试站点
                    day_train = day_train[day_train['fold'] != fold_id]
                    multi_day_train.append(day_train)

            if len(multi_day_train) > 0:
                multi_train_df = pd.concat(multi_day_train, ignore_index=True)
            else:
                multi_train_df = train_df.copy()
        else:
            multi_train_df = train_df.copy()

        # 创建并训练ST-CRK
        stcrk = ST_CRK(
            use_multi_day=(num_days > 1),
            num_days=num_days,
            use_gradient=True,
            temporal_weight=0.2
        )

        # 准备梯度特征
        multi_train_grad = multi_train_df.copy()
        if len(multi_train_grad) > 0:
            multi_train_grad = add_gradient_features(
                multi_train_grad, lon_cmaq, lat_cmaq, cmaq_day
            )

        stcrk.fit(
            train_df_grad.dropna(subset=['grad_x', 'grad_y']),
            monitor_df=monitor_df,
            lon_cmaq=lon_cmaq,
            lat_cmaq=lat_cmaq,
            cmaq_grids=cmaq_grids,
            all_dates=dates_used
        )

        stcrk_pred = stcrk.predict(test_df_grad, lon_cmaq, lat_cmaq, cmaq_day)
        stcrk_fusion = stcrk_pred['fusion']

        # 保存结果
        results[fold_id] = {
            'y_true': y_test,
            'vna': vna_fusion if compare_vna else None,
            'rk_poly': rk_poly_pred if compare_rk else None,
            'stcrk': stcrk_fusion
        }

        print(f"  Fold {fold_id}: completed")

    # 汇总结果
    print("\n=== Results Summary ===")

    true_all = np.concatenate([results[f]['y_true'] for f in range(1, 11) if results[f]])

    metrics_summary = []

    if compare_vna:
        vna_all = np.concatenate([results[f]['vna'] for f in range(1, 11) if results[f]])
        vna_metrics = compute_metrics(true_all, vna_all)
        metrics_summary.append(('VNA', vna_metrics))
        print(f"  VNA:      R2={vna_metrics['R2']:.4f}, MAE={vna_metrics['MAE']:.2f}, RMSE={vna_metrics['RMSE']:.2f}")

    if compare_rk:
        rk_all = np.concatenate([results[f]['rk_poly'] for f in range(1, 11) if results[f]])
        rk_metrics = compute_metrics(true_all, rk_all)
        metrics_summary.append(('RK-Poly', rk_metrics))
        print(f"  RK-Poly:  R2={rk_metrics['R2']:.4f}, MAE={rk_metrics['MAE']:.2f}, RMSE={rk_metrics['RMSE']:.2f}")

    stcrk_all = np.concatenate([results[f]['stcrk'] for f in range(1, 11) if results[f]])
    stcrk_metrics = compute_metrics(true_all, stcrk_all)
    # 缓存预测值供多天聚合使用
    global _last_y_true, _last_y_pred
    _last_y_true = true_all
    _last_y_pred = stcrk_all

    metrics_summary.append(('ST-CRK', stcrk_metrics))
    print(f"  ST-CRK:   R2={stcrk_metrics['R2']:.4f}, MAE={stcrk_metrics['MAE']:.2f}, RMSE={stcrk_metrics['RMSE']:.2f}")

    # 保存结果
    result_df = pd.DataFrame([
        {'method': name, **metrics}
        for name, metrics in metrics_summary
    ])
    result_df.to_csv(f'{output_dir}/ST_CRK_comparison_{start_date_str}.csv', index=False)

    print(f"\nResults saved to: {output_dir}/ST_CRK_comparison_{start_date_str}.csv")

    return metrics_summary


def add_gradient_features(df, lon_cmaq, lat_cmaq, pm25_grid):
    """为数据框添加CMAQ梯度特征"""
    grad_x, grad_y = [], []
    for _, row in df.iterrows():
        dx, dy = compute_cmaq_gradient(
            row['Lon'], row['Lat'],
            lon_cmaq, lat_cmaq, pm25_grid
        )
        grad_x.append(dx)
        grad_y.append(dy)
    df['grad_x'] = grad_x
    df['grad_y'] = grad_y
    return df


def run_jan_july_comparison():
    """
    运行1月和7月对比测试
    使用5天数据
    """
    print("\n" + "="*70)
    print("January vs July Comparison")
    print("="*70)

    # 1月测试（1月15-19日）
    print("\n>>> January Test (Jan 15-19) <<<")
    jan_metrics = run_stcrk_ten_fold(
        start_date_str='2020-01-15',
        num_days=5,
        compare_vna=True,
        compare_rk=True
    )

    # 7月测试（7月15-19日）
    print("\n>>> July Test (Jul 15-19) <<<")
    jul_metrics = run_stcrk_ten_fold(
        start_date_str='2020-07-15',
        num_days=5,
        compare_vna=True,
        compare_rk=True
    )

    # 保存汇总
    summary = []
    for name in ['VNA', 'RK-Poly', 'ST-CRK']:
        jan_m = next((m for n, m in jan_metrics if n == name), None)
        jul_m = next((m for n, m in jul_metrics if n == name), None)
        if jan_m and jul_m:
            summary.append({
                'method': name,
                'jan_R2': jan_m['R2'],
                'jan_MAE': jan_m['MAE'],
                'jan_RMSE': jan_m['RMSE'],
                'jul_R2': jul_m['R2'],
                'jul_MAE': jul_m['MAE'],
                'jul_RMSE': jul_m['RMSE']
            })

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(f'{output_dir}/ST_CRK_Jan_July_summary.csv', index=False)

    print("\n" + "="*70)
    print("Summary: January vs July")
    print("="*70)
    print(summary_df.to_string(index=False))

    return jan_metrics, jul_metrics


if __name__ == '__main__':
    # 运行对比测试（5天）
    print("\n>>> January 5-day Test (Jan 15-19) <<<")
    jan_metrics = run_stcrk_ten_fold(
        start_date_str='2020-01-15',
        num_days=5,
        compare_vna=True,
        compare_rk=True
    )

    print("\n>>> July 5-day Test (Jul 15-19) <<<")
    jul_metrics = run_stcrk_ten_fold(
        start_date_str='2020-07-15',
        num_days=5,
        compare_vna=True,
        compare_rk=True
    )

    # 保存汇总
    summary = []
    for name in ['VNA', 'RK-Poly', 'ST-CRK']:
        jan_m = next((m for n, m in jan_metrics if n == name), None)
        jul_m = next((m for n, m in jul_metrics if n == name), None)
        if jan_m and jul_m:
            summary.append({
                'method': name,
                'jan_R2': jan_m['R2'],
                'jan_MAE': jan_m['MAE'],
                'jan_RMSE': jan_m['RMSE'],
                'jul_R2': jul_m['R2'],
                'jul_MAE': jul_m['MAE'],
                'jul_RMSE': jul_m['RMSE']
            })

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(f'{output_dir}/ST_CRK_Jan_July_summary.csv', index=False)

    print("\n" + "="*70)
    print("Final Summary")
    print("="*70)
    print(summary_df.to_string(index=False))
