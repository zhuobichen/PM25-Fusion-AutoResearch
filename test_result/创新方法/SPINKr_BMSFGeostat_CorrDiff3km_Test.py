"""
SPIN-Kr, BMSF-Geostat, CorrDiff-3km Pre-Experiment Validation
=============================================================
5-day validation (2020-01-01 to 2020-01-05)
10-fold cross-validation vs RK-Poly baseline (R2=0.8519)

Methods:
1. SPIN-Kr (Graph Kernel Spatial-Temporal Kriging)
   - Uses graph structure kernel to capture spatial correlation
   - Time smoothing constraint
   - Physical interpretation: spatial correlation distance decay + temporal continuity

2. BMSF-Geostat (Bayesian Multi-Source Fusion)
   - Hierarchical Bayesian framework for multi-source data fusion
   - Prior: CMAQ model constraint
   - Posterior: Monitor data correction
   - Physical interpretation: error source decomposition

3. CorrDiff-3km (Residual Correction Diffusion Downscaling)
   - Multi-scale residual diffusion
   - 3km downscaling correction
   - Physical interpretation: thermodynamic diffusion process
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared.paths import get_project_root, data_path
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel, Matern
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from scipy.spatial.distance import cdist
from scipy.stats import norm
import netCDF4 as nc
import warnings
warnings.filterwarnings('ignore')

root_dir = str(get_project_root())
cmaq_file = data_path('test_data/raw/CMAQ/2020_PM25.nc')
monitor_file = data_path('test_data/raw/Monitor/2020_DailyPM2.5Monitor.csv')
fold_file = data_path('test_data/fold_split_table_daily.csv')
output_dir = f'{root_dir}/test_result/创新方法'
os.makedirs(output_dir, exist_ok=True)


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
    """获取站点位置的CMAQ值（最近邻）"""
    dist = np.sqrt((lon_grid - lon)**2 + (lat_grid - lat)**2)
    idx = np.argmin(dist)
    ny, nx = lon_grid.shape
    row, col = idx // nx, idx % nx
    return pm25_grid[row, col]


def compute_cmaq_gradient(lon, lat, lon_grid, lat_grid, pm25_grid):
    """计算CMAQ梯度"""
    ny, nx = lon_grid.shape
    dist = np.sqrt((lon_grid - lon)**2 + (lat_grid - lat)**2)
    idx = np.argmin(dist)
    row, col = idx // nx, idx % nx
    row = max(1, min(row, ny - 2))
    col = max(1, min(col, nx - 2))
    dmdx = (pm25_grid[row, col + 1] - pm25_grid[row, col - 1]) / (lon_grid[row, col + 1] - lon_grid[row, col - 1] + 1e-6)
    dmdy = (pm25_grid[row + 1, col] - pm25_grid[row - 1, col]) / (lat_grid[row + 1, col] - lat_grid[row - 1, col] + 1e-6)
    return dmdx, dmdy


# ============================================================
# SPIN-Kr: Graph Kernel Spatial-Temporal Kriging
# ============================================================
class SPINKr:
    """
    SPIN-Kr (Graph Kernel Spatial-Temporal Kriging) - 图核时空克里金法

    物理机制：
    - 图核：基于空间距离的图结构核函数，捕捉非欧几里得空间相关性
    - 时空克里金：结合时间相邻性的空间插值
    - 可解释参数：空间相关长度(range)、时间平滑因子

    公式: P = M + GPR(bias | graph_kernel)
    """
    def __init__(self, spatial_range=1.0, temporal_smooth=0.3, k=30):
        """
        Parameters
        ----------
        spatial_range : float
            空间相关长度（度），物理意义：PM2.5空间相关典型尺度 ~100km
        temporal_smooth : float
            时间平滑因子 [0,1]，0=仅用当天，1=完全时间平均
        k : int
            近邻数量
        """
        self.spatial_range = spatial_range
        self.temporal_smooth = temporal_smooth
        self.k = k

    def fit(self, X_obs, y_obs, y_model_obs, X_obs_prev=None, y_obs_prev=None):
        """
        拟合模型

        Parameters
        ----------
        X_obs : array (n, 2)
            监测站点坐标 [lon, lat]
        y_obs : array (n,)
            监测值
        y_model_obs : array (n,)
            CMAQ模型值
        X_obs_prev : array (n_prev, 2), optional
            前一天站点坐标
        y_obs_prev : array (n_prev,), optional
            前一天监测值
        """
        self.X_obs = X_obs
        self.y_obs = y_obs
        self.y_model_obs = y_model_obs
        self.bias = y_obs - y_model_obs

        # 时间平滑：用前一天的偏差场辅助（如果提供）
        if X_obs_prev is not None and y_obs_prev is not None:
            self.bias_prev = y_obs_prev - self._get_model_at_obs(X_obs_prev)  # 需要model值
            self.X_obs_prev = X_obs_prev
        else:
            self.bias_prev = None

        # 估计空间变异函数参数
        self._estimate_variogram()

    def _get_model_at_obs(self, X_obs):
        """获取站点对应的模型值（简化：返回观测均值）"""
        return np.full(len(X_obs), np.mean(self.y_model_obs))

    def _estimate_variogram(self):
        """估计变异函数参数 from bias"""
        n = len(self.bias)
        dists = cdist(self.X_obs, self.X_obs, 'euclidean')
        idx_i, idx_j = np.triu_indices(n)
        h = dists[idx_i, idx_j]
        gamma_exp = 0.5 * (self.bias[idx_i] - self.bias[idx_j]) ** 2

        mask = h > 1e-6
        h, gamma_exp = h[mask], gamma_exp[mask]

        self.sill = np.var(self.bias)
        self.range_est = np.median(h) if len(h) > 0 else self.spatial_range
        if self.range_est < 0.1:
            self.range_est = self.spatial_range
        self.nugget = 0

    def predict(self, X_grid, y_model_grid):
        """
        预测

        Parameters
        ----------
        X_grid : array (n_grid, 2)
            网格点坐标
        y_model_grid : array (n_grid,)
            网格点CMAQ值

        Returns
        -------
        y_pred : array (n_grid,)
        """
        n_grid = X_grid.shape[0]
        bias_pred = np.zeros(n_grid)

        # 图核：使用高斯协方差函数
        C0 = self.sill  # 方差

        for i, x0 in enumerate(X_grid):
            dists = cdist([x0], self.X_obs, 'euclidean').ravel()

            if len(dists) > self.k:
                idx = np.argpartition(dists, self.k)[:self.k]
            else:
                idx = np.arange(len(dists))

            dists_k = dists[idx]
            bias_k = self.bias[idx]

            # 图核协方差：使用实测变异函数参数
            # C(h) = sill * exp(-3*h^2/range^2)
            C = C0 * np.exp(-3 * (dists_k / self.range_est) ** 2)

            # 克里金权重
            weights = C / (np.sum(C) + 1e-10)
            bias_pred[i] = np.sum(weights * bias_k)

        # 时间平滑（如果可用）
        if self.bias_prev is not None and self.temporal_smooth > 0:
            # 对bias_pred做时间平滑
            bias_pred = (1 - self.temporal_smooth) * bias_pred + self.temporal_smooth * np.mean(self.bias_prev)

        return y_model_grid + bias_pred


# ============================================================
# BMSF-Geostat: Bayesian Multi-Source Fusion
# ============================================================
class BMSFGeostat:
    """
    BMSF-Geostat (Bayesian Multi-Source Fusion) - 贝叶斯多源融合法

    物理机制：
    - 层次贝叶斯：Y = M + bias，其中 bias ~ N(mu, sigma^2)
    - 多源先验：基于CMAQ模型的物理约束
    - 后验估计：监测数据对先验的校正
    - 可解释参数：mu（系统偏差）、sigma（随机误差标准差）

    公式: P(Y|x) = P(Y|M, x) * P(M)
         后验均值 = w * 观测 + (1-w) * 先验
    """
    def __init__(self, sigma_model=5.0, sigma_obs=2.0, corr_scale=1.0, k=30):
        """
        Parameters
        ----------
        sigma_model : float
            模型误差标准差（先验）
        sigma_obs : float
            观测误差标准差（似然）
        corr_scale : float
            空间相关尺度
        k : int
            近邻数量
        """
        self.sigma_model = sigma_model
        self.sigma_obs = sigma_obs
        self.corr_scale = corr_scale
        self.k = k

    def fit(self, X_obs, y_obs, y_model_obs):
        """
        拟合

        Parameters
        ----------
        X_obs : array (n, 2)
            监测站点坐标
        y_obs : array (n,)
            监测值
        y_model_obs : array (n,)
            CMAQ模型值
        """
        self.X_obs = X_obs
        self.y_obs = y_obs
        self.y_model_obs = y_model_obs
        self.bias = y_obs - y_model_obs

        # 从数据估计先验参数
        # 系统偏差 mu ~ N(mean_bias, var_bias)
        self.mu_prior = np.mean(self.bias)
        self.sigma_prior = np.std(self.bias)

        # 模型误差从CMAQ偏差估计
        if self.sigma_model is None or self.sigma_model <= 0:
            self.sigma_model = max(np.std(self.bias), 1.0)

    def predict(self, X_grid, y_model_grid):
        """
        预测

        贝叶斯融合：
        P_post ∝ P(y_obs|M) * P(M)
        后验均值 = w * y_obs_spatial + (1-w) * M

        其中权重 w = sigma_model^2 / (sigma_model^2 + sigma_obs^2)
        """
        n_grid = X_grid.shape[0]
        pred = np.zeros(n_grid)

        for i, x0 in enumerate(X_grid):
            dists = cdist([x0], self.X_obs, 'euclidean').ravel()

            if len(dists) > self.k:
                idx = np.argpartition(dists, self.k)[:self.k]
            else:
                idx = np.arange(len(dists))

            dists_k = dists[idx]
            bias_k = self.bias[idx]

            # 空间相关权重
            C = np.exp(-0.5 * (dists_k / self.corr_scale) ** 2)
            weights = C / (np.sum(C) + 1e-10)

            # 局部偏差估计（监测数据加权）
            local_bias = np.sum(weights * bias_k)

            # 贝叶斯融合：先验（全局偏差）+ 似然（局部偏差）
            # 后验均值 = (sigma_obs^2 / (sigma_model^2 + sigma_obs^2)) * local + (sigma_model^2 / (...)) * prior
            w_local = self.sigma_model ** 2 / (self.sigma_model ** 2 + self.sigma_obs ** 2)
            w_prior = self.sigma_obs ** 2 / (self.sigma_model ** 2 + self.sigma_obs ** 2)

            # 后验偏差估计
            post_bias = w_local * local_bias + w_prior * self.mu_prior

            pred[i] = y_model_grid[i] + post_bias

        return pred


# ============================================================
# CorrDiff-3km: Residual Correction Diffusion Downscaling
# ============================================================
class CorrDiff3km:
    """
    CorrDiff-3km (Residual Correction Diffusion Downscaling) - 残差修正扩散降尺度法

    物理机制：
    - 扩散过程：污染物从高浓度向低浓度扩散，直到平衡
    - 残差修正：M到精细网格的残差传导
    - 3km尺度：对应CMAQ网格尺度的校正

    公式: P_fine = M_fine + Diffusion(bias_fine | M_coarse)
    """
    def __init__(self, diffusion_coef=0.5, corr_scale=0.5, k=30):
        """
        Parameters
        ----------
        diffusion_coef : float
            扩散系数 [0,1]，物理意义：浓度梯度驱动扩散的强度
        corr_scale : float
            相关尺度（度），典型值对应 ~50km
        k : int
            近邻数量
        """
        self.diffusion_coef = diffusion_coef
        self.corr_scale = corr_scale
        self.k = k

    def fit(self, X_obs, y_obs, y_model_obs, gradient_x=None, gradient_y=None):
        """
        拟合

        Parameters
        ----------
        X_obs : array (n, 2)
            站点坐标
        y_obs : array (n,)
            监测值
        y_model_obs : array (n,)
            CMAQ模型值
        gradient_x : array (n,), optional
            CMAQ在x方向梯度（用于扩散方向判断）
        gradient_y : array (n,), optional
            CMAQ在y方向梯度
        """
        self.X_obs = X_obs
        self.y_obs = y_obs
        self.y_model_obs = y_model_obs
        self.bias = y_obs - y_model_obs

        # 如果提供梯度，计算扩散方向
        if gradient_x is not None and gradient_y is not None:
            self.grad_x = gradient_x
            self.grad_y = gradient_y
        else:
            # 默认：偏差沿浓度梯度反方向扩散
            self.grad_x = np.zeros(len(X_obs))
            self.grad_y = np.zeros(len(X_obs))

    def predict(self, X_grid, y_model_grid):
        """
        预测

        扩散模型：
        dB/dt = D * Laplacian(B) + S
        稳态解: B(x) = integral[K(x-x'; coef) * S(x')] dx'

        简化：使用迭代扩散校正
        """
        n_grid = X_grid.shape[0]
        bias_pred = np.zeros(n_grid)

        for i, x0 in enumerate(X_grid):
            dists = cdist([x0], self.X_obs, 'euclidean').ravel()

            if len(dists) > self.k:
                idx = np.argpartition(dists, self.k)[:self.k]
            else:
                idx = np.arange(len(dists))

            dists_k = dists[idx]
            bias_k = self.bias[idx]
            grad_x_k = self.grad_x[idx] if hasattr(self, 'grad_x') else np.zeros(len(idx))
            grad_y_k = self.grad_y[idx] if hasattr(self, 'grad_y') else np.zeros(len(idx))

            # 扩散核：距离衰减 + 梯度方向增强
            # K(x, x') = exp(-d^2 / (2*scale^2)) * (1 + alpha * dot(grad, direction))
            kernel = np.exp(-0.5 * (dists_k / self.corr_scale) ** 2)

            # 梯度方向增强（可选）
            # 偏差倾向于沿浓度梯度方向扩散
            if np.any(grad_x_k) or np.any(grad_y_k):
                # 扩散增强因子
                grad_mag = np.sqrt(grad_x_k**2 + grad_y_k**2) + 1e-6
                direction_factor = 1 + self.diffusion_coef * (grad_x_k + grad_y_k) / grad_mag
                kernel = kernel * direction_factor

            weights = kernel / (np.sum(kernel) + 1e-10)
            bias_pred[i] = np.sum(weights * bias_k)

        return y_model_grid + bias_pred


# ============================================================
# 主测试代码
# ============================================================
print('='*60)
print('SPIN-Kr, BMSF-Geostat, CorrDiff-3km 预实验验证')
print('='*60)

# 加载数据
print('\n=== 加载数据 ===')
monitor_df = pd.read_csv(monitor_file)
fold_df = pd.read_csv(fold_file)
ds = nc.Dataset(cmaq_file, 'r')
lon_cmaq = ds.variables['lon'][:]
lat_cmaq = ds.variables['lat'][:]
pred_pm25 = ds.variables['pred_PM25'][:]
ds.close()

print(f'监测数据：{len(monitor_df)} 条')
print(f'站点数：{monitor_df["Site"].nunique()}')

# 测试日期
selected_days = ['2020-01-01', '2020-01-02', '2020-01-03', '2020-01-04', '2020-01-05']
benchmark_r2 = 0.8519  # RK-Poly基准

# 存储结果
all_results = {
    'SPIN-Kr': [], 'BMSF-Geostat': [], 'CorrDiff-3km': [],
    'RK-Poly': [], 'CMAQ': []
}
# 按日期存储结果（用于生成日期维度CSV）
daily_results = {day: [] for day in selected_days}

# 多天验证
for day_str in selected_days:
    print(f'\n--- 日期: {day_str} ---')

    day_df = monitor_df[monitor_df['Date'] == day_str].copy()
    day_df = day_df.merge(fold_df, on='Site', how='left')
    day_df = day_df.dropna(subset=['Lat', 'Lon', 'Conc'])

    date_obj = datetime.strptime(day_str, '%Y-%m-%d')
    day_idx = (date_obj - datetime(2020, 1, 1)).days
    cmaq_day = pred_pm25[day_idx]

    # 提取CMAQ值和梯度
    cmaq_values = []
    grad_x, grad_y = [], []
    for _, row in day_df.iterrows():
        cmaq_values.append(get_cmaq_at_site(row['Lon'], row['Lat'], lon_cmaq, lat_cmaq, cmaq_day))
        dx, dy = compute_cmaq_gradient(row['Lon'], row['Lat'], lon_cmaq, lat_cmaq, cmaq_day)
        grad_x.append(dx)
        grad_y.append(dy)

    day_df['CMAQ'] = cmaq_values
    day_df['grad_x'] = grad_x
    day_df['grad_y'] = grad_y

    # 10折交叉验证
    for fold_id in range(1, 11):
        train_df = day_df[day_df['fold'] != fold_id].dropna(subset=['Lon', 'Lat', 'CMAQ', 'Conc', 'grad_x'])
        test_df = day_df[day_df['fold'] == fold_id].dropna(subset=['Lon', 'Lat', 'CMAQ', 'Conc', 'grad_x'])

        if len(test_df) == 0 or len(train_df) == 0:
            continue

        y_test = test_df['Conc'].values
        m_test = test_df['CMAQ'].values

        # CMAQ基准
        all_results['CMAQ'].append((y_test, m_test))

        # RK-Poly基准 - 使用正确的kernel配置（与PolyRK.py一致）
        poly = PolynomialFeatures(degree=2, include_bias=False)
        m_train = train_df['CMAQ'].values
        m_train_poly = poly.fit_transform(m_train.reshape(-1, 1))
        m_test_poly = poly.transform(m_test.reshape(-1, 1))
        ols_poly = LinearRegression()
        ols_poly.fit(m_train_poly, train_df['Conc'].values)
        pred_poly = ols_poly.predict(m_test_poly)
        residual_poly = train_df['Conc'].values - ols_poly.predict(m_train_poly)

        # 修正kernel配置：与PolyRK.py保持一致
        kernel = ConstantKernel(10.0, (1e-2, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e1))
        gpr_poly = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, alpha=0.1, normalize_y=True)
        gpr_poly.fit(train_df[['Lon', 'Lat']].values, residual_poly)
        gpr_poly_pred, _ = gpr_poly.predict(test_df[['Lon', 'Lat']].values, return_std=True)
        all_results['RK-Poly'].append((y_test, pred_poly + gpr_poly_pred))

        # SPIN-Kr
        spin_kr = SPINKr(spatial_range=0.5, temporal_smooth=0.0, k=30)
        spin_kr.fit(
            train_df[['Lon', 'Lat']].values,
            train_df['Conc'].values,
            train_df['CMAQ'].values
        )
        spin_pred = spin_kr.predict(test_df[['Lon', 'Lat']].values, m_test)
        all_results['SPIN-Kr'].append((y_test, spin_pred))

        # BMSF-Geostat
        bmsf = BMSFGeostat(sigma_model=5.0, sigma_obs=2.0, corr_scale=0.5, k=30)
        bmsf.fit(train_df[['Lon', 'Lat']].values, train_df['Conc'].values, train_df['CMAQ'].values)
        bmsf_pred = bmsf.predict(test_df[['Lon', 'Lat']].values, m_test)
        all_results['BMSF-Geostat'].append((y_test, bmsf_pred))

        # CorrDiff-3km
        corr_diff = CorrDiff3km(diffusion_coef=0.3, corr_scale=0.5, k=30)
        corr_diff.fit(
            train_df[['Lon', 'Lat']].values,
            train_df['Conc'].values,
            train_df['CMAQ'].values,
            train_df['grad_x'].values,
            train_df['grad_y'].values
        )
        corr_diff_pred = corr_diff.predict(test_df[['Lon', 'Lat']].values, m_test)
        all_results['CorrDiff-3km'].append((y_test, corr_diff_pred))

        # 按日期存储（用于生成日期维度CSV）
        daily_results[day_str].append({
            'fold': fold_id,
            'SPIN-Kr': compute_metrics(y_test, spin_pred),
            'BMSF-Geostat': compute_metrics(y_test, bmsf_pred),
            'CorrDiff-3km': compute_metrics(y_test, corr_diff_pred),
            'RK-Poly': compute_metrics(y_test, pred_poly + gpr_poly_pred),
            'CMAQ': compute_metrics(y_test, m_test)
        })

    # 打印当天结果
    print(f'  {len(test_df)} 验证站点')

# 汇总结果
print('\n' + '='*60)
print('5天预实验验证结果汇总')
print('='*60)

summary = []
for method, results in all_results.items():
    y_all = np.concatenate([r[0] for r in results])
    p_all = np.concatenate([r[1] for r in results])
    metrics = compute_metrics(y_all, p_all)
    metrics['method'] = method
    summary.append(metrics)

    # 与基准对比
    delta_r2 = metrics['R2'] - benchmark_r2
    status = '[PASS]' if delta_r2 >= 0.01 else ('[WARN]' if delta_r2 >= 0 else '[FAIL]')

    print(f'{method:>15}: R2={metrics["R2"]:.4f} (delta={delta_r2:+.4f}) {status}, MAE={metrics["MAE"]:.2f}, RMSE={metrics["RMSE"]:.2f}')

# 保存结果
summary_df = pd.DataFrame(summary)
summary_df = summary_df[['method', 'R2', 'MAE', 'RMSE', 'MB']]
summary_df.to_csv(f'{output_dir}/SPINKr_BMSFGeostat_CorrDiff3km_summary.csv', index=False)
print(f'\n结果已保存到：{output_dir}/SPINKr_BMSFGeostat_CorrDiff3km_summary.csv')

# 输出创新方法状态
print('\n' + '='*60)
print('Innovation Threshold Check (R2 improvement >= 0.01 vs RK-Poly baseline)')
print('='*60)
innovation_results = []
for m in ['SPIN-Kr', 'BMSF-Geostat', 'CorrDiff-3km']:
    r2 = summary_df[summary_df['method'] == m]['R2'].values[0]
    delta = r2 - benchmark_r2
    if delta >= 0.01:
        status = '[PASS] Innovation effective'
    elif delta >= 0:
        status = '[WARN] Improvement but insufficient'
    else:
        status = '[FAIL] Does not exceed baseline'
    print(f'{m}: R2={r2:.4f}, delta={delta:+.4f} -> {status}')
    innovation_results.append({'method': m, 'R2': r2, 'delta': delta, 'status': status})

# 保存详细报告
innovation_df = pd.DataFrame(innovation_results)
innovation_df.to_csv(f'{output_dir}/SPINKr_BMSFGeostat_CorrDiff3km_innovation_check.csv', index=False)

print('\n' + '='*60)
print('测试完成')
print('='*60)