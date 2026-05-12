"""
V1_Kriging伪标签增强法 - 克里金伪标签增强法
============================================
Reproduction of Kriging-based Pseudo-Label Augmentation

核心思想：
  1. 普通克里金插值：用变异函数建模空间相关性，生成插值场
  2. 伪标签生成：对高置信度（低插值方差）位置生成伪标签
  3. 数据增强：用伪标签扩充训练集
  4. 回归预测：用增强数据训练模型进行融合预测

文献来源：
  Kriging-based Pseudo-Label Augmentation for PM2.5 Estimation
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.paths import get_project_root, data_path
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.ensemble import RandomForestRegressor
import netCDF4 as nc

root_dir = str(get_project_root())
cmaq_file = data_path('test_data/raw/CMAQ/2020_PM25.nc')
monitor_file = data_path('test_data/raw/Monitor/2020_DailyPM2.5Monitor.csv')
fold_file = data_path('test_data/fold_split_table_daily.csv')
output_dir = f'{root_dir}/test_result/复现方法'
os.makedirs(output_dir, exist_ok=True)


def compute_metrics(y_true, y_pred):
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
    dist = np.sqrt((lon_grid - lon)**2 + (lat_grid - lat)**2)
    idx = np.argmin(dist)
    ny, nx = lon_grid.shape
    row, col = idx // nx, idx % nx
    return pm25_grid[row, col]


class KrigingPseudoLabel:
    """
    克里金伪标签增强法

    步骤:
    1. 使用GPR作为克里金代理进行空间插值
    2. 对CMAQ网格点生成伪标签（高置信度区域）
    3. 用原始+伪标签数据训练RF回归器
    4. 预测测试站点

    变异函数建模：
    gamma(h) = c0 + c * Sph(h/a)
    GPR的RBF核近似球状变异函数的空间相关结构
    """

    def __init__(self, kriging_range=200.0, min_stations=10,
                 nugget=0.1, sill=1.0,
                 confidence_threshold=0.8,
                 augmentation_ratio=0.3,
                 n_estimators=100, max_depth=10):
        """
        Parameters:
        -----------
        kriging_range : float
            克里金搜索半径 (km)
        min_stations : int
            最少站点数
        nugget : float
            块金值
        sill : float
            拱高
        confidence_threshold : float
            伪标签置信度阈值
        augmentation_ratio : float
            伪标签样本比例
        n_estimators : int
            RF决策树数量
        max_depth : int
            RF最大深度
        """
        self.kriging_range = kriging_range
        self.min_stations = min_stations
        self.nugget = nugget
        self.sill = sill
        self.confidence_threshold = confidence_threshold
        self.augmentation_ratio = augmentation_ratio
        self.n_estimators = n_estimators
        self.max_depth = max_depth

    def _build_kriging_model(self, station_coords, station_obs):
        """
        构建克里金模型（使用GPR作为代理）

        变异函数：球状模型 gamma(h) = c0 + c * Sph(h/a)
        GPR核函数：ConstantKernel * RBF + WhiteKernel 近似
        """
        kernel = (ConstantKernel(self.sill, (1e-2, 1e2)) *
                  RBF(length_scale=self.kriging_range / 3,
                      length_scale_bounds=(1e-2, 1e3)) +
                  WhiteKernel(noise_level=self.nugget))

        gpr = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=3,
            alpha=self.nugget,
            normalize_y=True
        )
        gpr.fit(station_coords, station_obs)
        return gpr

    def _generate_pseudo_labels(self, gpr, candidate_coords, candidate_cmaq):
        """
        生成伪标签

        对候选位置进行克里金预测，筛选高置信度（低方差）结果作为伪标签。
        伪标签值：克里金插值结果（而非CMAQ值）
        """
        pred_mean, pred_std = gpr.predict(candidate_coords, return_std=True)

        # 置信度：标准化为0-1
        confidence = 1.0 / (1.0 + pred_std)

        # 筛选高置信度样本
        high_conf_mask = confidence > self.confidence_threshold

        if np.sum(high_conf_mask) == 0:
            # 降低阈值
            threshold = np.percentile(confidence, 70)
            high_conf_mask = confidence > threshold

        # 限制伪标签数量
        n_pseudo = max(1, int(len(candidate_coords) * self.augmentation_ratio))
        if np.sum(high_conf_mask) > n_pseudo:
            # 按置信度排序，取top-n
            conf_values = confidence.copy()
            conf_values[~high_conf_mask] = -1
            top_indices = np.argsort(conf_values)[-n_pseudo:]
            high_conf_mask = np.zeros(len(candidate_coords), dtype=bool)
            high_conf_mask[top_indices] = True

        pseudo_coords = candidate_coords[high_conf_mask]
        pseudo_values = pred_mean[high_conf_mask]  # 伪标签 = 克里金插值值

        return pseudo_coords, pseudo_values

    def fit_predict(self, train_coords, train_obs, train_cmaq,
                    test_coords, test_cmaq):
        """
        训练并预测

        Parameters:
        -----------
        train_coords : array (n_train, 2) - 训练站坐标 [lon, lat]
        train_obs : array (n_train,) - 训练站观测值
        train_cmaq : array (n_train,) - 训练站CMAQ值
        test_coords : array (n_test, 2) - 测试站坐标 [lon, lat]
        test_cmaq : array (n_test,) - 测试站CMAQ值

        Returns:
        --------
        predictions : array (n_test,) - 融合预测值
        """
        # 阶段1: 构建克里金模型
        gpr = self._build_kriging_model(train_coords, train_obs)

        # 阶段2: 在训练站附近生成伪标签候选点
        # 使用训练站坐标作为候选位置（模拟卫星覆盖区域）
        candidate_coords = train_coords.copy()
        candidate_cmaq = train_cmaq.copy()

        # 阶段3: 生成伪标签
        pseudo_coords, pseudo_values = self._generate_pseudo_labels(
            gpr, candidate_coords, candidate_cmaq
        )

        # 阶段4: 构建增强训练集
        # 原始数据: 特征 = [lon, lat, CMAQ]
        X_orig = np.column_stack([train_coords, train_cmaq])
        y_orig = train_obs

        # 伪标签数据: 特征 = [lon, lat, CMAQ_at_pseudo]
        # 伪标签位置的CMAQ值使用最近邻从原始训练数据插值
        if len(pseudo_coords) > 0:
            # 简化：伪标签位置的CMAQ值用最近训练站的CMAQ值
            dist_to_train = np.sqrt(
                ((pseudo_coords[:, 0:1] - train_coords[:, 0:1].T)**2 +
                 (pseudo_coords[:, 1:2] - train_coords[:, 1:2].T)**2)
            )
            nearest_idx = np.argmin(dist_to_train, axis=1)
            pseudo_cmaq = train_cmaq[nearest_idx]

            X_pseudo = np.column_stack([pseudo_coords, pseudo_cmaq])
            y_pseudo = pseudo_values

            # 合并
            X_aug = np.vstack([X_orig, X_pseudo])
            y_aug = np.concatenate([y_orig, y_pseudo])
        else:
            X_aug = X_orig
            y_aug = y_orig

        # 阶段5: 训练RF回归器
        rf = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=42
        )
        rf.fit(X_aug, y_aug)

        # 阶段6: 预测测试点
        X_test = np.column_stack([test_coords, test_cmaq])
        predictions = rf.predict(X_test)

        # 非负约束
        predictions = np.maximum(predictions, 0)

        return predictions


def run_KrigingPseudoLabel_ten_fold(selected_day='2020-01-01'):
    """
    运行克里金伪标签增强法十折交叉验证

    Parameters:
    -----------
    selected_day : str
        验证日期，格式 'YYYY-MM-DD'
    """
    print("=" * 60)
    print("Kriging Pseudo-Label Ten-Fold Cross Validation")
    print(f"Date: {selected_day}")
    print("=" * 60)

    # 加载数据
    print("\n=== Loading Data ===")
    monitor_df = pd.read_csv(monitor_file)
    fold_df = pd.read_csv(fold_file)

    day_df = monitor_df[monitor_df['Date'] == selected_day].copy()
    day_df = day_df.merge(fold_df, on='Site', how='left')
    day_df = day_df.dropna(subset=['Lat', 'Lon', 'Conc'])

    # 加载CMAQ数据
    ds = nc.Dataset(cmaq_file, 'r')
    lon_cmaq = ds.variables['lon'][:]
    lat_cmaq = ds.variables['lat'][:]
    pred_pm25 = ds.variables['pred_PM25'][:]
    ds.close()

    from datetime import datetime
    date_obj = datetime.strptime(selected_day, '%Y-%m-%d')
    day_idx = (date_obj - datetime(2020, 1, 1)).days
    pred_day = pred_pm25[day_idx]

    # 提取站点CMAQ值
    print("=== Extracting CMAQ at Sites ===")
    cmaq_values = []
    for _, row in day_df.iterrows():
        val = get_cmaq_at_site(row['Lon'], row['Lat'], lon_cmaq, lat_cmaq, pred_day)
        cmaq_values.append(val)
    day_df['CMAQ'] = cmaq_values

    print(f"Data loaded: {len(day_df)} monitoring records")

    # 十折交叉验证
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

        # 训练模型
        model = KrigingPseudoLabel(
            kriging_range=200.0,
            confidence_threshold=0.8,
            augmentation_ratio=0.3,
            n_estimators=100,
            max_depth=10
        )
        y_pred = model.fit_predict(X_train, y_train, m_train, X_test, m_test)

        results[fold_id] = {
            'y_true': y_test,
            'y_pred': y_pred
        }

        fold_metrics = compute_metrics(y_test, y_pred)
        print(f"  Fold {fold_id}: R2={fold_metrics['R2']:.4f}, "
              f"RMSE={fold_metrics['RMSE']:.2f}, N={len(test_df)}")

    # 汇总
    y_pred_all = np.concatenate([results[f]['y_pred'] for f in range(1, 11) if results[f]])
    y_true_all = np.concatenate([results[f]['y_true'] for f in range(1, 11) if results[f]])

    metrics = compute_metrics(y_true_all, y_pred_all)

    print("\n=== Results ===")
    print(f"  KrigingPseudoLabel: R2={metrics['R2']:.4f}, MAE={metrics['MAE']:.2f}, "
          f"RMSE={metrics['RMSE']:.2f}, MB={metrics['MB']:.4f}")

    # 保存结果
    result_df = pd.DataFrame([{
        'method': 'KrigingPseudoLabel',
        **metrics
    }])
    result_df.to_csv(f'{output_dir}/KrigingPseudoLabel_folds.csv', index=False)

    print(f"\nResults saved to: {output_dir}/")

    return metrics


if __name__ == '__main__':
    metrics = run_KrigingPseudoLabel_ten_fold('2020-01-01')
    print(f"\nKrigingPseudoLabel: R2={metrics['R2']:.4f}")
