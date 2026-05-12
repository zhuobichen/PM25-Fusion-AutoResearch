"""
V1_RF-Kriging随机森林克里金残差校正法 - 随机森林-克里金残差校正法
=================================================================
Reproduction of RF-Kriging (Random Forest with Kriging Residual Correction)

核心思想：
  两步法：
  1. 随机森林学习CMAQ->监测的非线性映射关系
  2. 克里金插值校正RF残差的空间结构

  物理可解释：RF捕获CMAQ系统偏差，克里金捕获空间相关残差

文献来源：
  Xue et al., "A three-step method to fuse satellite, CMAQ, and observation data"
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.paths import get_project_root, data_path
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
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


class RF_Kriging:
    """
    随机森林-克里金残差校正法

    步骤:
    1. 特征工程：构建 X = [CMAQ, lon, lat]
    2. 随机森林预测：y_RF = RF(X)
    3. 残差计算：r = y_obs - y_RF
    4. 变异函数拟合：gamma(h) = c0 + c * [1 - exp(-3h/a)]
       使用GPR的RBF核近似指数变异函数
    5. 克里金残差插值：r_hat = Kriging(r)
    6. 最终融合：y_final = y_RF + r_hat
    """

    def __init__(self, n_estimators=100, max_depth=10, min_samples_leaf=5,
                 variogram_range=50.0, nugget=0.1, sill=1.0):
        """
        Parameters:
        -----------
        n_estimators : int
            RF决策树数量
        max_depth : int
            RF最大深度
        min_samples_leaf : int
            叶节点最小样本数
        variogram_range : float
            变异函数变程 (km)
        nugget : float
            块金效应
        sill : float
            基台值
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.variogram_range = variogram_range
        self.nugget = nugget
        self.sill = sill

    def _build_features(self, coords, cmaq_values):
        """
        构建特征向量 X = [CMAQ, lon, lat]

        Parameters:
        -----------
        coords : array (n, 2) - 坐标 [lon, lat]
        cmaq_values : array (n,) - CMAQ值

        Returns:
        --------
        X : array (n, 3) - 特征矩阵
        """
        return np.column_stack([cmaq_values, coords])

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
        # 步骤1: 特征工程
        X_train = self._build_features(train_coords, train_cmaq)
        X_test = self._build_features(test_coords, test_cmaq)

        # 步骤2: 训练随机森林
        rf = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=42
        )
        rf.fit(X_train, train_obs)

        # 步骤3: RF预测
        rf_pred_train = rf.predict(X_train)
        rf_pred_test = rf.predict(X_test)

        # 步骤4: 计算训练残差
        residual = train_obs - rf_pred_train

        # 步骤5: 克里金插值残差（使用GPR作为代理）
        # 变异函数 gamma(h) = c0 + c * [1 - exp(-3h/a)]
        # GPR核函数：ConstantKernel * RBF + WhiteKernel 近似
        kernel = (ConstantKernel(self.sill, (1e-2, 1e2)) *
                  RBF(length_scale=self.variogram_range / 3,
                      length_scale_bounds=(1e-2, 1e3)) +
                  WhiteKernel(noise_level=self.nugget))

        gpr = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=3,
            alpha=self.nugget,
            normalize_y=True
        )
        gpr.fit(train_coords, residual)

        # 步骤6: 预测测试点残差
        residual_pred, _ = gpr.predict(test_coords, return_std=True)

        # 步骤7: 最终融合 y_final = y_RF + r_hat
        predictions = rf_pred_test + residual_pred

        # 非负约束
        predictions = np.maximum(predictions, 0)

        return predictions


def run_RF_Kriging_ten_fold(selected_day='2020-01-01'):
    """
    运行RF-Kriging十折交叉验证

    Parameters:
    -----------
    selected_day : str
        验证日期，格式 'YYYY-MM-DD'
    """
    print("=" * 60)
    print("RF-Kriging Ten-Fold Cross Validation")
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
        model = RF_Kriging(
            n_estimators=100,
            max_depth=10,
            min_samples_leaf=5,
            variogram_range=50.0,
            nugget=0.1,
            sill=1.0
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
    print(f"  RF-Kriging: R2={metrics['R2']:.4f}, MAE={metrics['MAE']:.2f}, "
          f"RMSE={metrics['RMSE']:.2f}, MB={metrics['MB']:.4f}")

    # 保存结果
    result_df = pd.DataFrame([{
        'method': 'RF-Kriging',
        **metrics
    }])
    result_df.to_csv(f'{output_dir}/RF-Kriging_folds.csv', index=False)

    print(f"\nResults saved to: {output_dir}/")

    return metrics


if __name__ == '__main__':
    metrics = run_RF_Kriging_ten_fold('2020-01-01')
    print(f"\nRF-Kriging: R2={metrics['R2']:.4f}")
