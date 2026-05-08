"""
PM2.5 CMAQ融合方法基准测试适配代码
===================================
将VNA/eVNA/aVNA适配系统输入格式：
- CMAQ: netCDF格式，需要按日期/站点读取
- Monitor: CSV格式，需要按日期/站点读取
- 支持十折交叉验证
"""

import sys


import os
from shared.paths import get_project_root, data_path
import numpy as np
import pandas as pd
import netCDF4 as nc
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# 导入NNA类（包含VNA/eVNA/aVNA实现）
from interpolation_methods import NNA


class DataLoader:
    """数据加载器：适配系统输入格式"""

    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.cmaq_dir = data_path('test_data/raw/CMAQ')
        self.monitor_dir = data_path('test_data/raw/Monitor')
        self.test_data_dir = os.path.join(root_dir, 'test_data')

    def load_monitor_data(self, selected_day=None):
        """
        加载监测数据

        Parameters
        ----------
        selected_day : str, optional
            筛选指定日期，格式'YYYY-MM-DD'

        Returns
        -------
        df : DataFrame
            监测数据，包含 Site, Date, Conc, Lat, Lon
        """
        csv_file = os.path.join(self.monitor_dir, '2020_DailyPM2.5Monitor.csv')
        df = pd.read_csv(csv_file)

        # 筛选指定日期
        if selected_day:
            df = df[df['Date'] == selected_day]

        return df

    def load_cmaq_data(self, selected_day=None):
        """
        加载CMAQ数据（netCDF格式）

        Parameters
        ----------
        selected_day : str, optional
            筛选指定日期，格式'YYYY-MM-DD'

        Returns
        -------
        grid_lons : array
            网格经度
        grid_lats : array
            网格纬度
        pm25_predictions : array
            CMAQ预测值 (time, y, x)
        time_index : array
            时间索引对应日期
        """
        nc_file = os.path.join(self.cmaq_dir, '2020_PM25.nc')
        ds = nc.Dataset(nc_file, 'r')

        # 读取变量
        grid_lons = ds.variables['lon'][:]
        grid_lats = ds.variables['lat'][:]
        pred_pm25 = ds.variables['pred_PM25'][:]  # (time, y, x)
        base_pm25 = ds.variables['base_PM25'][:]  # (time, y, x)

        # 读取时间
        time_var = ds.variables['time']
        # 假设time是days since 2020-01-01的格式
        time_units = time_var.units
        time_data = nc.num2date(time_var[:], time_units)

        ds.close()

        # 筛选指定日期
        if selected_day:
            time_strs = [str(t)[:10] for t in time_data]
            if selected_day in time_strs:
                idx = time_strs.index(selected_day)
                pred_pm25 = pred_pm25[idx]
                base_pm25 = base_pm25[idx]
            else:
                raise ValueError(f"日期 {selected_day} 不在CMAQ数据中")

        return grid_lons, grid_lats, pred_pm25, base_pm25, time_data

    def get_grid_points(self, grid_lons, grid_lats):
        """
        获取网格点坐标和CMAQ预测值

        Returns
        -------
        X_grid : array (n_grid_points, 2)
            网格点坐标 [lon, lat]
        y_grid_model : array
            网格点上的CMAQ预测值
        """
        ny, nx = grid_lons.shape
        # 创建网格点坐标
        X_grid = np.zeros((ny * nx, 2))
        y_grid_model = np.zeros(ny * nx)

        idx = 0
        for i in range(ny):
            for j in range(nx):
                X_grid[idx, 0] = grid_lons[i, j]
                X_grid[idx, 1] = grid_lats[i, j]
                idx += 1

        return X_grid

    def extract_cmaq_at_sites(self, grid_lons, grid_lats, pm25_data, monitor_df):
        """
        在监测站点位置提取CMAQ预测值（双线性插值近似）

        Parameters
        ----------
        grid_lons, grid_lats : array
            CMAQ网格经纬度
        pm25_data : array (y, x)
            CMAQ PM2.5数据
        monitor_df : DataFrame
            监测数据

        Returns
        -------
        cmaq_at_sites : array
            各站点对应的CMAQ值
        """
        ny, nx = grid_lons.shape
        cmaq_at_sites = np.zeros(len(monitor_df))

        for i, row in enumerate(monitor_df.itertuples()):
            lon, lat = row.Lon, row.Lat

            # 找到最近的网格点
            dist_lon = np.abs(grid_lons - lon)
            dist_lat = np.abs(grid_lats - lat)
            min_idx = np.unravel_index(np.argmin(dist_lon + dist_lat), grid_lons.shape)

            cmaq_at_sites[i] = pm25_data[min_idx[0], min_idx[1]]

        return cmaq_at_sites


class FusionBenchmark:
    """融合方法基准测试"""

    def __init__(self, data_loader):
        self.dl = data_loader

    def run_method(self, method_name, X_obs, y_obs, y_model_obs, X_grid, y_grid_model, k=30, power=-2):
        """
        运行指定的融合方法

        Parameters
        ----------
        method_name : str
            方法名：'VNA', 'eVNA', 'aVNA', 'CMAQ'
        X_obs : array (n_obs, 2)
            监测站点坐标 [lon, lat]
        y_obs : array
            监测站点观测值
        y_model_obs : array
            监测站点对应的CMAQ模型值
        X_grid : array (n_grid, 2)
            目标网格点坐标
        y_grid_model : array
            目标网格点对应的CMAQ模型值
        k, power : NNA参数

        Returns
        -------
        y_pred : array
            预测结果
        """
        nn = NNA(k=k, power=power, method='nearest')

        if method_name == 'CMAQ':
            # 直接返回CMAQ原始值
            return y_grid_model.copy()

        # 拟合融合模型
        nn.fit_GAT(X_obs, y_model=y_model_obs, y_obs=y_obs)

        # 预测
        if method_name == 'VNA':
            return nn.predict_GAT(X_grid, y_grid_model, adjustment_method='VNA')
        elif method_name == 'eVNA':
            return nn.predict_GAT(X_grid, y_grid_model, adjustment_method='eVNA')
        elif method_name == 'aVNA':
            return nn.predict_GAT(X_grid, y_grid_model, adjustment_method='aVNA')
        else:
            raise ValueError(f"Unknown method: {method_name}")

    def cross_validate_fold(self, method_name, monitor_df, fold_df, X_grid, y_grid_model,
                            selected_day, k=30, power=-2):
        """
        执行单折交叉验证

        Parameters
        ----------
        method_name : str
            方法名
        monitor_df : DataFrame
            监测数据
        fold_df : DataFrame
            站点划分表
        X_grid, y_grid_model : array
            CMAQ网格数据
        selected_day : str
            测试日期

        Returns
        -------
        results : dict
            包含 y_true, y_pred, metrics
        """
        # 获取该折的测试站点
        test_sites = fold_df[fold_df['fold'] == fold]['Site'].values
        train_sites = fold_df[fold_df['fold'] != fold]['Site'].values

        # 划分训练/测试数据
        train_mask = monitor_df['Site'].isin(train_sites)
        test_mask = monitor_df['Site'].isin(test_sites)

        train_df = monitor_df[train_mask]
        test_df = monitor_df[test_mask]

        # 监测站点坐标和值
        X_obs_train = train_df[['Lon', 'Lat']].values
        y_obs_train = train_df['Conc'].values

        # 需要从CMAQ提取训练站点的模型值（简化：假设已有y_model_obs）
        # 实际需要用DataLoader.extract_cmaq_at_sites
        y_model_train = np.ones(len(train_df)) * train_df['Conc'].mean() * 0.9  # TODO: 真实CMAQ值

        X_obs_test = test_df[['Lon', 'Lat']].values
        y_obs_test = test_df['Conc'].values

        # 获取测试站点的CMAQ值
        # TODO: 真实实现需要提取CMAQ在测试站点的值

        # 运行方法
        y_pred = self.run_method(method_name, X_obs_train, y_obs_train,
                                 y_model_train, X_grid, y_grid_model, k, power)

        # 计算指标（在网格点上）
        metrics = {
            'R2': r2_score(y_grid_model[:len(y_pred)], y_pred) if len(y_pred) == len(y_grid_model) else np.nan,
            'MAE': mean_absolute_error(y_grid_model[:len(y_pred)], y_pred) if len(y_pred) == len(y_grid_model) else np.nan,
            'RMSE': np.sqrt(mean_squared_error(y_grid_model[:len(y_pred)], y_pred)) if len(y_pred) == len(y_grid_model) else np.nan
        }

        return {
            'method': method_name,
            'fold': fold,
            'y_true': y_grid_model,
            'y_pred': y_pred,
            'metrics': metrics
        }


def run_benchmark(root_dir, output_dir, selected_day='2020-01-01'):
    """
    运行基准方法测试

    Parameters
    ----------
    root_dir : str
        根目录
    output_dir : str
        输出目录
    selected_day : str
        测试日期
    """
    print(f"=== 基准方法测试开始 ===")
    print(f"测试日期: {selected_day}")

    # 加载数据
    dl = DataLoader(root_dir)
    monitor_df = dl.load_monitor_data(selected_day)
    grid_lons, grid_lats, pred_pm25, base_pm25, _ = dl.load_cmaq_data(selected_day)

    print(f"监测站点数: {len(monitor_df)}")
    print(f"CMAQ网格大小: {pred_pm25.shape}")

    # 获取网格数据
    X_grid = dl.get_grid_points(grid_lons, grid_lats)
    y_grid_model = pred_pm25.ravel()

    # 加载十折划分
    fold_df = pd.read_csv(data_path('test_data/fold_split_table_daily.csv'))

    # 初始化基准测试
    fb = FusionBenchmark(dl)

    # 测试方法列表
    methods = ['CMAQ', 'VNA', 'eVNA', 'aVNA']

    results = {}
    for method in methods:
        print(f"\n--- 测试方法: {method} ---")
        fold_metrics = []

        for fold_id in range(1, 11):
            result = fb.cross_validate_fold(method, monitor_df, fold_df,
                                           X_grid, y_grid_model, selected_day,
                                           k=30, power=-2)
            fold_metrics.append(result['metrics'])
            print(f"  Fold {fold_id}: R2={result['metrics']['R2']:.4f}, "
                  f"MAE={result['metrics']['MAE']:.4f}, RMSE={result['metrics']['RMSE']:.4f}")

        # 汇总
        results[method] = {
            'fold_metrics': fold_metrics,
            'mean_metrics': {
                'R2': np.mean([m['R2'] for m in fold_metrics]),
                'MAE': np.mean([m['MAE'] for m in fold_metrics]),
                'RMSE': np.mean([m['RMSE'] for m in fold_metrics])
            }
        }

    # 保存结果
    os.makedirs(output_dir, exist_ok=True)
    for method, res in results.items():
        # 保存详细结果
        result_file = os.path.join(output_dir, f'{method}_fold_results.csv')
        df = pd.DataFrame(res['fold_metrics'])
        df.to_csv(result_file, index=False)

        # 保存汇总
        summary_file = os.path.join(output_dir, f'{method}_summary.csv')
        pd.DataFrame([res['mean_metrics']]).to_csv(summary_file, index=False)

    print(f"\n=== 基准方法测试完成 ===")
    print("结果保存至:", output_dir)

    return results


if __name__ == '__main__':
    root_dir = str(get_project_root())
    output_dir = 'E:/CodeProject/ClaudeRoom/Data_Fusion_AutoResearch/test_result/基准方法'

    results = run_benchmark(root_dir, output_dir, selected_day='2020-01-01')

    # 打印汇总表
    print("\n========== 基准方法汇总 ==========")
    print(f"{'方法':<10} {'R2':>10} {'MAE':>10} {'RMSE':>10}")
    print("-" * 42)
    for method, res in results.items():
        m = res['mean_metrics']
        print(f"{method:<10} {m['R2']:>10.4f} {m['MAE']:>10.4f} {m['RMSE']:>10.4f}")
