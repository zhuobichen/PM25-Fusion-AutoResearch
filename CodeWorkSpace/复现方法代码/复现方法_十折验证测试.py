"""
复现方法十折验证测试
=====================
对LocalPaperLibrary论文中的融合方法进行十折交叉验证

方法列表：
1. OMA - Observation Model Aggregation
2. SMA - Statistical Model Aggregation
3. MMA - Mixed Model Aggregation
4. BC_Mean - 均值偏差校正
5. BC_Spatial - 空间偏差校正
6. BC_Scale - 缩放偏差校正
7. BC_Linear - 线性偏差校正
8. QuantileMapping - 分位数映射
9. SpatialKrigingBC - 空间克里金偏差校正
10. ODI - Observation Deviation Index
11. EnsembleMean - 集合平均
12. OptimumInterpolation - 最优插值
13. ThreeDVar - 三维变分同化
14. STK - 北京STK时空克里金方法
15. NC - 华北多源融合方法
"""

import sys
sys.path.insert(0, 'E:/CodeProject/ClaudeRoom/Data_Fusion_AutoResearch/CodeWorkSpace/复现方法代码')

import os
from shared.paths import get_project_root, data_path
import numpy as np
import pandas as pd
import netCDF4 as nc
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 导入复现方法
from BaseFusionMethods import (
    OMA, SMA, MMA, BC, QuantileMapping, SpatialKrigingBC,
    ODI, EnsembleMean, OptimumInterpolation, ThreeDVar
)
from STK import SpatioTemporalKriging, SimpleSTK
from NC import NorthChinaFusion, NC_IDW


class DataLoader:
    """数据加载器"""

    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.cmaq_dir = data_path('test_data/raw/CMAQ')
        self.monitor_dir = data_path('test_data/raw/Monitor')

    def load_monitor_data(self, selected_day=None):
        """加载监测数据"""
        csv_file = os.path.join(self.monitor_dir, '2020_DailyPM2.5Monitor.csv')
        df = pd.read_csv(csv_file)
        if selected_day:
            df = df[df['Date'] == selected_day]
        # 过滤无效坐标
        df = df.dropna(subset=['Lon', 'Lat', 'Conc'])
        return df

    def load_cmaq_data(self, selected_day=None):
        """加载CMAQ数据"""
        nc_file = os.path.join(self.cmaq_dir, '2020_PM25.nc')
        ds = nc.Dataset(nc_file, 'r')

        grid_lons = ds.variables['lon'][:]
        grid_lats = ds.variables['lat'][:]
        pred_pm25 = ds.variables['pred_PM25'][:]

        # 获取时间索引
        time_data = ds.variables['time'][:]
        # 假设time是day number from 2020-01-01
        time_index = np.arange(len(time_data))

        ds.close()

        if selected_day:
            # 解析日期
            import datetime
            start_date = datetime.datetime(2020, 1, 1)
            selected_date = datetime.datetime.strptime(selected_day, '%Y-%m-%d')
            day_offset = (selected_date - start_date).days
            if 0 <= day_offset < len(time_index):
                idx = day_offset
                pred_pm25 = pred_pm25[idx]
            else:
                raise ValueError(f"Date {selected_day} not in range")

        return grid_lons, grid_lats, pred_pm25, time_index

    def get_grid_points(self, grid_lons, grid_lats):
        """获取网格点坐标"""
        ny, nx = grid_lons.shape
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
        """提取CMAQ在监测站点的值"""
        ny, nx = grid_lons.shape
        cmaq_at_sites = np.zeros(len(monitor_df))

        for i, row in enumerate(monitor_df.itertuples()):
            lon, lat = row.Lon, row.Lat
            dist_lon = np.abs(grid_lons - lon)
            dist_lat = np.abs(grid_lats - lat)
            min_idx = np.unravel_index(np.argmin(dist_lon + dist_lat), grid_lons.shape)
            cmaq_at_sites[i] = pm25_data[min_idx[0], min_idx[1]]

        return cmaq_at_sites


class ReproductionValidator:
    """复现方法验证器"""

    def __init__(self, data_loader):
        self.dl = data_loader

    def validate_method(self, method_name, method_obj, monitor_df, fold_df,
                       X_grid, y_grid_model, selected_day):
        """
        验证单个方法

        Returns
        -------
        dict : 包含metrics和fold信息
        """
        fold_metrics = []

        for fold_id in range(1, 11):
            # 获取该折的测试站点
            test_sites = fold_df[fold_df['fold'] == fold_id]['Site'].values
            train_sites = fold_df[fold_df['fold'] != fold_id]['Site'].values

            # 划分数据
            train_mask = monitor_df['Site'].isin(train_sites)
            test_mask = monitor_df['Site'].isin(test_sites)

            train_df = monitor_df[train_mask]
            test_df = monitor_df[test_mask]

            # 训练数据
            X_obs_train = train_df[['Lon', 'Lat']].values
            y_obs_train = train_df['Conc'].values
            y_model_train = train_df['CMAQ'].values

            # 测试数据
            X_obs_test = test_df[['Lon', 'Lat']].values
            y_obs_test = test_df['Conc'].values
            y_model_test = test_df['CMAQ'].values

            # 拟合方法
            try:
                method_obj.fit(X_obs_train, y_obs_train, y_model_train)
                y_pred_test = method_obj.predict(X_obs_test, y_model_test)
            except Exception as e:
                print(f"  Fold {fold_id} error: {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
                continue

            # 计算指标（在测试站点）
            y_pred_test = np.array(y_pred_test).flatten()
            y_obs_test = np.array(y_obs_test).flatten()

            # 过滤无效值
            valid_mask = np.isfinite(y_pred_test) & np.isfinite(y_obs_test)
            if valid_mask.sum() < 5:
                continue

            y_pred_valid = y_pred_test[valid_mask]
            y_obs_valid = y_obs_test[valid_mask]

            metrics = {
                'fold': fold_id,
                'R2': r2_score(y_obs_valid, y_pred_valid),
                'MAE': mean_absolute_error(y_obs_valid, y_pred_valid),
                'RMSE': np.sqrt(mean_squared_error(y_obs_valid, y_pred_valid)),
                'MB': np.mean(y_pred_valid - y_obs_valid),
                'N': valid_mask.sum()
            }
            fold_metrics.append(metrics)

        if len(fold_metrics) == 0:
            return None

        # 汇总
        mean_metrics = {
            'R2': np.mean([m['R2'] for m in fold_metrics]),
            'MAE': np.mean([m['MAE'] for m in fold_metrics]),
            'RMSE': np.mean([m['RMSE'] for m in fold_metrics]),
            'MB': np.mean([m['MB'] for m in fold_metrics])
        }

        return {
            'method': method_name,
            'fold_metrics': fold_metrics,
            'mean_metrics': mean_metrics
        }

    def run_all_validations(self, root_dir, output_dir, selected_day='2020-01-01'):
        """运行所有方法的验证"""
        print("=" * 60)
        print("复现方法十折验证测试")
        print("=" * 60)
        print(f"Test date: {selected_day}")

        # 加载数据
        monitor_df = self.dl.load_monitor_data(selected_day)
        grid_lons, grid_lats, pred_pm25, _ = self.dl.load_cmaq_data(selected_day)

        # 获取网格数据
        X_grid = self.dl.get_grid_points(grid_lons, grid_lats)
        y_grid_model = pred_pm25.ravel()

        # 添加CMAQ列到monitor_df
        cmaq_at_sites = self.dl.extract_cmaq_at_sites(grid_lons, grid_lats, pred_pm25, monitor_df)
        monitor_df = monitor_df.copy()
        monitor_df['CMAQ'] = cmaq_at_sites

        print(f"Monitoring stations: {len(monitor_df)}")
        print(f"CMAQ grid size: {pred_pm25.shape}")
        print(f"Grid points: {len(X_grid)}")

        # 加载十折划分
        fold_df = pd.read_csv(data_path('test_data/fold_split_table_daily.csv'))

        # 定义方法
        methods = {
            'OMA': OMA(alpha=0.5, method='local', k=30, power=-2),
            'SMA_Linear': SMA(regression_type='linear'),
            'SMA_Poly': SMA(regression_type='polynomial', poly_degree=2),
            'MMA': MMA(beta=0.5, k=30, power=-2),
            'BC_Mean': BC(method='mean'),
            'BC_Spatial': BC(method='spatial', k=30, power=-2),
            'BC_Scale': BC(method='scale'),
            'BC_Linear': BC(method='linear'),
            'QuantileMapping': QuantileMapping(n_quantiles=10, method='linear'),
            'SpatialKrigingBC': SpatialKrigingBC(variogram_model='spherical', k=30),
            'ODI': ODI(gamma=0.1, k=30, power=2, normalize=True),
            'OptimumInterpolation': OptimumInterpolation(),
            'ThreeDVar': ThreeDVar(),
            # 北京STK时空克里金方法
            'STK': SimpleSTK(k=30, spatial_scale=0.5, temporal_scale=7.0, power=-2),
            # 华北多源融合方法
            'NC': NC_IDW(k=30, power=-2),
        }

        results = {}

        for method_name, method_obj in methods.items():
            print(f"\n--- Testing: {method_name} ---")
            result = self.validate_method(
                method_name, method_obj, monitor_df, fold_df,
                X_grid, y_grid_model, selected_day
            )

            if result:
                results[method_name] = result
                m = result['mean_metrics']
                print(f"  Mean R2: {m['R2']:.4f}, MAE: {m['MAE']:.4f}, RMSE: {m['RMSE']:.4f}")

        # 保存结果
        os.makedirs(output_dir, exist_ok=True)

        # 保存汇总表
        summary_data = []
        for method_name, result in results.items():
            m = result['mean_metrics']
            summary_data.append({
                'Method': method_name,
                'R2': m['R2'],
                'MAE': m['MAE'],
                'RMSE': m['RMSE'],
                'MB': m['MB']
            })

        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('R2', ascending=False)
        summary_df.to_csv(os.path.join(output_dir, 'reproduction_summary.csv'), index=False)

        # 保存详细结果
        for method_name, result in results.items():
            fold_df_result = pd.DataFrame(result['fold_metrics'])
            fold_df_result.to_csv(
                os.path.join(output_dir, f'{method_name}_folds.csv'),
                index=False
            )

        # 打印汇总表
        print("\n" + "=" * 60)
        print("验证结果汇总 (按R2排序)")
        print("=" * 60)
        print(f"{'Method':<20} {'R2':>10} {'MAE':>10} {'RMSE':>10} {'MB':>10}")
        print("-" * 60)
        for _, row in summary_df.iterrows():
            print(f"{row['Method']:<20} {row['R2']:>10.4f} {row['MAE']:>10.4f} "
                  f"{row['RMSE']:>10.4f} {row['MB']:>10.4f}")

        print(f"\nResults saved to: {output_dir}")

        return results


def main():
    root_dir = str(get_project_root())
    output_dir = 'E:/CodeProject/ClaudeRoom/Data_Fusion_AutoResearch/test_result/复现方法'

    dl = DataLoader(root_dir)
    validator = ReproductionValidator(dl)

    results = validator.run_all_validations(root_dir, output_dir, selected_day='2020-01-01')


if __name__ == '__main__':
    main()
