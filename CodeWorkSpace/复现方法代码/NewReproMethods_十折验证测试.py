"""
新复现方法十折验证测试
======================
对9个文献方法进行十折交叉验证

方法列表：
1. BayesianDA - 贝叶斯数据同化法
2. GPDownscaling - GP降尺度法
3. HDGC - HDGC监测偏差检测法
4. UniversalKriging - 通用克里金PM25映射法
5. IDWBias - IDW偏差加权融合法
6. GenFriberg - GenFriberg广义融合法
7. FC1 - FC1克里金插值法
8. FC2 - FC2尺度CMAQ法
9. FCopt - FCopt优化融合法
"""

import sys
sys.path.insert(0, 'E:/CodeProject/ClaudeRoom/Data_Fusion_AutoResearch/CodeWorkSpace/复现方法代码')

import os
from shared.paths import get_project_root, data_path
import numpy as np
import pandas as pd
import netCDF4 as nc
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# 导入复现方法
from BayesianDA import BayesianDA
from GPDownscaling import GPDownscaling
from HDGC import HDGC
from UniversalKriging import UniversalKriging
from IDWBias import IDWBias
from GenFriberg import GenFribergFusion
from FC1 import FC1Kriging
from FC2 import FC2ScaleCMAQ
from FCopt import FCoptFusion


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
        df = df.dropna(subset=['Lon', 'Lat', 'Conc'])
        return df

    def load_cmaq_data(self, selected_day=None):
        """加载CMAQ数据"""
        nc_file = os.path.join(self.cmaq_dir, '2020_PM25.nc')
        ds = nc.Dataset(nc_file, 'r')

        grid_lons = ds.variables['lon'][:]
        grid_lats = ds.variables['lat'][:]
        pred_pm25 = ds.variables['pred_PM25'][:]
        time_data = ds.variables['time'][:]
        time_index = np.arange(len(time_data))

        ds.close()

        if selected_day:
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
            test_sites = fold_df[fold_df['fold'] == fold_id]['Site'].values
            train_sites = fold_df[fold_df['fold'] != fold_id]['Site'].values

            train_mask = monitor_df['Site'].isin(train_sites)
            test_mask = monitor_df['Site'].isin(test_sites)

            train_df = monitor_df[train_mask]
            test_df = monitor_df[test_mask]

            X_obs_train = train_df[['Lon', 'Lat']].values
            y_obs_train = train_df['Conc'].values
            y_model_train = train_df['CMAQ'].values

            X_obs_test = test_df[['Lon', 'Lat']].values
            y_obs_test = test_df['Conc'].values
            y_model_test = test_df['CMAQ'].values

            try:
                method_obj.fit(X_obs_train, y_obs_train, y_model_train)
                y_pred_test = method_obj.predict(X_obs_test, y_model_test)
            except Exception as e:
                print(f"  Fold {fold_id} error: {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
                continue

            y_pred_test = np.array(y_pred_test).flatten()
            y_obs_test = np.array(y_obs_test).flatten()

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
        print("=" * 70)
        print("新复现方法十折验证测试")
        print("=" * 70)
        print(f"Test date: {selected_day}")

        # 加载数据
        monitor_df = self.dl.load_monitor_data(selected_day)
        grid_lons, grid_lats, pred_pm25, _ = self.dl.load_cmaq_data(selected_day)

        X_grid = self.dl.get_grid_points(grid_lons, grid_lats)
        y_grid_model = pred_pm25.ravel()

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
            'BayesianDA': BayesianDA(omega=1.0, epsilon=1e-2, delta=1e-2),
            'GPDownscaling': GPDownscaling(n_latent=3, nu=1.0, k=20),
            'HDGC': HDGC(max_iter=200, tol=1e-4, k=10),
            'UniversalKriging': UniversalKriging(variogram_model='exponential', k=20),
            'IDWBias': IDWBias(power=2.0, max_distance=100.0, min_neighbors=3),
            'GenFriberg': GenFribergFusion(regression_mode='auto', variogram_model='exponential', k=20),
            'FC1': FC1Kriging(variogram_model='exponential', k=20),
            'FC2': FC2ScaleCMAQ(seasonal_correction=True, k=20),
            'FCopt': FCoptFusion(W_min=0.0, variogram_model='exponential', k=20),
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
                print(f"  Mean R2: {m['R2']:.4f}, MAE: {m['MAE']:.4f}, RMSE: {m['RMSE']:.4f}, MB: {m['MB']:.4f}")

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
        summary_df.to_csv(os.path.join(output_dir, 'new_reproduction_summary.csv'), index=False)

        # 保存详细结果
        for method_name, result in results.items():
            fold_df_result = pd.DataFrame(result['fold_metrics'])
            fold_df_result.to_csv(
                os.path.join(output_dir, f'{method_name}_folds.csv'),
                index=False
            )

        # 打印汇总表
        print("\n" + "=" * 70)
        print("验证结果汇总 (按R2排序)")
        print("=" * 70)
        print(f"{'Method':<25} {'R2':>10} {'MAE':>10} {'RMSE':>10} {'MB':>10}")
        print("-" * 70)
        for _, row in summary_df.iterrows():
            print(f"{row['Method']:<25} {row['R2']:>10.4f} {row['MAE']:>10.4f} "
                  f"{row['RMSE']:>10.4f} {row['MB']:>10.4f}")

        # 添加基准对比
        print("\n" + "-" * 70)
        print("基准对比:")
        print(f"{'eVNA':<25} {'R2':>10} (已有基准)")
        print(f"{'SuperStackingEnsemble':<25} {'R2':>10} (最佳已有基准)")
        print("-" * 70)

        print(f"\nResults saved to: {output_dir}")

        return results


def main():
    root_dir = str(get_project_root())
    output_dir = 'E:/CodeProject/ClaudeRoom/Data_Fusion_AutoResearch/test_result/新复现方法'

    dl = DataLoader(root_dir)
    validator = ReproductionValidator(dl)

    results = validator.run_all_validations(root_dir, output_dir, selected_day='2020-01-01')


if __name__ == '__main__':
    main()
