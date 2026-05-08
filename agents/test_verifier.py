"""
测试验证员 Agent
================
职责：
1. 数据质量校验
2. 十折交叉验证
3. 计算R²、MAE、RMSE指标
4. 验证创新优越性
5. 生成comparison_report.md

测试规则：
- 确定性方法：直接十折验证
- 非确定性方法：2次稳定性测试 → R²>0.999则十折验证，否则抛弃
"""

import os
from shared.paths import get_project_root, data_path
import sys
import json
import numpy as np
import pandas as pd
import netCDF4 as nc
from datetime import datetime
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# 添加Code路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Code.VNAeVNAaVNA.nna_methods import NNA


class TestVerifier:
    """测试验证员 Agent"""

    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.test_data_dir = os.path.join(root_dir, 'test_data')
        self.test_result_dir = os.path.join(root_dir, 'test_result')
        self.error_dir = os.path.join(root_dir, 'error')

        # 子目录
        self.benchmark_result_dir = os.path.join(self.test_result_dir, '基准方法')
        self.reproduce_result_dir = os.path.join(self.test_result_dir, '复现方法')
        self.innovation_result_dir = os.path.join(self.test_result_dir, '创新方法')

        os.makedirs(self.test_result_dir, exist_ok=True)
        os.makedirs(self.error_dir, exist_ok=True)

    def validate_data_quality(self):
        """
        数据质量校验
        """
        print("=== [测试验证员] 数据质量校验 ===")

        issues = []

        # 检查目录
        raw_cmaq = os.path.join(self.test_data_dir, 'raw/CMAQ')
        raw_monitor = os.path.join(self.test_data_dir, 'raw/Monitor')

        if not os.path.exists(raw_cmaq):
            issues.append("CMAQ目录不存在")
        if not os.path.exists(raw_monitor):
            issues.append("Monitor目录不存在")

        if issues:
            self._log_error('data_quality', issues)
            return {'status': 'error', 'issues': issues}

        print("  数据目录结构正常")
        return {'status': 'ok'}

    def _log_error(self, prefix, content):
        """记录错误到error目录"""
        error_file = os.path.join(self.error_dir, f'{prefix}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        with open(error_file, 'w', encoding='utf-8') as f:
            if isinstance(content, list):
                f.write('\n'.join(content))
            else:
                f.write(str(content))

    def compute_metrics(self, y_true, y_pred):
        """计算R²、MAE、RMSE、MB"""
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

    def get_cmaq_at_sites(self, lon_array, lat_array, pm25_grid, lon_sites, lat_sites):
        """获取站点位置的CMAQ值（最近邻）"""
        cmaq_values = np.zeros(len(lon_sites))
        for i in range(len(lon_sites)):
            dist = np.sqrt((lon_array - lon_sites[i])**2 + (lat_array - lat_sites[i])**2)
            idx = np.argmin(dist)
            ny, nx = lon_array.shape
            row, col = idx // nx, idx % nx
            cmaq_values[i] = pm25_grid[row, col]
        return cmaq_values

    def run_benchmark_ten_fold(self, selected_day='2020-01-01'):
        """
        运行基准方法十折交叉验证
        """
        print(f"\n=== [测试验证员] 基准方法十折验证 ({selected_day}) ===")

        # 加载数据
        monitor_file = os.path.join(self.test_data_dir, 'raw/Monitor/2020_DailyPM2.5Monitor.csv')
        cmaq_file = os.path.join(self.test_data_dir, 'raw/CMAQ/2020_PM25.nc')
        fold_file = os.path.join(self.test_data_dir, 'fold_split_table_daily.csv')

        monitor_df = pd.read_csv(monitor_file)
        fold_df = pd.read_csv(fold_file)

        # 筛选日期
        day_df = monitor_df[monitor_df['Date'] == selected_day].copy()
        day_df = day_df.merge(fold_df, on='Site', how='left')
        day_df = day_df.dropna(subset=['Lat', 'Lon', 'Conc'])

        # 加载CMAQ数据
        ds = nc.Dataset(cmaq_file, 'r')
        lon_cmaq = ds.variables['lon'][:]
        lat_cmaq = ds.variables['lat'][:]
        pred_pm25 = ds.variables['pred_PM25'][:]
        ds.close()

        # 获取日期索引（time是0-364的整数）
        from datetime import datetime
        date_obj = datetime.strptime(selected_day, '%Y-%m-%d')
        day_idx = (date_obj - datetime(2020, 1, 1)).days
        pred_day = pred_pm25[day_idx]

        # 提取站点CMAQ值
        day_df['CMAQ'] = self.get_cmaq_at_sites(
            lon_cmaq, lat_cmaq, pred_day,
            day_df['Lon'].values, day_df['Lat'].values
        )

        print(f"  数据加载完成：{len(day_df)} 条监测数据")

        # 方法列表
        methods = ['CMAQ', 'VNA', 'aVNA', 'eVNA']
        results = {m: [] for m in methods}

        # 十折验证
        for fold_id in range(1, 11):
            train_df = day_df[day_df['fold'] != fold_id].copy()
            test_df = day_df[day_df['fold'] == fold_id].copy()

            if len(test_df) == 0:
                continue

            # 训练数据准备
            train_df['x'] = train_df['Lon']
            train_df['y'] = train_df['Lat']
            train_df['mod'] = train_df['CMAQ']
            train_df['bias'] = train_df['Conc'] - train_df['CMAQ']
            train_df['rn'] = train_df['Conc'] / train_df['CMAQ']

            # NNA拟合
            nn = NNA(method='voronoi', k=30, power=-2)
            nn.fit(train_df[['x', 'y']], train_df[['Conc', 'mod', 'bias', 'rn']])

            # CMAQ
            results['CMAQ'].append({
                'fold': fold_id,
                'y_true': test_df['Conc'].values,
                'y_pred': test_df['CMAQ'].values
            })

            # VNA/aVNA/eVNA预测 - 标准模式：直接对验证站点所在的CMAQ网格坐标预测
            # 获取验证站点所在的CMAQ网格坐标
            test_cmaq_lon = np.zeros(len(test_df))
            test_cmaq_lat = np.zeros(len(test_df))
            for i, (_, row) in enumerate(test_df.iterrows()):
                dist = np.sqrt((lon_cmaq - row['Lon'])**2 + (lat_cmaq - row['Lat'])**2)
                idx = np.argmin(dist)
                ny, nx = lon_cmaq.shape
                row_idx, col_idx = idx // nx, idx % nx
                test_cmaq_lon[i] = lon_cmaq[row_idx, col_idx]
                test_cmaq_lat[i] = lat_cmaq[row_idx, col_idx]

            # 直接对验证站点所在的CMAQ网格坐标预测
            X_test_grid = np.column_stack([test_cmaq_lon, test_cmaq_lat])
            zdf_test = nn.predict(X_test_grid)

            vna_pred = zdf_test[:, 0]
            vna_bias_pred = zdf_test[:, 2]
            vna_rn_pred = zdf_test[:, 3]

            results['VNA'].append({
                'fold': fold_id,
                'y_true': test_df['Conc'].values,
                'y_pred': vna_pred
            })

            # aVNA = M + bias
            results['aVNA'].append({
                'fold': fold_id,
                'y_true': test_df['Conc'].values,
                'y_pred': test_df['CMAQ'].values + vna_bias_pred
            })

            # eVNA = M * rn
            results['eVNA'].append({
                'fold': fold_id,
                'y_true': test_df['Conc'].values,
                'y_pred': test_df['CMAQ'].values * vna_rn_pred
            })

            # 打印本折结果
            print(f"  Fold {fold_id}: ", end='')
            for m in methods:
                metrics = self.compute_metrics(results[m][-1]['y_true'], results[m][-1]['y_pred'])
                print(f"{m} R2={metrics['R2']:.3f} ", end='')
            print()

        # 汇总
        print("\n=== 汇总结果 ===")
        summary = {}
        for m in methods:
            all_true = np.concatenate([r['y_true'] for r in results[m]])
            all_pred = np.concatenate([r['y_pred'] for r in results[m]])
            metrics = self.compute_metrics(all_true, all_pred)
            summary[m] = metrics
            print(f"  {m:>8}: R2={metrics['R2']:.4f}, MAE={metrics['MAE']:.2f}, RMSE={metrics['RMSE']:.2f}, MB={metrics['MB']:.2f}")

        # 保存结果
        os.makedirs(self.benchmark_result_dir, exist_ok=True)
        summary_df = pd.DataFrame(summary).T
        summary_df.to_csv(os.path.join(self.benchmark_result_dir, 'benchmark_summary.csv'))

        return summary

    def verify_innovation(self, innovation_metrics, benchmark_metrics):
        """
        验证创新是否成立
        条件：R²提升≥0.01 且 RMSE≤最优基准 且 |MB|≤基准最优
        """
        # 找到最优基准
        best_r2 = max(b['R2'] for b in benchmark_metrics.values())
        best_rmse = min(b['RMSE'] for b in benchmark_metrics.values())
        best_mb_abs = min(abs(b['MB']) for b in benchmark_metrics.values())

        r2_improvement = innovation_metrics['R2'] - best_r2
        rmse_improvement = best_rmse - innovation_metrics['RMSE']  # 负值表示变差
        innovation_mb_abs = abs(innovation_metrics['MB'])
        mb_ok = innovation_mb_abs <= best_mb_abs

        innovation_ok = r2_improvement >= 0.01 and innovation_metrics['RMSE'] <= best_rmse and mb_ok

        return {
            'innovation_ok': innovation_ok,
            'r2_improvement': r2_improvement,
            'rmse_improvement': rmse_improvement,
            'mb_improvement': best_mb_abs - innovation_mb_abs,
            'best_benchmark_r2': best_r2,
            'best_benchmark_rmse': best_rmse,
            'best_benchmark_mb_abs': best_mb_abs,
            'innovation_mb_abs': innovation_mb_abs,
            'mb_ok': mb_ok
        }

    def generate_comparison_report(self, all_metrics, innovation_verification=None):
        """生成对比报告"""
        report = f"""# PM2.5 CMAQ融合方法对比报告

生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 一、全方法指标对比表

| 方法 | R² | MAE | RMSE | MB |
|------|-----|-----|------|-----|
"""

        for method, metrics in all_metrics.items():
            report += f"| {method} | {metrics['R2']:.4f} | {metrics['MAE']:.2f} | {metrics['RMSE']:.2f} | {metrics['MB']:.2f} |\n"

        if innovation_verification:
            report += f"""

## 二、创新验证

- R²提升：{innovation_verification['r2_improvement']:.4f} {'≥ 0.01 ✓' if innovation_verification['r2_improvement'] >= 0.01 else '< 0.01 ✗'}
- RMSE相比最优基准：{innovation_verification['rmse_improvement']:.2f} {'✓' if innovation_verification['rmse_improvement'] >= 0 else '✗'}
- |MB|：{innovation_verification['innovation_mb_abs']:.2f}，基准最优：{innovation_verification['best_benchmark_mb_abs']:.2f} {'✓' if innovation_verification['mb_ok'] else '✗'}
- 创新结论：**{'成立' if innovation_verification['innovation_ok'] else '不足'}**
"""

        report += """

---

"""

        report_file = os.path.join(self.test_result_dir, 'comparison_report.md')
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)

        return report_file

    def run(self, method_code=None, method_type='benchmark'):
        """
        执行测试验证流程
        """
        print("\n" + "="*60)
        print("[测试验证员] 开始工作")
        print("="*60)

        # 1. 数据质量校验
        quality = self.validate_data_quality()
        if quality['status'] != 'ok':
            print("数据质量校验失败")
            return quality

        # 2. 运行基准方法十折验证
        if method_type == 'benchmark':
            summary = self.run_benchmark_ten_fold()

            # 3. 生成报告
            self.generate_comparison_report(summary)

            print(f"\n=== [测试验证员] 完成 ===")
            return {'status': 'done', 'summary': summary}

        return {'status': 'done'}


if __name__ == '__main__':
    root_dir = str(get_project_root())
    agent = TestVerifier(root_dir)
    result = agent.run(method_type='benchmark')
