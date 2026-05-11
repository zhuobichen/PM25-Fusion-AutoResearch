# -*- coding: utf-8 -*-
"""
UltimateStackingEnsemble 十折交叉验证（标准模式）
=====================================
自动生成时间: 2026-05-11T09:30:28.641998

验证流程对齐设计文档《十折交叉验证架构文档.md》9.4：
- pre_exp 主级未通过且 R² ≤ 基线 -> 停止
- pre_exp 次级创新（R² 达标） -> 继续 stage1
- stage1  主级未通过且 R² ≤ 基线 -> 停止，标记 seasonally_limited
- stage1  次级创新（R² 达标） -> 继续 stage2
- stage2  失败 -> 继续（不阻止 stage3）
- stage3  主级创新（三条件全满足） -> fully_established
- stage3  次级创新（R2 > 基线，其余未达标） -> secondary_innovation
- stage3  未通过 -> partially_established
"""

import os
import sys
import json
import importlib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# 路径设置
PROJECT_ROOT = r'E:/CodeProject/ClaudeRoom/Data_Fusion_AutoResearch'
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'Code', 'Downscaler'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'Code'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'CodeWorkSpace', '新融合方法代码'))

from shared.paths import data_path
from shared.metrics import compute_metrics

METHOD_NAME = 'UltimateStackingEnsemble'


def _find_run_ten_fold():
    """查找方法模块中的 run_*_ten_fold 函数。"""
    mod = importlib.import_module(METHOD_NAME)
    for name in dir(mod):
        if name.startswith('run_') and name.endswith('_ten_fold') and callable(getattr(mod, name)):
            return getattr(mod, name)
    return None


def _get_cached_predictions():
    """从方法模块中读取缓存的预测值。"""
    mod = importlib.import_module(METHOD_NAME)
    y_true = getattr(mod, '_last_y_true', None)
    y_pred = getattr(mod, '_last_y_pred', None)
    return y_true, y_pred


def run_stage_agg(start_date, end_date):
    """多天聚合验证：遍历日期范围，每天调用方法的 ten_fold 函数，合并预测值。"""
    func = _find_run_ten_fold()
    if func is None:
        print(f"  无法找到 {METHOD_NAME} 中的 run_*_ten_fold 函数")
        return None

    all_y_true = []
    all_y_pred = []
    current = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    days_run = 0

    while current <= end:
        day_str = current.strftime('%Y-%m-%d')
        try:
            func(day_str)
            yt, yp = _get_cached_predictions()
            if yt is not None and yp is not None and len(yt) > 0:
                all_y_true.extend(yt)
                all_y_pred.extend(yp)
                days_run += 1
        except Exception as e:
            print(f"  {day_str} 异常: {e}")
        current += timedelta(days=1)

    if not all_y_true:
        return None

    metrics = compute_metrics(np.array(all_y_true), np.array(all_y_pred))
    metrics['days_run'] = days_run
    return metrics


def run_multistage():
    """运行多阶段验证（对齐设计文档 9.4 分阶段执行流程）。"""
    # VNA 基线阈值（来自十折交叉验证架构文档 9.2）
    BASELINE = {
        'pre_exp': {'R2': 0.8941, 'RMSE': 16.42, 'MB': 0.76},
        'stage1':  {'R2': 0.9057, 'RMSE': 16.28, 'MB': 0.50},
        'stage2':  {'R2': 0.8458, 'RMSE': 4.97,  'MB': 0.04},
        'stage3':  {'R2': 0.9078, 'RMSE': 11.90, 'MB': 0.36},
    }

    # 验证阶段定义（对齐设计文档 9.1）
    STAGES = {
        'pre_exp': ('2020-01-01', '2020-01-05'),
        'stage1':  ('2020-01-01', '2020-01-31'),
        'stage2':  ('2020-07-01', '2020-07-31'),
        'stage3':  ('2020-12-01', '2020-12-31'),
    }

    results = {}
    outcome = 'unknown'

    for stage_name, (start, end) in STAGES.items():
        print(f"\n--- {stage_name} ({start} ~ {end}) ---")
        metrics = run_stage_agg(start, end)

        if metrics:
            baseline = BASELINE[stage_name]
            r2_pass = metrics['R2'] > baseline['R2'] + 0.01
            r2_above_baseline = metrics['R2'] > baseline['R2']
            rmse_pass = metrics['RMSE'] <= baseline['RMSE']
            mb_pass = abs(metrics['MB']) <= abs(baseline['MB'])
            innovation_pass = r2_pass and rmse_pass and mb_pass

            results[stage_name] = {
                'metrics': {k: v for k, v in metrics.items() if k != 'days_run'},
                'days_run': metrics.get('days_run', 0),
                'innovation_pass': innovation_pass
            }

            print(f"  R2={metrics['R2']:.4f} (阈值>{baseline['R2'] + 0.01:.4f}) {'PASS' if r2_pass else 'FAIL'}")
            print(f"  RMSE={metrics['RMSE']:.2f} (阈值<={baseline['RMSE']:.2f}) {'PASS' if rmse_pass else 'FAIL'}")
            print(f"  |MB|={abs(metrics['MB']):.2f} (阈值<={abs(baseline['MB']):.2f}) {'PASS' if mb_pass else 'FAIL'}")

            # 对齐设计文档 9.4 分阶段执行流程
            # pre_exp: R² ≤ 基线 -> 停止
            if stage_name == 'pre_exp' and not innovation_pass and not r2_above_baseline:
                outcome = 'failed'
                print(f"\n  [停止] pre_exp 未通过 -> outcome={outcome}")
                break
            if stage_name == 'pre_exp' and not innovation_pass and r2_above_baseline:
                print(f"\n  [继续] pre_exp 次级创新（R2 达标）-> 继续 stage1")
            # stage1: R² ≤ 基线 -> 停止
            if stage_name == 'stage1' and not innovation_pass and not r2_above_baseline:
                outcome = 'seasonally_limited'
                print(f"\n  [停止] stage1 未通过 -> outcome={outcome}")
                break
            if stage_name == 'stage1' and not innovation_pass and r2_above_baseline:
                print(f"\n  [继续] stage1 次级创新（R2 达标）-> 继续 stage2")
            # stage2 失败不阻止 stage3
            if stage_name == 'stage3':
                if innovation_pass:
                    outcome = 'fully_established'
                elif r2_above_baseline:
                    # 次级创新：R² > 基线，但 RMSE 或 MB 未达标
                    outcome = 'secondary_innovation'
                else:
                    outcome = 'partially_established'
                print(f"\n  [完成] stage3 -> outcome={outcome}")
        else:
            results[stage_name] = {
                'metrics': None,
                'days_run': 0,
                'innovation_pass': False
            }
            if stage_name == 'pre_exp':
                outcome = 'failed'
                print(f"\n  [停止] pre_exp 无数据 -> outcome={outcome}")
                break
            if stage_name == 'stage1':
                outcome = 'seasonally_limited'
                print(f"\n  [停止] stage1 无数据 -> outcome={outcome}")
                break
            if stage_name == 'stage3':
                outcome = 'partially_established'
                print(f"\n  [完成] stage3 无数据 -> outcome={outcome}")

    results['outcome'] = outcome
    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--pre-only', action='store_true', help='只运行 pre_exp 第一天预验证')
    args = parser.parse_args()

    if args.pre_only:
        # 快速预验证：只跑 pre_exp 第一天（2020-01-01）
        print(f"\n=== {METHOD_NAME} 预验证 (pre_exp day1) ===")
        func = _find_run_ten_fold()
        if func:
            func('2020-01-01')
            yt, yp = _get_cached_predictions()
            if yt is not None and yp is not None and len(yt) > 0:
                metrics = compute_metrics(np.array(yt), np.array(yp))
                baseline = {'R2': 0.8941, 'RMSE': 16.42, 'MB': 0.76}
                r2_pass = metrics['R2'] > baseline['R2'] + 0.01
                rmse_pass = metrics['RMSE'] <= baseline['RMSE']
                mb_pass = abs(metrics['MB']) <= abs(baseline['MB'])
                passed = r2_pass and rmse_pass and mb_pass
                print(f"  预验证{'通过' if passed else '失败'}: R2={metrics['R2']:.4f}")
                pre_path = os.path.join(PROJECT_ROOT, 'test_result', '创新方法', f'{METHOD_NAME}_pre_exp.json')
                with open(pre_path, 'w', encoding='utf-8') as f:
                    json.dump({'passed': passed, 'metrics': metrics}, f, indent=2)
            else:
                print("  预验证失败: 无数据")
        else:
            print("  无法找到 run_*_ten_fold 函数")
    else:
        # 运行完整多阶段验证
        results = run_multistage()

        # 保存结果
        output_dir = os.path.join(PROJECT_ROOT, 'test_result', '创新方法')
        os.makedirs(output_dir, exist_ok=True)

        json_path = os.path.join(output_dir, f'{METHOD_NAME}_all_stages.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存: {json_path}")

        csv_path = os.path.join(output_dir, f'{METHOD_NAME}_summary.csv')
        with open(csv_path, 'w', encoding='utf-8') as f:
            f.write('Method,Stage,R2,MAE,RMSE,MB,DaysRun,Pass\n')
            for stage_name, stage_data in results.items():
                if stage_name == 'outcome':
                    continue
                m = stage_data.get('metrics')
                if m:
                    f.write(f'{METHOD_NAME},{stage_name},{m["R2"]},{m["MAE"]},{m["RMSE"]},{m["MB"]},{stage_data.get("days_run",0)},{stage_data.get("innovation_pass",False)}\n')
            f.write(f'{METHOD_NAME},outcome,,,,,,{results["outcome"]}\n')
        print(f"结果已保存: {csv_path}")
