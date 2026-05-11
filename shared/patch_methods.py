# -*- coding: utf-8 -*-
"""
批量给方法文件添加预测值缓存
============================
在每个方法的 run_*_ten_fold 函数中，找到最后一个 compute_metrics(X, Y) 调用，
在其后插入缓存 Y 到模块级变量 _last_y_true/_last_y_pred。

用法：
    python -m shared.patch_methods           # 执行补丁
    python -m shared.patch_methods --dry-run  # 预览不修改
"""

import os
import re
import argparse
from shared.paths import get_project_root


CACHE_HEADER = '''
# === 预测值缓存（由 patch_methods.py 自动添加）===
_last_y_true = None
_last_y_pred = None
'''

CACHE_TEMPLATE = '''
    # 缓存预测值供多天聚合使用
    global _last_y_true, _last_y_pred
    _last_y_true = {true_var}
    _last_y_pred = {pred_var}
'''


def find_methods_dir():
    return os.path.join(str(get_project_root()), 'CodeWorkSpace', '新融合方法代码')


def find_ten_fold_function(content):
    """找到 run_*_ten_fold 函数的起始行号。"""
    match = re.search(r'^def (run_\w*_ten_fold)\s*\(', content, re.MULTILINE)
    if match:
        return match.group(1), match.start()
    return None, None


def find_last_compute_metrics(content, func_start):
    """在 ten_fold 函数中找到最后一个 compute_metrics(X, Y) 调用。"""
    # 从函数定义开始搜索
    func_content = content[func_start:]
    pattern = r'compute_metrics\s*\(\s*(\w+)\s*,\s*(\w+)\s*\)'
    matches = list(re.finditer(pattern, func_content))
    if not matches:
        return None, None, None
    last = matches[-1]
    true_var = last.group(1)
    pred_var = last.group(2)
    absolute_pos = func_start + last.end()
    return true_var, pred_var, absolute_pos


def has_cache(content):
    """检查文件是否已有缓存代码。"""
    return '_last_y_true' in content


def patch_file(filepath, dry_run=False):
    """给单个方法文件添加缓存。"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    if has_cache(content):
        return 'skipped', '已有缓存'

    # 找 ten_fold 函数
    func_name, func_start = find_ten_fold_function(content)
    if not func_name:
        return 'skipped', '无 ten_fold 函数'

    # 找最后一个 compute_metrics 调用
    true_var, pred_var, insert_pos = find_last_compute_metrics(content, func_start)
    if not true_var:
        return 'skipped', '无 compute_metrics 调用'

    if dry_run:
        return 'would_patch', f'{func_name}: compute_metrics({true_var}, {pred_var})'

    # 插入缓存代码（在 compute_metrics 调用之后）
    cache_code = CACHE_TEMPLATE.format(true_var=true_var, pred_var=pred_var)
    new_content = content[:insert_pos] + cache_code + content[insert_pos:]

    # 添加模块级变量（在第一个 import 之前或文件开头）
    # 找到第一个非注释、非空行的位置
    lines = new_content.split('\n')
    insert_line = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped and not stripped.startswith('#') and not stripped.startswith('"""') and not stripped.startswith("'''"):
            insert_line = i
            break

    lines.insert(insert_line, CACHE_HEADER)
    new_content = '\n'.join(lines)

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(new_content)

    return 'patched', f'{func_name}: compute_metrics({true_var}, {pred_var})'


def main():
    parser = argparse.ArgumentParser(description='批量给方法文件添加预测值缓存')
    parser.add_argument('--dry-run', action='store_true', help='预览模式')
    args = parser.parse_args()

    methods_dir = find_methods_dir()
    skip_prefixes = ('compare_', 'find_best_', 'validate_', 'lambda_', 'spatial_stat_',
                     'statistical_', 'robust_variogram_', 'mle_', 'elegant_', 'adaptive_',
                     'CrossDayValidation', 'patch_')

    files = sorted(f for f in os.listdir(methods_dir)
                   if f.endswith('.py') and not f.startswith(skip_prefixes))

    stats = {'patched': 0, 'skipped': 0, 'would_patch': 0, 'error': 0}

    for fname in files:
        filepath = os.path.join(methods_dir, fname)
        try:
            status, detail = patch_file(filepath, dry_run=args.dry_run)
            stats[status] = stats.get(status, 0) + 1
            if status != 'skipped':
                print(f"  [{'预览' if args.dry_run else '补丁'}] {fname}: {detail}")
        except Exception as e:
            stats['error'] += 1
            print(f"  [错误] {fname}: {e}")

    print(f"\n统计: {stats}")


if __name__ == '__main__':
    main()
