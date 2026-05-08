# -*- coding: utf-8 -*-
"""
迁移脚本：从已有文件构建全局方法注册表
======================================

扫描优先级：
1. Innovation/success/*/_all_stages.json → state=verified_pass
2. Innovation/failed/*/_all_stages.json → state=verified_fail
3. test_result/创新方法/*_summary.csv → state=verified_pass/fail
4. CodeWorkSpace/新融合方法代码/*.py → state=implemented
5. SmartToCode/创新方法指令/*.md → state=designed
6. SmartToCode/method_fingerprint.md5 → 匹配指纹

用法：
    python -m shared.build_registry              # 从头构建
    python -m shared.build_registry --dry-run    # 预览不写入
    python -m shared.build_registry --merge      # 合并已有注册表
"""

import os
import re
import csv
import glob as _glob
import argparse
from datetime import datetime

from shared.paths import get_project_root
from shared.method_registry import (
    MethodRegistry, STATE_DEIGNED, STATE_IMPLEMENTED,
    STATE_VERIFIED_PASS, STATE_VERIFIED_FAIL, STAGE_ORDER,
)


def _project_root():
    return str(get_project_root())


# ============================================================
# 1. Innovation/ 目录扫描
# ============================================================

def scan_innovation_dir(registry: MethodRegistry, root: str):
    """
    扫描 Innovation/success/ 和 Innovation/failed/ 中的 _all_stages.json。

    状态判定逻辑：
    - 有判定字段（Schema A/C）：根据实际 stages_passed 判断
    - 无判定字段（Schema B 扁平）：用目录位置作为判定（success→pass, failed→fail）
    - 同一方法在 success/ 和 failed/ 都有时，取 success
    """
    count = 0
    for verdict, default_state in [('success', STATE_VERIFIED_PASS), ('failed', STATE_VERIFIED_FAIL)]:
        base = os.path.join(root, 'Innovation', verdict)
        if not os.path.isdir(base):
            continue
        for method_dir in sorted(os.listdir(base)):
            method_path = os.path.join(base, method_dir)
            if not os.path.isdir(method_path):
                continue
            json_files = [f for f in os.listdir(method_path) if f.endswith('_all_stages.json')]
            for jf in json_files:
                json_path = os.path.join(method_path, jf)
                name = method_dir  # 用目录名作为方法名

                # success 优先：已有 verified_pass 则不覆盖
                if registry.method_exists(name) and registry.get_method(name).get('state') == STATE_VERIFIED_PASS:
                    continue

                # 先解析，不传 state，让 update_from_all_stages_json 自动判断
                registry.update_from_all_stages_json(name, json_path, state=None)

                # 对于 Schema B（无判定字段，stages_passed=0），用目录位置作为判定
                entry = registry.get_method(name)
                if entry and entry.get('stages_passed', 0) == 0:
                    # 检查是否有判定字段（Schema A/C 有判定但全部为 False）
                    has_judgment = _check_has_judgment(json_path)
                    if not has_judgment:
                        # Schema B：无判定字段，用目录位置
                        entry['state'] = default_state
                    # else: Schema A/C 有判定但 0 passed，保持 verified_fail

                registry.update_source_files(name, result_json=os.path.relpath(json_path, root).replace('\\', '/'))
                count += 1

    return count


def _check_has_judgment(json_path: str) -> bool:
    """检查 JSON 文件中是否有判定字段。"""
    import json as _json
    with open(json_path, 'r', encoding='utf-8') as f:
        d = _json.load(f)
    for stage_name in STAGE_ORDER:
        if stage_name in d:
            stage_data = d[stage_name]
            if isinstance(stage_data, dict) and '判定' in stage_data:
                return True
            # 也检查扁平结构中的 innovation_verified
            if 'innovation_verified' in stage_data:
                return True
    return False


# ============================================================
# 2. test_result/ CSV 扫描
# ============================================================

def scan_test_result_csv(registry: MethodRegistry, root: str):
    """
    扫描 test_result/创新方法/*_summary.csv。

    注意：CSV 只包含测试指标，不代表创新验证通过。
    真正的 verified_pass/verified_fail 由 Innovation/ 目录决定。
    CSV 中的方法仅补充指标数据，不改变状态。
    """
    csv_dir = os.path.join(root, 'test_result', '创新方法')
    if not os.path.isdir(csv_dir):
        return 0

    count = 0
    for csv_file in sorted(os.listdir(csv_dir)):
        if not csv_file.endswith('_summary.csv'):
            continue
        csv_path = os.path.join(csv_dir, csv_file)

        try:
            rows = _parse_summary_csv(csv_path)
        except Exception:
            continue

        for method_name, metrics in rows:
            if registry.method_exists(method_name):
                existing = registry.get_method(method_name)
                # 已有 Innovation/ 目录的判定结果（verified_pass/verified_fail），仅补充 CSV 指标
                if existing.get('state') in (STATE_VERIFIED_PASS, STATE_VERIFIED_FAIL):
                    if 'all_stages' not in existing:
                        existing.setdefault('metrics_csv', {}).update(metrics)
                    entry = existing
                else:
                    entry = existing
            else:
                # CSV 中有但 Innovation/ 中没有 → 仅标记为 implemented（有代码和测试结果，但未走创新验证流程）
                registry.add_method(method_name, state=STATE_IMPLEMENTED)
                entry = registry.get_method(method_name)

            entry.setdefault('source_files', {})['summary_csv'] = os.path.relpath(csv_path, root).replace('\\', '/')
            count += 1

    return count


def _parse_summary_csv(csv_path: str):
    """
    解析 *_summary.csv，兼容多种列格式。

    返回: [(method_name, {R2, MAE, RMSE, MB}), ...]
    """
    results = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # 方法名：优先 'Method'，其次 'method'
            name = row.get('Method') or row.get('method') or ''
            name = name.strip()
            if not name:
                continue

            # 指标：直接取 R2, MAE, RMSE, MB
            metrics = {}
            for key in ('R2', 'MAE', 'RMSE', 'MB'):
                val = row.get(key)
                if val is not None:
                    try:
                        metrics[key] = float(val)
                    except (ValueError, TypeError):
                        metrics[key] = 0.0
                else:
                    metrics[key] = 0.0

            results.append((name, metrics))

    return results


# ============================================================
# 3. CodeWorkSpace/ 代码扫描
# ============================================================

def scan_codeworkspace(registry: MethodRegistry, root: str):
    """扫描 CodeWorkSpace/新融合方法代码/*.py，标记为 implemented。"""
    code_dir = os.path.join(root, 'CodeWorkSpace', '新融合方法代码')
    if not os.path.isdir(code_dir):
        return 0

    # 跳过辅助脚本
    skip_prefixes = ('compare_', 'find_best_', 'validate_', 'lambda_', 'spatial_stat_',
                     'statistical_', 'robust_variogram_', 'mle_', 'elegant_', 'adaptive_')

    count = 0
    for py_file in sorted(os.listdir(code_dir)):
        if not py_file.endswith('.py'):
            continue
        if py_file.startswith(skip_prefixes):
            continue

        method_name = py_file[:-3]  # 去掉 .py

        if registry.method_exists(method_name):
            # 已有更高级状态，跳过
            existing = registry.get_method(method_name)
            if existing.get('state') in (STATE_VERIFIED_PASS, STATE_VERIFIED_FAIL):
                continue
            # 如果是 designed，升级为 implemented
            if existing.get('state') == STATE_DEIGNED:
                registry.update_state(method_name, STATE_IMPLEMENTED)
        else:
            registry.add_method(method_name, state=STATE_IMPLEMENTED)

        registry.update_source_files(method_name, code=f'CodeWorkSpace/新融合方法代码/{py_file}')
        count += 1

    return count


# ============================================================
# 4. SmartToCode/ 设计指令扫描
# ============================================================

def scan_smarttocode(registry: MethodRegistry, root: str):
    """扫描 SmartToCode/创新方法指令/*.md，标记为 designed。"""
    design_dir = os.path.join(root, 'SmartToCode', '创新方法指令')
    if not os.path.isdir(design_dir):
        return 0

    count = 0
    for md_file in sorted(os.listdir(design_dir)):
        if not md_file.endswith('.md'):
            continue

        # 从文件名提取方法名
        method_name = _extract_method_name_from_design(md_file)
        if not method_name:
            continue

        if registry.method_exists(method_name):
            # 已有更高级状态，跳过
            continue

        registry.add_method(method_name, state=STATE_DEIGNED)
        registry.update_source_files(method_name, design=f'SmartToCode/创新方法指令/{md_file}')
        count += 1

    return count


def _extract_method_name_from_design(filename: str) -> str:
    """从设计指令文件名提取方法名。"""
    name = filename[:-3]  # 去掉 .md

    # Innovation_XXX → XXX
    if name.startswith('Innovation_'):
        return name[len('Innovation_'):]

    # PG-STGAT... → PG-STGAT
    # VCFFM... → VCFFM
    # 直接返回（去掉中文描述部分）
    # 例如: "PG-STGAT物理引导时空图注意力网络法" → "PG-STGAT"
    m = re.match(r'^([A-Za-z][A-Za-z0-9_-]+)', name)
    if m:
        return m.group(1)

    return name


# ============================================================
# 5. 指纹文件扫描
# ============================================================

def scan_fingerprints(registry: MethodRegistry, root: str):
    """扫描 SmartToCode/method_fingerprint.md5，补充指纹信息。"""
    fp_path = os.path.join(root, 'SmartToCode', 'method_fingerprint.md5')
    if not os.path.exists(fp_path):
        return 0

    count = 0
    with open(fp_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = [p.strip() for p in line.split('|')]
            if len(parts) < 2:
                continue
            fp_hash, method_name = parts[0], parts[1]

            if registry.method_exists(method_name):
                registry.set_fingerprint(method_name, fp_hash)
                count += 1

    return count


# ============================================================
# 主入口
# ============================================================

def build_registry(merge: bool = False, dry_run: bool = False) -> MethodRegistry:
    """构建全局方法注册表。"""
    root = _project_root()
    registry = MethodRegistry(project_root=root)

    if merge:
        registry.load()  # 加载已有注册表
    else:
        # 从头构建：初始化空注册表（不读磁盘旧文件）
        registry._data = {
            '_meta': {
                'version': '1.0.0',
                'created_at': datetime.now().isoformat(),
                'last_updated': datetime.now().isoformat(),
            },
            'methods': {},
        }

    print('=' * 60)
    print('构建方法注册表')
    print('=' * 60)
    print(f'项目根目录: {root}')
    print(f'模式: {"合并" if merge else "全新构建"}')
    print()

    # 1. Innovation/ 目录
    n1 = scan_innovation_dir(registry, root)
    print(f'[1/5] Innovation/ 目录: {n1} 个方法')

    # 2. test_result/ CSV
    n2 = scan_test_result_csv(registry, root)
    print(f'[2/5] test_result/ CSV: {n2} 个方法')

    # 3. CodeWorkSpace/
    n3 = scan_codeworkspace(registry, root)
    print(f'[3/5] CodeWorkSpace/: {n3} 个方法')

    # 4. SmartToCode/
    n4 = scan_smarttocode(registry, root)
    print(f'[4/5] SmartToCode/: {n4} 个方法')

    # 5. 指纹
    n5 = scan_fingerprints(registry, root)
    print(f'[5/5] 指纹匹配: {n5} 个方法')

    # 汇总
    methods = registry.get_all_methods()
    by_state = {}
    for m in methods:
        s = m.get('state', 'unknown')
        by_state.setdefault(s, []).append(m)

    print()
    print('--- 汇总 ---')
    for state, label in [
        (STATE_VERIFIED_PASS, '验证通过'),
        (STATE_VERIFIED_FAIL, '验证失败'),
        (STATE_IMPLEMENTED, '已实现'),
        (STATE_DEIGNED, '已设计'),
    ]:
        group = by_state.get(state, [])
        print(f'  {label}: {len(group)}')

    print(f'  总计: {len(methods)}')

    if dry_run:
        print()
        print('[DRY RUN] 不写入文件。')
    else:
        registry.save()
        print()
        print(f'已写入: {registry._path}')

    return registry


def main():
    parser = argparse.ArgumentParser(description='构建全局方法注册表')
    parser.add_argument('--dry-run', action='store_true', help='预览模式，不写入文件')
    parser.add_argument('--merge', action='store_true', help='合并已有注册表（而非全新构建）')
    args = parser.parse_args()

    registry = build_registry(merge=args.merge, dry_run=args.dry_run)

    if not args.dry_run:
        print()
        registry.print_summary()


if __name__ == '__main__':
    main()
