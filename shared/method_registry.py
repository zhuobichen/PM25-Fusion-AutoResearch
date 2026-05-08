# -*- coding: utf-8 -*-
"""
全局方法注册表
==============
单一数据源记录每个融合方法的生命周期状态，供 Pipeline 各 Phase 查询去重。

生命周期: designed → implemented → verified_pass | verified_fail | excluded

使用方式:
    from shared.method_registry import MethodRegistry
    registry = MethodRegistry()
    if registry.is_method_tested('PolyRK'):
        print('跳过')
"""

import os
import json
import math
from datetime import datetime
from typing import Dict, List, Optional

from shared.paths import get_project_root

# 项目根目录
_PROJECT_ROOT = str(get_project_root())

# 注册表文件名
_REGISTRY_FILENAME = 'method_registry.json'

# 生命周期状态常量
STATE_DEIGNED = 'designed'
STATE_IMPLEMENTED = 'implemented'
STATE_VERIFIED_PASS = 'verified_pass'
STATE_VERIFIED_FAIL = 'verified_fail'
STATE_EXCLUDED = 'excluded'

TESTED_STATES = {STATE_VERIFIED_PASS, STATE_VERIFIED_FAIL}
ALL_STATES = {STATE_DEIGNED, STATE_IMPLEMENTED, STATE_VERIFIED_PASS, STATE_VERIFIED_FAIL, STATE_EXCLUDED}

# 阶段名列表（用于排序）
STAGE_ORDER = ['pre_exp', 'stage1', 'stage2', 'stage3']


def _clean_for_json(obj):
    """递归清理对象，将 NaN/Inf 转为 None，确保 JSON 序列化安全。"""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _clean_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean_for_json(v) for v in obj]
    return obj


class MethodRegistry:
    """
    全局方法注册表。

    读写 method_registry.json，提供查询、写入、汇总等方法。
    文件不存在时自动创建空注册表。
    """

    def __init__(self, project_root: str = None):
        self._root = project_root or _PROJECT_ROOT
        self._path = os.path.join(self._root, _REGISTRY_FILENAME)
        self._data = None  # lazy load

    # ============================================================
    # 读写
    # ============================================================

    def load(self) -> dict:
        """从磁盘加载注册表。文件不存在时返回空注册表。"""
        if self._data is not None:
            return self._data
        if os.path.exists(self._path):
            with open(self._path, 'r', encoding='utf-8') as f:
                self._data = json.load(f)
        else:
            self._data = {
                '_meta': {
                    'version': '1.0.0',
                    'created_at': datetime.now().isoformat(),
                    'last_updated': datetime.now().isoformat(),
                },
                'methods': {},
            }
        return self._data

    def save(self):
        """将注册表写入磁盘。"""
        data = self.load()
        data['_meta']['last_updated'] = datetime.now().isoformat()
        data = _clean_for_json(data)
        with open(self._path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    # ============================================================
    # 查询
    # ============================================================

    def get_method(self, name: str) -> Optional[dict]:
        """按名称获取方法条目，不存在返回 None。"""
        return self.load()['methods'].get(name)

    def get_all_methods(self) -> List[dict]:
        """获取所有方法条目列表。"""
        return list(self.load()['methods'].values())

    def get_methods_by_state(self, state: str) -> List[dict]:
        """按状态筛选方法。"""
        return [m for m in self.get_all_methods() if m.get('state') == state]

    def method_exists(self, name: str) -> bool:
        """方法是否已注册。"""
        return name in self.load()['methods']

    def is_method_tested(self, name: str) -> bool:
        """方法是否已经过验证（pass 或 fail）。"""
        m = self.get_method(name)
        return m is not None and m.get('state') in TESTED_STATES

    # ============================================================
    # 便捷查询（供 Phase prompt 使用）
    # ============================================================

    def get_untested_methods(self) -> List[str]:
        """获取尚未验证的方法名列表（designed + implemented）。"""
        return [m['name'] for m in self.get_all_methods()
                if m.get('state') in (STATE_DEIGNED, STATE_IMPLEMENTED)]

    def get_pending_verification(self) -> List[str]:
        """获取待验证的方法名列表（仅 implemented）。"""
        return [m['name'] for m in self.get_methods_by_state(STATE_IMPLEMENTED)]

    def get_tested_method_names(self) -> set:
        """获取所有已验证方法名集合（用于快速查重）。"""
        return {m['name'] for m in self.get_all_methods()
                if m.get('state') in TESTED_STATES}

    # ============================================================
    # 写入
    # ============================================================

    def add_method(self, name: str, **kwargs) -> dict:
        """
        注册新方法。如已存在则更新。

        kwargs 可包含: fingerprint, state, category, metrics,
        all_stages, source_files, notes 等。
        """
        data = self.load()
        now = datetime.now().isoformat()

        if name in data['methods']:
            # 已存在，合并更新
            entry = data['methods'][name]
            for k, v in kwargs.items():
                if k == 'timestamps':
                    entry.setdefault('timestamps', {}).update(v)
                elif k == 'source_files':
                    entry.setdefault('source_files', {}).update(v)
                else:
                    entry[k] = v
        else:
            entry = {'name': name, 'timestamps': {'created_at': now}}
            for k, v in kwargs.items():
                if k == 'timestamps':
                    entry.setdefault('timestamps', {}).update(v)
                elif k == 'source_files':
                    entry.setdefault('source_files', {}).update(v)
                else:
                    entry[k] = v
            data['methods'][name] = entry

        return entry

    def update_state(self, name: str, new_state: str):
        """更新方法状态。"""
        entry = self.get_method(name)
        if entry is None:
            raise KeyError(f'方法 {name} 未注册')
        entry['state'] = new_state
        now = datetime.now().isoformat()
        if new_state == STATE_DEIGNED:
            entry.setdefault('timestamps', {})['designed_at'] = now
        elif new_state == STATE_IMPLEMENTED:
            entry.setdefault('timestamps', {})['implemented_at'] = now
        elif new_state in TESTED_STATES:
            entry.setdefault('timestamps', {})['verified_at'] = now

    def update_metrics(self, name: str, metrics: dict, stage: str = None):
        """
        更新方法指标。

        stage 不为 None 时更新 all_stages[stage] 并重新计算 best metrics。
        stage 为 None 时直接更新顶层 metrics。
        """
        entry = self.get_method(name)
        if entry is None:
            raise KeyError(f'方法 {name} 未注册')

        if stage is not None:
            entry.setdefault('all_stages', {})[stage] = metrics
            # 重新计算 best stage
            self._compute_best_metrics(entry)
        else:
            entry['metrics'] = metrics

    def update_source_files(self, name: str, **file_paths):
        """更新来源文件路径。"""
        entry = self.get_method(name)
        if entry is None:
            raise KeyError(f'方法 {name} 未注册')
        entry.setdefault('source_files', {}).update(file_paths)

    def set_fingerprint(self, name: str, fingerprint: str):
        """设置方法指纹。"""
        entry = self.get_method(name)
        if entry is None:
            raise KeyError(f'方法 {name} 未注册')
        entry['fingerprint'] = fingerprint

    # ============================================================
    # 从已有文件更新（供 build_registry 和 Phase 5 使用）
    # ============================================================

    def update_from_all_stages_json(self, name: str, json_path: str, state: str = None):
        """
        从 *_all_stages.json 文件解析指标并更新注册表。

        兼容三种 JSON Schema：
        A: {stage: {metrics: {R2,...}, 判定: {innovation_verified}}}
        B: {stage: {R2, MAE, RMSE, MB}}  (扁平)
        C: {stage: {metrics: {ols: {...}, huber: {...}}, 判定: {ols_..., huber_...}}}
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            raw = json.load(f)

        all_stages = {}
        stages_passed = 0
        stages_verified = 0

        for stage_name in STAGE_ORDER:
            if stage_name not in raw:
                continue
            stage_data = raw[stage_name]
            parsed = self._parse_stage(stage_data)
            if parsed is None:
                continue
            all_stages[stage_name] = parsed
            stages_verified += 1
            if parsed.get('innovation_verified'):
                stages_passed += 1

        if not all_stages:
            return

        # 计算 best metrics（R2 最高的阶段）
        best_stage = max(all_stages.keys(), key=lambda s: all_stages[s].get('R2', -999))
        best_metrics = {
            'R2': all_stages[best_stage].get('R2'),
            'MAE': all_stages[best_stage].get('MAE'),
            'RMSE': all_stages[best_stage].get('RMSE'),
            'MB': all_stages[best_stage].get('MB'),
        }

        # 确定状态
        if state is None:
            state = STATE_VERIFIED_PASS if stages_passed > 0 else STATE_VERIFIED_FAIL

        # 注册或更新
        if not self.method_exists(name):
            self.add_method(name)

        entry = self.get_method(name)
        entry['state'] = state
        entry['metrics'] = best_metrics
        entry['best_stage'] = best_stage
        entry['all_stages'] = all_stages
        entry['stages_verified'] = stages_verified
        entry['stages_passed'] = stages_passed
        entry.setdefault('source_files', {})['result_json'] = os.path.relpath(json_path, self._root).replace('\\', '/')
        entry.setdefault('timestamps', {})['verified_at'] = datetime.now().isoformat()

    # ============================================================
    # 内部解析
    # ============================================================

    def _parse_stage(self, stage_data: dict) -> Optional[dict]:
        """
        解析单个阶段数据，兼容 Schema A/B/C。

        返回统一格式: {R2, MAE, RMSE, MB, innovation_verified}
        """
        if not isinstance(stage_data, dict):
            return None

        # Schema A: 有 metrics 子键
        if 'metrics' in stage_data:
            metrics_raw = stage_data['metrics']
            judgment = stage_data.get('判定', {})

            # Schema C: metrics 是 dict of dict（如 {ols: {...}, huber: {...}}）
            if isinstance(metrics_raw, dict) and any(isinstance(v, dict) for v in metrics_raw.values()):
                # 选择第一个子方法
                sub_name = list(metrics_raw.keys())[0]
                m = metrics_raw[sub_name]
                # 对应判定键
                verified_key = f'{sub_name}_innovation_verified'
                verified = judgment.get(verified_key, False)
            else:
                # Schema A 标准
                m = metrics_raw
                verified = judgment.get('innovation_verified', False)

        # Schema B: 扁平，直接有 R2 等键
        elif 'R2' in stage_data:
            m = stage_data
            # 没有判定字段，需要外部计算
            verified = False
        else:
            return None

        return {
            'R2': float(m.get('R2', 0)),
            'MAE': float(m.get('MAE', 0)),
            'RMSE': float(m.get('RMSE', 0)),
            'MB': float(m.get('MB', 0)),
            'innovation_verified': bool(verified),
        }

    def _compute_best_metrics(self, entry: dict):
        """根据 all_stages 重新计算 best_stage 和顶层 metrics。"""
        stages = entry.get('all_stages', {})
        if not stages:
            return
        best_stage = max(stages.keys(), key=lambda s: stages[s].get('R2', -999))
        entry['best_stage'] = best_stage
        entry['metrics'] = {
            'R2': stages[best_stage].get('R2'),
            'MAE': stages[best_stage].get('MAE'),
            'RMSE': stages[best_stage].get('RMSE'),
            'MB': stages[best_stage].get('MB'),
        }
        entry['stages_verified'] = len(stages)
        entry['stages_passed'] = sum(1 for s in stages.values() if s.get('innovation_verified'))

    # ============================================================
    # 汇总输出
    # ============================================================

    def print_summary(self):
        """打印注册表摘要到控制台。"""
        data = self.load()
        methods = data.get('methods', {})
        if not methods:
            print('注册表为空。运行 python -m shared.build_registry 构建。')
            return

        # 按状态分组统计
        by_state = {}
        for m in methods.values():
            s = m.get('state', 'unknown')
            by_state.setdefault(s, []).append(m)

        print('=' * 65)
        print('方法注册表摘要')
        print('=' * 65)
        print(f'总方法数: {len(methods)}')
        print()

        state_labels = {
            STATE_DEIGNED: '已设计 (designed)',
            STATE_IMPLEMENTED: '已实现 (implemented)',
            STATE_VERIFIED_PASS: '验证通过 (verified_pass)',
            STATE_VERIFIED_FAIL: '验证失败 (verified_fail)',
            STATE_EXCLUDED: '已排除 (excluded)',
        }

        for state in ALL_STATES:
            label = state_labels.get(state, state)
            group = by_state.get(state, [])
            print(f'  {label}: {len(group)}')
            if state in TESTED_STATES and group:
                # 按 R2 排序
                group_sorted = sorted(group, key=lambda m: m.get('metrics', {}).get('R2', 0), reverse=True)
                for m in group_sorted[:10]:
                    r2 = m.get('metrics', {}).get('R2', 0) or 0
                    stages_p = m.get('stages_passed', '?')
                    stages_v = m.get('stages_verified', '?')
                    print(f'    - {m["name"]:<25s} R2={r2:.4f}  ({stages_p}/{stages_v} stages passed)')
                if len(group) > 10:
                    print(f'    ... 及 {len(group) - 10} 个其他方法')
            print()

        print(f'注册表文件: {self._path}')
        print(f'最后更新: {data.get("_meta", {}).get("last_updated", "N/A")}')

    def to_dict(self) -> dict:
        """返回完整注册表字典。"""
        return self.load()
