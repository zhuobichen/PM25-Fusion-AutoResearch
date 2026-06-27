# -*- coding: utf-8 -*-
"""
PM2.5 CMAQ融合方法自动研究流程 v14.0
=====================================
闭环工作流版：支持多 Agent 后端（TRAE / CLI / Manual）

使用方式：
    # 闭环模式（默认，推荐）：生成任务描述，由 TRAE Agent 执行
    python run_pipeline.py --auto
    python run_pipeline.py --auto --skip 1
    python run_pipeline.py --auto --only 3,4,5

    # 指定后端
    python run_pipeline.py --auto --backend trae      # TRAE Agent（默认）
    python run_pipeline.py --auto --backend cli        # Claude CLI（向后兼容）
    python run_pipeline.py --auto --backend manual     # 手动指引

    # 向后兼容：--agent 等价于 --backend cli
    python run_pipeline.py --auto --agent
    python run_pipeline.py --auto --agent --budget 5.0

    # 直接跑验证脚本（不需要 agent）
    python run_pipeline.py --auto --only 5

    # Profile 管理
    python run_pipeline.py --list-profiles
    python run_pipeline.py --save-profile my-flow --skip 1

闭环工作流原理：
    1. run_pipeline.py 遇到 LLM Phase → 生成任务描述 → 返回
    2. TRAE Agent 读取任务描述并执行
    3. 产物产生后 → 重新运行 run_pipeline.py → 自动检测产物并推进
    4. 遇到下一个 LLM Phase → 重复步骤 1-3
    5. Phase 5（验证）始终直接执行 Python 脚本，无需 Agent
"""

import os
import sys
import json
import time
import signal
import glob
import io
import argparse
import subprocess
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime

# --- 路径初始化 ---
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)
from shared.paths import get_project_root, data_path

PROJECT_ROOT = str(get_project_root())

# --- Windows 终端中文支持 ---
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

# --- Phase 元数据 ---
PHASE_INFO = {
    0: ('organize', '项目整理'),
    1: ('download', '文献下载'),
    2: ('analyze',  '文献分析'),
    3: ('design',   '方案设计'),
    4: ('code',     '代码实现'),
    5: ('verify',   '测试验证'),
    6: ('write',    '论文写作'),
}

_REGISTRY_SYNCED = False


def _normalize_method_name(name):
    """统一方法名，便于在脚本文件名、注册表、状态文件之间对齐。"""
    return str(name).strip().replace('-', '_').replace(' ', '_')


def _load_json_file(file_path):
    if not os.path.exists(file_path):
        return None
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None


def _sync_method_registry(force=False, silent=True):
    """从现有产物重建方法注册表，确保每次启动都基于最新结果继续。"""
    global _REGISTRY_SYNCED
    if _REGISTRY_SYNCED and not force:
        return True

    try:
        from shared.build_registry import build_registry
        if silent:
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                build_registry(merge=False, dry_run=False)
        else:
            build_registry(merge=False, dry_run=False)
        _REGISTRY_SYNCED = True
        return True
    except Exception as e:
        if not silent:
            print(f"[警告] 方法注册表同步失败: {e}")
        return False


def _get_method_registry(sync=True):
    """获取方法注册表实例。"""
    if sync:
        _sync_method_registry(silent=True)
    try:
        from shared.method_registry import MethodRegistry
        return MethodRegistry(project_root=PROJECT_ROOT)
    except Exception:
        return None


def _load_research_state():
    """读取历史研究状态，供多次启动时延续优化方向。"""
    state_path = os.path.join(PROJECT_ROOT, 'test_result', '.state', 'research_state.json')
    return _load_json_file(state_path) or {}


def _get_registry_state_names(state_name):
    registry = _get_method_registry(sync=True)
    if registry is None:
        return []
    try:
        methods = registry.get_methods_by_state(state_name)
        return sorted(m['name'] for m in methods if m.get('name'))
    except Exception:
        return []


def _get_resume_context():
    """汇总当前最佳方法、失败历史和待处理方法，用于迭代式优化。"""
    research_state = _load_research_state()
    pending_implemented = _get_registry_state_names('implemented')
    pending_designed = _get_registry_state_names('designed')
    verified_fail = _get_registry_state_names('verified_fail')

    failed_methods = []
    for item in research_state.get('failed_methods', [])[-5:]:
        method = item.get('method')
        reason = item.get('reason', '')
        if method:
            failed_methods.append((method, reason))

    best_method = research_state.get('current_best_method') or PipelineState.load().get('current_best_method')
    best_r2 = None
    if research_state.get('current_best_metrics'):
        best_r2 = research_state['current_best_metrics'].get('R2')
    if best_r2 is None:
        best_r2 = PipelineState.load().get('current_best_r2')

    return {
        'best_method': best_method or 'None',
        'best_r2': best_r2,
        'research_status': research_state.get('status', 'unknown'),
        'iteration': research_state.get('iteration', 0),
        'pending_designed': pending_designed,
        'pending_implemented': pending_implemented,
        'verified_fail': verified_fail,
        'failed_methods': failed_methods,
        'next_direction': research_state.get('status', ''),
    }


def _build_resume_prompt_block():
    """为 Agent prompt 注入跨多次启动的历史上下文。"""
    context = _get_resume_context()
    best_r2_str = f"{context['best_r2']:.4f}" if isinstance(context['best_r2'], (int, float)) else "N/A"

    lines = [
        "## 历史迭代上下文（自动注入）",
        "",
        f"- 当前最佳方法: {context['best_method']}",
        f"- 当前最佳R²: {best_r2_str}",
        f"- 历史迭代轮次: {context['iteration']}",
        f"- 研究状态: {context['research_status']}",
    ]

    if context['failed_methods']:
        lines.append("- 最近失败方法（避免重复尝试同一路线）:")
        for method, reason in context['failed_methods'][:5]:
            reason_text = reason or "无记录原因"
            lines.append(f"  - {method}: {reason_text}")

    if context['pending_designed']:
        lines.append(f"- 待实现方法: {', '.join(context['pending_designed'][:12])}")
    else:
        lines.append("- 待实现方法: 无")

    if context['pending_implemented']:
        lines.append(f"- 待验证方法: {', '.join(context['pending_implemented'][:12])}")
    else:
        lines.append("- 待验证方法: 无")

    if context['verified_fail']:
        lines.append(f"- 已验证失败方法数: {len(context['verified_fail'])}")

    lines += [
        "",
        "要求：",
        "- 本次执行必须基于以上历史结果继续推进，而不是重复从零开始。",
        "- 优先处理待实现/待验证方法；只有在这些队列为空时才设计全新方法。",
        "- 已验证通过、已验证失败的方法都不要重复跑同一流程，除非明确是在做定向改造。",
    ]
    return "\n".join(lines)


def _recommended_phases_from_history():
    """
    根据历史结果给出默认执行阶段。
    目标：多次启动时优先“继续上次未完成的迭代”，而不是重复全流程。
    """
    _sync_method_registry(silent=True)
    context = _get_resume_context()
    phases = []

    paper_pdf = os.path.exists(os.path.join(PROJECT_ROOT, 'paper_output', 'paper.pdf'))
    innovation_established = bool(PipelineState.load().get('innovation_established'))
    if not innovation_established:
        innovation_established = context['research_status'] == 'converged'

    if context['pending_designed']:
        phases.extend([4, 5])
    elif context['pending_implemented']:
        phases.append(5)
    elif innovation_established:
        if not paper_pdf:
            phases.append(6)
    else:
        phases.extend([3, 4, 5])

    deduped = []
    for phase in phases:
        if phase not in deduped:
            deduped.append(phase)
    return deduped


# ============================================================
# 状态管理
# ============================================================

class PipelineState:
    """Pipeline state manager"""

    STATE_FILE = os.path.join(PROJECT_ROOT, '.agent_state.json')

    @classmethod
    def load(cls, include_inferred=True):
        if os.path.exists(cls.STATE_FILE):
            with open(cls.STATE_FILE, 'r', encoding='utf-8') as f:
                old_state = json.load(f)
            # Migrate old format -> new format
            state = cls._migrate(old_state)
        else:
            state = cls._default_state()

        if include_inferred:
            return cls._merge_inferred_state(state)
        return state

    @classmethod
    def _migrate(cls, old):
        """Migrate from old agent-based state to new phase-based state"""
        state = cls._default_state()
        state['round'] = old.get('round', 0)
        state['innovation_established'] = old.get('innovation_established', False)
        state['current_best_method'] = old.get('baseline_method')
        state['current_best_r2'] = old.get('baseline_r2')
        state['last_run'] = old.get('last_run')

        # Map old agent statuses to phases
        agent_phase_map = {
            'organizer': 'organize',
            'dl_1': 'download', 'dl_2': 'download', 'dl_3': 'download',
            'analyzer': 'analyze',
            'designer': 'design',
            'engineer': 'code',
            'verifier': 'verify',
            'writer': 'write',
        }
        old_agents = old.get('agents', {})
        for agent_id, info in old_agents.items():
            phase = agent_phase_map.get(agent_id)
            if phase and info.get('status') == 'completed':
                state['phases'][phase] = {
                    'status': 'completed',
                    'completed_at': info.get('completed_at', '')
                }

        return state

    @classmethod
    def save(cls, state):
        state = cls._strip_inferred_fields(state)
        state['last_run'] = datetime.now().isoformat()
        with open(cls.STATE_FILE, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2, ensure_ascii=False)

    @staticmethod
    def _default_state():
        return {
            'round': 0,
            'phases': {},
            'innovation_established': False,
            'current_best_method': None,
            'current_best_r2': None,
            'last_run': None,
        }

    @classmethod
    def _strip_inferred_fields(cls, state):
        """仅保存显式状态，不把自动推断的字段回写到文件。"""
        clean = {
            'round': state.get('round', 0),
            'phases': {},
            'innovation_established': state.get('innovation_established', False),
            'current_best_method': state.get('current_best_method'),
            'current_best_r2': state.get('current_best_r2'),
            'last_run': state.get('last_run'),
        }
        for phase_name, info in state.get('phases', {}).items():
            if info.get('inferred') and not info.get('completed_at'):
                continue
            clean['phases'][phase_name] = {
                'status': info.get('status', 'completed'),
                'completed_at': info.get('completed_at', '')
            }
        return clean

    @classmethod
    def _merge_inferred_state(cls, state):
        """将磁盘产物推断出的状态与显式状态合并。"""
        merged = {
            'round': state.get('round', 0),
            'phases': dict(state.get('phases', {})),
            'innovation_established': state.get('innovation_established', False),
            'current_best_method': state.get('current_best_method'),
            'current_best_r2': state.get('current_best_r2'),
            'last_run': state.get('last_run'),
        }

        inferred = cls._infer_from_artifacts()

        for phase_name, info in inferred.get('phases', {}).items():
            existing = merged['phases'].get(phase_name)
            if existing and existing.get('status') == 'completed':
                continue
            merged['phases'][phase_name] = info

        if inferred.get('innovation_established'):
            merged['innovation_established'] = True

        inferred_r2 = inferred.get('current_best_r2')
        current_r2 = merged.get('current_best_r2')
        if inferred.get('current_best_method') and (
            current_r2 is None or (inferred_r2 is not None and inferred_r2 > current_r2)
        ):
            merged['current_best_method'] = inferred['current_best_method']
            merged['current_best_r2'] = inferred_r2

        return merged

    @classmethod
    def _infer_from_artifacts(cls):
        """基于现有产物推断各阶段状态，避免只依赖 .agent_state.json。

        v14 改进：在文件计数之外，交叉验证 MethodRegistry 的方法状态，
        使 Phase 完成判断更准确（不再仅靠文件数量阈值）。
        """
        inferred = {
            'phases': {},
            'innovation_established': False,
            'current_best_method': None,
            'current_best_r2': None,
        }

        def exists(rel_path):
            return os.path.exists(os.path.join(PROJECT_ROOT, rel_path))

        def count_matches(pattern):
            return len(glob.glob(os.path.join(PROJECT_ROOT, pattern), recursive=True))

        def mark_completed(phase_name):
            inferred['phases'][phase_name] = {
                'status': 'completed',
                'completed_at': '',
                'inferred': True,
            }

        # 尝试从 MethodRegistry 获取方法状态
        registry = None
        try:
            from shared.method_registry import MethodRegistry
            registry = MethodRegistry()
        except Exception:
            pass

        if exists('INVENTORY.md'):
            mark_completed('organize')

        if count_matches('PaperDownload/**/*.pdf') > 0:
            mark_completed('download')

        method_docs = count_matches('MethodToSmart/*.md')
        if method_docs >= 3:
            mark_completed('analyze')

        # design: 方案指令文件 + 注册表 designed 状态方法
        design_docs = (
            count_matches('SmartToCode/创新方法指令/*.md') +
            count_matches('SmartToCode/复现方法指令/*.md')
        )
        designed_count = len(registry.get_methods_by_state('designed')) if registry else 0
        if design_docs >= 3 or designed_count > 0:
            mark_completed('design')

        # code: 代码文件 + 注册表 implemented 状态方法
        code_files = (
            count_matches('CodeWorkSpace/新融合方法代码/*.py') +
            count_matches('CodeWorkSpace/复现方法代码/*.py')
        )
        implemented_count = len(registry.get_methods_by_state('implemented')) if registry else 0
        if code_files >= 3 or implemented_count > 0:
            mark_completed('code')

        # verify: 验证结果 + 注册表 verified 状态方法
        verified_pass = len(registry.get_methods_by_state('verified_pass')) if registry else 0
        verified_fail = len(registry.get_methods_by_state('verified_fail')) if registry else 0
        if (
            exists('test_result/基准方法/benchmark_multistage.json') or
            count_matches('Innovation/success/**/*_all_stages.json') > 0 or
            count_matches('test_result/创新方法/*_summary.csv') > 0 or
            verified_pass + verified_fail > 0
        ):
            mark_completed('verify')

        if exists('paper_output/paper.tex') or exists('paper_output/paper.pdf'):
            mark_completed('write')

        best_method, best_r2 = cls._infer_best_method_from_results()

        # v14: 如果 MethodRegistry 有更准确的 best method，优先使用
        if registry and not best_method:
            try:
                verified = registry.get_methods_by_state('verified_pass')
                if verified:
                    # 从 verified_pass 方法中选 R² 最高的
                    best = max(verified, key=lambda m: m.get('metrics', {}).get('R2', -1))
                    best_method = best['name']
                    best_r2 = best.get('metrics', {}).get('R2')
            except Exception:
                pass

        if best_method:
            inferred['innovation_established'] = True
            inferred['current_best_method'] = best_method
            inferred['current_best_r2'] = best_r2

        return inferred

    @classmethod
    def _infer_best_method_from_results(cls):
        """
        从 Innovation/success 中挑选当前最佳方法。
        规则：排除同时出现在 failed/ 的方法，按 stage1~3 的平均 R² 排序。
        """
        failed_methods = set()
        for path in glob.glob(os.path.join(PROJECT_ROOT, 'Innovation', 'failed', '*')):
            if os.path.isdir(path):
                failed_methods.add(os.path.basename(path).replace('-', '_'))

        candidates = []
        pattern = os.path.join(PROJECT_ROOT, 'Innovation', 'success', '**', '*_all_stages.json')
        for json_path in glob.glob(pattern, recursive=True):
            method_name = os.path.basename(os.path.dirname(json_path))
            normalized_name = method_name.replace('-', '_')
            if normalized_name in failed_methods:
                continue

            r2_values = cls._extract_stage_r2_values(json_path)
            if len(r2_values) < 3:
                continue

            avg_r2 = sum(r2_values) / len(r2_values)
            stage1_r2 = r2_values[0]
            candidates.append((avg_r2, stage1_r2, method_name))

        if not candidates:
            return None, None

        candidates.sort(reverse=True)
        best_avg_r2, _stage1_r2, best_method = candidates[0]
        return best_method, best_avg_r2

    @staticmethod
    def _extract_stage_r2_values(json_path):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception:
            return []

        values = []
        for stage_name in ('stage1', 'stage2', 'stage3'):
            stage = data.get(stage_name, {})
            metrics = stage.get('metrics', stage)
            r2 = metrics.get('R2')
            if isinstance(r2, (int, float)):
                values.append(float(r2))
        return values

    @classmethod
    def phase_completed(cls, phase_name):
        state = cls.load(include_inferred=False)
        state['phases'][phase_name] = {
            'status': 'completed',
            'completed_at': datetime.now().isoformat()
        }
        cls.save(state)

    @classmethod
    def reset_phase(cls, phase_name):
        """重置指定 Phase 的完成状态，允许闭环迭代中重新执行。"""
        state = cls.load(include_inferred=False)
        if phase_name in state.get('phases', {}):
            state['phases'][phase_name] = {
                'status': 'pending',
                'completed_at': '',
                'reset_at': datetime.now().isoformat()
            }
            cls.save(state)

    @classmethod
    def is_phase_done(cls, phase_name, include_inferred=True):
        state = cls.load(include_inferred=include_inferred)
        return state['phases'].get(phase_name, {}).get('status') == 'completed'

    @classmethod
    def mark_innovation(cls, method_name, r2):
        state = cls.load(include_inferred=False)
        if state.get('current_best_r2') is None or r2 > state['current_best_r2']:
            state['current_best_method'] = method_name
            state['current_best_r2'] = r2
            state['innovation_established'] = True
        cls.save(state)

    @classmethod
    def print_status(cls):
        state = cls.load()
        resume_context = _get_resume_context()
        pending_design = len(resume_context.get('pending_designed', []))
        pending_verify = len(resume_context.get('pending_implemented', []))
        research_status = resume_context.get('research_status', 'unknown')
        print(f"""
=== 流水线状态 ===
轮次:       {state.get('round', 'N/A')}
最佳方法:   {state.get('current_best_method', 'None')} (R²={state.get('current_best_r2', 'N/A')})
创新成立:   {state.get('innovation_established', False)}
上次运行:   {state.get('last_run', '从未')}
研究状态:   {research_status}
待实现方法: {pending_design}
待验证方法: {pending_verify}

各 Phase 状态:
  Phase 0 (整理):   {cls._phase_status(state, 'organize')}
  Phase 1 (下载):   {cls._phase_status(state, 'download')}
  Phase 2 (分析):   {cls._phase_status(state, 'analyze')}
  Phase 3 (设计):   {cls._phase_status(state, 'design')}
  Phase 4 (编码):   {cls._phase_status(state, 'code')}
  Phase 5 (验证):   {cls._phase_status(state, 'verify')}
  Phase 6 (写作):   {cls._phase_status(state, 'write')}
""")

    @staticmethod
    def _phase_status(state, phase):
        p = state['phases'].get(phase, {})
        status = p.get('status', '未开始')
        if status == 'completed' and p.get('inferred') and not p.get('completed_at'):
            return '已完成(推断)'
        return status


# ============================================================
# Pipeline 配置管理
# ============================================================

class PipelineConfig:
    """Pipeline profile and configuration manager"""

    CONFIG_FILE = os.path.join(PROJECT_ROOT, 'pipeline_config.json')

    @classmethod
    def load(cls):
        if os.path.exists(cls.CONFIG_FILE):
            with open(cls.CONFIG_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        return cls._default_config()

    @classmethod
    def save(cls, config):
        with open(cls.CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

    @classmethod
    def _default_config(cls):
        return {"default_profile": None, "profiles": {}}

    @classmethod
    def get_profile(cls, name):
        config = cls.load()
        return config.get('profiles', {}).get(name)

    @classmethod
    def get_default_phases(cls):
        config = cls.load()
        default_name = config.get('default_profile')
        if default_name:
            profile = config.get('profiles', {}).get(default_name)
            if profile:
                return profile['phases']
        return None

    @classmethod
    def list_profiles(cls):
        config = cls.load()
        profiles = config.get('profiles', {})
        default = config.get('default_profile')
        if not profiles:
            print("没有已保存的 profile。")
            print(f"配置文件: {cls.CONFIG_FILE}")
            return
        print("\n可用 Profiles:")
        print("-" * 50)
        for name, info in profiles.items():
            marker = " (默认)" if name == default else ""
            phases_str = ", ".join(str(p) for p in info['phases'])
            desc = info.get('description', '')
            print(f"  {name}{marker}")
            print(f"    {desc}")
            print(f"    Phases: [{phases_str}]")
        print()

    @classmethod
    def save_profile(cls, name, phases, description=""):
        config = cls.load()
        config.setdefault('profiles', {})[name] = {
            "description": description,
            "phases": sorted(phases)
        }
        cls.save(config)
        print(f"Profile '{name}' 已保存 (phases: {sorted(phases)})")


# ============================================================
# Claude CLI 执行器
# ============================================================

# ============================================================
# Agent 后端（通过 task_dispatcher 支持多后端）
# ============================================================

from agents.task_dispatcher import TaskDispatcher, build_prompt_for_phase
from agents.artifact_detector import ArtifactDetector

# 向后兼容：ClaudeExecutor 已迁移至 task_dispatcher.CLIBackend
# 旧代码如需引用，可通过以下方式获取：
#   from agents.task_dispatcher import get_backend
#   cli_backend = get_backend('cli', project_root, budget=5.0)


# ============================================================
# Phase 选择与解析
# ============================================================

def parse_phase_list(value, flag_name):
    """解析逗号分隔的 phase 编号"""
    phases = []
    for part in value.split(','):
        part = part.strip()
        if not part:
            continue
        if not part.isdigit() or int(part) not in range(7):
            print(f"错误: {flag_name} 中的无效 phase: '{part}' (可选: 0-6)")
            sys.exit(1)
        phases.append(int(part))
    return sorted(set(phases))


def resolve_phases(args):
    """
    确定要执行的 phase 集合。
    优先级: --only > --profile > config default > 全部 0-6
    --from / --to / --skip 在选定集合上过滤。
    """
    if args.only is not None:
        return parse_phase_list(args.only, '--only')

    phases = None

    if args.profile is not None:
        profile = PipelineConfig.get_profile(args.profile)
        if not profile:
            print(f"错误: 未找到 profile '{args.profile}'")
            PipelineConfig.list_profiles()
            sys.exit(1)
        phases = list(profile['phases'])

    if phases is None:
        phases = PipelineConfig.get_default_phases()

    if phases is None:
        phases = list(range(7))

    if args.from_phase is not None:
        phases = [p for p in phases if p >= args.from_phase]
    if args.to_phase is not None:
        phases = [p for p in phases if p <= args.to_phase]

    if args.skip is not None:
        skip_set = set(parse_phase_list(args.skip, '--skip'))
        phases = [p for p in phases if p not in skip_set]

    if not phases:
        print("错误: 没有 phase 需要执行（过滤后为空）")
        sys.exit(1)

    return sorted(phases)


def print_plan(phases):
    """执行前展示计划"""
    state = PipelineState.load()
    print("\n执行计划:")
    print("-" * 45)
    for p in range(7):
        name, desc = PHASE_INFO[p]
        if p in phases:
            phase_info = state['phases'].get(name, {})
            if phase_info.get('status') == 'completed':
                if phase_info.get('inferred') and not phase_info.get('completed_at'):
                    status = " [已完成(推断),仍可执行]"
                else:
                    status = " [已完成,将跳过]"
            else:
                status = " [待执行]"
            print(f"  Phase {p} ({desc}):  >>>{status}")
        else:
            print(f"  Phase {p} ({desc}):  跳过")
    print("-" * 45)


# ============================================================
# Phase 执行器
# ============================================================

class PhaseRunner:
    """各 Phase 的 prompt 生成 + 指引打印双模式"""

    # Phase → role 映射
    ROLE_MAP = {
        0: 'organizer',
        1: 'literature_downloader',
        2: 'literature_analyzer',
        3: 'method_designer',
        4: 'code_engineer',
        5: 'test_verifier',
        6: 'technical_writer',
    }

    @classmethod
    def get_prompt(cls, phase_num):
        """获取指定 phase 的 agent prompt（用于 ClaudeExecutor 执行）"""
        role = cls.ROLE_MAP.get(phase_num)
        if not role:
            return None
        from agents.role_templates import get_spawn_prompt
        base_prompt = get_spawn_prompt(role, PROJECT_ROOT)
        if phase_num in (2, 3, 4, 6):
            return f"{base_prompt}\n\n{_build_resume_prompt_block()}\n"
        return base_prompt

    @staticmethod
    def print_guidance(phase_num):
        """打印手动执行指引（原有行为）"""
        header = f"\n{'=' * 60}\nPhase {phase_num}: {PHASE_INFO[phase_num][1]}\n{'=' * 60}"

        if phase_num == 0:
            print(header)
            print("""
此阶段需要 Claude Code Agent 执行以下任务：

1. 扫描 PaperDownload/, MethodToSmart/, SmartToCode/, CodeWorkSpace/,
   test_result/, Innovation/ 等目录
2. 更新 INVENTORY.md，反映当前文件状态
3. 清理临时文件和重复文件

请在 Claude Code 中执行：
  "整理 Data_Fusion_AutoResearch 项目，更新 INVENTORY.md"
""")

        elif phase_num == 1:
            print(header)
            paper_dir = data_path('PaperDownload')
            existing = len([f for f in os.listdir(paper_dir) if f.endswith('.pdf')]) if os.path.exists(paper_dir) else 0
            print(f"  当前已有论文: {existing} 篇")
            print(f"""
如需搜索新论文，请在 Claude Code 中执行：
  "搜索 PM2.5 CMAQ 数据融合相关的最新论文，下载到 PaperDownload/"

如果论文已足够，使用 --skip 1 跳过此阶段。
""")

        elif phase_num == 2:
            print(header)
            method_dir = data_path('MethodToSmart')
            existing_methods = len(os.listdir(method_dir)) if os.path.exists(method_dir) else 0
            print(f"  已有方法文档: {existing_methods} 个")
            print(f"""
请在 Claude Code 中执行：
  "读取 PaperDownload/ 中的论文，提炼融合方法，
   生成结构化方法文档到 MethodToSmart/。
   使用【可执行方法规范】模板。"
""")

        elif phase_num == 3:
            print(header)
            print(f"""
请在 Claude Code 中执行：
  "基于 MethodToSmart/ 中的方法文档和 Innovation/success/ 中的已有创新，
   设计下一个改进方案，输出到 SmartToCode/。
   避免 Stacking/Ensemble 类方法（无可解释性）。"
""")

        elif phase_num == 4:
            print(header)
            print(f"""
请在 Claude Code 中执行：
  "根据 SmartToCode/ 中最新的方案指令，
   在 CodeWorkSpace/新融合方法代码/ 中实现 Python 代码。
   参考已有的 PolyRK.py 和 AdvancedRK.py 的十折验证模式。"
""")

        elif phase_num == 5:
            print(header)
            innovation_dir = data_path('test_result/创新方法')
            scripts = []
            if os.path.exists(innovation_dir):
                for f in os.listdir(innovation_dir):
                    if f.endswith('_十折标准模式.py') or f.endswith('_十折验证.py'):
                        scripts.append(f)
            if scripts:
                print(f"  找到 {len(scripts)} 个验证脚本:")
                for i, s in enumerate(scripts):
                    print(f"    [{i + 1}] {s}")
            print(f"""
运行验证：
  python run_pipeline.py --auto --only 5     # 直接运行验证脚本
  或手动执行:
  python test_result/基准方法/validate_baseline_multistage.py
  python test_result/创新方法/<方法名>_十折标准模式.py
""")

        elif phase_num == 6:
            print(header)
            print(f"""
请在 Claude Code 中执行：
  "基于 Innovation/success/ 中验证通过的方法，
   生成学术论文到 paper_output/，包含：
   - 摘要、引言、方法、实验、结论
   - LaTeX 格式 (paper.tex)
   - 参考文献 (references.bib)"
""")


# ============================================================
# 主流程
# ============================================================

def _generate_validation_scripts(method_names: list, env: dict):
    """使用模板生成器为新方法生成验证脚本。"""
    if not method_names:
        return

    print(f"  [生成] 为 {len(method_names)} 个新方法生成验证脚本...")
    try:
        from shared.generate_validation_scripts import generate_script
        generated = 0
        for method_name in method_names:
            try:
                generate_script(method_name, PROJECT_ROOT, dry_run=False)
                generated += 1
            except Exception as e:
                print(f"    [错误] {method_name}: {e}")
        print(f"  [完成] 成功生成 {generated}/{len(method_names)} 个验证脚本")
    except Exception as e:
        print(f"  [错误] 验证脚本生成失败: {e}")


def _get_pending_verification_targets():
    """从方法注册表中获取真正待验证的方法。"""
    registry = _get_method_registry(sync=True)
    if registry is None:
        return []
    try:
        return registry.get_pending_verification()
    except Exception:
        return []


def _build_validation_script_map(innovation_dir):
    """扫描现有验证脚本，建立 方法名 -> 脚本路径 的映射。"""
    script_map = {}
    if not os.path.exists(innovation_dir):
        return script_map

    for f in os.listdir(innovation_dir):
        if not (f.endswith('_十折标准模式.py') or f.endswith('_十折验证.py')):
            continue
        method_name = f.split('_十折')[0] if '_十折' in f else f.rsplit('_', 1)[0]
        script_map[_normalize_method_name(method_name)] = os.path.join(innovation_dir, f)
    return script_map


def _pick_best_verified_method():
    """
    从方法注册表中选择当前最佳已验证方法，供多次启动时延续优化方向。
    规则：仅在 verified_pass 中选择，按 R2 从高到低排序。
    """
    registry = _get_method_registry(sync=True)
    if registry is None:
        return None, None

    try:
        candidates = registry.get_methods_by_state('verified_pass')
    except Exception:
        return None, None

    best_name = None
    best_r2 = None
    for entry in candidates:
        metrics = entry.get('metrics', {})
        r2 = metrics.get('R2')
        if isinstance(r2, (int, float)) and (best_r2 is None or r2 > best_r2):
            best_name = entry.get('name')
            best_r2 = float(r2)
    return best_name, best_r2


def _refresh_pipeline_best_method():
    """将注册表中的最佳已验证方法同步回 PipelineState。"""
    best_name, best_r2 = _pick_best_verified_method()
    if best_name and isinstance(best_r2, (int, float)):
        PipelineState.mark_innovation(best_name, best_r2)


def run_verify_phase():
    """Phase 5 特殊处理：直接运行 Python 验证脚本，不经过 Claude agent"""
    print(f"\n{'=' * 60}")
    print("Phase 5: 测试验证（直接执行）")
    print("=" * 60)

    _sync_method_registry(silent=True)

    # 确保子进程能找到 shared、Code/Downscaler 等模块
    extra_paths = [
        PROJECT_ROOT,
        os.path.join(PROJECT_ROOT, 'Code', 'Downscaler'),
        os.path.join(PROJECT_ROOT, 'Code'),
    ]
    env = os.environ.copy()
    env['PYTHONPATH'] = os.pathsep.join(extra_paths) + os.pathsep + env.get('PYTHONPATH', '')

    # 1. 先跑基准验证
    baseline_script = data_path('test_result/基准方法/validate_baseline_multistage.py')
    baseline_results = data_path('test_result/基准方法/benchmark_multistage.json')
    if os.path.exists(baseline_results):
        print(f"\n[1/2] 基准验证已有结果，跳过: {os.path.basename(baseline_results)}")
    elif os.path.exists(baseline_script):
        print(f"\n[1/2] 运行基准验证: {os.path.basename(baseline_script)}")
        result = subprocess.run(
            [sys.executable, baseline_script],
            cwd=PROJECT_ROOT, env=env,
            text=True, encoding='utf-8', errors='replace'
        )
        if result.returncode != 0:
            print(f"  [失败] 基准验证脚本退出码: {result.returncode}")
            return False
        print("  [完成] 基准验证")
    else:
        print(f"  [跳过] 未找到基准验证脚本: {baseline_script}")

    # 2. 运行创新方法验证
    innovation_dir = data_path('test_result/创新方法')
    script_map = _build_validation_script_map(innovation_dir)
    pending_methods = _get_pending_verification_targets()

    # 2.0 检查是否有新方法需要生成验证脚本
    code_dir = data_path('CodeWorkSpace/新融合方法代码')
    if os.path.exists(code_dir):
        new_methods = []
        existing_script_names = set(script_map.keys())
        for method_name in pending_methods:
            if _normalize_method_name(method_name) not in existing_script_names:
                new_methods.append(method_name)

        if new_methods:
            print(f"\n  发现 {len(new_methods)} 个新方法需要生成验证脚本: {', '.join(new_methods[:5])}{'...' if len(new_methods) > 5 else ''}")
            # 使用模板生成器生成验证脚本
            _generate_validation_scripts(new_methods, env)
            script_map = _build_validation_script_map(innovation_dir)

    if not pending_methods:
        print("\n  当前没有待验证方法。")
        print("  本次启动将基于已有结果继续，不会重复重跑历史验证。")
        _refresh_pipeline_best_method()
        PipelineState.phase_completed('verify')
        return True

    scripts_to_run = []
    missing_scripts = []
    for method_name in pending_methods:
        script_path = script_map.get(_normalize_method_name(method_name))
        if script_path:
            scripts_to_run.append(script_path)
        else:
            missing_scripts.append(method_name)

    if missing_scripts:
        print(f"\n  以下待验证方法尚无验证脚本: {', '.join(missing_scripts[:10])}")
        print("  请先完成 Phase 4 或检查脚本生成器。")
        return False

    if not scripts_to_run:
        print("\n  没有可执行的待验证脚本")
        return False

    print(f"\n[信息] 本次仅验证待处理方法: {', '.join(pending_methods[:12])}{'...' if len(pending_methods) > 12 else ''}")

    # === 阶段 2a: 预验证 (pre_exp) ===
    print(f"\n[2a/3] 预验证 (pre_exp) — {len(scripts_to_run)} 个方法:")
    pre_passed = []
    pre_failed = []
    for i, script in enumerate(scripts_to_run):
        basename = os.path.basename(script)
        method_name = basename.split('_十折')[0] if '_十折' in basename else basename.rsplit('_', 1)[0]
        print(f"  [{i + 1}/{len(scripts_to_run)}] {method_name}", end=" ")
        result = subprocess.run(
            [sys.executable, script, '--pre-only'],
            cwd=PROJECT_ROOT, env=env,
            text=True, encoding='utf-8', errors='replace'
        )
        # 检查预验证结果
        pre_json = data_path(f'test_result/创新方法/{method_name}_pre_exp.json')
        if os.path.exists(pre_json):
            with open(pre_json, 'r', encoding='utf-8') as f:
                pre_data = json.load(f)
            if pre_data.get('passed'):
                pre_passed.append((method_name, script))
                print("→ 通过")
            else:
                pre_failed.append(method_name)
                print("→ 未通过")
        else:
            pre_failed.append(method_name)
            print("→ 无结果")

    print(f"\n  预验证结果: {len(pre_passed)} 通过, {len(pre_failed)} 未通过")
    if pre_failed:
        print(f"  未通过方法: {', '.join(pre_failed[:10])}{'...' if len(pre_failed) > 10 else ''}")

    if not pre_passed:
        print("\n  所有方法预验证均未通过，无需正式验证")
        PipelineState.phase_completed('verify')
        return True

    # === 阶段 2b: 正式验证 (stage1/stage2/stage3) ===
    print(f"\n[2b/3] 正式验证 — {len(pre_passed)} 个方法通过预验证:")
    for i, (method_name, script) in enumerate(pre_passed):
        print(f"  [{i + 1}/{len(pre_passed)}] {method_name}")
        result = subprocess.run(
            [sys.executable, script],
            cwd=PROJECT_ROOT, env=env,
            text=True, encoding='utf-8', errors='replace'
        )
        if result.returncode != 0:
            print(f"  [失败] {method_name} 退出码: {result.returncode}")
            # 不返回 False，继续验证其他方法
            continue
        print(f"  [完成] {method_name}")

        # 验证后更新注册表
        try:
            _update_registry_after_verify(method_name)
        except Exception as e:
            print(f"  [警告] 注册表更新失败: {e}")

    _sync_method_registry(force=True, silent=True)
    _refresh_pipeline_best_method()

    # === 闭环关键：将本轮验证结果写入 StateTracker，实现经验沉淀 ===
    _record_iteration_to_state_tracker(pre_passed, scripts_to_run)

    PipelineState.phase_completed('verify')
    print(f"\n[完成] Phase 5 (verify) — 预验证 {len(scripts_to_run)} 个, 正式验证 {len(pre_passed)} 个")
    return True


def _record_iteration_to_state_tracker(pre_passed, scripts_to_run):
    """
    将本轮验证结果写入 StateTracker，实现经验沉淀。

    - 通过的方法 → accept_mutation()：更新最佳方法
    - 失败的方法 → reject_mutation()：记录失败原因
    - 检查收敛/耗尽条件 → 判断是否需要继续迭代
    """
    try:
        from agents.research_state_tracker import StateTracker, ResearchStatus
        state_dir = os.path.join(PROJECT_ROOT, 'test_result', '.state')
        tracker = StateTracker(state_dir)

        # 如果状态从未初始化（iteration=0），用基准指标初始化
        if tracker.state.iteration == 0:
            baseline_r2 = _get_baseline_r2()
            if baseline_r2 is not None:
                tracker.initialize(
                    baseline_metrics={'R2': baseline_r2},
                    baseline_method='VNA_baseline'
                )
                print(f"  [StateTracker] 初始化基准: R²={baseline_r2:.4f}")

        # 收集本轮所有已验证方法的结果
        registry = _get_method_registry(sync=True)
        if registry is None:
            return

        # 正式验证通过的方法 → accept
        for method_name, script in pre_passed:
            try:
                method_info = registry.get_method(method_name)
                if method_info is None:
                    continue
                metrics = method_info.get('metrics', {})
                r2 = metrics.get('R2')
                if r2 is None:
                    continue

                # 判断是否优于当前最佳
                current_best_r2 = tracker.state.current_best_metrics.get('R2', 0)
                if r2 > current_best_r2:
                    tracker.start_iteration(method_name)
                    tracker.update_metrics(metrics, method_name)
                    tracker.accept_mutation(
                        method_name=method_name,
                        metrics=metrics,
                        description=method_info.get('description', ''),
                        code_diff=''
                    )
                    print(f"  [StateTracker] 接受变异: {method_name} R²={r2:.4f} (↑ from {current_best_r2:.4f})")
                else:
                    # 通过但未超越最佳 → 记录但标记为非最佳
                    print(f"  [StateTracker] {method_name} R²={r2:.4f} 未超越最佳 {current_best_r2:.4f}")
            except Exception as e:
                print(f"  [StateTracker] 记录 {method_name} 时出错: {e}")

        # 预验证未通过的方法 → reject
        all_tested = set(_normalize_method_name(m) for m, _ in pre_passed)
        for method_name in scripts_to_run:
            try:
                basename = os.path.basename(method_name)
                mn = basename.split('_十折')[0] if '_十折' in basename else basename.rsplit('_', 1)[0]
                if _normalize_method_name(mn) in all_tested:
                    continue
                # 未通过 pre_exp 的方法
                pre_json = data_path(f'test_result/创新方法/{mn}_pre_exp.json')
                fail_reason = "预验证未通过"
                fail_metrics = {}
                if os.path.exists(pre_json):
                    with open(pre_json, 'r', encoding='utf-8') as f:
                        pre_data = json.load(f)
                    fail_metrics = pre_data.get('metrics', {})
                    fail_reason = pre_data.get('fail_reason', fail_reason)

                tracker.start_iteration(mn)
                tracker.reject_mutation(
                    method_name=mn,
                    metrics=fail_metrics or {'R2': 0},
                    reason=fail_reason,
                    description=''
                )
                print(f"  [StateTracker] 拒绝变异: {mn} ({fail_reason})")
            except Exception as e:
                pass  # 静默跳过

        # 打印迭代摘要
        summary = tracker.get_current_state()
        direction = tracker.get_next_optimization_direction()
        print(f"\n  [迭代摘要] 第 {summary['iteration']} 轮 | 状态: {summary['status']}")
        print(f"  最佳方法: {summary['current_best_method']} R²={summary['current_best_r2']:.4f}")
        print(f"  累计接受: {summary['accepted_count']} | 拒绝: {summary['rejected_count']}")
        print(f"  连续无改进: {summary['consecutive_no_improvement']}/{tracker.state.max_consecutive_no_improvement}")
        if tracker.should_continue():
            print(f"  下一步方向: {direction}")
        else:
            print(f"  [收敛] 研究状态: {summary['status']}，停止迭代")

    except ImportError:
        print(f"  [StateTracker] 模块未找到，跳过经验沉淀")
    except Exception as e:
        print(f"  [StateTracker] 经验沉淀失败: {e}")


def _get_baseline_r2():
    """获取基准方法的 R² 作为迭代基线。"""
    try:
        benchmark_json = data_path('test_result/基准方法/benchmark_multistage.json')
        if os.path.exists(benchmark_json):
            with open(benchmark_json, 'r', encoding='utf-8') as f:
                data = json.load(f)
            # 取 VNA 的 pre_exp R² 作为基线
            vna = data.get('VNA', {})
            pre_exp = vna.get('pre_exp', {})
            r2 = pre_exp.get('R2')
            if r2 is not None:
                return float(r2)
            # 或取所有方法的最佳 pre_exp R²
            best_r2 = 0
            for method_name, method_data in data.items():
                if isinstance(method_data, dict):
                    r2_val = method_data.get('pre_exp', {}).get('R2', 0)
                    if isinstance(r2_val, (int, float)) and r2_val > best_r2:
                        best_r2 = r2_val
            return best_r2 if best_r2 > 0 else None
    except Exception:
        return None
    return None


def _update_registry_after_verify(method_name: str, registry=None):
    """验证完成后更新注册表：从结果 JSON 中解析指标并写入。"""
    if registry is None:
        try:
            from shared.method_registry import MethodRegistry
            registry = MethodRegistry()
        except Exception:
            return

    # 查找结果 JSON（优先 test_result/创新方法/，其次 Innovation/）
    import glob as _glob
    test_result_json = os.path.join(PROJECT_ROOT, 'test_result', '创新方法', f'{method_name}_all_stages.json')
    patterns = [
        test_result_json,
        os.path.join(PROJECT_ROOT, 'Innovation', 'success', method_name, f'{method_name}_all_stages.json'),
        os.path.join(PROJECT_ROOT, 'Innovation', 'failed', method_name, f'{method_name}_all_stages.json'),
        os.path.join(PROJECT_ROOT, 'Innovation', '*', method_name, '*_all_stages.json'),
    ]
    for pattern in patterns:
        matches = _glob.glob(pattern)
        if matches:
            json_path = matches[0]
            registry.update_from_all_stages_json(method_name, json_path)
            registry.save()
            print(f"  [注册表] 已更新 {method_name}")
            return

    # 没有 all_stages JSON，跳过（模板始终生成 JSON，CSV fallback 已废弃）


def run_phase(phase_num, dispatcher=None):
    """
    运行指定 Phase。

    - dispatcher=None: 等价于 manual 后端（打印指引）
    - dispatcher=TaskDispatcher(backend='trae'): 生成任务描述，由外部 Agent 执行
    - dispatcher=TaskDispatcher(backend='cli'): 通过 Claude CLI 执行（同步）
    - Phase 5 始终直接运行 Python 脚本
    """
    if phase_num not in range(7):
        print(f"无效 Phase: {phase_num} (可选: 0-6)")
        return False

    if phase_num in (2, 3, 4, 5, 6):
        _sync_method_registry(silent=True)

    name, desc = PHASE_INFO[phase_num]
    if PipelineState.is_phase_done(name, include_inferred=False):
        print(f"Phase {phase_num} ({name}) 已完成，跳过。")
        return True

    # Phase 5: 直接运行验证脚本（不受后端影响）
    if phase_num == 5:
        return run_verify_phase()

    # 先检测产物是否已存在（闭环关键：可能有外部 Agent 已完成）
    detector = ArtifactDetector(PROJECT_ROOT)
    artifact = detector.check(phase_num)
    if artifact['done']:
        print(f"Phase {phase_num} ({name}) 产物已检测到，标记完成。")
        print(f"  详情: {artifact['details']}")
        PipelineState.phase_completed(name)
        return True

    # 构建 prompt
    from agents.task_dispatcher import build_prompt_for_phase
    resume_context = _get_resume_context()
    prompt = build_prompt_for_phase(phase_num, PROJECT_ROOT, resume_context)
    if not prompt:
        print(f"  Phase {phase_num} 无 prompt 模板，跳过")
        return True

    # 分发任务
    if dispatcher is not None:
        success, output, task_desc = dispatcher.dispatch(phase_num, prompt, resume_context)

        # 同步后端（如 CLI）：执行完立即检测产物
        if success and dispatcher.backend.is_sync():
            artifact = detector.check(phase_num)
            if artifact['done']:
                print(f"\n[产物检测] Phase {phase_num} 产物已生成: {artifact['details']}")
                PipelineState.phase_completed(name)
            else:
                print(f"\n[警告] Phase {phase_num} 后端执行完成，但未检测到预期产物")
                print(f"  详情: {artifact['details']}")
            return success
        # 异步后端（如 TRAE/Manual）：返回等待外部执行
        return success
    else:
        # 无 dispatcher：使用手动指引模式
        PhaseRunner.print_guidance(phase_num)
        print(f"\n[提示] Phase {phase_num} ({name}) 仅输出指引，未自动标记为完成。")
        print("      实际产物会在 `--status` 中自动识别。")
        return True


def run_auto(phases=None, dispatcher=None, max_iterations=20):
    """
    自动运行指定或全部 Phase，支持闭环迭代。

    v14 闭环模式：
    - 遇到 LLM Phase → 分发任务 → 等待外部执行 → 重新运行检测产物
    - Phase 5 验证后 → 写入 StateTracker → 检查是否需继续迭代
    - 如果未收敛（should_continue()=True）→ 自动推进下一轮设计+编码+验证
    - 如果收敛或耗尽 → 结束

    - dispatcher=None: 手动指引模式
    - dispatcher=TaskDispatcher(backend='trae'): 闭环模式（默认推荐）
    - dispatcher=TaskDispatcher(backend='cli'): CLI 模式（同步阻塞）
    - max_iterations: 最大迭代轮次（防止无限循环）
    """
    if phases is None:
        phases = _recommended_phases_from_history() or list(range(7))

    backend_name = dispatcher.backend_name if dispatcher else "manual"
    print("=" * 60)
    print(f"PM2.5 CMAQ 融合方法自动研究流程 v14.0 [{backend_name} 后端]")
    print("=" * 60)

    PipelineState.print_status()
    print_plan(phases)

    # === 闭环迭代循环 ===
    iteration = 0
    while iteration < max_iterations:
        iteration += 1
        print(f"\n{'─' * 60}")
        print(f"迭代轮次 {iteration}/{max_iterations}")
        print(f"{'─' * 60}")

        # 执行本轮各 Phase
        phase_executed = False
        for phase_num in phases:
            success = run_phase(phase_num, dispatcher=dispatcher)
            if not success and dispatcher is not None:
                print(f"\n[中断] Phase {phase_num} 失败，停止后续 phase。")
                break

            # 异步后端：任务已分发，等待外部执行后重新运行
            if dispatcher is not None and not dispatcher.backend.is_sync():
                if not PipelineState.is_phase_done(PHASE_INFO[phase_num][0]):
                    print(f"\n[等待] Phase {phase_num} 任务已分发到 {backend_name} 后端。")
                    print(f"  请执行任务后重新运行 run_pipeline.py 自动推进。")
                    _print_loop_summary(phases, dispatcher, iteration, max_iterations)
                    return

            phase_executed = True

        # === 迭代判定 ===
        # Phase 5 验证完成后，检查是否需要继续迭代
        should_continue, reason = _check_iteration_status()
        if not should_continue:
            print(f"\n[迭代终止] {reason}")
            break

        # 如果当前 Phase 列表不含验证（Phase 5），不需要继续迭代
        if 5 not in phases:
            break

        # 检查是否还有待验证方法（有则继续，无则可能需要设计新方法）
        pending_implemented = _get_registry_state_names('implemented')
        pending_designed = _get_registry_state_names('designed')

        if pending_implemented:
            # 有待验证方法 → 下一轮只跑验证
            next_phases = [5]
            print(f"\n[继续迭代] 还有 {len(pending_implemented)} 个待验证方法，自动进入下一轮验证")
        elif pending_designed:
            # 有待实现方法 → 下一轮跑编码+验证
            next_phases = [4, 5]
            print(f"\n[继续迭代] 还有 {len(pending_designed)} 个待实现方法，自动进入下一轮编码+验证")
        else:
            # 队列空了，需要设计新方法 → 下一轮跑设计+编码+验证
            next_phases = [3, 4, 5]
            print(f"\n[继续迭代] 方法队列已空，自动进入下一轮设计+编码+验证")

        # 重置 Phase 完成状态以允许重新执行
        for p in next_phases:
            PipelineState.reset_phase(PHASE_INFO[p][0])

        phases = next_phases
        print(f"  下一轮 Phase: {', '.join(PHASE_INFO[p][0] for p in phases)}")

    _print_loop_summary(phases, dispatcher, iteration, max_iterations)


def _check_iteration_status():
    """
    检查是否应该继续迭代。

    Returns:
        tuple: (should_continue: bool, reason: str)
    """
    try:
        from agents.research_state_tracker import StateTracker
        state_dir = os.path.join(PROJECT_ROOT, 'test_result', '.state')
        tracker = StateTracker(state_dir)

        if not tracker.should_continue():
            status = tracker.state.status
            if status == 'converged':
                return False, f"创新已收敛 (R²={tracker.state.current_best_metrics.get('R2', 0):.4f} ≥ 阈值)"
            elif status == 'exhausted':
                return False, f"创新力耗尽 (连续 {tracker.state.consecutive_no_improvement} 次无改进)"
            elif status == 'max_iterations':
                return False, f"已达迭代上限 ({tracker.state.iteration} 轮)"
            else:
                return False, f"状态: {status}"

        return True, "继续迭代"
    except Exception as e:
        # StateTracker 不可用时，默认不循环（安全降级）
        return False, f"StateTracker 不可用 ({e})"


def _print_loop_summary(phases, dispatcher, iteration, max_iterations):
    """打印闭环迭代摘要"""
    print("\n" + "=" * 60)
    executed = [p for p in phases if not PipelineState.is_phase_done(PHASE_INFO[p][0])]
    completed = [p for p in phases if PipelineState.is_phase_done(PHASE_INFO[p][0])]
    skipped = [p for p in range(7) if p not in phases]

    print(f"已完成: {len(completed)} | 待执行: {len(executed)} | 跳过: {len(skipped)}")
    print(f"迭代轮次: {iteration}/{max_iterations}")
    if skipped:
        skipped_names = [PHASE_INFO[p][1] for p in skipped]
        print(f"跳过的 Phase: {', '.join(skipped_names)}")

    # 打印研究状态摘要
    try:
        from agents.research_state_tracker import StateTracker
        state_dir = os.path.join(PROJECT_ROOT, 'test_result', '.state')
        tracker = StateTracker(state_dir)
        summary = tracker.get_current_state()
        print(f"\n研究状态: {summary['status']}")
        print(f"最佳方法: {summary['current_best_method']} (R²={summary['current_best_r2']:.4f})")
        print(f"累计接受: {summary['accepted_count']} | 拒绝: {summary['rejected_count']}")
        print(f"连续无改进: {summary['consecutive_no_improvement']}")
        if tracker.should_continue():
            direction = tracker.get_next_optimization_direction()
            print(f"下一步方向: {direction}")
    except Exception:
        pass

    if dispatcher:
        if dispatcher.backend.is_sync():
            print("\nAgent 执行完毕。")
        else:
            remaining = [PHASE_INFO[p][0] for p in executed]
            if remaining:
                print(f"\n待执行 Phase: {', '.join(remaining)}")
                print("请在外部 Agent 完成任务后重新运行。")
    else:
        print("\n请按指引在 IDE 中执行各 LLM 依赖的 Phase。")
    print("=" * 60)


def run_interactive():
    """交互式运行"""
    print("=" * 60)
    print("PM2.5 CMAQ 融合方法自动研究流程 v14.0")
    print("=" * 60)

    PipelineState.print_status()

    while True:
        print("""
可选操作:
  [0] Phase 0: 项目整理
  [1] Phase 1: 文献下载
  [2] Phase 2: 文献分析
  [3] Phase 3: 方案设计
  [4] Phase 4: 代码实现
  [5] Phase 5: 测试验证
  [6] Phase 6: 论文写作
  [s] 查看状态
  [q] 退出
""")
        choice = input("请选择 (0-6/s/q): ").strip().lower()

        if choice == 'q':
            print("退出。状态已保存。")
            break
        elif choice == 's':
            PipelineState.print_status()
        elif choice.isdigit() and 0 <= int(choice) <= 6:
            run_phase(int(choice))
        else:
            print("无效选择")


def _create_dispatcher(args):
    """
    根据命令行参数创建 TaskDispatcher。

    优先级: --backend > --agent(向后兼容) > None(手动模式)

    --backend trae:   闭环模式，生成任务描述由外部 Agent 执行
    --backend cli:    Claude CLI 同步执行
    --backend manual: 手动指引
    --agent:          等价于 --backend cli（向后兼容）
    无参数:           等价于 --backend manual
    """
    # 确定后端名称
    if args.backend:
        backend_name = args.backend
    elif args.agent:
        backend_name = 'cli'  # 向后兼容
    else:
        return None  # 手动指引模式

    # 构建 backend kwargs
    backend_kwargs = {}
    if backend_name == 'cli':
        if args.budget:
            backend_kwargs['budget'] = args.budget
        if args.model:
            backend_kwargs['model'] = args.model
        if args.timeout:
            backend_kwargs['timeout'] = args.timeout

    try:
        from agents.task_dispatcher import TaskDispatcher
        return TaskDispatcher(PROJECT_ROOT, backend=backend_name, **backend_kwargs)
    except Exception as e:
        print(f"[警告] 创建 {backend_name} 后端失败: {e}")
        print(f"  回退到手动指引模式")
        return None


def main():
    parser = argparse.ArgumentParser(
        description='PM2.5 CMAQ融合方法自动研究流程',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 闭环模式（默认推荐）：生成任务描述，由 TRAE Agent 执行
  python run_pipeline.py --auto                              # 运行全部
  python run_pipeline.py --auto --skip 1                     # 跳过下载
  python run_pipeline.py --auto --only 3,4,5                 # 只跑设计+编码+验证
  python run_pipeline.py --auto --backend trae               # 显式指定 TRAE 后端

  # CLI 模式（向后兼容，调用 Claude CLI）
  python run_pipeline.py --auto --backend cli                # 使用 CLI 后端
  python run_pipeline.py --auto --agent                      # 等价于 --backend cli
  python run_pipeline.py --auto --agent --budget 5.0         # 限制每个 phase 花费

  # 手动指引模式
  python run_pipeline.py --auto --backend manual             # 只打印指引

  # 直接跑验证（不需要 agent）
  python run_pipeline.py --auto --only 5

  # Profile 管理
  python run_pipeline.py --list-profiles
  python run_pipeline.py --save-profile my-flow --skip 1
        """)

    # 原有参数
    parser.add_argument('--status', action='store_true', help='查看当前状态')
    parser.add_argument('--phase', type=int, choices=range(7), help='执行指定 Phase (0-6)')
    parser.add_argument('--auto', action='store_true', help='自动运行 Phase')
    parser.add_argument('--reset', action='store_true', help='重置流水线状态')

    # Agent 模式（向后兼容）
    parser.add_argument('--agent', action='store_true',
                        help='使用 Claude CLI 自动执行（等价于 --backend cli）')
    parser.add_argument('--backend', type=str, default=None, metavar='NAME',
                        choices=['trae', 'cli', 'manual'],
                        help='指定 Agent 后端 (trae: 闭环模式[默认], cli: Claude CLI, manual: 手动指引)')
    parser.add_argument('--budget', type=float, metavar='USD',
                        help='每个 Phase 的最大花费（美元，仅 cli 后端）')
    parser.add_argument('--model', type=str, metavar='MODEL',
                        help='指定 Claude 模型（仅 cli 后端）')
    parser.add_argument('--timeout', type=int, default=None, metavar='SEC',
                        help='每个 Phase 的超时秒数（默认无限制，仅 cli 后端）')
    parser.add_argument('--max-iterations', type=int, default=20, metavar='N',
                        help='闭环迭代最大轮次（默认 20，防止无限循环）')

    # Phase 选择
    parser.add_argument('--skip', type=str, metavar='N,N,...',
                        help='跳过指定 phase（逗号分隔）')
    parser.add_argument('--from', dest='from_phase', type=int, metavar='N',
                        help='从指定 phase 开始（含）')
    parser.add_argument('--to', dest='to_phase', type=int, metavar='N',
                        help='到指定 phase 结束（含）')
    parser.add_argument('--only', type=str, metavar='N,N,...',
                        help='只运行指定 phase（逗号分隔，覆盖其他选项）')

    # Profile 管理
    parser.add_argument('--profile', type=str, metavar='NAME',
                        help='使用指定 profile')
    parser.add_argument('--list-profiles', action='store_true',
                        help='列出所有可用 profile')
    parser.add_argument('--save-profile', type=str, metavar='NAME',
                        help='将当前选择保存为 profile')
    parser.add_argument('--profile-desc', type=str, default='',
                        help='配合 --save-profile 使用，设置描述')

    # 注册表管理
    parser.add_argument('--registry-status', action='store_true',
                        help='查看方法注册表摘要')
    parser.add_argument('--registry-sync', action='store_true',
                        help='重建方法注册表（扫描 Innovation/test_result/CodeWorkSpace/SmartToCode）')

    args = parser.parse_args()

    # --- 特殊命令（不执行 phase）---

    if args.reset:
        state = PipelineState._default_state()
        PipelineState.save(state)
        print("状态已重置。")
        return

    if args.status:
        PipelineState.print_status()
        # 同时显示产物检测结果
        from agents.artifact_detector import ArtifactDetector, format_detection_report
        detector = ArtifactDetector(PROJECT_ROOT)
        detections = detector.check_all()
        print()
        print(format_detection_report(detections))
        pending = detector.get_pending_phases()
        if pending:
            pending_names = [PHASE_INFO[p][1] for p in pending]
            print(f"产物待完成 Phase: {', '.join(pending_names)}")
        # 显示待处理任务
        try:
            from agents.task_dispatcher import TaskDispatcher
            dispatcher = TaskDispatcher(PROJECT_ROOT, backend='manual')
            task = dispatcher.get_pending_task()
            if task:
                print(f"\n待处理任务: Phase {task.get('phase')} ({task.get('title')})")
                print(f"  分发时间: {task.get('dispatched_at')}")
                print(f"  后端: {task.get('backend')}")
        except Exception:
            pass
        return

    if args.list_profiles:
        PipelineConfig.list_profiles()
        return

    # --- 注册表管理 ---
    if args.registry_status:
        try:
            from shared.method_registry import MethodRegistry
            registry = MethodRegistry()
            registry.print_summary()
        except FileNotFoundError:
            print("注册表不存在。运行 --registry-sync 构建。")
        except Exception as e:
            print(f"读取注册表失败: {e}")
        return

    if args.registry_sync:
        try:
            from shared.build_registry import build_registry
            build_registry(merge=False, dry_run=False)
        except Exception as e:
            print(f"构建注册表失败: {e}")
            sys.exit(1)
        return

    # --- --save-profile ---
    if args.save_profile:
        has_selection = any([args.only, args.skip, args.from_phase is not None,
                            args.to_phase is not None, args.profile])
        if not has_selection:
            print("错误: --save-profile 需要配合 --only/--skip/--from/--to/--profile 使用")
            sys.exit(1)
        phases = resolve_phases(args)
        PipelineConfig.save_profile(args.save_profile, phases, args.profile_desc)
        return

    # --- 单个 phase ---
    if args.phase is not None:
        dispatcher = _create_dispatcher(args)
        run_phase(args.phase, dispatcher=dispatcher)
        return

    # --- --auto（带可选 phase 选择）---
    if args.auto:
        has_selection = any([args.only, args.skip, args.from_phase is not None,
                            args.to_phase is not None, args.profile])
        if has_selection:
            phases = resolve_phases(args)
        else:
            phases = _recommended_phases_from_history()
            if not phases:
                phases = PipelineConfig.get_default_phases() or list(range(7))

        dispatcher = _create_dispatcher(args)
        run_auto(phases, dispatcher=dispatcher, max_iterations=args.max_iterations)
        return

    # 默认：交互式
    run_interactive()


if __name__ == '__main__':
    main()
