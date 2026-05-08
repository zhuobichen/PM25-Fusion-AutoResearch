# -*- coding: utf-8 -*-
"""
PM2.5 CMAQ融合方法自动研究流程 v13.0
=====================================
Agent 自动执行版：支持 Claude CLI 串联执行 + profile 灵活配置

使用方式：
    # 手动指引模式（只打印指令）
    python run_pipeline.py --auto --skip 1
    python run_pipeline.py --auto --only 3,4,5

    # Agent 自动执行模式（调用 Claude CLI 串联执行）
    python run_pipeline.py --auto --agent
    python run_pipeline.py --auto --agent --skip 1
    python run_pipeline.py --auto --agent --budget 5.0

    # 直接跑验证脚本（不需要 agent）
    python run_pipeline.py --auto --only 5

    # Profile 管理
    python run_pipeline.py --list-profiles
    python run_pipeline.py --save-profile my-flow --skip 1
"""

import os
import sys
import json
import time
import signal
import glob
import argparse
import subprocess
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


# ============================================================
# 状态管理
# ============================================================

class PipelineState:
    """Pipeline state manager"""

    STATE_FILE = os.path.join(PROJECT_ROOT, '.agent_state.json')

    @classmethod
    def load(cls):
        if os.path.exists(cls.STATE_FILE):
            with open(cls.STATE_FILE, 'r', encoding='utf-8') as f:
                old_state = json.load(f)
            # Migrate old format -> new format
            return cls._migrate(old_state)
        return cls._default_state()

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
    def phase_completed(cls, phase_name):
        state = cls.load()
        state['phases'][phase_name] = {
            'status': 'completed',
            'completed_at': datetime.now().isoformat()
        }
        cls.save(state)

    @classmethod
    def is_phase_done(cls, phase_name):
        state = cls.load()
        return state['phases'].get(phase_name, {}).get('status') == 'completed'

    @classmethod
    def mark_innovation(cls, method_name, r2):
        state = cls.load()
        if state.get('current_best_r2') is None or r2 > state['current_best_r2']:
            state['current_best_method'] = method_name
            state['current_best_r2'] = r2
        cls.save(state)

    @classmethod
    def print_status(cls):
        state = cls.load()
        print(f"""
=== 流水线状态 ===
轮次:       {state.get('round', 'N/A')}
最佳方法:   {state.get('current_best_method', 'None')} (R²={state.get('current_best_r2', 'N/A')})
创新成立:   {state.get('innovation_established', False)}
上次运行:   {state.get('last_run', '从未')}

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
        return p.get('status', '未开始')


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

class ClaudeExecutor:
    """通过 claude CLI 非交互模式执行 agent prompt"""

    def __init__(self, project_root, budget=None, model=None, timeout=600):
        self.project_root = project_root
        self.budget = budget
        self.model = model
        self.timeout = timeout
        self.claude_cmd = self._find_claude()

    @staticmethod
    def _find_claude():
        """查找 claude CLI 的完整路径"""
        import shutil
        claude_path = shutil.which('claude')
        if claude_path:
            return claude_path
        # Windows npm 全局目录常见路径
        npm_paths = [
            os.path.expandvars(r'%APPDATA%\npm\claude.CMD'),
            os.path.expandvars(r'%APPDATA%\npm\claude.cmd'),
            os.path.expandvars(r'%APPDATA%\npm\claude'),
        ]
        for p in npm_paths:
            if os.path.exists(p):
                return p
        return 'claude'  # fallback，让报错信息更明确

    def execute(self, prompt, phase_name=""):
        """
        调用 claude -p 执行 prompt，通过 stdin 传递（避免命令行长度限制）。
        返回 (success: bool, output: str)
        """
        cmd = [self.claude_cmd, "-p", "--output-format", "text"]
        if self.budget:
            cmd += ["--max-budget-usd", str(self.budget)]
        if self.model:
            cmd += ["--model", self.model]

        # 确保子进程能找到项目模块和 claude CLI
        env = os.environ.copy()
        env['PYTHONPATH'] = self.project_root + os.pathsep + env.get('PYTHONPATH', '')
        npm_bin = os.path.expandvars(r'%APPDATA%\npm')
        if npm_bin not in env.get('PATH', ''):
            env['PATH'] = npm_bin + os.pathsep + env.get('PATH', '')

        label = f"[Agent:{phase_name}] " if phase_name else "[Agent] "
        print(f"\n{label}启动 Claude CLI...")
        timeout_str = f"超时: {self.timeout}s" if self.timeout else "无超时限制"
        print(f"{label}{timeout_str}" + (f" | 预算: ${self.budget}" if self.budget else ""))
        print("-" * 60)

        try:
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=self.project_root,
                env=env,
                text=True,
                encoding='utf-8',
                errors='replace',
            )

            # 通过 stdin 传递 prompt，然后关闭 stdin
            stdout_data, _ = process.communicate(input=prompt, timeout=self.timeout if self.timeout else None)

            # 实时打印输出
            print(stdout_data, end='')

            success = process.returncode == 0

            print("-" * 60)
            if success:
                print(f"{label}执行成功")
            else:
                print(f"{label}执行失败 (exit code: {process.returncode})")
            return success, stdout_data

        except subprocess.TimeoutExpired:
            process.kill()
            print(f"\n{label}超时 ({self.timeout}s)，已终止")
            return False, "timeout"
        except FileNotFoundError:
            print(f"\n错误: 未找到 'claude' CLI。请确保 Claude Code 已安装。")
            return False, "claude not found"
        except Exception as e:
            print(f"\n{label}执行异常: {e}")
            return False, str(e)


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
    print("\n执行计划:")
    print("-" * 45)
    for p in range(7):
        name, desc = PHASE_INFO[p]
        if p in phases:
            done = PipelineState.is_phase_done(name)
            status = " [已完成,将跳过]" if done else " [待执行]"
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
        return get_spawn_prompt(role, PROJECT_ROOT)

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
    """调用 Claude CLI 为新方法生成验证脚本。"""
    if not method_names:
        return

    # 构建 prompt
    method_list = '\n'.join(f'- {name}' for name in method_names)
    prompt = f"""你是一个严格的机器学习算法评测专家。

## 任务

为以下新方法生成十折交叉验证脚本：

{method_list}

## 要求

1. 读取 {PROJECT_ROOT}/CodeWorkSpace/新融合方法代码/ 中每个方法的代码
2. 参考 {PROJECT_ROOT}/test_result/创新方法/AdvancedRK_十折标准模式.py 的结构
3. 为每个方法生成 test_result/创新方法/{{方法名}}_十折标准模式.py
4. 脚本必须包含：
   - 十折交叉验证（fold_split_table_daily.csv）
   - 多阶段验证（pre_exp/stage1/stage2/stage3）
   - 指标计算（R2, MAE, RMSE, MB）
   - 结果保存为 {{方法名}}_all_stages.json 和 {{方法名}}_summary.csv

## 参考脚本结构

读取 AdvancedRK_十折标准模式.py 了解完整结构，包括：
- 数据加载（CMAQ + Monitor）
- 十折循环（train/test split）
- 模型拟合和预测
- 指标计算和保存

完成后退出。
"""

    # 调用 Claude CLI
    claude_cmd = _find_claude()
    cmd = [claude_cmd, "-p", "--output-format", "text"]

    print(f"  [生成] 调用 Claude CLI 生成 {len(method_names)} 个验证脚本...")
    try:
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=PROJECT_ROOT,
            env=env,
            text=True,
            encoding='utf-8',
            errors='replace',
        )
        stdout_data, _ = process.communicate(input=prompt, timeout=600)

        if process.returncode == 0:
            print(f"  [完成] 验证脚本生成成功")
        else:
            print(f"  [警告] Claude CLI 返回码: {process.returncode}")
    except subprocess.TimeoutExpired:
        process.kill()
        print(f"  [超时] 验证脚本生成超时")
    except Exception as e:
        print(f"  [错误] 验证脚本生成失败: {e}")


def _find_claude():
    """查找 claude CLI 路径。"""
    import shutil
    claude_path = shutil.which('claude')
    if claude_path:
        return claude_path
    npm_paths = [
        os.path.expandvars(r'%APPDATA%\npm\claude.CMD'),
        os.path.expandvars(r'%APPDATA%\npm\claude.cmd'),
        os.path.expandvars(r'%APPDATA%\npm\claude'),
    ]
    for p in npm_paths:
        if os.path.exists(p):
            return p
    return 'claude'


def run_verify_phase():
    """Phase 5 特殊处理：直接运行 Python 验证脚本，不经过 Claude agent"""
    print(f"\n{'=' * 60}")
    print("Phase 5: 测试验证（直接执行）")
    print("=" * 60)

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
    scripts = []
    if os.path.exists(innovation_dir):
        for f in os.listdir(innovation_dir):
            if f.endswith('_十折标准模式.py') or f.endswith('_十折验证.py'):
                scripts.append(os.path.join(innovation_dir, f))

    # 2.0 检查是否有新方法需要生成验证脚本
    code_dir = data_path('CodeWorkSpace/新融合方法代码')
    if os.path.exists(code_dir):
        existing_script_names = set()
        for s in scripts:
            basename = os.path.basename(s)
            name = basename.split('_十折')[0] if '_十折' in basename else basename.rsplit('_', 1)[0]
            existing_script_names.add(name.replace('-', '_'))

        # 找出有代码但无验证脚本的方法
        skip_prefixes = ('compare_', 'find_best_', 'validate_', 'lambda_', 'spatial_stat_',
                         'statistical_', 'robust_variogram_', 'mle_', 'elegant_', 'adaptive_')
        new_methods = []
        for py_file in os.listdir(code_dir):
            if not py_file.endswith('.py'):
                continue
            if py_file.startswith(skip_prefixes):
                continue
            method_name = py_file[:-3]
            if method_name.replace('-', '_') not in existing_script_names:
                new_methods.append(method_name)

        if new_methods:
            print(f"\n  发现 {len(new_methods)} 个新方法需要生成验证脚本: {', '.join(new_methods[:5])}{'...' if len(new_methods) > 5 else ''}")
            # 调用 Claude CLI 生成验证脚本
            _generate_validation_scripts(new_methods, env)
            # 重新扫描脚本
            scripts = []
            if os.path.exists(innovation_dir):
                for f in os.listdir(innovation_dir):
                    if f.endswith('_十折标准模式.py') or f.endswith('_十折验证.py'):
                        scripts.append(os.path.join(innovation_dir, f))

    if not scripts:
        print("\n  未找到创新方法验证脚本")
        print("  请先完成 Phase 4 (代码实现)")
        return False

    # 2.1 读取注册表，跳过已验证方法
    skip_methods = set()
    skip_methods_normalized = {}  # normalized_name → original_name
    try:
        from shared.method_registry import MethodRegistry
        registry = MethodRegistry()
        skip_methods = registry.get_tested_method_names()
        for m in skip_methods:
            skip_methods_normalized[m.replace('-', '_')] = m
        if skip_methods:
            print(f"\n  注册表中已有 {len(skip_methods)} 个已验证方法，将跳过: {', '.join(sorted(skip_methods))}")
    except Exception:
        pass  # 注册表不存在时正常执行

    # 2.2 过滤脚本（归一化连字符/下划线以匹配注册表）
    scripts_to_run = []
    for script in scripts:
        basename = os.path.basename(script)
        # 从文件名提取方法名：PolyRK_十折标准模式.py → PolyRK
        method_name = basename.split('_十折')[0] if '_十折' in basename else basename.rsplit('_', 1)[0]
        method_name_normalized = method_name.replace('-', '_')
        if method_name_normalized in skip_methods_normalized:
            original = skip_methods_normalized[method_name_normalized]
            print(f"  [跳过] {basename} — 方法 {original} 已有验证结果")
        else:
            scripts_to_run.append(script)

    if not scripts_to_run:
        print(f"\n  所有 {len(scripts)} 个创新方法均已验证，无需重复运行")
        PipelineState.phase_completed('verify')
        print(f"\n[完成] Phase 5 (verify) — 全部方法已验证")
        return True

    print(f"\n[2/2] 运行 {len(scripts_to_run)} 个创新方法验证脚本 (跳过 {len(scripts) - len(scripts_to_run)} 个):")
    for i, script in enumerate(scripts_to_run):
        basename = os.path.basename(script)
        method_name = basename.split('_十折')[0] if '_十折' in basename else basename.rsplit('_', 1)[0]
        print(f"  [{i + 1}/{len(scripts_to_run)}] {basename}")
        result = subprocess.run(
            [sys.executable, script],
            cwd=PROJECT_ROOT, env=env,
            text=True, encoding='utf-8', errors='replace'
        )
        if result.returncode != 0:
            print(f"  [失败] {basename} 退出码: {result.returncode}")
            return False
        print(f"  [完成] {basename}")

        # 验证后更新注册表
        try:
            _update_registry_after_verify(method_name, registry)
        except Exception as e:
            print(f"  [警告] 注册表更新失败: {e}")

    PipelineState.phase_completed('verify')
    print(f"\n[完成] Phase 5 (verify) 全部验证通过")
    return True


def _update_registry_after_verify(method_name: str, registry=None):
    """验证完成后更新注册表：从结果 JSON 中解析指标并写入。"""
    if registry is None:
        try:
            from shared.method_registry import MethodRegistry
            registry = MethodRegistry()
        except Exception:
            return

    # 查找结果 JSON
    import glob as _glob
    patterns = [
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

    # 没有 all_stages JSON，尝试从 summary CSV 更新
    csv_path = os.path.join(PROJECT_ROOT, 'test_result', '创新方法', f'{method_name}_summary.csv')
    if os.path.exists(csv_path):
        import csv as _csv
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = _csv.DictReader(f)
            for row in reader:
                name = (row.get('Method') or row.get('method') or '').strip()
                if name == method_name:
                    metrics = {}
                    for key in ('R2', 'MAE', 'RMSE', 'MB'):
                        try:
                            metrics[key] = float(row.get(key, 0))
                        except (ValueError, TypeError):
                            metrics[key] = 0.0
                    if not registry.method_exists(method_name):
                        registry.add_method(method_name)
                    r2 = metrics.get('R2', 0) or 0
                    from shared.method_registry import STATE_VERIFIED_PASS, STATE_VERIFIED_FAIL
                    registry.update_state(method_name, STATE_VERIFIED_PASS if r2 > 0.8 else STATE_VERIFIED_FAIL)
                    registry.update_metrics(method_name, metrics)
                    registry.save()
                    print(f"  [注册表] 已从 CSV 更新 {method_name}")
                    return


def run_phase(phase_num, executor=None):
    """
    运行指定 Phase。

    - executor=None: 打印手动指引
    - executor=ClaudeExecutor: 调用 Claude CLI 自动执行
    - Phase 5 始终直接运行 Python 脚本
    """
    if phase_num not in range(7):
        print(f"无效 Phase: {phase_num} (可选: 0-6)")
        return False

    name, desc = PHASE_INFO[phase_num]
    if PipelineState.is_phase_done(name):
        print(f"Phase {phase_num} ({name}) 已完成，跳过。")
        return True

    # Phase 5: 直接运行验证脚本
    if phase_num == 5:
        return run_verify_phase()

    # Agent 模式
    if executor is not None:
        prompt = PhaseRunner.get_prompt(phase_num)
        if not prompt:
            print(f"  Phase {phase_num} 无 prompt 模板，跳过")
            return True
        success, output = executor.execute(prompt, phase_name=name)
        if success:
            PipelineState.phase_completed(name)
            print(f"\n[完成] Phase {phase_num} ({name})")
        return success

    # 手动指引模式
    PhaseRunner.print_guidance(phase_num)
    PipelineState.phase_completed(name)
    print(f"\n[完成] Phase {phase_num} ({name}) 已标记为完成")
    return True


def run_auto(phases=None, executor=None):
    """
    自动运行指定或全部 Phase。

    - executor=None: 打印手动指引
    - executor=ClaudeExecutor: 调用 Claude CLI 自动串联执行
    """
    if phases is None:
        phases = list(range(7))

    mode = "Agent 自动执行" if executor else "手动指引"
    print("=" * 60)
    print(f"PM2.5 CMAQ 融合方法自动研究流程 v13.0 [{mode}]")
    print("=" * 60)

    PipelineState.print_status()
    print_plan(phases)

    for phase_num in phases:
        success = run_phase(phase_num, executor=executor)
        if not success and executor is not None:
            print(f"\n[中断] Phase {phase_num} 失败，停止后续 phase。")
            break

    print("\n" + "=" * 60)
    executed = [p for p in phases if not PipelineState.is_phase_done(PHASE_INFO[p][0])]
    completed = [p for p in phases if PipelineState.is_phase_done(PHASE_INFO[p][0])]
    skipped = [p for p in range(7) if p not in phases]

    print(f"已完成: {len(completed)} | 待执行: {len(executed)} | 跳过: {len(skipped)}")
    if skipped:
        skipped_names = [PHASE_INFO[p][1] for p in skipped]
        print(f"跳过的 Phase: {', '.join(skipped_names)}")

    if executor:
        print("Agent 执行完毕。")
    else:
        print("请按指引在 Claude Code 中执行各 LLM 依赖的 Phase。")
    print("=" * 60)


def run_interactive():
    """交互式运行"""
    print("=" * 60)
    print("PM2.5 CMAQ 融合方法自动研究流程 v11.0")
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


def main():
    parser = argparse.ArgumentParser(
        description='PM2.5 CMAQ融合方法自动研究流程',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 手动指引模式（只打印指令，不执行）
  python run_pipeline.py --auto                         # 运行全部（或 config default）
  python run_pipeline.py --auto --skip 1                # 跳过下载
  python run_pipeline.py --auto --only 3,4,5            # 只跑设计+编码+验证

  # Agent 自动执行模式（调用 Claude CLI 串联执行）
  python run_pipeline.py --auto --agent                 # 全自动
  python run_pipeline.py --auto --agent --skip 1        # 跳过下载
  python run_pipeline.py --auto --agent --budget 5.0    # 限制每个 phase 花费

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

    # Agent 模式
    parser.add_argument('--agent', action='store_true',
                        help='使用 Claude Agent 自动执行（默认：只打印指引）')
    parser.add_argument('--budget', type=float, metavar='USD',
                        help='每个 Phase 的最大花费（美元）')
    parser.add_argument('--model', type=str, metavar='MODEL',
                        help='指定 Claude 模型')
    parser.add_argument('--timeout', type=int, default=None, metavar='SEC',
                        help='每个 Phase 的超时秒数（默认无限制）')

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
        executor = ClaudeExecutor(PROJECT_ROOT, args.budget, args.model, args.timeout) if args.agent else None
        run_phase(args.phase, executor=executor)
        return

    # --- --auto（带可选 phase 选择）---
    if args.auto:
        has_selection = any([args.only, args.skip, args.from_phase is not None,
                            args.to_phase is not None, args.profile])
        if has_selection:
            phases = resolve_phases(args)
        else:
            phases = PipelineConfig.get_default_phases() or list(range(7))

        executor = None
        if args.agent:
            executor = ClaudeExecutor(PROJECT_ROOT, args.budget, args.model, args.timeout)

        run_auto(phases, executor=executor)
        return

    # 默认：交互式
    run_interactive()


if __name__ == '__main__':
    main()
