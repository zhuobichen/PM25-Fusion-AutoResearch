# -*- coding: utf-8 -*-
"""
任务调度器
==========
闭环工作流的核心：生成结构化任务描述，分发给可用的 Agent 后端执行。

设计理念：
  - 不依赖外部 CLI（如 claude -p），实现自包含闭环
  - 支持多种 Agent 后端（TRAE / CLI / Manual / 自定义扩展）
  - 通过文件系统通信（任务文件 + 产物检测），天然适配异步执行

闭环工作流：
  1. run_pipeline.py 遇到 LLM Phase → TaskDispatcher.dispatch()
  2. TaskDispatcher 生成标准化任务描述 → 写入 .state/pending_task.json
  3. 后端执行任务（TRAE Agent 读取任务文件并执行）
  4. 产物产生后 → ArtifactDetector 检测到 → PipelineState 推进
  5. 下一次 run_pipeline.py 调用自动跳过已完成 Phase

扩展新后端：
  from agents.task_dispatcher import register_backend, BackendBase

  class MyBackend(BackendBase):
      def execute(self, task: dict) -> tuple:
          # 自定义执行逻辑
          return True, "执行完成"

  register_backend('my_backend', MyBackend)
"""

import os
import sys
import json
import subprocess
from datetime import datetime

# ---- Phase 元数据（与 run_pipeline.py 保持一致）----

PHASE_INFO = {
    0: ('organize', '项目整理'),
    1: ('download', '文献下载'),
    2: ('analyze',  '文献分析'),
    3: ('design',   '方案设计'),
    4: ('code',     '代码实现'),
    5: ('verify',   '测试验证'),
    6: ('write',    '论文写作'),
}

ROLE_MAP = {
    0: 'organizer',
    1: 'literature_downloader',
    2: 'literature_analyzer',
    3: 'method_designer',
    4: 'code_engineer',
    5: 'test_verifier',
    6: 'technical_writer',
}

# ---- 各 Phase 的输入/输出产物定义 ----

PHASE_ARTIFACTS = {
    0: {
        'inputs': [],
        'outputs': ['INVENTORY.md'],
        'output_dirs': [],
    },
    1: {
        'inputs': [],
        'outputs': [],
        'output_dirs': ['PaperDownload/', 'PaperDownloadMd/paper_list.json'],
    },
    2: {
        'inputs': ['PaperDownload/', 'PaperDownloadMd/paper_list.json'],
        'outputs': [],
        'output_dirs': ['MethodToSmart/'],
    },
    3: {
        'inputs': ['MethodToSmart/', 'Innovation/success/'],
        'outputs': [],
        'output_dirs': ['SmartToCode/创新方法指令/', 'SmartToCode/复现方法指令/'],
    },
    4: {
        'inputs': ['SmartToCode/'],
        'outputs': [],
        'output_dirs': ['CodeWorkSpace/新融合方法代码/', 'CodeWorkSpace/复现方法代码/'],
    },
    5: {
        'inputs': ['CodeWorkSpace/'],
        'outputs': [],
        'output_dirs': ['test_result/创新方法/', 'test_result/基准方法/'],
    },
    6: {
        'inputs': ['Innovation/success/', 'test_result/'],
        'outputs': ['paper_output/paper.tex'],
        'output_dirs': ['paper_output/'],
    },
}


# ============================================================
# 后端接口定义
# ============================================================

class BackendBase:
    """Agent 后端抽象基类，所有后端必须实现此接口"""

    name = 'base'

    def __init__(self, project_root, **kwargs):
        self.root = project_root
        self.kwargs = kwargs

    def execute(self, task):
        """
        执行任务。

        Args:
            task: 标准化任务描述字典，包含:
                - phase: Phase 编号
                - phase_name: Phase 名称
                - role: 角色名
                - title: 中文标题
                - prompt: 完整 prompt 文本
                - context: 历史上下文 (best_method, failed_methods 等)
                - inputs: 输入产物路径列表
                - expected_outputs: 预期输出产物路径列表
                - dispatched_at: 分发时间

        Returns:
            tuple: (success: bool, output: str)
                - success=True 表示任务已分发/执行
                - output 为人类可读的执行信息
        """
        raise NotImplementedError

    def is_sync(self):
        """
        是否为同步后端（执行完立即返回结果）。
        - True: CLI 后端等，execute() 返回时产物已生成
        - False: TRAE/Manual 后端等，execute() 返回时代理理可能还在执行
        """
        return True


# ============================================================
# 内置后端实现
# ============================================================

class TRAEBackend(BackendBase):
    """
    TRAE IDE Agent 后端（推荐）。

    生成标准化任务文件，打印 prompt 供 TRAE Agent 读取执行。
    非阻塞：dispatch 后立即返回，由外部 Agent 执行后再推进。
    """

    name = 'trae'

    def __init__(self, project_root, **kwargs):
        super().__init__(project_root, **kwargs)
        self.state_dir = os.path.join(project_root, 'test_result', '.state')
        os.makedirs(self.state_dir, exist_ok=True)

    def execute(self, task):
        task_file = os.path.join(self.state_dir, 'pending_task.json')

        # 写入任务文件
        task['dispatched_at'] = datetime.now().isoformat()
        task['status'] = 'pending'
        with open(task_file, 'w', encoding='utf-8') as f:
            json.dump(task, f, indent=2, ensure_ascii=False)

        # 打印任务信息（供 TRAE Agent 读取）
        phase = task.get('phase', '?')
        title = task.get('title', '')
        role = task.get('role', '')
        inputs = task.get('inputs', [])
        outputs = task.get('expected_outputs', [])

        print(f"\n{'=' * 60}")
        print(f"[TRAE Agent 任务] Phase {phase}: {title}")
        print(f"角色: {role}")
        print(f"任务文件: {task_file}")
        if inputs:
            print(f"输入产物: {', '.join(inputs)}")
        if outputs:
            print(f"预期输出: {', '.join(outputs)}")
        print(f"{'=' * 60}")
        print(f"\n--- Prompt ---\n")
        print(task.get('prompt', ''))
        print(f"\n--- Prompt 结束 ---\n")
        print(f"请执行上述任务。完成后重新运行 run_pipeline.py 自动检测产物并推进。")
        print(f"任务文件已保存到: {task_file}")

        return True, f"任务已分发到 TRAE Agent (Phase {phase}: {title})"

    def is_sync(self):
        return False


class CLIBackend(BackendBase):
    """
    Claude CLI 后端（向后兼容）。

    通过 subprocess 调用 claude -p 执行任务。
    阻塞式：execute() 返回时产物已生成（或执行失败）。
    """

    name = 'cli'

    def __init__(self, project_root, **kwargs):
        super().__init__(project_root, **kwargs)
        self.budget = kwargs.get('budget')
        self.model = kwargs.get('model')
        self.timeout = kwargs.get('timeout', 600)
        self.claude_cmd = self._find_claude()

    @staticmethod
    def _find_claude():
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

    def execute(self, task):
        cmd = [self.claude_cmd, "-p", "--output-format", "text"]
        if self.budget:
            cmd += ["--max-budget-usd", str(self.budget)]
        if self.model:
            cmd += ["--model", self.model]

        env = os.environ.copy()
        env['PYTHONPATH'] = self.root + os.pathsep + env.get('PYTHONPATH', '')
        npm_bin = os.path.expandvars(r'%APPDATA%\npm')
        if npm_bin not in env.get('PATH', ''):
            env['PATH'] = npm_bin + os.pathsep + env.get('PATH', '')

        phase = task.get('phase', '?')
        title = task.get('title', '')
        prompt = task.get('prompt', '')

        print(f"\n[CLI Agent] Phase {phase}: {title}")
        print(f"  命令: {' '.join(cmd[:3])}...")
        timeout_str = f"超时: {self.timeout}s" if self.timeout else "无超时限制"
        print(f"  {timeout_str}" + (f" | 预算: ${self.budget}" if self.budget else ""))
        print("-" * 60)

        try:
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=self.root,
                env=env,
                text=True,
                encoding='utf-8',
                errors='replace',
            )
            stdout_data, _ = process.communicate(
                input=prompt, timeout=self.timeout if self.timeout else None
            )
            print(stdout_data, end='')

            success = process.returncode == 0
            print("-" * 60)
            if success:
                print(f"[CLI Agent] 执行成功")
            else:
                print(f"[CLI Agent] 执行失败 (exit code: {process.returncode})")
            return success, stdout_data

        except subprocess.TimeoutExpired:
            process.kill()
            print(f"\n[CLI Agent] 超时 ({self.timeout}s)，已终止")
            return False, "timeout"
        except FileNotFoundError:
            print(f"\n错误: 未找到 'claude' CLI。建议使用 --backend trae 模式。")
            return False, "claude not found"
        except Exception as e:
            print(f"\n[CLI Agent] 执行异常: {e}")
            return False, str(e)


class ManualBackend(BackendBase):
    """
    手动指引后端（默认）。

    打印操作指引，由用户手动在 IDE 中执行。
    非阻塞：dispatch 后立即返回。
    """

    name = 'manual'

    def execute(self, task):
        phase = task.get('phase', '?')
        title = task.get('title', '')
        role = task.get('role', '')
        prompt = task.get('prompt', '')
        inputs = task.get('inputs', [])
        outputs = task.get('expected_outputs', [])

        print(f"\n{'=' * 60}")
        print(f"[手动指引] Phase {phase}: {title}")
        print(f"角色: {role}")
        if inputs:
            print(f"输入: {', '.join(inputs)}")
        if outputs:
            print(f"预期输出: {', '.join(outputs)}")
        print(f"{'=' * 60}")
        print(f"\n请在 IDE 中执行以下任务:\n")
        print(prompt[:500] + '...' if len(prompt) > 500 else prompt)
        print(f"\n[提示] 完成后重新运行 run_pipeline.py 自动检测产物并推进。")

        return True, f"手动指引已输出 (Phase {phase}: {title})"

    def is_sync(self):
        return False


# ============================================================
# 后端注册表
# ============================================================

_BACKEND_REGISTRY = {}


def register_backend(name, backend_class):
    """注册自定义后端"""
    _BACKEND_REGISTRY[name] = backend_class


def get_backend(name, project_root, **kwargs):
    """获取后端实例"""
    # 注册内置后端
    _register_builtin_backends()

    backend_class = _BACKEND_REGISTRY.get(name)
    if backend_class is None:
        raise ValueError(
            f"未知后端: '{name}'。可用: {', '.join(_BACKEND_REGISTRY.keys())}"
        )
    return backend_class(project_root, **kwargs)


def list_backends():
    """列出所有已注册的后端"""
    _register_builtin_backends()
    return list(_BACKEND_REGISTRY.keys())


def _register_builtin_backends():
    """注册内置后端（仅首次调用时执行）"""
    if _BACKEND_REGISTRY:
        return
    register_backend('trae', TRAEBackend)
    register_backend('cli', CLIBackend)
    register_backend('manual', ManualBackend)


# ============================================================
# 任务调度器
# ============================================================

class TaskDispatcher:
    """
    任务调度器：生成任务描述 → 分发给后端 → 检测产物完成。

    使用方式：
        dispatcher = TaskDispatcher(project_root, backend='trae')
        result = dispatcher.dispatch(phase_num=3, prompt=..., context=...)
    """

    def __init__(self, project_root, backend='trae', **backend_kwargs):
        self.root = project_root
        self.backend_name = backend
        self.backend = get_backend(backend, project_root, **backend_kwargs)
        self.state_dir = os.path.join(project_root, 'test_result', '.state')
        os.makedirs(self.state_dir, exist_ok=True)

    def dispatch(self, phase_num, prompt, context=None):
        """
        分发一个 Phase 的任务。

        Args:
            phase_num: Phase 编号 (0-6)
            prompt: 完整的 prompt 文本
            context: 历史上下文字典 (可选)

        Returns:
            tuple: (success, output, task_desc)
        """
        phase_name, phase_label = PHASE_INFO.get(phase_num, ('?', '?'))
        role = ROLE_MAP.get(phase_num, '?')
        artifacts = PHASE_ARTIFACTS.get(phase_num, {})

        task_desc = {
            'phase': phase_num,
            'phase_name': phase_name,
            'phase_label': phase_label,
            'role': role,
            'title': phase_label,
            'prompt': prompt,
            'context': context or {},
            'inputs': artifacts.get('inputs', []),
            'expected_outputs': artifacts.get('outputs', []) + artifacts.get('output_dirs', []),
            'dispatched_at': datetime.now().isoformat(),
            'backend': self.backend_name,
            'status': 'dispatching',
        }

        # 分发给后端
        success, output = self.backend.execute(task_desc)

        # 更新任务状态
        task_desc['status'] = 'dispatched' if success else 'failed'
        task_desc['dispatch_result'] = output

        # 保存任务记录
        self._save_task_record(task_desc)

        return success, output, task_desc

    def _save_task_record(self, task_desc):
        """保存任务记录到文件系统"""
        task_file = os.path.join(self.state_dir, 'pending_task.json')
        with open(task_file, 'w', encoding='utf-8') as f:
            json.dump(task_desc, f, indent=2, ensure_ascii=False)

        # 同时追加到历史记录
        history_file = os.path.join(self.state_dir, 'task_history.jsonl')
        with open(history_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(task_desc, ensure_ascii=False) + '\n')

    def get_pending_task(self):
        """读取当前待处理的任务"""
        task_file = os.path.join(self.state_dir, 'pending_task.json')
        if not os.path.exists(task_file):
            return None
        try:
            with open(task_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return None

    def clear_pending_task(self):
        """清除待处理任务（产物检测通过后调用）"""
        task_file = os.path.join(self.state_dir, 'pending_task.json')
        if os.path.exists(task_file):
            os.remove(task_file)


# ============================================================
# Prompt 构建
# ============================================================

def build_prompt_for_phase(phase_num, project_root, resume_context=None):
    """
    为指定 Phase 构建 prompt。

    Args:
        phase_num: Phase 编号
        project_root: 项目根目录
        resume_context: 历史上下文字典 (可选)

    Returns:
        str: 完整的 prompt 文本
    """
    role = ROLE_MAP.get(phase_num)
    if not role:
        return None

    try:
        import sys
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        from agents.role_templates import get_spawn_prompt
        base_prompt = get_spawn_prompt(role, project_root)
    except Exception:
        return None

    # 为需要历史上下文的 Phase 注入 resume 信息
    if phase_num in (2, 3, 4, 6) and resume_context:
        context_block = _build_context_block(resume_context)
        return f"{base_prompt}\n\n{context_block}\n"

    return base_prompt


def _build_context_block(context):
    """构建历史上下文 prompt 块"""
    best_r2 = context.get('best_r2')
    best_r2_str = f"{best_r2:.4f}" if isinstance(best_r2, (int, float)) else "N/A"

    lines = [
        "## 历史迭代上下文（自动注入）",
        "",
        f"- 当前最佳方法: {context.get('best_method', 'None')}",
        f"- 当前最佳R²: {best_r2_str}",
        f"- 历史迭代轮次: {context.get('iteration', 0)}",
        f"- 研究状态: {context.get('research_status', 'unknown')}",
    ]

    failed = context.get('failed_methods', [])
    if failed:
        lines.append("- 最近失败方法（避免重复尝试同一路线）:")
        for method, reason in failed[:5]:
            lines.append(f"  - {method}: {reason or '无记录原因'}")

    pending_designed = context.get('pending_designed', [])
    if pending_designed:
        lines.append(f"- 待实现方法: {', '.join(pending_designed[:12])}")

    pending_implemented = context.get('pending_implemented', [])
    if pending_implemented:
        lines.append(f"- 待验证方法: {', '.join(pending_implemented[:12])}")

    verified_fail = context.get('verified_fail', [])
    if verified_fail:
        lines.append(f"- 已验证失败方法数: {len(verified_fail)}")

    lines += [
        "",
        "要求：",
        "1. 基于已有结果继续优化，不要从零开始",
        "2. 避免重复已失败的方法路线",
        "3. 新方法必须有明确物理可解释性",
    ]

    return '\n'.join(lines)
