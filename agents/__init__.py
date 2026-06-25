# -*- coding: utf-8 -*-
# Agents Package - 闭环工作流 Agent 系统
#
# v14.0 架构：
#   - task_dispatcher:  任务调度器（生成任务描述 + 分发给后端）
#   - artifact_detector: 产物检测器（检测各 Phase 产物完成状态）
#   - role_templates:   角色模板（生成 Agent prompt）
#
# 多 Agent 后端支持：
#   from agents.task_dispatcher import TaskDispatcher, register_backend, BackendBase
#
#   # 使用内置后端
#   dispatcher = TaskDispatcher(project_root, backend='trae')  # 闭环模式
#   dispatcher = TaskDispatcher(project_root, backend='cli')   # Claude CLI
#   dispatcher = TaskDispatcher(project_root, backend='manual') # 手动指引
#
#   # 注册自定义后端
#   class MyBackend(BackendBase):
#       def execute(self, task): ...
#   register_backend('my_backend', MyBackend)

from .task_dispatcher import (
    TaskDispatcher,
    BackendBase,
    TRAEBackend,
    CLIBackend,
    ManualBackend,
    register_backend,
    get_backend,
    list_backends,
    build_prompt_for_phase,
)
from .artifact_detector import (
    ArtifactDetector,
    format_detection_report,
    PHASE_NAMES,
    PHASE_LABELS,
)

__all__ = [
    # Task Dispatcher
    'TaskDispatcher',
    'BackendBase',
    'TRAEBackend',
    'CLIBackend',
    'ManualBackend',
    'register_backend',
    'get_backend',
    'list_backends',
    'build_prompt_for_phase',
    # Artifact Detector
    'ArtifactDetector',
    'format_detection_report',
    'PHASE_NAMES',
    'PHASE_LABELS',
]
