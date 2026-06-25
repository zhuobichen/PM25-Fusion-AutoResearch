# -*- coding: utf-8 -*-
"""
产物检测器
==========
检测各 Phase 的产物是否已完成，用于闭环工作流的自动推进。

每个 Phase 有对应的检测逻辑：
- Phase 0 (organize): INVENTORY.md 存在
- Phase 1 (download):  PaperDownload/ 有 PDF 文件
- Phase 2 (analyze):   MethodToSmart/ 有方法文档
- Phase 3 (design):    SmartToCode/ 有方案指令 + MethodRegistry 有 designed 状态
- Phase 4 (code):      CodeWorkSpace/ 有代码文件 + MethodRegistry 有 implemented 状态
- Phase 5 (verify):    由 run_verify_phase() 直接执行，不需要检测
- Phase 6 (write):     paper_output/paper.tex 存在
"""

import os
import glob
import json
from datetime import datetime

PHASE_NAMES = {
    0: 'organize',
    1: 'download',
    2: 'analyze',
    3: 'design',
    4: 'code',
    5: 'verify',
    6: 'write',
}

PHASE_LABELS = {
    0: '项目整理',
    1: '文献下载',
    2: '文献分析',
    3: '方案设计',
    4: '代码实现',
    5: '测试验证',
    6: '论文写作',
}


class ArtifactDetector:
    """检测各 Phase 产物完成状态"""

    def __init__(self, project_root):
        self.root = project_root

    # ---- 公共接口 ----

    def check(self, phase_num):
        """
        检测指定 Phase 的产物是否已完成。

        Returns:
            dict: {
                'done': bool,        # 产物是否已存在
                'details': str,      # 人类可读的检测详情
                'artifacts': list,   # 检测到的产物文件列表
            }
        """
        checker = {
            0: self._check_organize,
            1: self._check_download,
            2: self._check_analyze,
            3: self._check_design,
            4: self._check_code,
            5: self._check_verify,
            6: self._check_write,
        }.get(phase_num)
        if checker is None:
            return {'done': False, 'details': f'未知 Phase: {phase_num}', 'artifacts': []}
        return checker()

    def check_all(self):
        """检测所有 Phase 的状态，返回 {phase_num: dict}"""
        return {p: self.check(p) for p in range(7)}

    def get_pending_phases(self):
        """获取产物尚未完成的 Phase 列表"""
        return [p for p in range(7) if not self.check(p)['done']]

    # ---- 逐 Phase 检测逻辑 ----

    def _check_organize(self):
        inventory = os.path.join(self.root, 'INVENTORY.md')
        done = os.path.exists(inventory)
        return {
            'done': done,
            'details': f'INVENTORY.md {"存在" if done else "不存在"}',
            'artifacts': [inventory] if done else [],
        }

    def _check_download(self):
        pdf_dir = os.path.join(self.root, 'PaperDownload')
        pdfs = []
        if os.path.isdir(pdf_dir):
            pdfs = glob.glob(os.path.join(pdf_dir, '**', '*.pdf'), recursive=True)
        done = len(pdfs) > 0
        return {
            'done': done,
            'details': f'PaperDownload/ 中有 {len(pdfs)} 个 PDF',
            'artifacts': pdfs[:5],
        }

    def _check_analyze(self):
        method_dir = os.path.join(self.root, 'MethodToSmart')
        docs = []
        if os.path.isdir(method_dir):
            docs = [f for f in os.listdir(method_dir)
                    if f.endswith('.md') and not f.startswith('INVENTORY')]
        done = len(docs) >= 3
        return {
            'done': done,
            'details': f'MethodToSmart/ 中有 {len(docs)} 个方法文档',
            'artifacts': docs[:5],
        }

    def _check_design(self):
        innovation_dir = os.path.join(self.root, 'SmartToCode', '创新方法指令')
        repro_dir = os.path.join(self.root, 'SmartToCode', '复现方法指令')
        designs = []
        for d in (innovation_dir, repro_dir):
            if os.path.isdir(d):
                designs.extend(f for f in os.listdir(d) if f.endswith('.md'))

        # 同时检查 MethodRegistry 是否有 designed 状态的方法
        registry_pending = self._get_registry_state_count('designed')
        done = len(designs) >= 3
        return {
            'done': done,
            'details': f'SmartToCode/ 中有 {len(designs)} 个方案指令, 注册表待实现: {registry_pending}',
            'artifacts': designs[:5],
        }

    def _check_code(self):
        new_dir = os.path.join(self.root, 'CodeWorkSpace', '新融合方法代码')
        repro_dir = os.path.join(self.root, 'CodeWorkSpace', '复现方法代码')
        code_files = []
        for d in (new_dir, repro_dir):
            if os.path.isdir(d):
                code_files.extend(f for f in os.listdir(d) if f.endswith('.py'))

        # 同时检查 MethodRegistry 是否有 implemented 状态的方法
        registry_pending = self._get_registry_state_count('implemented')
        done = len(code_files) >= 3
        return {
            'done': done,
            'details': f'CodeWorkSpace/ 中有 {len(code_files)} 个代码文件, 注册表待验证: {registry_pending}',
            'artifacts': code_files[:5],
        }

    def _check_verify(self):
        # Phase 5 由 run_verify_phase() 直接执行，这里只检查是否有结果
        benchmark = os.path.join(self.root, 'test_result', '基准方法', 'benchmark_multistage.json')
        success_jsons = glob.glob(
            os.path.join(self.root, 'Innovation', 'success', '**', '*_all_stages.json'),
            recursive=True
        )
        done = os.path.exists(benchmark) or len(success_jsons) > 0
        return {
            'done': done,
            'details': f'基准结果: {"存在" if os.path.exists(benchmark) else "无"}, '
                       f'创新验证 JSON: {len(success_jsons)} 个',
            'artifacts': [benchmark] if os.path.exists(benchmark) else [],
        }

    def _check_write(self):
        tex = os.path.join(self.root, 'paper_output', 'paper.tex')
        pdf = os.path.join(self.root, 'paper_output', 'paper.pdf')
        done = os.path.exists(tex) or os.path.exists(pdf)
        return {
            'done': done,
            'details': f'paper.tex: {"存在" if os.path.exists(tex) else "无"}, '
                       f'paper.pdf: {"存在" if os.path.exists(pdf) else "无"}',
            'artifacts': [f for f in (tex, pdf) if os.path.exists(f)],
        }

    # ---- 辅助方法 ----

    def _get_registry_state_count(self, state_name):
        """从 MethodRegistry 获取指定状态的方法数量"""
        try:
            import sys
            if self.root not in sys.path:
                sys.path.insert(0, self.root)
            from shared.method_registry import MethodRegistry
            registry = MethodRegistry(project_root=self.root)
            methods = registry.get_methods_by_state(state_name)
            return len(methods)
        except Exception:
            return 0


def format_detection_report(detections):
    """格式化检测结果为可读文本"""
    lines = ["产物检测结果:", "-" * 50]
    for phase_num in range(7):
        name = PHASE_NAMES.get(phase_num, '?')
        label = PHASE_LABELS.get(phase_num, '?')
        det = detections.get(phase_num, {})
        done = det.get('done', False)
        details = det.get('details', '')
        status = "✅ 已完成" if done else "⏳ 待完成"
        lines.append(f"  Phase {phase_num} ({label}): {status} — {details}")
    lines.append("-" * 50)
    return '\n'.join(lines)
