# -*- coding: utf-8 -*-
from shared.paths import get_project_root, data_path
"""
Agent Spawn 工作流编排器 (v9)
===========================
真正的 AI Agent spawn 模式 - 使用 Agent 工具启动子代理

核心变化：
- v8 及之前：Python 类直接调用（主会话执行所有任务）
- v9：使用 Agent 工具真正 spawn 子 AI Agent

流程：
文献下载员(并行) → 文献分析员 → 方案设计师 → 代码工程师 → 测试验证员 → 技术写作员

迭代：创新不足时打回方案设计师重新设计
终止：连续3轮无提升 或 人类明确停止
"""

import os
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Any

# 导入角色模板
from agents.role_templates import get_spawn_prompt, ROLE_TEMPLATES


class AgentSpawnOrchestrator:
    """
    Agent Spawn 工作流编排器

    使用 Agent 工具真正 spawn 子 AI Agent，而不是 Python 类调用
    """

    def __init__(self, project_root: str):
        self.project_root = project_root
        self.state_file = os.path.join(project_root, '.agent_state.json')
        self.error_dir = os.path.join(project_root, 'error')
        self.state = self._load_state()

        # 确保目录存在
        os.makedirs(self.error_dir, exist_ok=True)

        # Agent 配置
        self.agents_config = {
            'downloader': {
                'role': 'literature_downloader',
                'template': 'literature_downloader',
                'parallel': True,
                'count': 3,  # 3个并行下载Agent
            },
            'analyzer': {
                'role': 'literature_analyzer',
                'template': 'literature_analyzer',
                'parallel': False,
                'depends_on': ['downloader'],
            },
            'designer': {
                'role': 'method_designer',
                'template': 'method_designer',
                'parallel': False,
                'depends_on': ['analyzer'],
            },
            'engineer': {
                'role': 'code_engineer',
                'template': 'code_engineer',
                'parallel': False,
                'depends_on': ['designer'],
            },
            'verifier': {
                'role': 'test_verifier',
                'template': 'test_verifier',
                'parallel': False,
                'depends_on': ['engineer'],
            },
            'writer': {
                'role': 'technical_writer',
                'template': 'technical_writer',
                'parallel': False,
                'depends_on': ['verifier'],
                'trigger': 'innovation_established',  # 创新成立才触发
            },
        }

        # 配置
        self.max_no_improvement_rounds = 3
        self.agent_timeout = 3600  # 1小时超时

    def _load_state(self) -> Dict:
        """加载工作流状态"""
        if os.path.exists(self.state_file):
            with open(self.state_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {
            'round': 0,
            'agents': {},
            'innovation_established': False,
            'iteration_count': 0,
            'no_improvement_count': 0,
            'terminated': False,
            'last_run': None,
        }

    def _save_state(self):
        """保存工作流状态"""
        self.state['last_run'] = datetime.now().isoformat()
        with open(self.state_file, 'w', encoding='utf-8') as f:
            json.dump(self.state, f, indent=2, ensure_ascii=False)

    def _log_error(self, agent_id: str, error_type: str, message: str):
        """记录错误日志"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(self.error_dir, f'{agent_id}_{error_type}_{timestamp}.log')
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(f"""# 错误日志

Agent ID: {agent_id}
错误类型: {error_type}
时间: {timestamp}

## 错误信息
```
{message}
```

## 当前状态
{json.dumps(self.state, indent=2, ensure_ascii=False)}
""")
        return log_file

    def spawn_agent(self, agent_id: str, role: str,
                    depends_on: List[str] = None,
                    background: bool = False,
                    trigger_condition: str = None) -> Dict:
        """
        Spawn 一个 AI Agent

        Parameters:
        -----------
        agent_id : str
            唯一标识，如 'dl_1', 'analyzer'
        role : str
            角色名，如 'literature_downloader'
        depends_on : list
            依赖的 agent ID 列表
        background : bool
            是否后台运行（用于并行Agent）
        trigger_condition : str
            触发条件，如 'innovation_established'

        Returns:
        --------
        dict : spawn 结果
        """
        # 检查依赖是否满足
        if depends_on:
            for dep_id in depends_on:
                dep_state = self.state['agents'].get(dep_id, {}).get('status')
                if dep_state != 'completed':
                    return {
                        'status': 'blocked',
                        'reason': f'dependency {dep_id} not completed (current: {dep_state})'
                    }

        # 检查触发条件
        if trigger_condition:
            if trigger_condition == 'innovation_established' and not self.state['innovation_established']:
                return {
                    'status': 'blocked',
                    'reason': 'innovation not established yet'
                }

        # 获取 spawn prompt
        prompt = get_spawn_prompt(role, self.project_root)

        # 记录状态
        self.state['agents'][agent_id] = {
            'role': role,
            'status': 'spawned',
            'spawned_at': datetime.now().isoformat(),
            'depends_on': depends_on,
            'background': background,
            'result': None,
            'prompt_length': len(prompt),
        }
        self._save_state()

        # 使用 Agent 工具 spawn
        # 注意：这里返回的是 spawn 指令，实际执行由调用者通过 Agent tool 启动
        return {
            'status': 'ready_to_spawn',
            'agent_id': agent_id,
            'role': role,
            'prompt': prompt,
            'background': background,
        }

    def mark_agent_completed(self, agent_id: str, result: Any = None):
        """标记 Agent 完成"""
        if agent_id in self.state['agents']:
            self.state['agents'][agent_id]['status'] = 'completed'
            self.state['agents'][agent_id]['completed_at'] = datetime.now().isoformat()
            self.state['agents'][agent_id]['result'] = result
            self._save_state()

    def mark_agent_failed(self, agent_id: str, error: str):
        """标记 Agent 失败"""
        if agent_id in self.state['agents']:
            self.state['agents'][agent_id]['status'] = 'failed'
            self.state['agents'][agent_id]['failed_at'] = datetime.now().isoformat()
            self.state['agents'][agent_id]['error'] = error
            self._save_state()
            self._log_error(agent_id, 'agent_failed', error)

    def check_all_agents_completed(self, agent_ids: List[str]) -> bool:
        """检查所有指定 Agent 是否完成"""
        for agent_id in agent_ids:
            state = self.state['agents'].get(agent_id, {}).get('status')
            if state != 'completed':
                return False
        return True

    def evaluate_innovation(self) -> bool:
        """
        评估创新是否成立

        读取 test_result/comparison_report.md 或历史最佳方案
        判断 R² 提升是否 >= 0.01
        """
        comparison_file = os.path.join(self.project_root, 'test_result', 'comparison_report.md')
        best_file = os.path.join(self.project_root, 'test_result', '历史最佳方案', 'best_metrics.json')

        # 尝试读取对比报告
        if os.path.exists(comparison_file):
            with open(comparison_file, 'r', encoding='utf-8') as f:
                content = f.read()
                # 简单检查是否有 "创新成立" 或 R² 提升标记
                if '创新成立' in content or 'R²提升' in content:
                    return True

        # 尝试读取最佳指标
        if os.path.exists(best_file):
            with open(best_file, 'r', encoding='utf-8') as f:
                metrics = json.load(f)
                # 检查是否有 R² 提升
                if metrics.get('r2_improvement', 0) >= 0.01:
                    return True

        return False

    def reset_for_new_iteration(self):
        """重置状态，准备新一轮迭代"""
        # 保留 agent 历史记录，但标记为上一轮
        for agent_id in self.state['agents']:
            if self.state['agents'][agent_id]['status'] == 'spawned':
                self.state['agents'][agent_id]['status'] = 'cancelled'

        self.state['innovation_established'] = False
        self._save_state()

    def run_phase_download(self) -> Dict[str, str]:
        """
        Phase 1: 并行下载

        Spawn 3个下载Agent（不同关键词）
        返回 spawn 的 agent ID 列表
        """
        print("\n" + "="*60)
        print("Phase 1: 并行文献下载")
        print("="*60)

        spawn_ids = []
        for i in range(1, 4):  # dl_1, dl_2, dl_3
            agent_id = f'dl_{i}'
            result = self.spawn_agent(
                agent_id=agent_id,
                role='literature_downloader',
                background=True
            )
            spawn_ids.append(agent_id)
            print(f"  Spawned: {agent_id} (background={result['background']})")

        print(f"  总计 Spawned {len(spawn_ids)} 下载Agent")
        return spawn_ids

    def run_phase_analyze(self, depends_on: List[str]) -> str:
        """
        Phase 2: 文献分析

        等下载完成后，启动分析Agent
        """
        print("\n" + "="*60)
        print("Phase 2: 文献分析")
        print("="*60)

        # 检查依赖
        if not self.check_all_agents_completed(depends_on):
            print(f"  等待下载Agent完成: {depends_on}")
            return None

        result = self.spawn_agent(
            agent_id='analyzer',
            role='literature_analyzer',
            depends_on=depends_on
        )
        print(f"  Spawned: analyzer")
        return 'analyzer'

    def run_phase_design(self, depends_on: List[str]) -> str:
        """
        Phase 3: 方案设计

        等分析完成后，启动设计Agent
        """
        print("\n" + "="*60)
        print("Phase 3: 方案设计")
        print("="*60)

        result = self.spawn_agent(
            agent_id='designer',
            role='method_designer',
            depends_on=depends_on
        )
        print(f"  Spawned: designer")
        return 'designer'

    def run_phase_code(self, depends_on: List[str]) -> str:
        """
        Phase 4: 代码实现

        等设计完成后，启动工程师Agent
        """
        print("\n" + "="*60)
        print("Phase 4: 代码实现")
        print("="*60)

        result = self.spawn_agent(
            agent_id='engineer',
            role='code_engineer',
            depends_on=depends_on
        )
        print(f"  Spawned: engineer")
        return 'engineer'

    def run_phase_test(self, depends_on: List[str]) -> str:
        """
        Phase 5: 测试验证

        等代码完成后，启动测试Agent
        """
        print("\n" + "="*60)
        print("Phase 5: 测试验证")
        print("="*60)

        result = self.spawn_agent(
            agent_id='verifier',
            role='test_verifier',
            depends_on=depends_on
        )
        print(f"  Spawned: verifier")
        return 'verifier'

    def run_phase_write(self, depends_on: List[str]) -> Optional[str]:
        """
        Phase 6: 技术写作

        创新成立时启动写作Agent
        """
        print("\n" + "="*60)
        print("Phase 6: 技术写作")
        print("="*60)

        result = self.spawn_agent(
            agent_id='writer',
            role='technical_writer',
            depends_on=depends_on,
            trigger_condition='innovation_established'
        )

        if result['status'] == 'blocked':
            print(f"  Blocked: {result['reason']}")
            return None

        print(f"  Spawned: writer")
        return 'writer'

    def check_termination(self) -> bool:
        """检查是否应该终止"""
        if self.state['no_improvement_count'] >= self.max_no_improvement_rounds:
            print(f"\n{'='*60}")
            print(f"连续{self.max_no_improvement_rounds}轮无提升，触发终止条件")
            print(f"{'='*60}")
            self.state['terminated'] = True
            self._save_state()
            return True

        if self.state['terminated']:
            return True

        return False

    def run_iteration(self) -> bool:
        """
        执行一次完整迭代

        Returns:
        --------
        bool : 是否继续迭代（False表示终止）
        """
        self.state['round'] += 1
        print(f"\n{'#'*60}")
        print(f"第 {self.state['round']} 轮迭代")
        print(f"{'#'*60}")

        # Phase 1: 并行下载
        dl_ids = self.run_phase_download()

        # Phase 2: 文献分析
        analyzer_id = self.run_phase_analyze(dl_ids)

        # Phase 3: 方案设计
        designer_id = self.run_phase_design(['analyzer'])

        # Phase 4: 代码实现
        engineer_id = self.run_phase_code(['designer'])

        # Phase 5: 测试验证
        verifier_id = self.run_phase_test(['engineer'])

        # Phase 6: 创新判定
        print("\n" + "="*60)
        print("Phase 6: 创新判定")
        print("="*60)

        innovation_established = self.evaluate_innovation()

        if innovation_established:
            self.state['innovation_established'] = True
            self._save_state()
            print("  创新成立！")

            # Phase 7: 技术写作
            writer_id = self.run_phase_write(['verifier'])
            if writer_id:
                print(f"  技术写作Agent已启动: {writer_id}")
            return False  # 创新成立，流程完成
        else:
            self.state['no_improvement_count'] += 1
            self._save_state()
            print(f"  创新不足，已连续{self.state['no_improvement_count']}轮无提升")

            if self.check_termination():
                return False

            # 打回重设
            print("  打回方案设计师重新设计...")
            self.reset_for_new_iteration()

        return True

    def run(self, max_iterations: int = 10) -> Dict:
        """
        运行完整工作流

        注意：这个方法生成 spawn 指令，
        实际的 Agent spawn 需要在主会话中通过 Agent 工具执行

        使用方式：
        1. 初始化 orchestrator
        2. 调用 run() 获取本轮 spawn 计划
        3. 在主会话中使用 Agent 工具执行 spawn
        4. 使用 mark_agent_completed/failed 更新状态
        5. 循环直到 terminated
        """
        print("\n" + "="*60)
        print("PM2.5 CMAQ融合方法自动研究系统启动 (Agent Spawn Mode)")
        print("="*60)
        print(f"项目目录: {self.project_root}")
        print(f"状态文件: {self.state_file}")
        print(f"最大迭代: {max_iterations}")

        self._save_state()

        # 生成执行计划
        plan = {
            'total_rounds': 0,
            'spawn_sequence': [],
            'state': self.state,
        }

        # 迭代
        iteration = 0
        while iteration < max_iterations:
            if self.state.get('terminated'):
                break

            iteration += 1
            self.state['iteration_count'] = iteration

            # Phase 1-6 spawn 计划
            round_plan = {
                'round': self.state['round'] + 1,
                'phases': {}
            }

            # 下载并行
            round_plan['phases']['download'] = {
                'agents': [f'dl_{i}' for i in range(1, 4)],
                'parallel': True,
            }

            # 分析顺序
            round_plan['phases']['analyze'] = {
                'agent': 'analyzer',
                'depends_on': [f'dl_{i}' for i in range(1, 4)],
            }

            # 设计顺序
            round_plan['phases']['design'] = {
                'agent': 'designer',
                'depends_on': ['analyzer'],
            }

            # 代码顺序
            round_plan['phases']['code'] = {
                'agent': 'engineer',
                'depends_on': ['designer'],
            }

            # 测试顺序
            round_plan['phases']['test'] = {
                'agent': 'verifier',
                'depends_on': ['engineer'],
            }

            # 写作（条件触发）
            round_plan['phases']['write'] = {
                'agent': 'writer',
                'depends_on': ['verifier'],
                'trigger': 'innovation_established',
            }

            plan['spawn_sequence'].append(round_plan)
            plan['total_rounds'] += 1

            # 这里应该由调用者执行实际的 spawn
            # 暂时返回计划，实际执行需要主会话配合

        plan['final_state'] = self.state

        print(f"\n生成了 {plan['total_rounds']} 轮迭代计划")
        print("请在主会话中执行 spawn")

        return plan


def generate_spawn_script(project_root: str) -> str:
    """
    生成 spawn 执行脚本

    这个脚本用于在主会话中执行真正的 Agent spawn
    """
    orchestrator = AgentSpawnOrchestrator(project_root)

    script = f'''#!/usr/bin/env python
"""
Agent Spawn 执行脚本
===================
自动执行工作流中的所有 Agent spawn

使用方式：
    python agents/spawn_executor.py

注意：实际 spawn 仍需要在 Claude Code 主会话中通过 Agent 工具执行
'''

# ===== 执行计划 =====

## Round 1

# Phase 1: 并行下载 (3个Agent)
!!SPAWN_AGENT:bg:id=dl_1!!
role: literature_downloader
prompt: |
  {orchestrator.state['agents']}
  你是一个专业的学术论文搜索专家。
  任务：搜索PM2.5 CMAQ数据融合相关论文并下载。
  输出：{project_root}/PaperDownload/
!!

!!SPAWN_AGENT:bg:id=dl_2!!
role: literature_downloader
prompt: |
  [同 dl_1，不同关键词] ...

!!SPAWN_AGENT:bg:id=dl_3!!
role: literature_downloader
prompt: |
  [同 dl_1，不同关键词] ...

# 等待下载完成
!!AGENT_WAIT:dl_1,dl_2,dl_3!!

# Phase 2: 文献分析
!!SPAWN_AGENT:wait_id=analyzer!!
role: literature_analyzer
depends_on: [dl_1, dl_2, dl_3]
prompt: |
  你是一个资深的空气质量数据融合研究专家。
  任务：分析论文，提炼融合方法。
  输入：{project_root}/LocalPaperLibrary/, {project_root}/PaperDownload/
  输出：{project_root}/MethodToSmart/
!!

# ... (后续Phase类似)
'''

    return script


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1:
        project_root = sys.argv[1]
    else:
        project_root = 'str(get_project_root())'

    print(f"初始化项目: {project_root}")

    orchestrator = AgentSpawnOrchestrator(project_root)

    print("\n当前状态:")
    print(json.dumps(orchestrator.state, indent=2, ensure_ascii=False))

    print("\nAgent 配置:")
    for name, config in orchestrator.agents_config.items():
        print(f"  {name}: {config['role']} (parallel={config.get('parallel', False)})")

    print("\n使用方法:")
    print("  1. 在 Claude Code 主会话中:")
    print("     from agents.workflow_orchestrator import AgentSpawnOrchestrator")
    print("     orchestrator = AgentSpawnOrchestrator(project_root)")
    print("     plan = orchestrator.run()")
    print("")
    print("  2. 使用 Agent 工具 spawn 子代理:")
    print("     Agent(tool_call, prompt=plan['spawn_sequence'][0]['phases']['download'])")
