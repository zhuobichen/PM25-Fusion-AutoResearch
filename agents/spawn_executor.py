from shared.paths import get_project_root, data_path
"""
Agent Spawn 执行器
================
在主会话中实际执行 Agent spawn 的脚本

使用方式：
    from agents.spawn_executor import SpawnExecutor
    executor = SpawnExecutor(project_root)

    # Phase 1: 并行下载
    executor.phase1_download()

    # Phase 2: 文献分析
    executor.phase2_analyze()

    # ... 后续 Phase

注意：实际 spawn 需要在 Claude Code 主会话中通过 Agent 工具执行
"""

import os
import json
import glob
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional

from agents.role_templates import get_spawn_prompt


class SpawnExecutor:
    """
    Agent Spawn 执行器

    管理工作流状态，并在主会话中协调 Agent spawn
    自动集成 StateTracker 更新研究状态
    """

    def __init__(self, project_root: str):
        self.project_root = project_root
        self.state_file = os.path.join(project_root, '.agent_state.json')
        self.error_dir = os.path.join(project_root, 'error')
        self.test_result_dir = os.path.join(project_root, 'test_result', '创新方法')

        # 确保目录存在
        os.makedirs(self.error_dir, exist_ok=True)

        self.state = self._load_state()

        # 初始化 StateTracker
        try:
            from agents.research_state_tracker import StateTracker
            state_tracker_dir = os.path.join(project_root, 'test_result', '.state')
            self.state_tracker = StateTracker(state_tracker_dir)
        except Exception as e:
            print(f"  [警告] StateTracker 初始化失败: {e}")
            self.state_tracker = None

    def _load_state(self) -> Dict:
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
        self.state['last_run'] = datetime.now().isoformat()
        with open(self.state_file, 'w', encoding='utf-8') as f:
            json.dump(self.state, f, indent=2, ensure_ascii=False)

    def get_agent_prompt(self, role: str) -> str:
        """获取指定角色的 spawn prompt"""
        return get_spawn_prompt(role, self.project_root)

    def spawn(self, agent_id: str, role: str, background: bool = False) -> Dict:
        """
        Spawn 一个 Agent

        返回 spawn 信息，供主会话使用 Agent 工具调用
        """
        prompt = self.get_agent_prompt(role)

        self.state['agents'][agent_id] = {
            'role': role,
            'status': 'spawned',
            'spawned_at': datetime.now().isoformat(),
            'background': background,
        }
        self._save_state()

        return {
            'agent_id': agent_id,
            'role': role,
            'background': background,
            'prompt': prompt,
        }

    def mark_completed(self, agent_id: str, result: any = None):
        """标记 Agent 完成，并自动触发 StateTracker 更新"""
        if agent_id in self.state['agents']:
            self.state['agents'][agent_id]['status'] = 'completed'
            self.state['agents'][agent_id]['completed_at'] = datetime.now().isoformat()
            if result:
                self.state['agents'][agent_id]['result'] = result
            self._save_state()

        # 自动更新 StateTracker（verifier 阶段）
        if agent_id == 'verifier' and self.state_tracker:
            self._auto_update_state_tracker()

    def _auto_update_state_tracker(self):
        """自动读取最新测试结果并更新 StateTracker"""
        if not os.path.exists(self.test_result_dir):
            return

        try:
            # 查找最新的测试结果文件
            csv_files = glob.glob(os.path.join(self.test_result_dir, '*_summary.csv'))
            if not csv_files:
                return

            # 读取所有 summary 文件，提取最新方法的结果
            latest_results = []
            for f in csv_files:
                df = pd.read_csv(f)
                for _, row in df.iterrows():
                    latest_results.append({
                        'method': row.get('method', ''),
                        'R2': row.get('R2', 0),
                        'MAE': row.get('MAE', 0),
                        'RMSE': row.get('RMSE', 0)
                    })

            if not latest_results:
                return

            # 找 R² 最高的方法作为当前最佳
            latest_results.sort(key=lambda x: x['R2'], reverse=True)
            best = latest_results[0]

            method_name = best['method']
            metrics = {
                'R2': float(best['R2']),
                'MAE': float(best['MAE']),
                'RMSE': float(best['RMSE'])
            }

            # 开始新迭代
            self.state_tracker.start_iteration(method_name)
            self.state_tracker.update_metrics(metrics, method_name)

            # 检查是否应该接受（需要对比历史最佳）
            current_best_r2 = self.state_tracker.state.current_best_metrics.get('R2', 0)
            improvement = metrics['R2'] - current_best_r2

            # 判断是否属于应排除的 Stacking 类方法
            stacking_keywords = ['Stacking', 'Ensemble', 'StackingEnsemble', 'SuperStacking']
            is_stacking = any(kw in method_name for kw in stacking_keywords)

            if is_stacking:
                # Stacking 类方法直接拒绝
                self.state_tracker.reject_mutation(
                    method_name=method_name,
                    metrics=metrics,
                    reason='Stacking类权重组合方法应排除（无物理可解释性）'
                )
                print(f"  [StateTracker] {method_name}: 标记为排除（Stacking类）")
            elif improvement >= 0.01:
                # 有显著提升，接受
                self.state_tracker.accept_mutation(method_name, metrics)
                print(f"  [StateTracker] {method_name}: R²={metrics['R2']:.4f} ✅ 接受")
            else:
                # 提升不足但非Stacking，记录但不接受
                self.state_tracker.state.iteration -= 1  # 不算有效迭代
                print(f"  [StateTracker] {method_name}: R²={metrics['R2']:.4f}, Δ={improvement:+.4f} (<0.01)")

            # 生成简短报告
            state = self.state_tracker.get_current_state()
            print(f"  [StateTracker] 当前最佳: {state['current_best_method']} (R²={state['current_best_r2']:.4f})")

        except Exception as e:
            print(f"  [StateTracker] 更新失败: {e}")

    def mark_failed(self, agent_id: str, error: str):
        """标记 Agent 失败"""
        if agent_id in self.state['agents']:
            self.state['agents'][agent_id]['status'] = 'failed'
            self.state['agents'][agent_id]['failed_at'] = datetime.now().isoformat()
            self.state['agents'][agent_id]['error'] = error
            self._save_state()

    def wait_and_check(self, agent_ids: List[str]) -> bool:
        """检查所有 Agent 是否完成（基于状态文件，非阻塞）"""
        for agent_id in agent_ids:
            state = self.state['agents'].get(agent_id, {}).get('status')
            if state != 'completed':
                return False
        return True

    def get_pending_agents(self) -> List[str]:
        """获取所有未完成的 Agent ID 列表"""
        pending = []
        for agent_id, info in self.state['agents'].items():
            if info.get('status') not in ['completed', 'failed']:
                pending.append(agent_id)
        return pending

    def is_any_running(self) -> bool:
        """检查是否有任何 Agent 还在运行"""
        for info in self.state['agents'].values():
            if info.get('status') not in ['completed', 'failed']:
                return True
        return False

    def get_state(self) -> Dict:
        """获取当前状态"""
        return self.state

    def verify_agent_output(self, agent_id: str, output_paths: List[str]) -> bool:
        """
        健康检查：验证 Agent 是否真正产生了预期输出

        Parameters:
        -----------
        agent_id : str
            Agent ID
        output_paths : List[str]
            预期输出文件路径列表

        Returns:
        --------
        bool : 所有文件都存在返回 True，否则返回 False
        """
        missing = []
        for path in output_paths:
            if not os.path.exists(path):
                missing.append(path)

        if missing:
            print(f"  [警告] {agent_id} 缺少输出文件:")
            for p in missing:
                print(f"    - {p}")
            return False
        return True

    def retry_agent(self, agent_id: str, role: str, max_retries: int = 3) -> Dict:
        """
        重试失败的 Agent

        Parameters:
        -----------
        agent_id : str
            Agent ID
        role : str
            角色名
        max_retries : int
            最大重试次数

        Returns:
        --------
        Dict : spawn 结果
        """
        # 标记当前为失败
        self.mark_failed(agent_id, "需要重试")

        for attempt in range(max_retries):
            print(f"  [重试] {agent_id} 第 {attempt + 1} 次重试...")

            # 生成新的 agent_id
            new_agent_id = f"{agent_id}_retry_{attempt + 1}"
            result = self.spawn(new_agent_id, role, background=True)

            # 更新原 agent_id 的重试信息
            if agent_id in self.state['agents']:
                self.state['agents'][agent_id]['retry_count'] = attempt + 1
                self.state['agents'][agent_id]['latest_retry_id'] = new_agent_id
                self._save_state()

            return result

        print(f"  [错误] {agent_id} 重试 {max_retries} 次后仍失败")
        return None

    def check_and_retry(self, agent_id: str, role: str, output_paths: List[str], max_retries: int = 3) -> Dict:
        """
        检查 Agent 输出，必要时重试

        Parameters:
        -----------
        agent_id : str
            Agent ID
        role : str
            角色名
        output_paths : List[str]
            预期输出文件路径
        max_retries : int
            最大重试次数

        Returns:
        --------
        Dict : spawn 结果（如果是新的 retry）
        """
        if self.verify_agent_output(agent_id, output_paths):
            return None  # 输出正常，不需要重试

        print(f"  [触发重试] {agent_id} 输出验证失败")
        return self.retry_agent(agent_id, role, max_retries)

    # ====== Phase 执行方法 ======

    def phase0_organize(self) -> Dict:
        """
        Phase 0: 项目整理
        进入项目后首先执行，整理前人遗留，生成盘点报告
        """
        print("\n" + "="*60)
        print("Phase 0: 项目整理")
        print("="*60)

        result = self.spawn('organizer', 'organizer', background=True)
        print(f"  [准备 Spawn] organizer (background=True)")
        print(f"    Role: organizer")
        print(f"    Prompt 长度: {len(result['prompt'])} chars")

        return result

    def phase1_download(self) -> Dict[str, Dict]:
        """
        Phase 1: 并行下载
        Spawn 3个下载Agent
        """
        print("\n" + "="*60)
        print("Phase 1: 并行文献下载")
        print("="*60)

        results = {}
        for i in range(1, 4):
            agent_id = f'dl_{i}'
            result = self.spawn(agent_id, 'literature_downloader', background=True)
            results[agent_id] = result
            print(f"  [准备 Spawn] {agent_id} (background=True)")
            print(f"    Role: literature_downloader")
            print(f"    Prompt 长度: {len(result['prompt'])} chars")

        print(f"\n  请在主会话中使用 Agent 工具执行上述 Agent")
        print(f"  执行完成后调用 mark_completed() 更新状态")

        return results

    def phase2_analyze(self) -> Dict:
        """
        Phase 2: 文献分析
        等下载完成后执行
        """
        print("\n" + "="*60)
        print("Phase 2: 文献分析")
        print("="*60)

        # 检查下载是否完成
        if not self.wait_and_check(['dl_1', 'dl_2', 'dl_3']):
            print("  错误: 下载阶段未完成")
            return None

        result = self.spawn('analyzer', 'literature_analyzer', background=True)
        print(f"  [准备 Spawn] analyzer (background=True)")
        print(f"    Role: literature_analyzer")
        print(f"    Prompt 长度: {len(result['prompt'])} chars")

        return result

    def phase3_design(self) -> Dict:
        """
        Phase 3: 方案设计
        等分析完成后执行
        """
        print("\n" + "="*60)
        print("Phase 3: 方案设计")
        print("="*60)

        if not self.wait_and_check(['analyzer']):
            print("  错误: 分析阶段未完成")
            return None

        result = self.spawn('designer', 'method_designer', background=True)
        print(f"  [准备 Spawn] designer (background=True)")
        print(f"    Role: method_designer")
        print(f"    Prompt 长度: {len(result['prompt'])} chars")

        return result

    def phase4_code(self) -> Dict:
        """
        Phase 4: 代码实现
        等设计完成后执行
        """
        print("\n" + "="*60)
        print("Phase 4: 代码实现")
        print("="*60)

        if not self.wait_and_check(['designer']):
            print("  错误: 设计阶段未完成")
            return None

        result = self.spawn('engineer', 'code_engineer', background=True)
        print(f"  [准备 Spawn] engineer (background=True)")
        print(f"    Role: code_engineer")
        print(f"    Prompt 长度: {len(result['prompt'])} chars")

        return result

    def phase5_test(self) -> Dict:
        """
        Phase 5: 测试验证
        等代码完成后执行
        """
        print("\n" + "="*60)
        print("Phase 5: 测试验证")
        print("="*60)

        if not self.wait_and_check(['engineer']):
            print("  错误: 代码阶段未完成")
            return None

        result = self.spawn('verifier', 'test_verifier', background=True)
        print(f"  [准备 Spawn] verifier (background=True)")
        print(f"    Role: test_verifier")
        print(f"    Prompt 长度: {len(result['prompt'])} chars")

        return result

    def phase6_write(self) -> Optional[Dict]:
        """
        Phase 6: 技术写作
        创新成立时执行
        """
        print("\n" + "="*60)
        print("Phase 6: 技术写作")
        print("="*60)

        if not self.state.get('innovation_established'):
            print("  跳过: 创新未成立")
            return None

        if not self.wait_and_check(['verifier']):
            print("  错误: 验证阶段未完成")
            return None

        result = self.spawn('writer', 'technical_writer', background=True)
        print(f"  [准备 Spawn] writer (background=True)")
        print(f"    Role: technical_writer")
        print(f"    Prompt 长度: {len(result['prompt'])} chars")

        return result

    def skip_phase1(self):
        """
        跳过 Phase 1（文献下载）
        当论文已足够时使用
        """
        print("\n" + "="*60)
        print("Phase 1: 跳过（论文已足够）")
        print("="*60)

        # 直接标记下载完成
        for agent_id in ['dl_1', 'dl_2', 'dl_3']:
            self.state['agents'][agent_id] = {
                'role': 'literature_downloader',
                'status': 'completed',
                'completed_at': datetime.now().isoformat(),
                'skipped': True
            }
        self._save_state()
        print("  已跳过下载阶段，所有下载Agent标记为完成")

    def run_all(self, skip_download: bool = False, max_iterations: int = 1):
        """
        多米诺骨牌式自动执行
        检查当前状态，自动执行可以继续的下一个 Phase
        返回下一个 Agent 的 spawn 信息
        """
        # Phase 0: 整理（如果还没做）
        if not self.wait_and_check(['organizer']):
            print("\n[Phase 0] 执行项目整理...")
            return self.phase0_organize()

        # Phase 1: 下载 或 跳过
        if not self.wait_and_check(['dl_1']):
            if skip_download:
                self.skip_phase1()
            else:
                print("\n[Phase 1] 执行文献下载...")
                return self.phase1_download()

        # Phase 2: 分析
        if not self.wait_and_check(['analyzer']):
            print("\n[Phase 2] 执行文献分析...")
            return self.phase2_analyze()

        # Phase 3: 设计
        if not self.wait_and_check(['designer']):
            print("\n[Phase 3] 执行方案设计...")
            return self.phase3_design()

        # Phase 4: 代码
        if not self.wait_and_check(['engineer']):
            print("\n[Phase 4] 执行代码实现...")
            return self.phase4_code()

        # Phase 5: 测试
        if not self.wait_and_check(['verifier']):
            print("\n[Phase 5] 执行测试验证...")
            return self.phase5_test()

        # Phase 6: 写作（如果创新成立）
        if self.state.get('innovation_established'):
            if not self.wait_and_check(['writer']):
                print("\n[Phase 6] 执行论文写作...")
                return self.phase6_write()

        # 所有 Phase 完成
        print("\n" + "="*60)
        print("所有 Phase 已完成！")
        print("="*60)
        print(f"当前最佳方法: {self.state_tracker.get_current_state()['current_best_method']}")
        print(f"R²: {self.state_tracker.get_current_state()['current_best_r2']:.4f}")
        return None

    def trigger_next(self):
        """
        触发下一个可执行的 Phase
        供外部调用（如在 Agent prompt 里调用 Bash 触发）

        用法：
          在 Agent prompt 末尾添加：
          完成后退出的理由：
          exit_reason: "任务完成"

          然后在当前会话执行：
          python -c "from agents.spawn_executor import SpawnExecutor; SpawnExecutor('.').trigger_next()"
        """
        result = self.run_all()
        if result:
            print("\n" + "="*60)
            print("下一步 Agent 信息:")
            print("="*60)
            print(f"Agent ID: {result['agent_id']}")
            print(f"Role: {result['role']}")
            print(f"\nPrompt ({len(result['prompt'])} chars):")
            print("-"*40)
            print(result['prompt'][:500] + "..." if len(result['prompt']) > 500 else result['prompt'])
            print("-"*40)
        return result

    def get_status(self) -> Dict:
        """获取当前工作流状态"""
        status = {
            'round': self.state.get('round', 0),
            'agents': {},
            'innovation_established': self.state.get('innovation_established', False)
        }

        for agent_id, info in self.state.get('agents', {}).items():
            status['agents'][agent_id] = {
                'role': info.get('role', ''),
                'status': info.get('status', ''),
                'completed_at': info.get('completed_at', ''),
                'skipped': info.get('skipped', False)
            }

        return status


def print_spawn_guide():
    """打印 Agent spawn 执行指南"""
    guide = """
================================================================================
                    Agent Spawn 执行指南
================================================================================

【核心概念】

  Claude Code 的 Agent 工具可以 spawn 子 Agent独立运行
  主会话负责任务协调，子 Agent负责具体执行

【执行模式】

  1. 初始化执行器
  2. 调用 Phase 方法获取 spawn 信息
  3. 使用 Agent 工具 spawn 子 Agent
  4. 子 Agent 完成后调用 mark_completed()
  5. 继续下一个 Phase

【代码示例】

  from agents.spawn_executor import SpawnExecutor

  executor = SpawnExecutor(project_root)

  # Phase 1: 并行下载
  spawns = executor.phase1_download()
  for agent_id, info in spawns.items():
      # 使用 Agent 工具 spawn
      Agent(tool_call="...", prompt=info['prompt'])

  # ... 等待完成后

  # Phase 2: 文献分析
  executor.mark_completed('dl_1')
  executor.mark_completed('dl_2')
  executor.mark_completed('dl_3')

  info = executor.phase2_analyze()
  Agent(tool_call="...", prompt=info['prompt'])

【Agent 工具调用格式】

  Agent(
      description="文献下载 Agent dl_1",
      prompt="你是一个专业的学术论文搜索专家。..."
  )

【注意事项】

  - 后台 Agent (background=True) 可以并行运行
  - 前台 Agent (background=False) 顺序执行
  - 每个 Phase 完成后需要调用 mark_completed()
  - 状态保存在 .agent_state.json

================================================================================
"""
    print(guide)


if __name__ == '__main__':
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from shared.paths import get_project_root
    project_root = str(get_project_root())

    print(f"初始化项目: {project_root}")
    executor = SpawnExecutor(project_root)

    print("\n当前状态:")
    print(json.dumps(executor.state, indent=2, ensure_ascii=False))

    print_spawn_guide()
