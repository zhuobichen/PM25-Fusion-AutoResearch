"""
代码工程师 Agent
================
职责：
1. 改造VNA/eVNA/aVNA/Downscaler代码适配系统输入格式
2. 实现复现方法代码
3. 实现创新方法代码

关键要求：
- 日期对齐由代码工程师负责
- 语义确认后开始实现
- 每次修改记录到WorkDocument/
"""

import os
from shared.paths import get_project_root, data_path
import json
from datetime import datetime

class CodeEngineer:
    """代码工程师 Agent"""

    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.code_ref_dir = os.path.join(root_dir, 'Code')
        self.workspace_dir = os.path.join(root_dir, 'CodeWorkSpace')
        self.work_doc_dir = os.path.join(root_dir, 'CodeWorkSpace/WorkDocument')
        self.error_dir = os.path.join(root_dir, 'error')

        # 子目录
        self.benchmark_code_dir = os.path.join(self.workspace_dir, '基准方法代码')
        self.reproduce_code_dir = os.path.join(self.workspace_dir, '复现方法代码')
        self.innovation_code_dir = os.path.join(self.workspace_dir, '新融合方法代码')

        os.makedirs(self.workspace_dir, exist_ok=True)
        os.makedirs(self.work_doc_dir, exist_ok=True)
        os.makedirs(self.error_dir, exist_ok=True)

    def semantic_confirmation(self, instruction_content):
        """
        语义确认环节
        代码工程师复述理解，系统自动验证

        确认内容：
        - 物理意义
        - 输入输出格式
        - 核心公式
        - 关键步骤
        """
        confirmation_template = f"""## 语义确认记录

日期：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
方法：[待填写]

代码工程师复述：
- 物理意义：[待填写]
- 输入输出：[待填写]
- 核心公式：[待填写]
- 关键步骤：[待填写]

系统验证：✓ 通过 / ✗ 不通过
"""

        # TODO: 解析instruction_content，复述理解
        # 自动验证四项是否完整

        return confirmation_template

    def adapt_benchmark_methods(self):
        """
        改造VNA/eVNA/aVNA代码适配系统输入格式
        """
        print("=== [代码工程师] 改造基准方法代码 ===")

        # 1. 读取原始Code/VNAeVNAaVNA代码
        # 2. 修改输入部分（适配netCDF/CSV格式）
        # 3. 输出到CodeWorkSpace/基准方法代码/

        output = os.path.join(self.benchmark_code_dir, 'benchmark_adapter.py')

        return {'status': 'done', 'output': output}

    def implement_reproduce_method(self, instruction_file):
        """
        实现复现方法代码
        """
        print(f"=== [代码工程师] 实现复现方法：{instruction_file} ===")

        # 1. 读取指令
        # 2. 按指令实现代码
        # 3. 记录改动到WorkDocument/

        output = os.path.join(self.reproduce_code_dir, f'reproduce_{datetime.now().strftime("%Y%m%d")}.py')

        # 记录改动
        self.log_change('reproduce', instruction_file, output)

        return {'status': 'done', 'output': output}

    def implement_innovation_method(self, instruction_file):
        """
        实现创新方法代码
        """
        print(f"=== [代码工程师] 实现创新方法：{instruction_file} ===")

        # 1. 语义确认
        # 2. 读取指令
        # 3. 按指令实现代码
        # 4. 记录改动到WorkDocument/

        output = os.path.join(self.innovation_code_dir, f'innovation_{datetime.now().strftime("%Y%m%d")}.py')

        # 记录改动
        self.log_change('innovation', instruction_file, output)

        return {'status': 'done', 'output': output}

    def log_change(self, method_type, instruction_file, output_file):
        """
        记录代码改动到WorkDocument/
        """
        log = f"""# 改动记录

## 日期
{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 方法类型
{method_type}

## 依据指令
{instruction_file}

## 输出文件
{output_file}

## 改动说明
[待填写]

---

"""
        log_file = os.path.join(self.work_doc_dir, f'{method_type}_改动记录_{datetime.now().strftime("%Y%m%d")}.md')
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(log)

        return log_file

    def date_alignment(self, selected_day):
        """
        日期对齐校验
        - 读取CMAQ netCDF和监测CSV
        - 检查指定日期是否存在
        - 对齐失败记录到error/
        """
        print(f"=== [代码工程师] 日期对齐校验：{selected_day} ===")

        from CodeWorkSpace.改造后VNA_eVNA_aVNA.benchmark_methods import DataLoader

        dl = DataLoader(self.root_dir)

        try:
            # 检查CMAQ数据
            grid_lons, grid_lats, pred_pm25, base_pm25, time_data = dl.load_cmaq_data(selected_day)
            print(f"  CMAQ数据：{pred_pm25.shape}")

            # 检查监测数据
            monitor_df = dl.load_monitor_data(selected_day)
            print(f"  监测站点数：{len(monitor_df)}")

            return {'status': 'ok', 'day': selected_day}

        except Exception as e:
            # 记录错误
            error_file = os.path.join(self.error_dir, f'date_alignment_{datetime.now().strftime("%Y%m%d")}.log')
            with open(error_file, 'w') as f:
                f.write(f"日期对齐失败：{selected_day}\n错误：{str(e)}\n")

            return {'status': 'error', 'day': selected_day, 'error': str(e)}

    def run(self, task='all'):
        """
        执行代码工程师任务
        """
        print("=== [代码工程师] 开始工作 ===")

        if task in ['all', 'benchmark']:
            self.adapt_benchmark_methods()

        if task in ['all', 'reproduce']:
            # 查找复现指令
            reproduce_dir = os.path.join(self.root_dir, 'SmartToCode/复现方法指令')
            if os.path.exists(reproduce_dir):
                for f in os.listdir(reproduce_dir):
                    if f.endswith('.md'):
                        self.implement_reproduce_method(os.path.join(reproduce_dir, f))

        if task in ['all', 'innovation']:
            # 查找创新指令
            innovation_dir = os.path.join(self.root_dir, 'SmartToCode/创新方法指令')
            if os.path.exists(innovation_dir):
                for f in os.listdir(innovation_dir):
                    if f.endswith('.md'):
                        self.implement_innovation_method(os.path.join(innovation_dir, f))

        print("=== [代码工程师] 完成 ===")

        return {'status': 'done'}


if __name__ == '__main__':
    root_dir = str(get_project_root())
    agent = CodeEngineer(root_dir)
    result = agent.run()
