"""
方案设计师 Agent
================
职责：
1. 复现现有融合方法（生成复现指令）
2. 提出创新融合方法（生成创新指令 + innovation_note + 方法指纹）

创新判定：
- 新颖性：指纹与已有方法不重复
- 优越性：R²提升 ≥ 0.01，RMSE ≤ 最优基准
"""

import os
from shared.paths import get_project_root, data_path
import hashlib
import json
from datetime import datetime

class MethodDesigner:
    """方案设计师 Agent"""

    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.method_input_dir = os.path.join(root_dir, 'MethodToSmart')
        self.output_dir = os.path.join(root_dir, 'SmartToCode')
        self.code_ref_dir = os.path.join(root_dir, 'Code')

        # 输出子目录
        self.reproduce_dir = os.path.join(self.output_dir, '复现方法指令')
        self.innovation_dir = os.path.join(self.output_dir, '创新方法指令')

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.reproduce_dir, exist_ok=True)
        os.makedirs(self.innovation_dir, exist_ok=True)

        # 方法指纹库
        self.fingerprint_file = os.path.join(self.output_dir, 'method_fingerprint.md5')
        self.fingerprint_db = self._load_fingerprint_db()

        # 已有方法指纹（VNA/eVNA/aVNA/Downscaler）
        self.baseline_fingerprints = [
            'vna_md5',  # TODO: 实际计算
            'evna_md5',
            'avna_md5',
            'downscaler_md5'
        ]

    def _load_fingerprint_db(self):
        """加载指纹库"""
        if os.path.exists(self.fingerprint_file):
            with open(self.fingerprint_file, 'r') as f:
                return json.load(f)
        return {}

    def _save_fingerprint_db(self):
        """保存指纹库"""
        with open(self.fingerprint_file, 'w') as f:
            json.dump(self.fingerprint_db, f, indent=2)

    def compute_fingerprint(self, formula, steps):
        """
        计算方法指纹
        格式：MD5(核心公式字符串 + 关键步骤结构化描述)
        """
        content = f"{formula}|{steps}"
        return hashlib.md5(content.encode()).hexdigest()

    def check_fingerprint(self, fingerprint):
        """
        检查指纹是否重复
        """
        # 检查已有方法
        if fingerprint in self.baseline_fingerprints:
            return False, "与基准方法重复"

        # 检查指纹库
        if fingerprint in self.fingerprint_db:
            return False, "与已有方法重复"

        return True, "指纹唯一"

    def read_method_documents(self):
        """
        读取MethodToSmart/下的所有方法文档
        """
        docs = []
        if os.path.exists(self.method_input_dir):
            for f in os.listdir(self.method_input_dir):
                if f.endswith('.md'):
                    with open(os.path.join(self.method_input_dir, f), 'r', encoding='utf-8') as fp:
                        docs.append({
                            'filename': f,
                            'content': fp.read()
                        })
        return docs

    def generate_reproduce_instruction(self, method_info):
        """
        生成复现方法指令
        """
        instruction = f"""# 复现方法指令

## 方法名称
{method_info.get('name', 'Unknown')}

## 文献来源
{method_info.get('source', 'N/A')}

## 复现要求
1. 保持方法核心逻辑不变
2. 适配系统输入格式（netCDF/CSV）
3. 支持十折交叉验证
4. 输出到CodeWorkSpace/复现方法代码/

## 核心公式
```
{method_info.get('formula', 'N/A')}
```

## 关键步骤
{method_info.get('steps', 'N/A')}

## 输入输出格式
- 输入：监测数据CSV + CMAQ netCDF
- 输出：融合网格PM2.5

生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

        return instruction

    def generate_innovation_instruction(self, method_info):
        """
        生成创新方法指令
        """
        # 检查指纹
        fingerprint = self.compute_fingerprint(
            method_info.get('formula', ''),
            method_info.get('steps', '')
        )
        is_unique, msg = self.check_fingerprint(fingerprint)

        if not is_unique:
            return None, f"指纹重复：{msg}"

        # 生成指令
        instruction = f"""# 创新方法指令

## 方法名称
{method_info.get('name', 'Unknown')}

## 创新核心
{method_info.get('innovation_core', 'N/A')}

## 核心公式
$$
{method_info.get('formula', 'N/A')}
$$

## 关键步骤
{method_info.get('steps', 'N/A')}

## 创新优势
- R²提升预期：≥ 0.01
- 相比现有方法的改进点：{method_info.get('improvement', 'N/A')}

## 方法指纹
MD5: {fingerprint}

## 输入输出格式
- 输入：监测数据CSV + CMAQ netCDF
- 输出：融合网格PM2.5
- 支持十折交叉验证

生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

        # 保存指纹
        self.fingerprint_db[fingerprint] = {
            'name': method_info.get('name', 'Unknown'),
            'created': datetime.now().isoformat()
        }
        self._save_fingerprint_db()

        return instruction, "创新方法可用"

    def generate_innovation_note(self, method_info):
        """
        生成创新自评文档 innovation_note.md
        """
        fingerprint = self.compute_fingerprint(
            method_info.get('formula', ''),
            method_info.get('steps', '')
        )

        note = f"""# 创新自评文档

## 方法名称
{method_info.get('name', 'Unknown')}

## ① 核心差异
{method_info.get('core_difference', 'N/A')}

## ② 预期更优原因
{method_info.get('expected_better_reason', 'N/A')}

## ③ 无人提及依据
{method_info.get('novelty_basis', 'N/A')}

## ④ 方法指纹
MD5: {fingerprint}

## 创新判定
- 新颖性：指纹与已有方法不重复 ✓
- 优越性：R²提升 ≥ 0.01，RMSE ≤ 最优基准 待验证

生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

        note_file = os.path.join(self.output_dir, 'innovation_note.md')
        with open(note_file, 'w', encoding='utf-8') as f:
            f.write(note)

        return note_file

    def run(self):
        """
        执行方案设计流程
        """
        print("=== [方案设计师] 开始工作 ===")

        # 1. 读取方法文档
        method_docs = self.read_method_documents()
        print(f"读取方法文档数：{len(method_docs)}")

        # 2. 生成复现指令
        reproduce_count = 0
        for doc in method_docs:
            # TODO: 解析文档内容，识别方法类型
            method_info = {
                'name': doc['filename'],
                'source': 'from literature',
                'formula': 'VNA formula',
                'steps': '1. compute weights 2. interpolate'
            }
            instruction = self.generate_reproduce_instruction(method_info)
            # TODO: 保存指令文件
            reproduce_count += 1

        # 3. 生成创新指令
        innovation_count = 0
        for doc in method_docs:
            # TODO: 分析是否可以创新
            method_info = {
                'name': f"Innovation_{doc['filename']}",
                'formula': 'new formula',
                'steps': '1. new step 2. new step',
                'innovation_core': 'improved interpolation',
                'improvement': 'dynamic weighting'
            }
            instruction, msg = self.generate_innovation_instruction(method_info)
            if instruction:
                self.generate_innovation_note(method_info)
                innovation_count += 1
            else:
                print(f"  跳过：{msg}")

        print(f"=== [方案设计师] 完成 ===")
        print(f"复现方法指令数：{reproduce_count}")
        print(f"创新方法指令数：{innovation_count}")

        return {
            'reproduce_count': reproduce_count,
            'innovation_count': innovation_count,
            'fingerprint_db': self.fingerprint_db
        }


if __name__ == '__main__':
    root_dir = str(get_project_root())
    agent = MethodDesigner(root_dir)
    result = agent.run()
