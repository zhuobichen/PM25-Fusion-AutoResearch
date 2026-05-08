"""
文献分析员 Agent
================
职责：读取LocalPaperLibrary/本地论文和PaperDownload/下载论文，
      提炼融合方法，生成结构化方法文档到MethodToSmart/

结构化输出要求：
- 使用【可执行方法规范】模板
- 包含：输入数据、输出数据、核心公式、关键参数、实现步骤
"""

import os
from shared.paths import get_project_root, data_path
import hashlib
from datetime import datetime

class LiteratureAnalyzer:
    """文献分析员 Agent"""

    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.local_paper_dir = os.path.join(root_dir, 'LocalPaperLibrary')
        self.paper_download_dir = os.path.join(root_dir, 'PaperDownload')
        self.paper_md_dir = os.path.join(root_dir, 'PaperDownloadMd')
        self.method_output_dir = os.path.join(root_dir, 'MethodToSmart')

        os.makedirs(self.method_output_dir, exist_ok=True)

        # 可执行方法规范模板
        self.template = """# 【可执行方法规范】

## 方法名称
[中文名] ([英文缩写])

## 文献来源
- 论文标题：《{title}》
- 作者/年份：{authors} / {year}年
- 关键章节：P.{page} / Section {section}

## 核心公式
$$
y_{{fused}} = \\text{{{formula}}}
$$

## 参数清单
{param_table}

## 数据规格

### 输入
{input_table}

### 输出
{output_table}

## 随机性
- [ ] 是（需设随机种子）
- [x] 否（确定性方法）

## 方法指纹
MD5: {md5_hash}

## 实现检查清单
- [ ] 核心公式已验证
- [ ] 边界条件已处理
- [ ] 单元测试通过
"""

    def read_local_papers(self):
        """
        读取LocalPaperLibrary/下所有PDF
        """
        papers = []
        if os.path.exists(self.local_paper_dir):
            for f in os.listdir(self.local_paper_dir):
                if f.endswith('.pdf'):
                    papers.append({
                        'path': os.path.join(self.local_paper_dir, f),
                        'title': f.replace('.pdf', ''),
                        'source': 'local'
                    })
        return papers

    def read_downloaded_papers(self):
        """
        读取PaperDownload/的论文（按相关度优先级）
        """
        papers = []
        md_file = os.path.join(self.paper_md_dir, 'paper_list.md')

        if os.path.exists(md_file):
            # 读取清单，按评分排序
            # TODO: 解析Markdown获取评分
            pass

        if os.path.exists(self.paper_download_dir):
            for f in os.listdir(self.paper_download_dir):
                if f.endswith('.pdf'):
                    papers.append({
                        'path': os.path.join(self.paper_download_dir, f),
                        'title': f.replace('.pdf', ''),
                        'source': 'downloaded'
                    })

        return papers

    def extract_method_from_paper(self, paper_path):
        """
        从论文提取融合方法
        TODO: 需要调用PDF解析
        """
        # 读取PDF内容
        # 识别方法章节
        # 提取公式、参数、数据规格
        pass

    def generate_method_document(self, paper_info, method_content):
        """
        生成结构化方法文档
        """
        filename = f"文献分析员_{paper_info['title'][:30]}_{datetime.now().strftime('%Y%m%d')}.md"
        filepath = os.path.join(self.method_output_dir, filename)

        doc = self.template.format(
            title=paper_info.get('title', 'Unknown'),
            authors=paper_info.get('authors', 'Unknown'),
            year=paper_info.get('year', 'Unknown'),
            page=paper_info.get('page', 'N/A'),
            section=paper_info.get('section', 'N/A'),
            formula=method_content.get('formula', 'N/A'),
            param_table=method_content.get('param_table', '| 参数名 | 类型 | 默认值 | 说明 |\n|-------|------|-------|------|'),
            input_table=method_content.get('input_table', '| 数据 | 格式 | 维度 | 单位 |\n|-----|------|-----|------|'),
            output_table=method_content.get('output_table', '| 数据 | 格式 | 单位 |\n|-----|------|------|'),
            md5_hash=method_content.get('md5', 'N/A')
        )

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(doc)

        return filepath

    def compute_fingerprint(self, formula, steps):
        """
        计算方法指纹
        """
        content = f"{formula}|{steps}"
        return hashlib.md5(content.encode()).hexdigest()

    def run(self):
        """
        执行文献分析流程
        """
        print("=== [文献分析员] 开始工作 ===")

        # 1. 读取本地论文（全读）
        local_papers = self.read_local_papers()
        print(f"本地论文数：{len(local_papers)}")

        # 2. 读取下载论文（按优先级）
        downloaded_papers = self.read_downloaded_papers()
        print(f"下载论文数：{len(downloaded_papers)}")

        # 3. 合并并按优先级排序
        all_papers = local_papers + downloaded_papers
        # TODO: 按评分排序

        # 4. 提取方法并生成文档
        output_files = []
        for paper in all_papers:
            method_content = self.extract_method_from_paper(paper['path'])
            if method_content:
                output_file = self.generate_method_document(paper, method_content)
                output_files.append(output_file)

        print(f"=== [文献分析员] 完成 ===")
        print(f"生成方法文档数：{len(output_files)}")

        return {
            'methods_extracted': len(output_files),
            'output_files': output_files
        }


if __name__ == '__main__':
    root_dir = str(get_project_root())
    agent = LiteratureAnalyzer(root_dir)
    result = agent.run()
