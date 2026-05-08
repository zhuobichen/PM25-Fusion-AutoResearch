"""
文献下载员 Agent
================
职责：搜索与PM2.5 CMAQ融合相关的论文，下载并存入PaperDownload/；
      对论文进行相关度评分（1-5分），生成清单并存入PaperDownloadMd/。

过滤规则：
- 仅保留：CMAQ / PM2.5 / 数据融合 / 空间插值
- 过滤掉：不相关领域、多源融合（含卫星AOD/气象/土地利用等额外数据）
"""

import os
from shared.paths import get_project_root, data_path
import json
from datetime import datetime

class LiteratureDownloader:
    """文献下载员 Agent"""

    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.paper_download_dir = os.path.join(root_dir, 'PaperDownload')
        self.paper_md_dir = os.path.join(root_dir, 'PaperDownloadMd')
        self.paper_md_file = os.path.join(self.paper_md_dir, 'paper_list.md')

        # 确保目录存在
        os.makedirs(self.paper_download_dir, exist_ok=True)
        os.makedirs(self.paper_md_dir, exist_ok=True)

        # 搜索关键词
        self.search_keywords = [
            'PM2.5 CMAQ data fusion',
            'PM2.5 spatial interpolation monitoring data',
            'CMAQ model data fusion air quality',
            'PM2.5 downscaling kriging'
        ]

        # 领域过滤
        self.allowed_domains = ['CMAQ', 'PM2.5', '数据融合', '空间插值']
        self.excluded_keywords = ['AOD', 'satellite', 'meteorological', 'land use', '土地利用', '气象', '卫星']

    def search_papers(self, keyword, max_results=10):
        """
        搜索论文（调用WebSearch）
        返回论文列表
        """
        # TODO: 调用WebSearch搜索
        pass

    def download_paper(self, paper_info):
        """
        下载论文到PaperDownload/
        """
        # TODO: 下载PDF
        pass

    def score_relevance(self, paper_title, abstract):
        """
        相关度评分（1-5分）
        5分：直接相关，方法明确
        4分：高度相关，略有扩展
        3分：部分相关，可参考
        2分：弱相关
        1分：不相关
        """
        score = 3  # 默认

        # 检查标题关键词
        title_lower = paper_title.lower()
        if 'pm2.5' in title_lower and ('cmaq' in title_lower or 'fusion' in title_lower or 'downscaler' in title_lower):
            score = 5
        elif 'pm2.5' in title_lower and 'cmaq' in title_lower:
            score = 5
        elif 'pm2.5' in title_lower and 'kriging' in title_lower:
            score = 4
        elif 'air quality' in title_lower and 'model' in title_lower:
            score = 3

        # 检查是否排除
        for kw in self.excluded_keywords:
            if kw.lower() in abstract.lower():
                score = max(1, score - 2)

        return score

    def generate_markdown_list(self, papers):
        """
        生成论文清单Markdown文档
        """
        md_content = f"""# 论文清单

生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 评分说明
- 5分：直接相关，方法明确
- 4分：高度相关，略有扩展
- 3分：部分相关，可参考
- 2分：弱相关
- 1分：不相关

---

"""

        for i, paper in enumerate(papers, 1):
            md_content += f"""## [{i}] {paper['title']}

- **作者/年份**：{paper.get('authors', 'Unknown')} / {paper.get('year', 'Unknown')}
- **相关度评分**：{paper['score']}分
- **领域标签**：{', '.join(paper.get('domains', []))}
- **论文简介**：{paper.get('abstract', 'N/A')}
- **下载状态**：{paper.get('download_status', '未下载')}

---

"""

        with open(self.paper_md_file, 'w', encoding='utf-8') as f:
            f.write(md_content)

        return self.paper_md_file

    def run(self):
        """
        执行文献下载流程
        """
        print("=== [文献下载员] 开始工作 ===")

        # 1. 搜索论文
        all_papers = []
        for keyword in self.search_keywords:
            papers = self.search_papers(keyword)
            all_papers.extend(papers)

        # 2. 去重
        seen_titles = set()
        unique_papers = []
        for p in all_papers:
            if p['title'] not in seen_titles:
                seen_titles.add(p['title'])
                unique_papers.append(p)

        # 3. 评分
        for paper in unique_papers:
            paper['score'] = self.score_relevance(paper['title'], paper.get('abstract', ''))

        # 4. 过滤（评分>=2才保留）
        filtered_papers = [p for p in unique_papers if p['score'] >= 2]

        # 5. 下载
        for paper in filtered_papers:
            self.download_paper(paper)

        # 6. 生成清单
        self.generate_markdown_list(filtered_papers)

        print(f"=== [文献下载员] 完成 ===")
        print(f"下载论文数：{len(filtered_papers)}")
        print(f"清单文件：{self.paper_md_file}")

        return {
            'papers_downloaded': len(filtered_papers),
            'paper_list_file': self.paper_md_file
        }


if __name__ == '__main__':
    root_dir = str(get_project_root())
    agent = LiteratureDownloader(root_dir)
    result = agent.run()
