# INVENTORY.md - paper_output 目录清单

生成时间：2026-05-08

## 已有文件

| 文件 | 大小 | 最后修改 | 说明 |
|------|------|----------|------|
| paper.tex | 16KB | 2026-04-14 | 主论文LaTeX源码（基于PRK方法） |
| paper.pdf | 224KB | 2026-04-14 | 编译后的PDF（旧版本） |
| paper_backup.tex | 25KB | 2026-04-09 | 旧版备份 |
| references.bib | 4.5KB | 2026-04-14 | BibTeX参考文献 |
| tech_report.md | 11KB | 2026-04-10 | 技术报告 |
| README.md | 11KB | 2026-04-15 | 项目说明 |
| build.bat | 246B | 2026-04-09 | 编译脚本 |
| figures/ | - | 2026-04-10 | 图表目录（含comparison.png） |

## 已有论文版本分析

### paper_output/paper.tex（旧版）
- 方法名称：PRK (Polynomial Residual Kriging)
- 核函数：RBF
- 最佳R²：0.9128 (stage1)
- 验证阶段：pre_exp + stage1 + stage3（stage2未通过）
- 状态：需更新为AdvancedRK方法

### Innovation/success/AdvancedRK/paper.tex（新版）
- 方法名称：AdvancedRK
- 核函数：Matern (ν=1.5)
- 最佳R²：0.9143 (stage1)
- 验证阶段：4/4全部通过
- 状态：内容更完整，应作为主版本

## 最新实验结果（来源：AdvancedRK_all_stages.json）

| 阶段 | R² | MAE | RMSE | MB | 判定 |
|------|-----|------|------|-----|------|
| pre_exp | 0.9015 | 10.07 | 15.84 | -0.05 | 通过 |
| stage1 | 0.9143 | 8.87 | 15.51 | -0.14 | 通过 |
| stage2 | 0.8501 | 3.32 | 4.90 | -0.04 | 通过 |
| stage3 | 0.9104 | 7.08 | 11.73 | 0.01 | 通过 |

## VNA基准（来源：benchmark_multistage.json）

| 阶段 | R² | MAE | RMSE | MB |
|------|------|------|------|------|
| pre_exp | 0.8907 | 10.32 | 16.68 | 0.70 |
| stage1 | 0.9034 | 9.07 | 16.48 | 0.50 |
| stage2 | 0.8408 | 3.39 | 5.05 | 0.05 |
| stage3 | 0.9031 | 7.22 | 12.20 | 0.42 |

## 更新计划

本次更新将：
1. 替换 paper.tex 为 AdvancedRK 版本（更新R²数据为最新JSON结果）
2. 更新 references.bib 补充缺失引用
3. 重新编译 paper.pdf
