<div align="center">

# PM2.5 · CMAQ Data Fusion

**自动化 PM2.5 数据融合研究系统**

`文献分析` → `方案设计` → `代码实现` → `测试验证` → `论文生成`

---

[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-Academic-4CAF50?style=for-the-badge)](#)

---

</div>

## Quick Start

```bash
# 查看流水线状态
python run_pipeline.py --status

# 一键运行（跳过文献下载，Agent 自动串联执行）
python run_pipeline.py --auto --agent --skip 1

# 只跑验证（直接执行 Python 脚本）
python run_pipeline.py --auto --only 5

# 使用预置配置
python run_pipeline.py --auto --agent --profile skip-download
```

---

## Pipeline Architecture

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Phase 2    │     │   Phase 3    │     │   Phase 4    │     │   Phase 5    │
│   文献分析    │────→│   方案设计    │────→│   代码实现    │────→│   测试验证    │
│   Claude CLI  │     │   Claude CLI  │     │   Claude CLI  │     │  Python 直跑  │
└──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
       │                    │                    │                    │
       ▼                    ▼                    ▼                    ▼
  MethodToSmart/       SmartToCode/        CodeWorkSpace/        test_result/
```

每个 Phase 通过 `claude -p` 启动独立 Claude 子进程，自动读取前序产出，无需手动干预。

---

## Phase Reference

| Phase | Name | Method | Output |
|:-----:|------|--------|--------|
| 0 | 项目整理 | `claude -p` | `INVENTORY.md` |
| 1 | 文献下载 | `claude -p` | `PaperDownload/` |
| 2 | 文献分析 | `claude -p` | `MethodToSmart/` |
| 3 | 方案设计 | `claude -p` | `SmartToCode/` |
| 4 | 代码实现 | `claude -p` | `CodeWorkSpace/` |
| 5 | 测试验证 | Python 脚本 | `test_result/` |
| 6 | 论文写作 | `claude -p` | `paper_output/` |

---

## Execution Modes

| Mode | Command | Use Case |
|------|---------|----------|
| **Guidance** | `--auto --skip 1` | 只打印执行指引 |
| **Agent** | `--auto --agent --skip 1` | Claude CLI 自动执行 |
| **Verify** | `--auto --only 5` | 直接运行验证脚本 |

---

## Key Results

### Baseline Methods (VNA Best)

| Stage | Period | R² | RMSE | \|MB\| | 主级阈值 |
|:-----:|--------|:---:|:----:|:------:|:--------:|
| pre_exp | Jan 1-5 2020 | 0.8907 | 16.68 | 0.70 | 0.9007 |
| stage1 | Jan 2020 | 0.9034 | 16.48 | 0.50 | 0.9134 |
| stage2 | Jul 2020 | 0.8408 | 5.05 | 0.05 | 0.8508 |
| stage3 | Dec 2020 | 0.9031 | 12.20 | 0.42 | 0.9131 |

### Innovative Methods (Four-Stage Verified)

| Method | Stage1 R² | Stage2 R² | Stage3 R² | Kernel | Status |
|--------|:---------:|:---------:|:---------:|--------|:------:|
| **AdvancedRK** | 0.9162 | 0.8526 | 0.9129 | GPR-Matérn | ✅ Best |
| **PolyRK** | 0.9105 | 0.8474 | 0.9060 | GPR-RBF | ✅ Core |

### Innovation Criteria

**主级创新** — 三条件必须同时满足：

| Metric | Requirement |
|--------|-------------|
| R² | ≥ best baseline R² + 0.01 |
| RMSE | ≤ best baseline RMSE |
| \|MB\| | ≤ best baseline \|MB\| |

**次级创新** — R² > baseline（只需大于基线，无需 +0.01）

**验证流程**：
```
pre_exp → stage1 → stage2 → stage3
  ↓         ↓        ↓        ↓
 失败     失败     继续    主级创新
        停止      ↓
                stage3通过→次级创新
                stage3未通过→创新失败
```

---

## Method Inventory

### Baseline Methods (4)

| Method | Location | Description |
|--------|----------|-------------|
| VNA | `Code/VNAeVNAaVNA/` | Voronoi Neighbor Average (spatial interpolation) |
| eVNA | `Code/VNAeVNAaVNA/` | 乘法偏差校正 |
| aVNA | `Code/VNAeVNAaVNA/` | 加法偏差校正 |
| Downscaler | `Code/Downscaler/` | MCMC 降尺度 |

### Reproduced Methods (25+)

来自 `LocalPaperLibrary/` 中 12 篇论文的融合方法复现，存放在 `CodeWorkSpace/复现方法代码/`。

| Category | Methods | Typical R² |
|----------|---------|:----------:|
| Spatial Kriging | SpatialKrigingBC, Cokriging | 0.55 ~ 0.58 |
| Model Aggregation | OMA, SMA, MMA | 0.30 ~ 0.49 |
| Bias Correction | QuantileMapping, ODI | 0.08 ~ 0.09 |

### Innovative Methods (30+)

已设计并实现的创新方法，存放在 `CodeWorkSpace/新融合方法代码/`。

| Method | Status | R² Range | Innovation |
|--------|:------:|:--------:|------------|
| **AdvancedRK** | ✅ Verified | 0.85 ~ 0.92 | GPR-Matérn kernel, best overall |
| **PolyRK** | ✅ Verified | 0.85 ~ 0.91 | Polynomial OLS + GPR-RBF |
| RobustRK | Partial | ~0.91 | Robust variant |
| MSEF | Tested | ~0.81 | Multi-source ensemble |
| Others | Various | — | 20+ methods in pipeline |

---

## Project Structure

```
Data_Fusion_AutoResearch/
├── run_pipeline.py                 # 流水线入口
├── pipeline_config.json            # Profile 配置
├── CLAUDE.md                       # Claude Code 项目说明
├── INVENTORY.md                    # 项目总清单
│
├── Code/                           # 参考实现
│   ├── VNAeVNAaVNA/                #   VNA / eVNA / aVNA
│   └── Downscaler/                 #   MCMC 降尺度
│
├── CodeWorkSpace/
│   ├── 复现方法代码/                #   25+ 复现方法
│   ├── 新融合方法代码/              #   30+ 创新方法
│   └── 年均融合方法/                #   年均数据融合
│
├── MethodToSmart/                  #   41 个方法文档（文献分析员输出）
├── SmartToCode/                    #   55+ 个设计指令（方案设计师输出）
│
├── PaperDownload/                  #   ~100 篇论文 PDF（按 score 分类）
├── PaperDownloadMd/                #   论文清单与分析报告
├── LocalPaperLibrary/              #   12 篇中文论文原文
│
├── test_data/
│   ├── raw/CMAQ/                   #   CMAQ 模型输出 (NetCDF)
│   ├── raw/Monitor/                #   地面监测数据
│   └── fold_split_table*.csv       #   十折交叉验证分配
│
├── test_result/
│   ├── 基准方法/                    #   基准方法验证结果
│   ├── 创新方法/                    #   创新方法验证结果
│   └── comparison_report.md        #   全方法对比报告
│
├── Innovation/
│   ├── success/                    #   已确认创新方法 (AdvancedRK, PolyRK, ...)
│   └── failed/                     #   验证失败方法 (GARK, PG-STGAT, ...)
│
├── paper_output/
│   ├── paper.tex                   #   论文主文件
│   ├── paper.pdf                   #   编译后 PDF
│   └── references.bib              #   参考文献
│
├── agents/                         #   Agent 模块
│   ├── role_templates.py           #     6 个 Agent 角色 Prompt
│   ├── spawn_executor.py           #     Agent spawn 执行器
│   └── workflow_orchestrator.py    #     工作流编排
│
└── shared/                         #   共享工具
    ├── paths.py                    #     路径解析
    ├── metrics.py                  #     评估指标
    └── geo_utils.py                #     地理空间工具
```

---

## CLI Reference

```
python run_pipeline.py [OPTIONS]

Phase Selection:
  --skip N,N,...          跳过指定 Phase
  --from N                从 Phase N 开始
  --to N                  到 Phase N 结束
  --only N,N,...          只运行指定 Phase
  --profile NAME          使用预置配置

Agent Mode:
  --agent                 启用 Claude CLI 自动执行
  --budget USD            每个 Phase 最大花费
  --model MODEL           指定 Claude 模型
  --timeout SEC           每个 Phase 超时（默认无限制）

Profile Management:
  --list-profiles         显示所有配置
  --save-profile NAME     保存当前选择为配置

Status:
  --status                查看流水线状态
  --reset                 重置流水线状态
```

---

## Profiles

预置在 `pipeline_config.json` 中：

| Profile | Phases | Description |
|---------|--------|-------------|
| `full` | 0-6 | 完整流程 |
| `skip-download` | 0, 2-6 | 已有文献，跳过下载 |
| `design-verify` | 3-5 | 方案设计 + 验证循环 |
| `verify-only` | 5 | 只跑验证 |
| `code-iterate` | 4-5 | 编码 + 验证迭代 |

```bash
python run_pipeline.py --auto --agent --profile skip-download
python run_pipeline.py --save-profile my-flow --skip 1,6
```

---

## Dependencies

```
numpy  pandas  scikit-learn  netCDF4  joblib
```

Agent 模式需要 [Claude Code CLI](https://claude.ai/code)。

---

<div align="center">

**Academic Research Use**

湖南大学 · 机械与运载工程学院

</div>
