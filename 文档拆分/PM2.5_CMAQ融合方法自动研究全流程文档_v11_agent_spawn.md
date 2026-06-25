# PM2.5 CMAQ融合方法自动研究全流程文档

> **版本**：v11（整合v10.1完整内容 + v11更新）
>
> **⚠️ v14 更新说明**：本文档为 v11 历史版本。v14 已将架构从 SpawnExecutor 链式触发改为
> TaskDispatcher + ArtifactDetector 闭环工作流。具体变更参见：
> - `文档拆分/01_项目概览与核心原则.md`（TaskDispatcher API）
> - `文档拆分/03_Agent_Spawn工作流.md`（闭环工作流图）
> - `文档拆分/05_运行与异常处理.md`（闭环主循环）
> - `文档拆分/06_系统目录结构.md`（agents/ 目录更新）
> - `README.md`（v14.0 完整架构说明）
>
> 本文档保留下方 v11 原始内容作为历史参考，不再作为最新规范。
>
> ---
>
> **核心原则**：**禁止停下来问问题**，遇到任何情况都必须自主决策或记录后继续
>
> **核心变化**：v11 基于v10.1，整合了所有更新：
> 1. 修正多阶段验证方式：每天十折验证，合并所有天预测计算整体R²
> 2. 明确4个基准方法：VNA, eVNA, aVNA, Downscaler
> 3. 增加初始化检查阶段（Phase X）
> 4. 要求基准方法提供完整4项指标
> 5. 保留v10.1完整内容（Agent Spawn详细职责、快照机制、Scout-Reflection闭环等）

---

## 关联文档

| 文档 | 位置 | 说明 |
|------|------|------|
| **十折交叉验证架构文档** | `E:/CodeProject/MEMORY/Agent工作流/十折交叉验证架构文档.md` | 十折验证核心架构、两种模式、标准/特例区分 |
| **创新排除文档** | `E:/CodeProject/MEMORY/Agent工作流/PM2.5_CMAQ融合方法创新排除.md` | Stacking等方法排除原则、创新成立条件 |

---

# 一、文档说明

## 1.1 核心目标

基于现有代码、本地原始论文及下载的论文，自动：
1. 复现现有融合方法
2. 提出全新且更优的融合方法
3. 与基准方法进行公平对比
4. **输出最优融合方法**
5. **将研究成果整理成学术论文（LaTeX → PDF）**

## 1.2 最高原则

```
┌────────────────────────────────────────────────────────────────┐
│                    ⚠️ 最高原则 ⚠️                               │
├────────────────────────────────────────────────────────────────┤
│  【禁止】停下来问问题                                           │
│                                                                 │
│  遇到任何情况（包括数据格式不熟、方法不清晰、不确定是否正确）：   │
│  → A) 自主做出合理假设并继续                                     │
│  → B) 记录到 error/xxx_YYYYMMDD.log 后继续                     │
│  → C) 使用默认值/行业标准做法继续                               │
│                                                                 │
│  【绝对禁止】说"我需要确认"、"请告诉我"、"等一下"               │
└────────────────────────────────────────────────────────────────┘
```

## 1.3 整理优先原则（v9.1 新增）

```
┌────────────────────────────────────────────────────────────────┐
│              ⚠️ 整理优先原则 ⚠️（v9.1 新增）                    │
├────────────────────────────────────────────────────────────────┤
│  【新任务开始前】                                                │
│                                                                 │
│  1. 先检查当前目录下的已有文件                                   │
│  2. 整理/规范化文件格式（符合项目文档模板规范）                  │
│  3. 生成文件清单和状态报告                                       │
│  4. 基于已有工作继续创新，而不是从零开始                          │
│                                                                 │
│  【禁止】直接覆盖或忽略已有文件                                  │
│  【禁止】从零开始创建重复内容                                    │
└────────────────────────────────────────────────────────────────┘
```

### 整理流程

```
新任务开始
    ↓
[Step 1] 扫描已有文件
    │
    └─→ 遍历项目目录，记录所有文件
    ↓
[Step 2] 生成 INVENTORY.md
    │
    └─→ 格式：
    ```markdown
    # 项目文件清单

    ## 目录结构
    [tree]

    ## 方法文档 (MethodToSmart/)
    | 文件名 | 方法名 | 日期 | 状态 |

    ## 代码文件 (CodeWorkSpace/)
    | 文件名 | 方法名 | 日期 | 状态 |

    ## 测试结果 (test_result/)
    | 文件名 | 方法名 | 日期 | R² | 状态 |
    ```
    ↓
[Step 3] 规范化检查
    │
    ├─ 文件名是否符合规范？
    ├─ 内容格式是否符合模板？
    ├─ 是否有重复/冗余文件？
    └─ 标记需要整理的文件
    ↓
[Step 4] 整理报告
    │
    └─→ 输出：INVENTORY.md + 整理建议.md
    ↓
[Step 5] 基于已有工作继续
    │
    └─→ 对接已有成果，继续创新研究
```

### 规范文件格式参考

| 文件类型 | 规范格式 | 存放位置 |
|----------|----------|----------|
| 方法文档 | 文献分析员_[方法名].md | MethodToSmart/ |
| 方案指令 | V1_[方法名].md / Innovation_[方法名].md | SmartToCode/复现方法指令/ / SmartToCode/创新方法指令/ |
| 代码文件 | [方法名].py | CodeWorkSpace/复现方法代码/ / CodeWorkSpace/新融合方法代码/ |
| 测试结果 | [方法名]_summary.csv | test_result/复现方法/ / test_result/创新方法/ |
| 对比报告 | comparison_report.md | test_result/ |

## 1.4 Agent Spawn 模式说明

### 架构对比

```
传统模式（错误）：                      Agent Spawn 模式（正确）：
┌─────────────────┐                  ┌─────────────────────────┐
│  主会话          │                  │  主Agent（任务协调者）     │
│  ├─ 文献下载     │                  │      │                   │
│  ├─ 文献分析     │                  │      ├─ dl_1 (并行)      │
│  ├─ 方案设计     │    ===改写==>    │      ├─ dl_2 (并行)      │
│  ├─ 代码实现     │                  │      ├─ dl_3 (并行)      │
│  ├─ 测试验证     │                  │      ├─ analyzer         │
│  └─ 技术写作     │                  │      ├─ designer         │
└─────────────────┘                  │      ├─ engineer         │
                                      │      ├─ verifier         │
                                      │      └─ writer           │
                                      └─────────────────────────┘
```

### 实际执行方式

Agent Spawn 通过 Claude Code 的 **Agent 工具** 实现。

**执行流程：**

```
1. 初始化 SpawnExecutor
   └─ executor = SpawnExecutor(project_root)

2. 调用 Phase 方法
   └─ spawns = executor.phase1_download()
   └─ 返回需要 spawn 的 Agent 列表

3. 在主会话中使用 Agent 工具 spawn
   └─ Agent(description="...", prompt=spawns['dl_1']['prompt'])

4. 子 Agent 完成后更新状态
   └─ executor.mark_completed('dl_1')

5. 继续下一个 Phase
```

**关键文件：**

| 文件 | 作用 |
|------|------|
| `agents/spawn_executor.py` | Agent spawn 执行器 |
| `agents/role_templates.py` | 各角色 prompt 模板 |
| `.agent_state.json` | 工作流状态文件 |

**SpawnExecutor API：**

```python
from agents.spawn_executor import SpawnExecutor

executor = SpawnExecutor(project_root)

# Phase 执行
executor.phase0_organize()  # 整理前人遗留（最先执行）
executor.phase1_download()  # 返回 spawn 信息
executor.phase2_analyze()
executor.phase3_design()
executor.phase4_code()
executor.phase5_test()
executor.phase6_write()

# 状态管理
executor.mark_completed(agent_id)   # 标记完成
executor.mark_failed(agent_id, err) # 标记失败
executor.get_state()                # 获取状态
```

### 触发词规范（原设计，已由 SpawnExecutor 替代）

| 触发词 | 含义 | 对应 SpawnExecutor 方法 |
|--------|------|------------------------|
| `!!SPAWN_AGENT:role!!` | 启动指定角色Agent | `spawn(role)` |
| `!!SPAWN_AGENT:bg:id=xxx!!` | 后台启动（并行） | `spawn(background=True)` |
| `!!AGENT_WAIT:ids!!` | 等待指定Agent完成 | `wait_and_check()` |
| `!!CHECK_TRIGGER:condition!!` | 检查触发条件 | `state['innovation_established']` |

---

# 二、初始化检查阶段（v11 新增）

## 2.1 目的

在工作流开始前，检查是否具备启动研究所需的基础设施，特别是**基准方法的完整指标**。

## 2.2 检查清单

```
┌────────────────────────────────────────────────────────────────┐
│                  初始化检查清单                                  │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  【P0 - 必须通过】                                               │
│                                                                 │
│  □ 1. 基准方法代码存在                                          │
│       └─ CodeWorkSpace/基准方法代码/ 目录下应有：                │
│          - VNA.py                                               │
│          - eVNA.py                                               │
│          - aVNA.py                                               │
│          - Downscaler.py                                         │
│                                                                 │
│  □ 2. 基准方法指标完整                                           │
│       └─ test_result/基准方法/benchmark_summary.csv 必须包含：  │
│          - VNA:  R², RMSE, MAE, MB                             │
│          - eVNA: R², RMSE, MAE, MB                             │
│          - aVNA: R², RMSE, MAE, MB                             │
│          - Downscaler: R², RMSE, MAE, MB                        │
│                                                                 │
│  □ 3. 十折划分表存在                                            │
│       └─ test_data/fold_split_table.csv                         │
│                                                                 │
│  □ 4. CMAQ数据文件存在                                          │
│       └─ test_data/raw/CMAQ/2020_PM25.nc                        │
│                                                                 │
│  □ 5. 监测站数据存在                                             │
│       └─ test_data/raw/Monitor/2020_DailyPM2.5Monitor.csv      │
│                                                                 │
│  【P1 - 建议通过】                                               │
│                                                                 │
│  □ 6. 创新排除规则文档存在                                       │
│  □ 7. 十折交叉验证架构文档存在                                   │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

## 2.3 基准方法指标要求

### 指标定义

| 指标 | 公式 | 说明 |
|------|------|------|
| **R²** | 1 - Σ(y-ŷ)² / Σ(y-ȳ)² | 决定系数，越接近1越好 |
| **MAE** | Σ|y-ŷ| / n | 平均绝对误差，越小越好 |
| **RMSE** | √(Σ(y-ŷ)² / n) | 均方根误差，越小越好 |
| **MB** | Σ(ŷ-y) / n | 平均偏差，接近0越好 |

### 基准方法当前参考值（2020-01-01单天）

| 方法 | R² | RMSE | MAE | MB |
|------|-----|------|-----|-----|
| CMAQ | -0.038 | 29.25 | 20.47 | -3.24 |
| VNA | 0.800 | 12.86 | 7.75 | 0.76 |
| aVNA | 0.794 | 13.03 | 8.10 | 0.10 |
| **eVNA** | **0.810** | **12.52** | **7.99** | **0.08** |
| Downscaler | 0.806 | 12.64 | 8.19 | 1.85 |

**注意**：上表为2020-01-01单天结果，各阶段的真实基准需要通过多阶段验证建立。

### 基准方法多阶段验证要求

基准方法必须在各阶段验证中提供**完整的4项指标**（R², RMSE, MAE, MB），用于：
1. 建立各阶段的真实基准线
2. 判断创新方法的RMSE和MB是否满足要求
3. 确保创新是"全面提升"而非"R²提升但RMSE下降"

## 2.4 初始化检查流程

```
项目启动
    ↓
检查 test_result/基准方法/benchmark_summary.csv 是否存在
    ↓
    ├─ 不存在 → 先跑基准方法多阶段验证
    │              ↓
    │           输出：test_result/基准方法/benchmark_summary.csv
    │              ↓
    │           包含：VNA, eVNA, aVNA, Downscaler 的 R², RMSE, MAE, MB
    │
    └─ 存在 → 检查指标是否完整（4方法 × 4指标 = 16项）
                ↓
                ├─ 不完整 → 补充验证缺失的指标
                └─ 完整 → 通过检查，继续主流程
```

## 2.5 初始化检查代码模板

```python
def check_initialization(project_root):
    """初始化检查"""
    issues = []

    # 1. 检查基准方法代码
    benchmark_codes = ['VNA.py', 'eVNA.py', 'aVNA.py', 'Downscaler.py']
    for code in benchmark_codes:
        path = f'{project_root}/CodeWorkSpace/基准方法代码/{code}'
        if not os.path.exists(path):
            issues.append(f'缺少基准方法代码: {code}')

    # 2. 检查基准方法指标
    benchmark_csv = f'{project_root}/test_result/基准方法/benchmark_summary.csv'
    if not os.path.exists(benchmark_csv):
        issues.append('缺少基准方法指标文件 benchmark_summary.csv')
    else:
        df = pd.read_csv(benchmark_csv)
        required_methods = ['CMAQ', 'VNA', 'aVNA', 'eVNA', 'Downscaler']
        required_metrics = ['R2', 'RMSE', 'MAE', 'MB']
        for method in required_methods:
            if method not in df['method'].values:
                issues.append(f'缺少基准方法: {method}')
        for metric in required_metrics:
            if metric not in df.columns:
                issues.append(f'缺少指标: {metric}')

    # 3. 检查数据文件
    data_files = [
        'test_data/fold_split_table.csv',
        'test_data/raw/CMAQ/2020_PM25.nc',
        'test_data/raw/Monitor/2020_DailyPM2.5Monitor.csv'
    ]
    for f in data_files:
        if not os.path.exists(f'{project_root}/{f}'):
            issues.append(f'缺少数据文件: {f}')

    return issues  # 空列表 = 通过
```

---

# 三、系统目录结构

```
E:\CodeProject\ClaudeRoom\Data_Fusion_AutoResearch\
├── PaperDownload/              # 论文PDF文件（按评分分类）
│   ├── score_5/              # 5分论文（直接相关）
│   ├── score_4/              # 4分论文（高度相关）
│   └── score_3/              # 3分论文（部分相关）
├── PaperDownloadMd/            # 论文清单 paper_list.json
├── LocalPaperLibrary/          # 本地原始论文库
├── MethodToSmart/              # 文献分析员输出（文献分析员_*.md）
├── SmartToCode/               # 方案设计师输出
│   ├── 复现方法指令/V1_方法名.md
│   ├── 创新方法指令/V1_方法名.md
│   └── innovation_note.md
├── Code/                      # 参考代码
│   ├── VNA_eVNA_aVNA/
│   └── Downscaler/
├── CodeWorkSpace/             # 工作区
│   ├── 基准方法代码/           # 【新增】VNA, eVNA, aVNA, Downscaler
│   ├── 复现方法代码/
│   └── 新融合方法代码/
├── test_data/                 # 测试数据
│   ├── fold_split_table.csv   # 十折交叉验证划分
│   └── selected_days.txt      # 测试日期列表
├── test_result/               # 测试结果【规范化目录结构】
│   ├── INVENTORY.md          # 结果清单（仅此一份）
│   ├── .state/               # 状态追踪目录
│   │   ├── ledger.jsonl      # 决策记录（每次决策追加）
│   │   └── research_status.md # 研究状态报告
│   ├── 基准方法/               # 【规范化】
│   │   ├── benchmark_summary.csv      # 基准方法指标
│   │   └── benchmark_multistage.json   # 基准方法多阶段验证结果
│   ├── 复现方法/
│   │   └── reproduction_summary.csv
│   ├── 创新方法/
│   │   ├── innovation_summary.csv  # 所有创新方法汇总
│   │   └── [方法名]_summary.csv    # 各方法详细结果
│   ├── 历史最佳方案/          # 当前最优方案
│   │   ├── best_method.py
│   │   └── best_metrics.json
│   ├── comparison_report.md   # 对比报告（最终版）
│   └── FINAL_REPORT.md       # 最终报告（最终版）
├── Innovation/                 # 【新增】已确认创新方法
│   ├── INVENTORY.md          # 创新方法总清单（成功+失败汇总）
│   ├── success/              # 已确认创新的方法
│   │   └── [方法名]/
│   │       ├── INVENTORY.md  # 方法详情
│   │       ├── [验证数据].json
│   │       ├── [验证代码].py
│   │       ├── paper.tex
│   │       ├── paper.pdf
│   │       └── references.bib
│   └── failed/               # 创新未通过的方法（避免重复踩坑）
│       └── [方法名]/
│           ├── INVENTORY.md  # 失败原因、失败阶段
│           ├── [验证代码].py
│           └── result.json
├── paper_output/              # 论文写作输出（技术写作Agent）
│   ├── paper.tex             # LaTeX源码
│   ├── paper.pdf             # 编译后PDF
│   ├── references.bib         # 参考文献
│   └── figures/              # 图表
├── agents/                    # Agent 相关模块
│   ├── role_templates.py     # Agent spawn prompt 模板
│   ├── workflow_orchestrator.py # 工作流编排器
│   ├── literature_downloader.py
│   ├── literature_analyzer.py
│   ├── method_designer.py
│   ├── code_engineer.py
│   ├── test_verifier.py
│   └── workflow_orchestrator.py
├── .agent_state.json          # 工作流状态文件（运行时生成）
└── error/                     # 错误日志（必须记录所有异常）
```

### 三.1 根目录文件规范（v10 新增）

**根目录只允许存在以下文件/目录**：

| 文件/目录 | 类型 | 说明 |
|-----------|------|------|
| `INVENTORY.md` | 文件 | 项目清单（由整理员生成） |
| `run_pipeline.py` | 文件 | 工作流启动脚本 |
| `PaperDownload/` | 目录 | 论文PDF下载 |
| `PaperDownloadMd/` | 目录 | 论文清单、分析报告 |
| `LocalPaperLibrary/` | 目录 | 本地原始论文库 |
| `MethodToSmart/` | 目录 | 文献分析员输出 |
| `SmartToCode/` | 目录 | 方案设计师输出 |
| `Code/` | 目录 | 参考代码 |
| `CodeWorkSpace/` | 目录 | 工作区代码 |
| `test_data/` | 目录 | 测试数据 |
| `test_result/` | 目录 | 测试结果 |
| `paper_output/` | 目录 | 论文输出 |
| `agents/` | 目录 | Agent模块 |
| `error/` | 目录 | 错误日志 |

**禁止在根目录放置的文件**：

| 类别 | 示例 | 处理方式 |
|------|------|----------|
| 临时文件 | `temp_*.txt`, `*.log` | 删除 |
| 测试脚本 | `test_*.py`, `*_cv.py` | 移至 `CodeWorkSpace/` |
| 报告文件 | `*.docx`, `*.xlsx` | 移至 `paper_output/` 或删除 |
| 工具脚本 | `download_*.py`, `generate_*.py` | 移至 `CodeWorkSpace/` |
| 独立文档 | `*架构文档.md`, `*排除.md` | 移至 `MEMORY/Agent工作流/` |

**验证命令**：
```bash
# 检查根目录是否有禁止文件
ls *.py *.txt *.docx 2>/dev/null | grep -v "run_pipeline.py" || echo "OK: 根目录干净"
```

---

# 四、多阶段验证规范（v11 修正）

> **重要**：十折交叉验证的核心架构、验证方式、判定条件详见 `十折交叉验证架构文档.md`

### 验证阶段定义

| 阶段 | 数据范围 | 验证方式 |
|------|----------|----------|
| **预实验** | 2020-01-01 ~ 2020-01-05 (5天) | 每天十折，合并预测 |
| **Stage 1** | 2020-01-01 ~ 2020-01-31 (31天) | 每天十折，合并预测 |
| **Stage 2** | 2020-07-01 ~ 2020-07-31 (31天) | 每天十折，合并预测 |
| **Stage 3** | 2020-12-01 ~ 2020-12-31 (31天) | 每天十折，合并预测 |

### 基准值来源

```
基准值存储位置：test_result/基准方法/benchmark_multistage.json
```

### 预实验规范

预实验通过 ≠ 创新成立，只是说明该方法值得进一步验证。

---

# 五、创新判定规范

> **重要**：创新判定条件、流程详见 `十折交叉验证架构文档.md`

### 基准方法明确

4个基准方法：VNA, eVNA, aVNA, Downscaler

### 多阶段验证流程

```
预实验(5天)
    │
    ├─ 通过 → Stage 1 (1月整月)
    │              │
    │              ├─ 通过 → Stage 2 (7月整月)
    │              │              │
    │              │              ├─ 通过 → Stage 3 (12月整月)
    │              │              │              │
    │              │              │              └─ 全部通过 → 创新成立
    │              │              │
    │              │              └─ 未通过 → 分析原因
    │              │
    │              └─ 未通过 → 创新不足，打回重设
    │
    └─ 未通过 → 打回 [Phase 3] 重新设计
```

---

# 六、Agent Spawn 全流程工作流图

## 6.1 完整执行流程

```
Phase 0: executor.phase0_organize()  → spawn 整理Agent（整理前人遗留，生成INVENTORY.md）
Phase X: executor.phaseX_check()     → 【新增】初始化检查
Phase 1: executor.phase1_download()  → 并行 spawn 3个下载Agent
Phase 2: executor.phase2_analyze()   → spawn 分析Agent
Phase 3: executor.phase3_design()   → spawn 设计Agent
Phase 4: executor.phase4_code()     → spawn 工程师Agent
Phase 5: executor.phase5_test()     → spawn 测试Agent
Phase 6: 创新判定
         ├─ 创新成立 → executor.phase6_write() → spawn 写作Agent
         └─ 创新不足 → 打回重设
```

## 6.2 初始化检查 Phase X

```python
def phaseX_check():
    """
    Phase X: 初始化检查（v11 新增）

    检查是否具备启动研究所需的基础设施
    """
    issues = check_initialization(project_root)

    if issues:
        print("初始化检查未通过:")
        for issue in issues:
            print(f"  - {issue}")
        print("\n请先完成初始化，再继续主流程")
        return {'status': 'failed', 'issues': issues}
    else:
        print("初始化检查通过！")
        return {'status': 'passed'}
```

## 6.3 主会话 Agent Spawn 伪代码

```python
from agents.spawn_executor import SpawnExecutor

executor = SpawnExecutor(project_root)

# Phase 0: 项目整理（先整理房间，再开工）
result = executor.phase0_organize()
Agent(description="整理Agent", prompt=result['prompt'])
executor.mark_completed('organizer')

# Phase X: 初始化检查
check_result = executor.phaseX_check()
if check_result['status'] == 'failed':
    # 处理初始化失败
    ...

# Phase 1: 并行下载
spawns = executor.phase1_download()
for agent_id, info in spawns.items():
    Agent(description=f"下载Agent {agent_id}", prompt=info['prompt'])
executor.mark_completed('dl_1')
executor.mark_completed('dl_2')
executor.mark_completed('dl_3')

# Phase 2: 文献分析
info = executor.phase2_analyze()
Agent(description="文献分析 Agent", prompt=info['prompt'])
executor.mark_completed('analyzer')

# Phase 3-6: 类似...
```

## 6.4 状态文件 (.agent_state.json)

工作流运行时会生成 `.agent_state.json` 记录状态：

```json
{
  "round": 1,
  "agents": {
    "dl_1": {"status": "completed", "completed_at": "2026-04-08T12:00:00"},
    "dl_2": {"status": "completed", "completed_at": "2026-04-08T12:01:00"},
    "dl_3": {"status": "completed", "completed_at": "2026-04-08T12:00:30"},
    "analyzer": {"status": "completed", "completed_at": "2026-04-08T12:05:00"},
    "designer": {"status": "completed", "completed_at": "2026-04-08T12:10:00"},
    "engineer": {"status": "completed", "completed_at": "2026-04-08T12:15:00"},
    "verifier": {"status": "completed", "completed_at": "2026-04-08T12:20:00"},
    "writer": {"status": "pending", "trigger": "innovation_established"}
  },
  "innovation_established": true,
  "iteration_count": 1,
  "no_improvement_count": 0,
  "terminated": false,
  "last_run": "2026-04-08T12:20:00"
}
```

---

# 七、七大角色详细职责

## 7.1 七大角色总览

| 角色 | Agent ID | 输入 | 输出 | 核心任务 | 并行度 |
|------|----------|------|------|----------|--------|
| **整理员** | organizer | 项目目录 | INVENTORY.md | 整理前人遗留，生成盘点报告 | **最开始** |
| **初始化检查员** | initializer | 目录/文件 | 检查报告 | 验证基准指标完整性 | **Phase X** |
| 文献下载员 | dl_1, dl_2, dl_3 | 无 | PaperDownload/ + paper_list.json | 下载论文，评分1-5 | **3并行** |
| 文献分析员 | analyzer | PaperDownload/ | MethodToSmart/文献分析员_*.md | 提炼方法为可执行规范 | 顺序 |
| 方案设计师 | designer | MethodToSmart/ | SmartToCode/复现/创新方法指令/ | 设计方案，生成方法指纹 | 顺序 |
| 代码工程师 | engineer | SmartToCode/指令 | CodeWorkSpace/ | 实现代码，运行验证 | 顺序 |
| 测试验证员 | verifier | CodeWorkSpace/代码 | test_result/ | 十折验证，计算指标 | 顺序 |
| **技术写作员** | writer | test_result/ + SmartToCode/ | paper_output/ | **整理成论文（LaTeX → PDF）** | 顺序 |

## 7.2 文献下载员

### 核心任务
1. 搜索PM2.5/CMAQ/数据融合/空间插值相关论文
2. 下载PDF到 PaperDownload/
3. 生成论文清单到 PaperDownloadMd/paper_list.json

### 去重机制（必须执行）

每篇论文入库前计算去重指纹：
```python
import hashlib
dedup_key = hashlib.md5(f"{title}{authors}".encode()).hexdigest()[:16]
```

已存在的 dedup_key 跳过下载，避免重复浪费存储空间。

### 异常决策

| 情况 | 决策 |
|------|------|
| 论文领域不确定 | 评分3，下载，不深究 |
| 下载失败 | 记录到 error/paper_download_YYYYMMDD.log，继续下一篇 |

---

## 7.3 文献分析员

### 核心任务
1. 读取 PaperDownloadMd/paper_list.json 获取**全部论文清单**
2. 交叉对比已有方法文档，确认**哪些论文还未被分析**
3. **必须继续分析未被文档化的论文**（"不要重复分析" ≠ "工作完成"）
4. 输出结构化方法文档到 MethodToSmart/

### 关键原则

| 情况 | 判定 |
|------|------|
| 已有方法文档 7 个，paper_list.json 共 37 篇 | **还有 30 篇未被分析，必须继续** |
| 已有方法文档已覆盖 paper_list.json 全部论文 | 工作完成，退出 |

### 输出模板

```markdown
# 【可执行方法规范】

## 方法名称
[中文名] ([英文缩写])

## 文献来源
- 论文标题：xxx
- 作者/年份：xxx / xxxx年
- 关键章节：P.xx / Section x

## 核心公式
$$
y_{fused} = \text{具体公式}
$$

## 参数清单
| 参数名 | 类型 | 默认值 | 说明 |
|-------|------|-------|------|
| k | float | 2.0 | 距离幂指数 |

## 数据规格
| 数据 | 格式 | 维度 |
|-----|------|-----|
| 监测站点坐标 | array | (n, 2) |
| 融合网格 | array | μg/m³ |

## 实现步骤
1. 步骤1
2. 步骤2
3. 步骤3

## 方法指纹
MD5: [自动生成]

## 随机性
- [ ] 是  - [x] 否
```

### 异常决策

| 情况 | 决策 |
|------|------|
| 方法描述不清晰 | 自行推断，标注【推断】，继续 |
| 公式无法理解 | 使用行业标准解释，记录推断过程，继续 |
| 数据格式不确定 | 参考 Code/ 推断；如无参考，使用netCDF/CSV，继续 |

---

## 7.4 方案设计师

### 核心任务
1. 基于 MethodToSmart/ 设计复现方案和创新方案
2. 生成方法指纹，比对去重
3. 判定方法是否成立

### 创新判定标准

| 指标 | 要求 | 不满足时 |
|------|------|----------|
| 新颖性 | 指纹与已有方法不重复 | 修改方法，重新生成指纹 |
| R²提升 | ≥ 0.01（相比最优基准） | 打回重设 |
| RMSE | ≤ 最优基准 | 打回重设 |

### 异常决策

| 情况 | 决策 |
|------|------|
| 不确定方法效果 | 记录为【风险假设】到 innovation_note.md，继续输出 |
| 方法与已有方法指纹相似 | 必须修改公式或步骤，生成新指纹，直到不重复 |
| 连续3轮无提升 | 尝试全新方法方向，继续迭代 |

---

## 7.5 代码工程师

### 核心任务
1. 实现方案设计师的指令
2. 适配数据格式
3. 运行并验证代码

### 语义确认环节（硬编码校验）

```
方案设计师输出指令
    ↓
代码工程师【复述理解】（书面形式）
    ↓
系统硬编码检查（必须全部满足）：
    ✓ 包含"方法名"字段
    ✓ 包含"输入数据"字段（维度明确）
    ✓ 包含"输出数据"字段（维度明确）
    ✓ 包含"核心公式"字段（LaTeX或Python表达式）
    ✓ 包含"关键步骤"字段（≥2步）
    ✓ 包含"参数清单"字段（表格形式）
    ✓ 方法指纹已生成
    ↓
通过 → 开始实现
不通过 → 记录到error/semantic_YYYYMMDD.log，重新生成
```

### 异常决策

| 情况 | 决策 |
|------|------|
| 代码报错 | 记录到 error/code_方法名_YYYYMMDD.log，自行尝试3次修复；失败则标记"待修复"，继续下一任务 |
| 数据格式转换问题 | 参考 Code/ 现有代码；如无参考，使用标准库（xarray/netCDF4），继续 |
| 参数值不确定 | 使用文献推荐值；如无推荐，使用默认值（k=2, power=1），继续 |

### 最低可运行标准（强制检查）

代码在进入测试验证阶段前，必须通过以下检查：

```python
def check_minimum_runnable(code_path):
    """强制检查：核心融合逻辑必须有语法错误以外的全部要素"""
    issues = []

    # 1. 语法检查（可修复）
    if has_syntax_error(code_path):
        issues.append("语法错误（非拼写类）")

    # 2. 核心函数存在性检查
    if not has_function(code_path, "fuse_method"):
        issues.append("缺少 fuse_method() 函数")
    if not has_function(code_path, "calculate_metrics"):
        issues.append("缺少 calculate_metrics() 函数")

    # 3. 输入输出接口完整性
    if not has_expected_params(code_path, "fuse_method", ["cmaq_data", "station_data", "station_coords", "params"]):
        issues.append("fuse_method() 参数不完整")

    return issues  # 空列表 = 通过
```

**判定规则**：
- issues 为空 → 允许进入测试验证阶段
- issues 包含"语法错误" → **直接打回代码工程师修复**，不得流入测试阶段
- issues 包含其他项 → 标记"警告"但允许继续，测试验证员会记录

**为什么这样设计**：
带语法错误的代码流入测试阶段会污染指标，导致基准对比失效。"语法错误直接打回"成本最低。

---

## 7.6 测试验证员

### 核心任务
1. 执行十折交叉验证
2. 计算R²、MAE、RMSE、MB指标
3. 验证创新是否成立

> **重要**：十折交叉验证的技术细节（验证流程、指标计算、异常决策等）详见 `十折交叉验证架构文档.md`

### 基准带校验机制

```
基准方法R²合理范围（10折验证）：
- VNA:     R² ∈ [0.70, 0.95]
- eVNA:    R² ∈ [0.65, 0.95]
- aVNA:    R² ∈ [0.65, 0.95]
- CMAQ:    R² ∈ [0.30, 0.80]

校验流程：
1. 测试前先运行VNA基准
2. 检查VNA R²是否在范围内
3. R² < 0.70 → 记录到error/，标记"数据异常"，停止测试
4. R² > 0.95 → 记录到error/，标注"疑似过拟合"，继续测试

【禁止】跳过基准带校验直接测试新方法
```

### 历史最佳保存机制

```
每次创新判定后：
1. 对比当前方案与历史最佳
2. 当前R²更高 → 更新历史最佳
3. 当前R²更低 → 保留历史最佳，不更新
4. 保存到 test_result/历史最佳方案/（best_method.py, best_metrics.json）
```

---

## 7.7 测试验证员的 Scout-Reflection 闭环

参考 AutoSOTA 架构，测试验证员在每次验证迭代中执行 **Plan-Execute-Reflect** 循环：

### 闭环流程

```
┌─────────────────────────────────────────────────────────────┐
│                   Plan-Execute-Reflect 闭环                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   ┌─────────┐    ┌─────────┐    ┌─────────┐               │
│   │  Scout  │───▶│ Execute │───▶│Reflect  │               │
│   │(分析机会)│    │(验证方法)│    │(评估决策)│               │
│   └─────────┘    └─────────┘    └─────────┘               │
│        ▲                               │                     │
│        │         loop                  │                     │
│        └───────────────────────────────┘                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Scout 阶段（分析优化机会）

**触发时机**：每次迭代开始前

**任务**：基于历史验证结果，分析下一个最有潜力的优化方向

**分析维度**：

| 维度 | 分析内容 |
|------|----------|
| 历史表现 | 已验证方法的效果排名、失败方法的失败原因 |
| 提升空间 | 当前最佳方法的薄弱环节（如高污染区、稀疏区） |
| 候选方向 | 基于文献分析提出的待验证假设 |

**StateTracker 查询**：

```python
# 查询当前状态
current = tracker.get_current_state()
# → {'iteration': 5, 'current_best_method': 'SuperStacking', 'current_best_r2': 0.8571, ...}

# 查询失败方法及原因
failed = tracker.get_failed_methods_summary()
# → [{'method': 'CSP-RK', 'reason': '参数迁移性差', ...}, ...]

# 获取下一个优化方向建议
next_direction = tracker.get_next_optimization_direction()
# → "建议：优先选择具有物理锚点的固定参数方法，减少数据依赖"
```

**输出**：下一个待验证的假设或方法

### Execute 阶段（执行验证）

**任务**：对 Scout 提出的候选方法/假设进行十折交叉验证

（执行现有测试验证流程不变）

### Reflection 阶段（评估决策）

**触发时机**：每次验证完成后

**任务**：评估验证结果，决定是否接受优化，并更新 StateTracker

**评估维度**：

| 评估项 | 标准 |
|--------|------|
| 定量提升 | R² 提升 ≥ 0.01 |
| 参数迁移性 | 跨天测试中参数是否稳定 |
| 物理可解释性 | 参数是否有物理锚点 |

**决策逻辑**：

```
验证结果评估
    │
    ├─ R² 提升 ≥ 0.01 AND 跨天稳定
    │   → 接受优化，更新 StateTracker
    │   → 记录到 Ledger
    │
    ├─ R² 提升 ≥ 0.01 BUT 跨天不稳定（参数敏感）
    │   → 拒绝优化
    │   → 记录失败原因：参数迁移性差
    │   → 更新失败方法列表
    │
    └─ R² 提升 < 0.01
        → 拒绝优化
        → 记录失败原因：提升不足
```

**StateTracker 更新示例**：

```python
# 接受优化
tracker.accept_mutation(
    method_name="SuperStackingEnsemble",
    metrics={"R2": 0.8571, "MAE": 6.95, "RMSE": 10.85},
    description="Ridge元学习器组合多个基础模型"
)

# 拒绝优化
tracker.reject_mutation(
    method_name="CSP-RK",
    metrics={"R2": 0.8535, "MAE": 7.07, "RMSE": 10.99},
    reason="参数迁移性差，跨天平均Δ=-0.00003 << 0.01阈值",
    description="浓度分层多项式克里金"
)
```

### StateTracker 集成

**初始化**：研究开始时，从 INVENTORY.md 加载历史最佳方法

```python
tracker = StateTracker("test_result/.state")
tracker.initialize(
    baseline_metrics={"R2": 0.8100, "MAE": 7.5, "RMSE": 11.0},
    baseline_method="eVNA"
)
```

**查询下一个优化方向**：

```python
direction = tracker.get_next_optimization_direction()
# 基于失败方法分析，给出建议
```

**生成状态报告**：

```python
tracker.save_report()  # → test_result/.state/research_status_report.md
```

### Ledger 格式

每次决策记录到 `test_result/.state/ledger.jsonl`：

```json
{"timestamp": "2026-04-10T09:00:35", "event_type": "mutation_accepted", "iteration": 1, "method_name": "ResidualKriging", "metrics": {"R2": 0.8273, ...}, "status": "accepted"}
{"timestamp": "2026-04-10T09:15:22", "event_type": "mutation_rejected", "iteration": 2, "method_name": "CSP-RK", "reason": "参数迁移性差", "status": "rejected"}
```

---

## 7.8 技术写作员（论文写作Agent）

### 核心任务

**当创新成立时**，技术写作Agent负责：
1. 收集研究过程中的决策记录和实验结果
2. 按照学术论文结构组织内容
3. 生成 **LaTeX 论文源码** 并编译为 **PDF**

### PDF解析步骤

**使用 opendataloader-pdf 解析论文 PDF**（安装：`pip install opendataloader-pdf`）

```
技术写作Agent启动
    ↓
[Step 1] 解析参考文献 PDF
    │
    └─→ opendataloader_pdf.convert(
    │       input_path=["ref1.pdf", "ref2.pdf", ...],
    │       output_dir="paper_output/references_json/",
    │       format="markdown,json"
    │   )
    ↓
[Step 2] 提取关键内容
    │
    ├─ 标题、作者、年份 → references.bib
    ├─ 方法描述 → 用于 Related Work 章节
    ├─ 公式 → 用于方法对比
    └─ 表格数据 → 用于实验对比
    ↓
[Step 3] 组织论文结构
    │
    └─→ 按论文模板填充内容
    ↓
[Step 4] 编译 LaTeX → PDF
    │
    └─→ xelatex paper.tex (×3)
```

### 输出产物

```
paper_output/
├── paper.tex              # 主论文 LaTeX 源码
├── paper.pdf             # 编译后的 PDF
├── references.bib         # BibTeX 参考文献
├── figures/              # 图表目录
├── references_json/      # 解析的参考文献 JSON
│   ├── ref1.json
│   └── ref1.md
└── tech_report.md        # 详细技术报告（过程记录）
```

### 论文结构要求

| 章节 | 内容来源 | 说明 |
|------|----------|------|
| Title | 最佳方法名称 | 简洁，包含核心创新点 |
| Abstract | 研究成果总结 | R²提升、RMSE改善 |
| Introduction | 研究背景 | PM2.5监测的重要性 |
| Related Work | LocalPaperLibrary/ + **opendataloader-pdf解析** | 引用已有方法论文 |
| Methodology | SmartToCode/创新方法指令/ | 详细描述提出的方法 |
| Experiments | test_result/历史最佳方案/ | 十折验证结果 |
| Results | comparison_report.md | 方法对比表 |
| Discussion | innovation_note.md | 分析成败原因 |
| Conclusion | 总结与展望 | 贡献总结 |

### 异常决策

| 情况 | 决策 |
|------|------|
| PDF解析失败 | 记录到 error/pdf_parse_YYYYMMDD.log，跳过该PDF |
| 缺少某个实验数据 | 使用已有数据，标注"待补充" |
| 图表缺失 | 生成占位符，标注需要手动添加 |
| 参考文献不全 | 从 LocalPaperLibrary/ 推断，标注需要核实 |
| LaTeX编译失败 | 检查语法错误，修复后重试（最多3次） |

### 触发条件

**自动触发**：当满足以下任一条件时，技术写作Agent自动启动
- 创新判定通过（R²提升≥0.01）
- 人类说"停止"并要求生成论文

**输出完成标志**：
```
paper_output/
├── paper.pdf  ← 此文件存在表示论文生成完成
└── tech_report.md  ← 详细过程记录
```

---

# 八、输出规范化要求

## 8.0 设计原则

```
┌────────────────────────────────────────────────────────────────┐
│                    ⚠️ 输出规范化原则 ⚠️                         │
├────────────────────────────────────────────────────────────────┤
│  【一角色一清单】                                                │
│  每个角色只在自己目录生成一份清单，不重复生成                     │
│                                                                 │
│  【一次一版本】                                                  │
│  报告类文件只保留最新版本，历史版本自动归档到历史目录             │
│                                                                 │
│  【机器可读】                                                    │
│  汇总数据必须是 CSV/JSON，供下游 Agent 直接读取                  │
│                                                                 │
│  【人类可读】                                                    │
│  报告必须是 Markdown/PDF，标注日期和版本                         │
└────────────────────────────────────────────────────────────────┘
```

## 8.0.1 各角色输出清单

| 角色 | 输出目录 | 文件规范 | 上游可读 |
|------|----------|----------|----------|
| **整理员** | 项目根目录 | `INVENTORY.md` | 所有Agent |
| **文献下载员** | PaperDownloadMd/ | `paper_list.json` | 文献分析员 |
| **文献分析员** | MethodToSmart/ | `文献分析员_[方法名].md` + `INVENTORY.md` | 方案设计师 |
| **方案设计师** | SmartToCode/ | `V1_[方法名].md` / `Innovation_[方法名].md` + `复现方法汇总.md` + `创新方法汇总.md` | 代码工程师 |
| **代码工程师** | CodeWorkSpace/ | `[方法名].py` + `code_manifest.json` | 测试验证员 |
| **测试验证员** | test_result/ | `[方法名]_summary.csv` + `innovation_summary.csv` + `comparison_report.md` | 技术写作员 |
| **技术写作员** | paper_output/ | `paper.tex` + `paper.pdf` | 人类 |

## 8.0.2 汇总文件格式

### 汇总 CSV 规范（所有 Agent 必须遵守）

```csv
# 文件名格式：[类型]_summary.csv
# 位置：对应角色输出目录

# 示例：test_result/创新方法/innovation_summary.csv
method_name,R2,RMSE,MAE,MB,round,date,status
RK-Poly,0.8519,8.45,6.12,0.23,1,2026-04-10,accepted
CSP-RK,0.8535,8.38,6.07,0.19,2,2026-04-10,rejected
```

### 清单 Markdown 规范

```markdown
# [目录名] 清单

生成时间：2026-04-10
角色：[角色名]

## 文件列表

| 文件名 | 方法名 | 日期 | 状态 | 备注 |
|--------|--------|------|------|------|
| xxx.md | yyy | 2026-04-10 | 正常 | - |
```

## 8.0.3 报告版本控制

**规则**：
- `comparison_report.md` 每次更新直接覆盖，不保留历史版本
- `FINAL_REPORT.md` 每次更新直接覆盖，不保留历史版本
- 需要历史记录时，查看 `.state/ledger.jsonl`

**禁止**：
- ❌ comparison_report_v1.md, comparison_report_v2.md
- ❌ FINAL_REPORT_v1.md, FINAL_REPORT_v2.md
- ❌ 任何带版本号的报告文件

## 8.0.4 test_result 目录结构（详细）

```
test_result/
├── INVENTORY.md                    # 【必须】结果清单
├── .state/                         # 【必须】状态追踪
│   ├── ledger.jsonl               # 【追加】每次决策记录
│   └── research_status.md         # 【覆盖】当前状态
├── 基准方法/
│   └── benchmark_summary.csv      # 【必须】VNA/eVNA/aVNA/CMAQ/Downscaler
├── 复现方法/
│   └── reproduction_summary.csv   # 【必须】所有复现方法汇总
├── 创新方法/
│   ├── innovation_summary.csv     # 【必须】所有创新方法汇总
│   ├── [方法A]_summary.csv        # 【可选】详细结果
│   └── [方法B]_summary.csv
├── 历史最佳方案/                   # 【更新】每次有新最佳时更新
│   ├── best_method.py
│   └── best_metrics.json
├── comparison_report.md           # 【覆盖】对比报告
└── FINAL_REPORT.md               # 【覆盖】最终报告
```

## 8.0.5 状态追踪格式

### ledger.jsonl（追加模式）

```jsonl
{"timestamp": "2026-04-10T09:00:35", "event_type": "mutation_accepted", "iteration": 1, "method_name": "RK-Poly", "metrics": {"R2": 0.8519, "RMSE": 8.45, "MAE": 6.12, "MB": 0.23}, "status": "accepted"}
{"timestamp": "2026-04-10T09:15:22", "event_type": "mutation_rejected", "iteration": 2, "method_name": "CSP-RK", "reason": "参数迁移性差", "metrics": {"R2": 0.8535, "RMSE": 8.38, "MAE": 6.07, "MB": 0.19}, "status": "rejected"}
```

### research_status.md（覆盖模式）

```markdown
# 研究状态报告

更新时间：2026-04-10

## 当前最佳方案

| 指标 | 值 |
|------|-----|
| 方法名 | RK-Poly |
| R² | 0.8519 |
| RMSE | 8.45 |

## 迭代历史

| 轮次 | 方法 | R² | 结论 |
|------|------|-----|------|
| 1 | RK-Poly | 0.8519 | 接受 |
| 2 | CSP-RK | 0.8535 | 拒绝：参数迁移性差 |

## 待处理

- [ ] 多日验证（1月全月）
```

## 8.0.6 禁止的输出模式

```
┌─────────────────────────────────────────────────────────────┐
│                      禁止清单                               │
├─────────────────────────────────────────────────────────────┤
│  ❌ 输出多个版本报告（comparison_report_v1/v2/v3.md）        │
│  ❌ 在错误目录生成文件                                        │
│  ❌ 生成机器无法解析的报告（纯图片、无表格）                   │
│  ❌ 输出 CSV 但不更新汇总清单                                 │
│  ❌ 跨目录生成重复内容（各 Agent 只管自己目录）               │
└─────────────────────────────────────────────────────────────┘
```

---

# 九、快照与迭代机制

## 9.1 核心问题

```
Run1 → 不满意 → 手动整理 → Run2 → 还是会重复阅读文献？
```

## 9.2 解决方案：快照管理器

```
test_result/
├── snapshots/                    # 【新增】快照目录
│   ├── round_0/                  # 初始状态
│   │   └── metadata.json
│   ├── round_1/                  # 第1次运行后
│   │   ├── metadata.json         # 去重状态
│   │   ├── CodeWorkSpace/
│   │   ├── SmartToCode/
│   │   └── test_result/
│   └── round_2/                  # 第2次运行后
├── .state/
│   ├── current.txt               # 指针：round_1
│   └── ledger.jsonl
└── ...
```

## 9.3 metadata.json 格式

```json
{
    "round": 1,
    "created": "2026-04-10T12:00:00",
    "parent": "round_0",
    "note": "预实验完成，效果不理想",
    "downloaded_papers": ["md5_abc123", "md5_def456"],
    "analyzed_methods": ["VNA", "eVNA", "RK-Poly"],
    "method_fingerprints": ["md5_xxx", "md5_yyy"],
    "best_method": "RK-Poly",
    "best_r2": 0.8519
}
```

## 9.4 快照管理器 API

```python
from test_result.snapshot_manager import SnapshotManager

manager = SnapshotManager(root)

# 查看当前状态
manager.print_status()

# 创建新快照（运行前必须）
manager.create_snapshot(round_num=1, note="开始第1次运行")

# ===== 运行 Agent 工作流 =====

# 更新最佳方法（运行后）
manager.update_best_method("RK-Poly", {"R2": 0.8519, "RMSE": 8.45})

# 添加已下载论文（下载员必须调用）
manager.add_downloaded_paper("论文标题", "作者列表")

# 添加已分析方法（分析员必须调用）
manager.add_analyzed_method("VNA")

# 添加方法指纹（设计师必须调用）
manager.add_fingerprint("md5_fingerprint")

# 检查是否重复
if not manager.check_paper_dedup(title, authors):
    download()  # 未下载才下载
```

## 9.5 去重检查流程

```
【文献下载员】
1. 读取 manager.load_dedup_state()["downloaded_papers"]
2. 对每篇论文计算 MD5(title+authors)
3. 如果已存在 → 跳过
4. 如果不存在 → 下载 → manager.add_downloaded_paper()

【文献分析员】
1. 读取 manager.load_dedup_state()["analyzed_methods"]
2. 检查 paper_list.json 中的论文是否已分析
3. 如果已分析 → 跳过
4. 如果未分析 → 分析 → manager.add_analyzed_method()

【方案设计师】
1. 读取 manager.load_dedup_state()["method_fingerprints"]
2. 生成新方法指纹
3. 如果指纹已存在 → 修改方法直到不重复
4. 如果指纹不存在 → 添加 → manager.add_fingerprint()
```

## 9.6 迭代工作流

```
┌─────────────────────────────────────────────────────────────┐
│                     完整迭代流程                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  [开始]                                                      │
│     │                                                        │
│     ▼                                                        │
│  ┌─────────────────┐                                        │
│  │ 读取当前快照     │ ← manager.restore_snapshot()           │
│  └────────┬────────┘                                        │
│           │                                                   │
│           ▼                                                   │
│  ┌─────────────────┐                                        │
│  │ 创建新快照      │ ← manager.create_snapshot(round_N)     │
│  └────────┬────────┘                                        │
│           │                                                   │
│           ▼                                                   │
│  ┌─────────────────┐                                        │
│  │ 运行Agent工作流  │                                        │
│  │ (带去重检查)     │                                        │
│  └────────┬────────┘                                        │
│           │                                                   │
│           ▼                                                   │
│      结果满意？                                               │
│           │                                                   │
│     ┌────┴────┐                                              │
│     │         │                                              │
│    Yes       No                                              │
│     │         │                                              │
│     ▼         ▼                                              │
│  [完成]   更新快照metadata                                    │
│           ↑                                                   │
│           │                                                   │
│     修改note/retry                                           │
│           │                                                   │
│           └─────────────────────────────────────────────────┘
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

# 十、全流程运行步骤

## 10.1 初始化（仅执行1次）

| Step | 任务 | 输出 | 约束 |
|------|------|------|------|
| 1 | 生成十折交叉验证表 | test_data/fold_split_table.csv | 禁止问"表怎么分" |
| 2 | 生成测试日期列表 | test_data/selected_days.txt | 禁止问"选哪些日期" |
| 3 | 运行基准方法测试 | test_result/基准方法/ | 禁止问"参数对不对" |

## 10.2 主循环 Spawn 执行

使用 SpawnExecutor 管理主循环：

```
executor = SpawnExecutor(project_root)

while True:
    # Phase X: 初始化检查
    check_result = executor.phaseX_check()
    if check_result['status'] == 'failed':
        # 处理初始化失败
        ...

    # Phase 1: 并行下载
    spawns = executor.phase1_download()
    for agent_id, info in spawns.items():
        Agent(description=f"下载Agent {agent_id}", prompt=info['prompt'])
    executor.mark_completed('dl_1')
    executor.mark_completed('dl_2')
    executor.mark_completed('dl_3')

    # Phase 2: 文献分析
    info = executor.phase2_analyze()
    Agent(description="文献分析 Agent", prompt=info['prompt'])
    executor.mark_completed('analyzer')

    # Phase 3: 方案设计
    info = executor.phase3_design()
    Agent(description="方案设计 Agent", prompt=info['prompt'])
    executor.mark_completed('designer')

    # Phase 4: 代码实现
    info = executor.phase4_code()
    Agent(description="代码实现 Agent", prompt=info['prompt'])
    executor.mark_completed('engineer')

    # Phase 5: 测试验证
    info = executor.phase5_test()
    Agent(description="测试验证 Agent", prompt=info['prompt'])
    executor.mark_completed('verifier')

    # Phase 6: 创新判定
    state = executor.get_state()
    if state['innovation_established']:
        # Phase 7: 技术写作
        info = executor.phase6_write()
        Agent(description="技术写作 Agent", prompt=info['prompt'])
        break  # 流程完成
    else:
        # 打回重设，继续下一轮
        continue
```

## 10.3 迭代优化规则

| 情况 | 决策 |
|------|------|
| 被打回时 | 立即开始新一轮设计，禁止问"为什么失败" |
| 连续3轮无提升 | 记录到 error/iteration_stagnation_YYYYMMDD.log，尝试全新方法方向 |
| 超过10轮未成功 | 记录到 error/iteration_max_YYYYMMDD.log，生成当前最优方案报告，**等待人工指示** |

---

# 十一、异常处理统一规范

## 11.1 统一异常决策表

| 异常类型 | 处理方式 | 后续动作 |
|----------|----------|----------|
| 数据下载失败 | 记录 → 继续下一篇 | 不停止 |
| 论文分析失败 | 记录 → 跳过该论文 | 不停止 |
| 公式无法理解 | 推断 → 标注【推断】 | 继续 |
| 数据格式不明 | 使用标准格式 | 继续 |
| 代码运行报错 | 记录 → 尝试3次修复 | 失败则标记"待修复" |
| 测试运行失败 | 记录 → 继续其他方法 | 不停止 |
| 数据对齐失败 | 跳过该日期 | 继续其他日期 |
| 方法指纹重复 | 修改方法 → 重新生成 | 不停止 |
| 指标计算异常 | 跳过NaN | 继续 |

## 11.2 错误日志格式

```markdown
# 错误日志

日期：20260408
时间：14:30:25
类型：code_runtime_error
Agent ID: engineer
方法：IDW_Kriging_Hybrid

## 错误信息
```
[完整错误堆栈]
```

## 尝试的修复
1. 检查数据维度...
2. 修改参数类型...
3. [均失败]

## 当前状态
标记为"待修复"，继续其他任务

## 处理时间
2分钟
```

## 11.3 禁止行为清单

```
┌─────────────────────────────────────────────────────────────┐
│                      禁止清单                               │
├─────────────────────────────────────────────────────────────┤
│  1. 禁止说"我需要确认"、"请告诉我"、"等一下"                 │
│  2. 禁止停下来问问题                                         │
│  3. 禁止等待人工输入才继续                                   │
│  4. 禁止跳过异常记录                                         │
│  5. 禁止在未尝试3次修复前标记失败                            │
│  6. 禁止自行决定"这是最后一代"                               │
│  7. 禁止自行终止流程（除非人类明确说"停止"）                 │
│  8. 禁止跳过基准带校验                                       │
└─────────────────────────────────────────────────────────────┘
```

---

# 十二、版本管理与终止条件

## 12.1 版本管理

| 对象 | 命名格式 | 保存规则 |
|------|----------|----------|
| 方法方案 | ideas_V1.json, V2... | 永久保存 |
| 代码 | workspace/innovation/V1/, V2/... | 永久保存 |
| 测试结果 | results/innovation/V1/, V2/... | 永久保存 |
| 错误日志 | error/xxx_YYYYMMDD.log | 永久保存 |

**【禁止】覆盖任何历史版本**

## 12.2 自动终止条件

| 条件 | 说明 |
|------|------|
| 创新成立 | R²提升≥0.01 且 RMSE≤最优基准 |
| 人类明确停止 | 人类说"停止" |

## 12.3 人类干预入口

### 方式一：修改配置文件（立即生效）

```json
// test_result/流程配置.json
{
  "max_iterations": 10,
  "min_r2_improvement": 0.01,
  "vna_r2_min": 0.70,
  "vna_r2_max": 0.95,
  "stagnation_threshold": 3,
  "test_days": ["2020-01-01", "2020-01-02"],
  "enabled_methods": ["VNA", "eVNA", "aVNA", "RK-Poly"]
}
```

### 方式二：强制采用特定方案

```json
// test_result/最终方案标记.json
{
  "final": true,
  "method": "RK-Poly",
  "round": 15,
  "note": "人工确认为最优方案"
}
```

### 方式三：跳过基准校验

```
创建文件：error/skip_baseline_check_YYYYMMDD.flag
（文件存在即表示跳过基准带校验）
```

### 方式四：手动指定基准方法

```json
// test_result/基准方法配置.json
{
  "primary_baseline": "eVNA",
  "secondary_baselines": ["VNA", "aVNA"]
}
```

## 12.4 【禁止】自行终止

```
即使连续多轮无提升、超过10轮仍未成功，也【禁止】自行终止。
正确做法：记录到error/，生成当前最优方案报告，等待人工指示。
```

---

# 附录A：创新方法排除规则

> **重要**：参见项目根目录 `PM2.5_CMAQ融合方法创新排除.md`

## A.1 需要排除的方法类型

**加权集成方法（Stacking/Ensemble with learned weights）**：

| 应排除的方法 |
|-------------|
| SuperStackingEnsemble | EnhancedStackingEnsemble | UltimateStackingEnsemble |
| FeatureStackingEnsemble | MultiLevelStackingEnsemble | LogRatioEnsemble |
| V6-Ensemble-V1 | OptimizedTripleEnsemble | TripleEnsemble |

**排除原因**：
| 原因 | 说明 |
|------|------|
| 迁移性能不强 | 权重在特定日期学习，不同日期可能需要不同权重 |
| 效果提升有限 | 往往比最优子方法提升很小(<0.005)，甚至不如 |
| 可解释性差 | 负权重含义不明确，物理意义不清楚 |
| 复杂但无用 | 实现复杂，参数众多，难以部署 |

## A.2 应该保留的方法类型

| 特征 | 描述 |
|------|------|
| 单模型 | 单一方法解决问题 |
| 物理可解释 | 数学/物理意义清晰 |
| 显著提升 | R²提升 ≥ 0.02 |
| 可迁移 | 权重不随日期剧烈变化 |

**鼓励的创新方向**：
| 类型 | 示例 | 物理意义 |
|------|------|---------|
| 新偏差校正 | RK-Poly | CMAQ偏差非线性 |
| 新空间建模 | GPR降尺度 | 残差空间相关性 |
| 新融合策略 | 协同克里金 | 多变量空间协同 |

## A.3 创新判定流程（补充）

```
1. 新方法是否使用权重学习（Ridge/Lasso等）？
   → 是 → 排除
   → 否 → 继续

2. 新方法是否有物理可解释性？
   → 否 → 排除
   → 是 → 保留（创新成立）
```

## A.4 典型案例

| 方法 | R² | 结论 |
|------|-----|------|
| SuperStackingEnsemble | 0.8571 | **应排除**（加权集成，RK-Poly效果更好） |
| RK-Poly | ~0.85-0.88 | **应保留**（物理意义清晰，单模型） |

---

# 附录B：快速参考

## B.1 Agent Spawn 触发词

| 触发词 | 含义 |
|--------|------|
| `!!SPAWN_AGENT:bg:id=xxx!!` | 后台启动（并行） |
| `!!AGENT_WAIT:ids!!` | 等待指定Agent完成 |
| `!!CHECK_TRIGGER:condition!!` | 检查触发条件 |

## B.2 统一异常决策表

| 情况 | 决策 |
|------|------|
| 公式参数文献没给 | 用默认值(k=2) |
| 数据格式不确定 | 用netCDF/CSV标准格式 |
| 方法描述模糊 | 自行推断，标注【推断】 |
| 下载失败 | 跳过，记录到error |
| 代码报错 | 尝试3次修复，失败则标记"待修复" |
| 测试失败 | 跳过该方法，继续其他 |
| 指纹重复 | 修改方法生成新指纹 |
| 指标NaN | 跳过该站点继续 |

## B.3 基准带校验标准

| 基准方法 | R²合理范围 |
|----------|-------------|
| VNA | [0.70, 0.95] |
| eVNA | [0.65, 0.95] |
| aVNA | [0.65, 0.95] |
| CMAQ | [0.30, 0.80] |

## B.4 基准方法多阶段验证要求

### 基准方法验证代码模板

```python
def validate_baseline_methods_multistage():
    """
    基准方法多阶段验证

    输出：test_result/基准方法/benchmark_multistage.json
    """
    methods = ['VNA', 'eVNA', 'aVNA', 'Downscaler']
    stages = {
        'pre_exp': ('2020-01-01', '2020-01-05'),
        'stage1':  ('2020-01-01', '2020-01-31'),
        'stage2':  ('2020-07-01', '2020-07-31'),
        'stage3':  ('2020-12-01', '2020-12-31'),
    }

    results = {}
    for method in methods:
        results[method] = {}
        for stage_name, (start, end) in stages.items():
            all_y_true, all_y_pred = [], []
            current_date = start
            while current_date <= end:
                # 每天十折验证
                daily_true, daily_pred = ten_fold_for_day(method, current_date)
                all_y_true.extend(daily_true)
                all_y_pred.extend(daily_pred)
                current_date += timedelta(days=1)

            # 合并计算整体指标
            metrics = compute_metrics(all_y_true, all_y_pred)
            results[method][stage_name] = metrics

    # 保存结果
    with open('benchmark_multistage.json', 'w') as f:
        json.dump(results, f, indent=2)

    return results
```

### 基准验证输出格式

```json
{
  "VNA": {
    "pre_exp": {"R2": 0.800, "RMSE": 12.86, "MAE": 7.75, "MB": 0.76},
    "stage1":  {"R2": 0.785, "RMSE": 13.45, "MAE": 8.12, "MB": 0.82},
    "stage2":  {"R2": 0.720, "RMSE": 8.34, "MAE": 5.67, "MB": 0.45},
    "stage3":  {"R2": 0.798, "RMSE": 14.23, "MAE": 9.34, "MB": 1.12}
  },
  "eVNA": {
    "pre_exp": {"R2": 0.810, "RMSE": 12.52, "MAE": 7.99, "MB": 0.08},
    "stage1":  {"R2": 0.795, "RMSE": 13.01, "MAE": 8.34, "MB": 0.12},
    "stage2":  {"R2": 0.738, "RMSE": 8.56, "MAE": 5.89, "MB": 0.34},
    "stage3":  {"R2": 0.805, "RMSE": 14.45, "MAE": 9.56, "MB": 0.95}
  },
  ...
}
```

---

# 附录C：参数迁移性评估

## C.1 核心矛盾

| 方案 | 问题 |
|------|------|
| 只跑5天预实验调参 | 参数只对这几天有效，其他天用同样参数结果变差 |
| 从第一天就跑全部天数 | 太贵，失去了预实验的意义 |
| 物理意义赋值 | 很多参数（尤其是偏差校正系数）根本没有物理意义可以查表 |

## C.2 两阶段参数策略

```
第一阶段：物理初始化
↓
用物理意义给参数赋"合理初值"
（比如 Kriging 的 range 初值，可以用 PM2.5 典型空间相关距离）
↓
第二阶段：有限数据验证
↓
在 1-2 天上做参数微调
→ 验证"这套初值 + 少量数据 → 能泛化到其他天"
↓
跨天测试（看参数是否迁移）
```

## C.3 四元数组的隐性第5维度

| 维度 | 评判标准 |
|------|---------|
| 1.定量提升 | R² ≥ 历史最佳 + 0.01 |
| 2.物理可解释性 | 参数能用物理意义初始化，或在有限数据下稳定收敛 |
| 3.跨日期一致性 | 在≥3个不同天测试，参数不剧烈震荡 |

**第2/5条重点**：不是"参数完全来自物理"，而是"**参数有物理锚点，不怕小数据扰动**"

## C.4 能排除的方法类型

| 类型 | 特征 | 论文风险 |
|------|------|---------|
| 数据饥渴型 | 依赖大量数据才能拟合，数据少了就崩 | 审稿人挑战 |
| 日期波动型 | 参数在好日期和坏日期之间剧烈波动 | 说明在拟合日期特征 |

## C.5 参数迁移性验证

| 验证场景 | 做法 |
|----------|------|
| 单日参数 vs 多日平均 | 对比同一方法在单日学习和多日联合学习下的表现 |
| 固定参数方法 | 如 T1=35, T2=75 基于国标，天然具有迁移性 |
| 建议策略 | 优先设计具有物理意义的固定参数，减少数据依赖 |

**关键洞见**：参数对某一天的数据特别敏感（小扰动导致大变化）→ 本质上是在"拟合那一天"而非"建模"

---

*文档版本：v11_agent_spawn*
*更新日期：2026-04-14*
*核心改变：整合v10.1完整内容 + v11多阶段验证更新 + 初始化检查阶段*

*版本历史：*
- v6_autonomous: 高自动化提示版
- v7: 新增技术写作Agent（第六个角色）
- v8: 集成 opendataloader-pdf 进行 PDF 解析
- v9: **Agent Spawn 执行模式**，每个角色作为独立子AI Agent运行
- v9.1: 新增整理优先原则，新任务开始前先整理已有文件
- v9.2: 新增创新方法排除规则，加权集成方法（Stacking类）应排除
- v9.3: 使用 SpawnExecutor 替代 !!SPAWN_AGENT!! 语法
- v9.4: 新增下载去重机制 + 代码工程师最低可运行标准检查
- v9.5: 移除冗余附录A，保留附录B快速启动命令
- v9.6: 新增 Phase 0 整理Agent
- v9.7: 新增失败原因分析机制，R²<0.70时必须诊断
- v10: 新增输出规范化要求
- **v10.1: 新增快照与迭代机制**
- **v11: 修正多阶段验证方式，明确4个基准方法，增加初始化检查阶段**
