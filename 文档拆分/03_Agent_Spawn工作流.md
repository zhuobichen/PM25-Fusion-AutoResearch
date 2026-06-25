# Agent Spawn 工作流

> **来源**：PM2.5_CMAQ融合方法自动研究全流程文档_v11_agent_spawn.md
> **版本**：v14

---

# 六、Agent Spawn 全流程工作流图

## 6.1 完整执行流程

```
Phase 0: dispatcher.dispatch(0, ...)  → 生成整理任务，TRAE Agent 执行
Phase X: 初始化检查（Python 直跑）
Phase 1: dispatcher.dispatch(1, ...)  → 生成下载任务
Phase 2: dispatcher.dispatch(2, ...)  → 生成分析任务
Phase 3: dispatcher.dispatch(3, ...)  → 生成设计任务
Phase 4: dispatcher.dispatch(4, ...)  → 生成编码任务
Phase 5: run_verify_phase()           → Python 直接运行验证脚本
Phase 6: 创新判定
         ├─ 创新成立 → dispatcher.dispatch(6, ...) → 生成写作任务
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
from agents.task_dispatcher import TaskDispatcher, build_prompt_for_phase
from agents.artifact_detector import ArtifactDetector

dispatcher = TaskDispatcher(project_root, backend='trae')
detector = ArtifactDetector(project_root)

for phase_num in range(7):
    # 先检测产物
    if detector.check(phase_num)['done']:
        continue

    # Phase 5: 直接运行
    if phase_num == 5:
        run_verify_phase()
        continue

    # LLM Phase: 分发任务
    prompt = build_prompt_for_phase(phase_num, project_root)
    dispatcher.dispatch(phase_num, prompt)

    # 异步后端：等待外部完成
    if not dispatcher.backend.is_sync():
        break  # 重新运行时自动检测产物并推进
```

## 6.4 状态文件 (.agent_state.json)

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

> **v14 补充说明**：除 `.agent_state.json`（记录 Pipeline 各 Phase 完成状态）之外，v14 额外引入 `test_result/.state/pending_task.json` 用于任务分发追踪。`pending_task.json` 记录当前 `TaskDispatcher` 派发给 TRAE 后端的待执行任务（含 phase_num、prompt 摘要、dispatch 时间等），与 `.agent_state.json` 配合实现"派发-产物检测-状态推进"的闭环：
> - `.agent_state.json`：Pipeline 级状态（各 Phase 是否完成、创新是否成立、迭代轮次等）
> - `test_result/.state/pending_task.json`：任务级状态（当前派发了什么任务、是否已被外部后端领取执行、是否产出产物）

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
│   ┌─────────┐    ┌─────────┐    ┌─────────┐               │
│   │  Scout  │───▶│ Execute │───▶│Reflect  │               │
│   │(分析机会)│    │(验证方法)│    │(评估决策)│               │
│   └─────────┘    └─────────┘    └─────────┘               │
│        ▲                               │                     │
│        │         loop                  │                     │
│        └───────────────────────────────┘                     │
└─────────────────────────────────────────────────────────────┘
```

### Scout 阶段（分析优化机会）

**触发时机**：每次迭代开始前

**任务**：基于历史验证结果，分析下一个最有潜力的优化方向

### Execute 阶段（执行验证）

**任务**：对 Scout 提出的候选方法/假设进行十折交叉验证

### Reflection 阶段（评估决策）

**触发时机**：每次验证完成后

**任务**：评估验证结果，决定是否接受优化，并更新 StateTracker

**决策逻辑**：

```
验证结果评估
    │
    ├─ R² 提升 ≥ 0.01 AND 跨天稳定
    │   → 接受优化，更新 StateTracker
    │
    ├─ R² 提升 ≥ 0.01 BUT 跨天不稳定（参数敏感）
    │   → 拒绝优化
    │   → 记录失败原因：参数迁移性差
    │
    └─ R² 提升 < 0.01
        → 拒绝优化
        → 记录失败原因：提升不足
```

---

## 7.8 技术写作员（论文写作Agent）

### 核心任务

**当创新成立时**，技术写作Agent负责：
1. 收集研究过程中的决策记录和实验结果
2. 按照学术论文结构组织内容
3. 生成 **LaTeX 论文源码** 并编译为 **PDF**

### 输出产物

```
paper_output/
├── paper.tex              # 主论文 LaTeX 源码
├── paper.pdf             # 编译后的 PDF
├── references.bib         # BibTeX 参考文献
├── figures/              # 图表目录
└── tech_report.md        # 详细技术报告（过程记录）
```

### 论文结构要求

| 章节 | 内容来源 | 说明 |
|------|----------|------|
| Title | 最佳方法名称 | 简洁，包含核心创新点 |
| Abstract | 研究成果总结 | R²提升、RMSE改善 |
| Introduction | 研究背景 | PM2.5监测的重要性 |
| Related Work | LocalPaperLibrary/ + **PyMuPDF/pdfplumber 解析** | 引用已有方法论文 |
| Methodology | SmartToCode/创新方法指令/ | 详细描述提出的方法 |
| Experiments | test_result/历史最佳方案/ | 十折验证结果 |
| Results | comparison_report.md | 方法对比表 |
| Discussion | innovation_note.md | 分析成败原因 |
| Conclusion | 总结与展望 | 贡献总结 |

### 触发条件

**自动触发**：当满足以下任一条件时，技术写作Agent自动启动
- 创新判定通过（R²提升≥0.01）
- 人类说"停止"并要求生成论文

---

*来源文档：PM2.5_CMAQ融合方法自动研究全流程文档_v11_agent_spawn.md*
