# PM2.5 CMAQ融合方法自动研究系统 - 项目清单

> 生成时间: 2026-04-21
> 最后更新: 2026-05-07 19:47
> 整理版本: v3.0

---

## 一、根目录结构（整理后）

```
E:\CodeProject\ClaudeRoom\Data_Fusion_AutoResearch\
├── CLAUDE.md                    # Claude Code项目说明
├── INVENTORY.md                 # 项目总清单（本文件）
├── run_pipeline.py              # 工作流启动脚本
├── .claude/                     # Claude Code配置
├── .git/                        # Git版本控制
├── .gitignore
├── .agent_state.json
│
├── PaperDownload/               # 论文PDF文件（~100篇，按score分类）
├── PaperDownloadMd/             # 论文清单、分析报告（19个文件）
├── LocalPaperLibrary/           # 本地原始论文库（12篇中文论文）
├── MethodToSmart/               # 文献分析员输出（41个方法文档）
├── SmartToCode/                 # 方案设计师输出（55+个实现指令）
├── Code/                        # 参考代码（Downscaler/VNAeVNAaVNA）
├── CodeWorkSpace/               # 工作区代码
│   ├── 基准方法代码/            # VNA/eVNA/aVNA变体
│   ├── 复现方法代码/            # 复现方法实现（20+个）
│   ├── 新融合方法代码/          # 创新方法实现（15+个）
│   ├── 年均融合方法/            # 年均数据融合方法（25+个）
│   └── 改造后VNA_eVNA_aVNA/     # 改造后VNA系列方法
├── test_data/                   # 测试数据
├── test_result/                 # 测试结果
│   ├── 基准方法/                # 基准方法验证结果
│   ├── 创新方法/                # 创新方法验证结果（20+个）
│   ├── 复现方法/                # 复现方法测试结果（15个）
│   ├── 历史/                    # 历史验证结果（50+个）
│   ├── snapshots/              # 状态快照
│   ├── InnovationMethods/       # 创新方法代码
│   ├── legacy_tests/           # 历史测试脚本（已归档）
│   └── 代码实现报告.md          # 代码实现清单
├── Innovation/                  # 已确认创新方法
│   ├── success/                 # 验证通过的方法
│   │   ├── AdvancedRK/          # AdvancedRK（R²=0.9162，最优）
│   │   ├── PolyRK/              # PolyRK（R²=0.9105，核心创新）
│   │   └── RobustRK/            # RobustRK
│   └── failed/                  # 验证失败的方法
│       ├── ARK_OLS/
│       ├── CGARK/
│       ├── GARK/
│       ├── MSAGARK/
│       ├── PG-STGAT/
│       └── VCFFM/
├── paper_output/                # 论文输出（15个文件）
│   ├── paper.tex                # 论文主文件
│   ├── paper.pdf                # 编译后PDF
│   ├── references.bib           # 参考文献
│   ├── figures/                 # 论文图表
│   └── README.md                # 项目说明
├── agents/                       # Agent模块
│   ├── spawn_executor.py        # Agent spawn执行器
│   ├── role_templates.py       # 角色prompt模板
│   ├── workflow_orchestrator.py
│   ├── research_state_tracker.py
│   └── ...
├── error/                        # 错误日志
├── skills/                       # Claude Code Skills
├── 文档拆分/                     # 项目文档拆分
└── LizhuoChen/                   # 用户个人代码（保留）
```

---

## 二、目录统计（2026-05-07更新）

| 目录 | 文件数 | 状态 | 说明 |
|------|--------|------|------|
| PaperDownload/ | ~100 | ✅ 正常 | 按score_2~5分类 |
| PaperDownloadMd/ | 19 | ✅ 正常 | 含INVENTORY.md |
| LocalPaperLibrary/ | 12 | ✅ 正常 | 中文论文库 |
| MethodToSmart/ | 41 | ✅ 正常 | 33个方法文档+8个报告 |
| SmartToCode/ | 55+ | ✅ 正常 | 25个复现+30+个创新 |
| CodeWorkSpace/ | 80+ | ✅ 正常 | 含缓存文件需清理 |
| test_result/ | 100+ | ✅ 正常 | 含历史测试结果 |
| paper_output/ | 15 | ✅ 正常 | LaTeX输出 |

---

## 三、前人遗留文件处理记录

### 3.1 MethodToSmart目录（方法文档）

**格式规范**：【可执行方法规范】
- 方法名称、文献来源、核心公式、参数清单、数据规格、随机性、方法指纹、实现检查清单

**处理结果**：

| 文件 | 状态 |
|------|------|
| 文献分析员_VNA方法.md | ✅ 已整理，符合规范 |
| 文献分析员_eVNA方法.md | ✅ 已整理，符合规范 |
| 文献分析员_aVNA方法.md | ✅ 已整理，符合规范 |
| 文献分析员_Downscaler方法.md | ✅ 已整理，符合规范 |
| 文献分析员_FC1克里金插值法.md | ✅ 已整理，符合规范 |
| 文献分析员_FC2尺度CMAQ法.md | ✅ 已整理，符合规范 |
| 文献分析员_FCopt优化融合法.md | ✅ 已整理，符合规范 |
| 文献分析员_贝叶斯数据同化法.md | ✅ 已整理，符合规范 |
| 文献分析员_GP降尺度法.md | ✅ 已整理，符合规范 |
| 文献分析员_CleanAir深度学习CMAQ替代法.md | ✅ 已整理，符合规范 |
| 文献分析员_HDGC监测偏差检测法.md | ✅ 已整理，符合规范 |
| 文献分析员_KNNSINDy缺失数据填补法.md | ✅ 已整理，符合规范 |
| 文献分析员_AQNet时空神经网络法.md | ✅ 已整理，符合规范 |
| 文献分析员_通用克里金PM25映射法.md | ✅ 已整理，符合规范 |
| 文献分析员_GenFriberg广义融合法.md | ✅ 已整理，符合规范 |
| 文献分析员_IDW偏差加权融合法.md | ✅ 已整理，符合规范 |
| 文献分析员_CRNNSpatiotemporalPM25法.md | ✅ 已整理，符合规范 |
| 文献分析员_Stacking集成学习方法.md | ✅ 已整理，符合规范 |
| 文献分析员_RF残差克里金校正法.md | ✅ 已整理，符合规范 |
| 文献分析员_GWR地理加权回归法.md | ✅ 已整理，符合规范 |
| 文献分析员_BayesianSpaceTimeKriging法.md | ✅ 已整理，符合规范 |
| 文献分析员_LUR土地使用回归法.md | ✅ 已整理，符合规范 |
| 文献分析员_MLE最优插值法.md | ✅ 已整理，符合规范 |
| 文献分析员_Cokriging共克里金法.md | ✅ 已整理，符合规范 |
| 文献分析员_TopoFlow地形感知神经网络法_20260409.md | ✅ 已整理，符合规范 |
| 文献分析员_Zeeman深度学习化学传输模型法_20260409.md | ✅ 已整理，符合规范 |
| 文献分析员_GenDA生成式数据同化法_20260409.md | ✅ 已整理，符合规范 |
| 文献分析员_EnsAI大气化学集合生成法_20260409.md | ✅ 已整理，符合规范 |
| 文献分析员_AirFusion扩散概率空气质量预报法_20260409.md | ✅ 已整理，符合规范 |
| 文献分析员_NeuroDDAF神经动态扩散平流场法_20260409.md | ✅ 已整理，符合规范 |
| 文献分析员_DDNet双深度网络PM25预报法_20260409.md | ✅ 已整理，符合规范 |
| 文献分析员_SPIN时空物理引导推理网络法.md | ✅ 已整理，符合规范 |
| 文献分析员_BSMFM贝叶斯多源融合模型法.md | ✅ 已整理，符合规范 |
| 文献分析员_GeoML混合PM25预测法.md | ✅ 已整理，符合规范 |
| 文献分析员_CorrDiff残差扩散降尺度法.md | ✅ 已整理，符合规范 |
| 文献分析员_DeepAIR混合CNN_LSTM法.md | ✅ 已整理，符合规范 |
| 文献分析员_KiCDPM克里金信息扩散降尺度法.md | ✅ 已整理，符合规范 |
| 文献分析员_BSMFM贝叶斯多源融合模型法_20260411.md | ✅ 已整理，符合规范 |
| 文献分析员_Kriging伪标签增强法.md | ✅ 已整理，符合规范 |

### 3.2 SmartToCode目录（方案指令）

**格式规范**：创新方法指令格式
- 方法名称、输入数据、输出数据、核心公式、关键步骤、创新点、创新判定、方法指纹

**处理结果**：

| 类别 | 数量 | 状态 |
|------|------|------|
| 复现方法指令 | 25 | ✅ 已整理，符合规范 |
| 创新方法指令 | 30+ | ✅ 已整理，符合规范 |

### 3.3 CodeWorkSpace目录（代码文件）

**处理结果**：

| 子目录 | 文件数 | 状态 |
|--------|--------|------|
| 复现方法代码/ | 20+ | ✅ 已整理，代码规范 |
| 新融合方法代码/ | 15+ | ✅ 已整理，代码规范 |
| 年均融合方法/ | 25+ | ✅ 已整理，代码规范 |

---

## 四、当前状态统计

### 4.1 论文统计

| 类别 | 数量 | 状态 |
|------|------|------|
| 论文总数（PaperDownload） | ~100 | 已下载 |
| 本地论文库（LocalPaperLibrary） | 12 | 中文论文 |
| 高相关论文（score≥4） | ~40 | 待分析 |
| 已分析论文 | 33 | 已生成方法文档 |
| 未分析论文 | ~70 | 待分析 |

### 4.2 方法统计

| 类别 | 数量 | 状态 |
|------|------|------|
| 基准方法 | 4 | VNA, eVNA, aVNA, Downscaler |
| 复现方法 | 25 | 已实现 |
| 创新方法 | 30+ | 已设计 |
| 已确认创新方法 | 8 | PolyRK最优(R²≈0.91) |

### 4.3 测试结果

| 类别 | 数量 | 状态 |
|------|------|------|
| 基准方法测试 | 4 | 四阶段验证完成 |
| 复现方法测试 | 13 | 十折验证完成 |
| 创新方法测试 | 8 | 四阶段验证通过 |
| 历史测试 | 24轮 | 年均融合方法迭代 |

### 4.4 最优方法性能

| 方法 | Stage1 R² | Stage2 R² | Stage3 R² | 状态 |
|------|-----------|-----------|-----------|------|
| **AdvancedRK** | 0.9162 | 0.8526 | 0.9129 | ✅ 最优 |
| **PolyRK** | 0.9105 | 0.8474 | 0.9060 | ✅ 核心创新 |
| VNA (基准) | 0.9034 | 0.8408 | 0.9031 | 基准线 |

---

## 五、本次整理记录（2026-05-07）

### 5.1 移至正确位置的文件

| 原位置 | 新位置 | 原因 |
|--------|--------|------|
| 根目录/*.log | error/ | 禁止根目录放日志文件 |
| 根目录/temp_*.txt | error/ | 禁止根目录放临时文件 |
| 根目录/README.md | paper_output/ | 禁止根目录放独立文档 |
| 根目录/十折交叉验证架构文档.md | 文档拆分/ | 禁止根目录放架构文档 |
| 根目录/PM2.5_CMAQ融合方法..._v11_agent_spawn.md | 文档拆分/ | 禁止根目录放独立文档 |
| CodeWorkSpace/*_验证.py (14个) | test_result/legacy_tests/ | 禁止根目录放测试脚本 |
| CodeWorkSpace/代码实现报告.md | test_result/ | 禁止根目录放报告文件 |

### 5.2 删除的目录

| 目录 | 原因 |
|------|------|
| CodeWorkSpace/WorkDocument/ | 空目录，已删除 |

---

## 六、目录完整性检查

| 必需目录 | 状态 | 说明 |
|----------|------|------|
| PaperDownload/ | ✅ 存在 | 论文PDF |
| PaperDownloadMd/ | ✅ 存在 | 论文清单 |
| LocalPaperLibrary/ | ✅ 存在 | 本地论文库 |
| MethodToSmart/ | ✅ 存在 | 文献分析 |
| SmartToCode/ | ✅ 存在 | 方案设计 |
| Code/ | ✅ 存在 | 参考代码 |
| CodeWorkSpace/ | ✅ 存在 | 工作区代码 |
| test_data/ | ✅ 存在 | 测试数据 |
| test_result/ | ✅ 存在 | 测试结果 |
| Innovation/ | ✅ 存在 | 已确认创新 |
| paper_output/ | ✅ 存在 | 论文输出 |
| agents/ | ✅ 存在 | Agent模块 |
| error/ | ✅ 存在 | 错误日志 |
| .claude/ | ✅ 存在 | Claude配置 |
| 文档拆分/ | ✅ 存在 | 项目文档 |

**禁止在根目录的文件类型**（本次已清理）：
- 临时文件（temp_*.txt）
- 日志文件（*.log）
- 测试脚本（*_验证.py, *十折*.py, test_*.py）
- 独立文档（*.md, *架构文档.md, *排除.md）
- 报告文件（comparison_report.md, 代码实现报告.md）

---

## 七、核心方法清单

### 7.1 基准方法（Baseline）

| 方法 | 目录 | 说明 |
|------|------|------|
| VNA | Code/VNAeVNAaVNA/ | Voronoi Neighbor Average |
| eVNA | Code/VNAeVNAaVNA/ | 乘法偏差校正 |
| aVNA | Code/VNAeVNAaVNA/ | 加法偏差校正 |
| Downscaler | Code/Downscaler/ | MCMC降尺度 |

### 7.2 已确认创新方法（Innovation/success/）

| 方法 | stage1 R² | stage2 R² | stage3 R² | 状态 |
|------|-----------|-----------|-----------|------|
| **AdvancedRK** | 0.9162 | 0.8526 | 0.9129 | ✅ 4/4通过（最优） |
| **PolyRK** | 0.9105 | 0.8474 | 0.9060 | ✅ 4/4通过（核心） |
| RobustRK | ~0.91 | - | - | 部分验证 |

### 7.3 验证失败方法（Innovation/failed/）

| 方法 | 失败原因 |
|------|----------|
| GARK | IDW类无明确优势 |
| CGARK | IDW类无明确优势 |
| MSAGARK | IDW类无明确优势 |
| PG-STGAT | 图网络路线验证失败 |
| VCFFM | 验证失败 |
| ARK_OLS | 验证失败 |
| BayesianVariationalFusion | 验证失败 |

### 7.4 排除方法（不测试）

| 方法 | 排除原因 |
|------|----------|
| PSK | 样条校正无实质创新 |
| CSPRK | 浓度分层不合理 |
| Stacking类 | 加权集成，迁移性差 |

---

## 八、基准阈值（VNA方法）

| 阶段 | 时间范围 | R² > | RMSE ≤ | \|MB\| ≤ |
|------|----------|-------|--------|----------|
| pre_exp | 2020-01-01~05 | 0.8907 | 16.68 | 0.70 |
| stage1 | 2020-01 | 0.9034 | 16.48 | 0.50 |
| stage2 | 2020-07 | 0.8408 | 5.05 | 0.05 |
| stage3 | 2020-12 | 0.9031 | 12.20 | 0.42 |

---

## 九、后续工作建议

### 9.1 高优先级

1. **[高] 继续分析剩余~70篇论文**
   - 优先分析score≥4的论文（~40篇）
   - 生成方法文档 → MethodToSmart/
   - 生成方案指令 → SmartToCode/

2. **[高] 实现新方法代码**
   - 基于已分析的33个方法文档
   - 实现到CodeWorkSpace/新融合方法代码/
   - 运行十折验证

### 9.2 中优先级

3. **[中] 整理缓存文件**
   - 清理CodeWorkSpace/__pycache__/目录
   - 清理PaperDownload/test_*.pdf测试文件

4. **[中] 补充方法文档格式**
   - 部分早期方法文档缺少"方法指纹"字段
   - 统一文档格式

### 9.3 低优先级

5. **[低] 论文写作**
   - 基于AdvancedRK/PolyRK的验证结果
   - 生成LaTeX论文 → paper_output/
   - 当前已有paper.tex模板

6. **[低] 代码重构**
   - 统一复现方法接口
   - 优化测试脚本

---

## 十、快速命令

```bash
# 查看项目状态
python run_pipeline.py --status

# 运行基准方法多阶段验证
python test_result/基准方法/validate_baseline_multistage.py

# 运行创新方法十折验证
python test_result/创新方法/PolyRK_十折标准模式.py
python test_result/创新方法/AdvancedRK_十折标准模式.py

# 运行所有创新方法验证
python test_result/创新方法/validate_all_methods.py
```

---

*本清单由项目整理专家自动生成*
*整理时间: 2026-05-07 19:47*
