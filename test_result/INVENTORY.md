# test_result 清单

生成时间：2026-05-08
最后更新：2026-05-08

## 目录结构

```
test_result/
├── INVENTORY.md              # 本清单
├── .state/                   # 状态追踪
├── 基准方法/                  # 基准方法验证结果
├── 复现方法/                 # 复现方法验证结果
├── 创新方法/                 # 创新方法验证结果
├── 历史/                     # 历史归档
│   ├── cross_day_validation/
│   └── 年平均融合测试/
├── legacy_tests/             # 已归档的历史测试
├── snapshots/                # 状态快照
├── comparison_report.md      # 对比报告
└── gVNA_full_domain/         # 全域融合结果（NetCDF）
```

## 基准方法验证结果

### 四阶段验证（最新）

| 方法 | pre_exp R² | Stage1 R² | Stage2 R² | Stage3 R² | 状态 |
|------|------------|-----------|-----------|-----------|------|
| VNA | 0.8907 | 0.9034 | 0.8408 | 0.9031 | ✅ 基准线 |
| aVNA | 0.8883 | 0.9014 | 0.8175 | 0.9007 | ✅ |
| eVNA | 0.8842 | 0.8913 | 0.7595 | 0.8924 | ✅ |
| Downscaler | - | - | - | - | 待验证 |

### 基准阈值（VNA方法）

| 阶段 | 时间范围 | R² > | RMSE ≤ | \|MB\| ≤ |
|------|----------|-------|--------|----------|
| pre_exp | 2020-01-01~05 | 0.8907 | 16.68 | 0.70 |
| stage1 | 2020-01 | 0.9034 | 16.48 | 0.50 |
| stage2 | 2020-07 | 0.8408 | 5.05 | 0.05 |
| stage3 | 2020-12 | 0.9031 | 12.20 | 0.42 |

## 复现方法验证结果

| 方法 | Stage1 R² | Stage2 R² | Stage3 R² | 状态 |
|------|-----------|-----------|-----------|------|
| OMA | - | - | - | 待验证 |
| SMA | - | - | - | 待验证 |
| MMA | - | - | - | 待验证 |
| QuantileMapping | - | - | - | 待验证 |
| SpatialKriging | - | - | - | 待验证 |
| ODI | - | - | - | 待验证 |
| BiasCorrection | - | - | - | 待验证 |
| EnsembleMean | - | - | - | 待验证 |
| OptimumInterpolation | - | - | - | 待验证 |
| DiffusionSmoothing | - | - | - | 待验证 |
| STK | - | - | - | 待验证 |
| NC | - | - | - | 待验证 |
| BayesianDA | - | - | - | 待验证 |
| GPDownscaling | - | - | - | 待验证 |
| HDGC | - | - | - | 待验证 |
| UniversalKriging | - | - | - | 待验证 |
| IDWBias | - | - | - | 待验证 |
| GenFriberg | - | - | - | 待验证 |
| FC1 | - | - | - | 待验证 |
| FC2 | - | - | - | 待验证 |
| FCopt | - | - | - | 待验证 |
| DDNet | - | - | - | 待验证 |
| BayesianSTK | - | - | - | 待验证 |
| NeuroDDAF | - | - | - | 待验证 |
| KiCDPM | - | - | - | 待验证 |
| BSMFM | - | - | - | 待验证 |
| KrigingPseudoLabel | - | - | - | 待验证 |
| RF-Kriging | - | - | - | 待验证 |
| MLE-OI | - | - | - | 待验证 |

## 创新方法验证结果

### 已确认创新（4/4阶段通过）

| 方法 | pre_exp R² | Stage1 R² | Stage2 R² | Stage3 R² | 状态 |
|------|------------|-----------|-----------|-----------|------|
| **AdvancedRK** | 0.9047 | 0.9162 | 0.8526 | 0.9129 | ✅ 最优 |
| **PolyRK** | - | 0.9105 | 0.8474 | 0.9060 | ✅ 核心创新 |

### 验证失败

| 方法 | 失败阶段 | 失败原因 |
|------|----------|----------|
| ARK_OLS | - | 验证失败 |
| BayesianVariationalFusion | - | 验证失败 |
| CGARK | - | IDW类无明确优势 |
| GARK | - | IDW类无明确优势 |
| MSAGARK | - | IDW类无明确优势 |
| PG-STGAT | - | 图网络路线验证失败 |
| VCFFM | - | 验证失败 |

### 待验证方法（42个）

| 方法 | 状态 |
|------|------|
| HybridEAVNA | 待验证 |
| ResidualKriging | 待验证 |
| PDEICNN | 待验证 |
| PolyGPRAdapt | 待验证 |
| ConservativeTransport | 待验证 |
| MSEF | 待验证 |
| VG_VNA | 待验证 |
| CR_ABC | 待验证 |
| SLOOCV_AK | 待验证 |
| GDIDW | 待验证 |
| gVNA | 待验证 |
| CopulaSpatialFusion | 待验证 |
| WaveletGPR | 待验证 |
| ... | 共42个方法 |

## 状态追踪

| 文件 | 说明 |
|------|------|
| .state/ledger.jsonl | 决策记录 |
| .state/research_status.md | 研究状态 |

## 规范说明

- **一角色一清单**：每个角色只在自己目录生成一份清单
- **一次一版本**：报告类文件只保留最新版本
- **机器可读**：汇总数据必须是 CSV/JSON
- **人类可读**：报告必须是 Markdown/PDF

---
更新时间: 2026-05-08
