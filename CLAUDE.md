# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an **automated PM2.5 CMAQ fusion method research system** that performs:
- Literature analysis → Method design → Code implementation → Testing → Paper generation
- Combines ground monitoring data with CMAQ (chemical transport model) simulations
- Best innovative method: **AdvancedRK** (R²=0.916 in stage1, 4/4 stages verified) or **PolyRK** (R²=0.911, 4/4 stages verified)

## Key Architecture

### Two-Layer Method System

**Baseline Methods** (`CodeWorkSpace/基准方法代码/`, `Code/Downscaler/`, `Code/VNAeVNAaVNA/`):
- VNA (Voronoi Neighbor Average) - spatial interpolation
- eVNA (enhanced VNA) - multiplicative bias correction
- aVNA (additive VNA) - additive bias correction
- Downscaler - MCMC-based downscaling

**Innovative Methods** (`CodeWorkSpace/新融合方法代码/`):
- RK-Poly, SuperEnsemble, PolyEnsemble, MSEF, ARK_OLS, AdvancedRK, RobustRK, PG_STGAT, VCFFM, etc.
- Core innovation: Polynomial OLS + GPR residual modeling + ensemble

### Ten-Fold Cross Validation

**Standard Mode** (VNA/eVNA/aVNA/RK-Poly):
```
Train: 9-fold monitoring data + 9-fold CMAQ grid values
Predict: CMAQ grid coordinates of 1-fold sites
```

**Special Mode** (Downscaler):
```
Train: 9-fold monitoring data + full grid CMAQ
Predict: Full grid (required), then extract test site values
```

### Multi-Stage Validation

| Stage | Period | Purpose |
|-------|--------|---------|
| pre_exp | 2020-01-01 ~ 2020-01-05 | Quick screening |
| stage1 | January 2020 | Winter validation |
| stage2 | July 2020 | Summer validation |
| stage3 | December 2020 | Winter validation |

### Innovation Criteria

**主级创新**（三条件必须同时满足）:
| Metric | Requirement |
|--------|-------------|
| R² | ≥ best baseline R² + 0.01 |
| RMSE | ≤ best baseline RMSE |
| \|MB\| | ≤ best baseline \|MB\| |

**次级创新**:
- 条件：R² > baseline（只需大于基线，无需+0.01）

**验证流程**:
```
pre_exp → stage1 → stage2 → stage3
  ↓         ↓        ↓        ↓
 失败     失败     继续    主级创新
        停止      ↓      
                stage3通过→次级创新
                stage3未通过→创新失败
```

## Common Commands

```bash
# Multi-stage baseline validation (4 stages × 5 methods)
python test_result/基准方法/validate_baseline_multistage.py

# Ten-fold validation for specific innovative method
python test_result/创新方法/PolyRK_十折标准模式.py
python test_result/创新方法/AdvancedRK_十折标准模式.py

# Run all innovative methods validation
python test_result/创新方法/validate_all_methods.py

# Agent workflow
python run_pipeline.py --status
```

## Test Data

- `test_data/fold_split_table.csv` - 10-fold cross-validation site assignments
- `test_data/fold_split_table_daily.csv` - Daily fold assignments for multi-stage
- `test_data/raw/CMAQ/2020_PM25.nc` - CMAQ model output (NetCDF)
- `test_data/raw/Monitor/2020_DailyPM2.5Monitor.csv` - Ground monitoring data

## Directory Structure

```
Code/                          # Core reference implementations
  ├── VNAeVNAaVNA/            # VNA/eVNA/aVNA fusion module
  └── Downscaler/             # Downscaler variants

CodeWorkSpace/
  ├── 基准方法代码/            # Baseline methods
  ├── 复现方法代码/            # Reproduced methods from literature
  └── 新融合方法代码/          # Innovative methods under test

test_result/
  ├── 基准方法/               # Baseline validation results
  ├── 创新方法/               # Innovative method results (/*_十折标准模式.py)
  └── snapshots/             # State snapshots for resume

Innovation/success/           # Confirmed innovative methods
Innovation/failed/            # Failed methods
```

## Code Import Patterns

```python
# 推荐方式：使用 shared.paths（自动解析项目根目录，无需硬编码）
from shared.paths import get_project_root, data_path
import sys, os
root = str(get_project_root())

# For baseline methods
sys.path.insert(0, os.path.join(root, 'Code/Downscaler'))
from pm25_downscaler import PM25Downscaler
from common_setting import CommonSetting
from Code.VNAeVNAaVNA.nna_methods import NNA

# For innovative methods
sys.path.insert(0, os.path.join(root, 'CodeWorkSpace/新融合方法代码'))
from PolyRK import PolyRK
from AdvancedRK import AdvancedRK

# For reproduced methods
sys.path.insert(0, os.path.join(root, 'CodeWorkSpace/复现方法代码'))
from ReproductionMethods import BayesianDataAssimilation, GPDownscaling
```

## Key Metrics (Four-Stage Validation)

**Baseline Methods** (VNA is best baseline):
| Stage | VNA R² | aVNA R² | eVNA R² | 主级阈值 |
|-------|--------|---------|---------|----------|
| pre_exp | 0.8907 | 0.8883 | 0.8842 | 0.9007 |
| stage1 (Jan) | 0.9034 | 0.9014 | 0.8913 | 0.9134 |
| stage2 (Jul) | 0.8408 | 0.8175 | 0.7595 | 0.8508 |
| stage3 (Dec) | 0.9031 | 0.9007 | 0.8924 | 0.9131 |

**Confirmed Innovative Methods** (✅ 4/4 stages verified):
| Method | Stage1 R² | Stage2 R² | Stage3 R² | Notes |
|--------|-----------|-----------|-----------|-------|
| **PolyRK** | 0.9105 | 0.8474 | 0.9060 | Core innovation (OLS+GPR-RBF) |
| **AdvancedRK** | 0.9162 | 0.8526 | 0.9129 | GPR-Matern kernel, best overall |

**Excluded Methods**: PSK, CSPRK (no real innovation), MSRK/GARK/CGARK (no clear advantage)
