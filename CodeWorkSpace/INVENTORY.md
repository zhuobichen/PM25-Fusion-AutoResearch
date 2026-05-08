# CodeWorkSpace 代码清单

生成时间: 2026-05-07

## 复现方法代码 (CodeWorkSpace/复现方法代码/)

| 文件名 | 方法简称 | 核心技术 | 状态 |
|--------|---------|---------|------|
| VNA.py | VNA | Voronoi邻域平均 | ✅ 已实现 |
| eVNA.py | eVNA | 乘性偏差校正 | ✅ 已实现 |
| aVNA.py | aVNA | 加性偏差校正 | ✅ 已实现 |
| STK.py | STK | 时空克里金 | ✅ 已实现 |
| GPDownscaling.py | GPD | 高斯过程降尺度 | ✅ 已实现 |
| HDGC.py | HDGC | 偏差检测 | ✅ 已实现 |
| IDWBias.py | IDWB | IDW偏差加权 | ✅ 已实现 |
| UniversalKriging.py | UK | 通用克里金 | ✅ 已实现 |
| BayesianDA.py | BDA | 贝叶斯同化 | ✅ 已实现 |
| GenFriberg.py | GF | 广义融合 | ✅ 已实现 |
| FC1.py | FC1 | 克里金插值 | ✅ 已实现 |
| FC2.py | FC2 | CMAQ尺度 | ✅ 已实现 |
| FCopt.py | FCopt | 优化融合 | ✅ 已实现 |
| DiffusionSmoothing.py | DS | 扩散平滑 | ✅ 已实现 |
| DDNet.py | DDNet | 双网络同化 | ✅ 已实现 |
| BayesianSTK.py | BSTK | 贝叶斯时空克里金 | ✅ 已实现 |
| NeuroDDAF.py | NDAF | 物理引导神经ODE | ✅ 已实现 |
| ReproductionMethods.py | 多方法 | 统一复现接口 | ✅ 已实现 |
| BaseFusionMethods.py | 基础 | 基础融合方法 | ✅ 已实现 |
| NC.py | NC | 华北多源融合 | ✅ 已实现 |
| **OMA_SMA_MMA.py** | OMA/SMA/MMA | 观测/统计/混合模型聚合 | ✅ 新增 |
| **QuantileMapping.py** | QM | 分位数映射偏差校正 | ✅ 新增 |
| **SpatialKriging_ODI.py** | SK/ODI | 空间克里金/观测偏差指示器 | ✅ 新增 |
| **BiasCorrection_EnsembleMean_OI.py** | BC/EM/OI | 偏差校正/集合平均/最优插值 | ✅ 新增 |
| **NC_KiCDPM_BSMFM.py** | NC/KiCDPM/BSMFM | 华北融合/克里金扩散/贝叶斯多源 | ✅ 新增 |
| **KrigingPseudoLabel_RF_Kriging_MLE_OI.py** | KPL/RF-K/MLE-OI | 伪标签/随机森林克里金/MLE最优插值 | ✅ 新增 |

## 创新方法代码 (CodeWorkSpace/新融合方法代码/)

| 文件名 | 方法简称 | 核心创新点 | 状态 |
|--------|---------|-----------|------|
| PolyRK.py | PolyRK | 多项式OLS+GPR残差 | ✅ 已实现 |
| AdvancedRK.py | AdvancedRK | GPR-Matern核 | ✅ 已实现 |
| RobustRK.py | RobustRK | 鲁棒残差克里金 | ✅ 已实现 |
| PG_STGAT.py | PG-STGAT | 物理引导图注意力 | ✅ 已实现 |
| VCFFM.py | VCFFM | 变分协方差场 | ✅ 已实现 |
| HybridEAVNA.py | HybridEAVNA | eVNA+aVNA混合 | ✅ 已实现 |
| ResidualKriging.py | RK | 自适应变异函数残差克里金 | ✅ 已实现 |
| PDEICNN.py | PDEICNN | PDE硬约束凸神经网络 | ✅ 已实现 |
| PolyGPRAdapt.py | PolyGPRAdapt | 大气稳定度自适应GPR | ✅ 已实现 |
| ConservativeTransport.py | CT | 质量守恒传输映射 | ✅ 已实现 |
| MSEF.py | MSEF | 多源融合 | ✅ 已实现 |
| VG_VNA.py | VG-VNA | 变异函数几何VNA | ✅ 已实现 |
| CR_ABC.py | CR-ABC | 浓度体制ABC | ✅ 已实现 |
| SLOOCV_AK.py | SLOOCV-AK | 空间LOO自适应克里金 | ✅ 已实现 |
| GDIDW.py | GDIDW | 梯度方向IDW | ✅ 已实现 |
| gVNA.py | gVNA | 广义VNA | ✅ 已实现 |
| ARK_OLS.py | ARK-OLS | 自适应RK-OLS | ✅ 已实现 |
| BayesianVariationalFusion.py | BVF | 贝叶斯变分融合 | ✅ 已实现 |
| BMSF_Geostat.py | BMSF | 贝叶斯多源地统计融合 | ✅ 已实现 |
| CGARK.py | CGARK | 浓度梯度自适应RK | ✅ 已实现 |
| CSPRK.py | CSPRK | 浓度分层多项式RK | ✅ 已实现 |
| CSPRKATO.py | CSPRK-ATO | 自适应阈值CSP-RK | ✅ 已实现 |
| CSPRKHLG.py | CSPRK-HLG | 混合层GPR CSP-RK | ✅ 已实现 |
| CSPRKINT.py | CSPRK-INT | 交互项CSP-RK | ✅ 已实现 |
| CorrDiff_Downscaling.py | CorrDiff | 残差扩散降尺度 | ✅ 已实现 |
| EnsembleRK.py | EnsembleRK | 集成RK | ✅ 已实现 |
| GARK.py | GARK | 梯度自适应RK | ✅ 已实现 |
| HGPRK.py | HGPRK | 异方差GPR-RK | ✅ 已实现 |
| LBGPR.py | LBGPR | 局部带宽GPR | ✅ 已实现 |
| MKGPRK.py | MKGPRK | 多核GPR-RK | ✅ 已实现 |
| MSAGARK.py | MSAGARK | 多尺度自适应GARK | ✅ 已实现 |
| MSAK.py | MSAK | 多尺度自适应克里金 | ✅ 已实现 |
| MSRK.py | MSRK | 多尺度残差克里金 | ✅ 已实现 |
| PSK.py | PSK | 多项式样条克里金 | ✅ 已实现 |
| RRK.py | RRK | 鲁棒残差克里金 | ✅ 已实现 |
| SPIN_GraphKernel_Kriging.py | SPIN-GKK | 图核克里金 | ✅ 已实现 |
| SQDM.py | SQDM | 空间分位数映射 | ✅ 已实现 |
| STRK.py | STRK | 时空残差克里金 | ✅ 已实现 |
| ST_CRK.py | ST-CRK | 时空共克里金 | ✅ 已实现 |
| **CopulaSpatialFusion.py** | CopulaFusion | Copula非高斯空间融合 | ✅ 新增 |
| **WaveletGPR.py** | WaveletGPR | 小波多尺度GPR残差 | ✅ 新增 |

## 统计摘要

| 类别 | 数量 |
|-----|------|
| 复现方法 | 26个 |
| 创新方法 | 42个 |
| **总计** | **68个方法** |

## 本轮新增 (2026-05-07)

### 复现方法 (6个文件，15个方法)
1. OMA_SMA_MMA.py - 观测模型聚合、统计模型聚合、混合模型聚合
2. QuantileMapping.py - 分位数映射偏差校正
3. SpatialKriging_ODI.py - 空间克里金、观测偏差指示器
4. BiasCorrection_EnsembleMean_OI.py - 偏差校正家族、集合平均、最优插值
5. NC_KiCDPM_BSMFM.py - 华北融合、克里金信息扩散、贝叶斯多源融合
6. KrigingPseudoLabel_RF_Kriging_MLE_OI.py - 伪标签增强、随机森林克里金、MLE最优插值

### 创新方法 (2个)
1. CopulaSpatialFusion.py - Copula非高斯空间融合法
2. WaveletGPR.py - 小波多尺度GPR残差融合法

## 方法分类汇总

| 类别 | 方法数 | 代表方法 |
|-----|-------|---------|
| 偏差校正类 | 10 | BC, QM, eVNA, aVNA, IDWB, ODI, VG-VNA, CR-ABC, OMA, SMA |
| 空间插值类 | 12 | VNA, SK, UK, OI, FC1, FC2, STK, KPL, KCDP, EM, NC, MMA |
| 贝叶斯/同化类 | 8 | BDA, BSTK, BSFM, BMF, BayesianDA, MLE-OI, BSMFM, KiCDPM |
| 克里金变体类 | 10 | ResidualKriging, GradAnisoKriging, MSAdaptKriging, SLOOCV-AK, RF-Kriging等 |
| 多项式+GPR类 | 8 | PolyRK, HGP-RK, MKGP-RK, CSP-RK系列, WaveletGPR |
| 神经网络类 | 4 | DDNet, NDAF, PG-STGAT, PDEICNN |
| 融合框架类 | 8 | OMA, SMA, MMA, EM, GF, 华北融合, CopulaFusion, VCFFM |
| 统计创新类 | 6 | CopulaFusion, WaveletGPR, VCFFM, CorrDiff, SQDM, STRK |

---
更新时间: 2026-05-07
