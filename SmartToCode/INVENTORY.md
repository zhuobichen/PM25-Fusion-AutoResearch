# SmartToCode 方案清单

生成时间: 2026-04-09
最后更新: 2026-05-07

## 复现方法指令（共31个）

| 序号 | 文件名 | 方法简称 | 核心技术 |
|-----|--------|---------|---------|
| 1 | OMA方法_观测模型聚合.md | OMA | 观测-模型聚合 |
| 2 | SMA方法_统计模型聚合.md | SMA | 统计模型聚合 |
| 3 | MMA方法_混合模型聚合.md | MMA | 混合模型聚合 |
| 4 | QuantileMapping方法_分位数映射.md | QM | 分位数映射校正 |
| 5 | SpatialKriging方法_空间克里金偏差校正.md | SK | 空间克里金 |
| 6 | ODI方法_观测偏差指示器融合.md | ODI | 偏差指示器 |
| 7 | BiasCorrection方法_简单偏差校正家族.md | BC | 简单偏差校正 |
| 8 | EnsembleMean方法_集合平均.md | EM | 集合平均 |
| 9 | OptimumInterpolation方法_最优插值.md | OI | 最优插值 |
| 10 | DiffusionSmoothing方法_扩散平滑.md | DS | 扩散平滑 |
| 11 | STK方法_时空克里金.md | STK | 时空克里金 |
| 12 | 华北多源融合方法_纯监测CMAQ.md | 华北融合 | 多源融合 |
| 13 | BayesianDA方法_贝叶斯数据同化法.md | BDA | 贝叶斯同化 |
| 14 | GPDownscaling方法_GP降尺度法.md | GPD | 高斯过程降尺度 |
| 15 | HDGC方法_HDGC监测偏差检测法.md | HDGC | 偏差检测 |
| 16 | UniversalKriging方法_通用克里金PM25映射法.md | UK | 通用克里金 |
| 17 | IDWBias方法_IDW偏差加权融合法.md | IDWB | IDW偏差加权 |
| 18 | GenFriberg方法_GenFriberg广义融合法.md | GF | 广义融合 |
| 19 | FC1方法_FC1克里金插值法.md | FC1 | 克里金插值 |
| 20 | FC2方法_FC2尺度CMAQ法.md | FC2 | CMAQ尺度 |
| 21 | FCopt方法_FCopt优化融合法.md | FCopt | 优化融合 |
| 22 | DDNet方法_双深度神经网络法.md | DDNet | 双网络同化 |
| 23 | BayesianSTK方法_贝叶斯时空克里金法.md | BSTK | 贝叶斯时空克里金 |
| 24 | NeuroDDAF方法_神经动态扩散平流场法.md | NDAF | 物理引导神经ODE |
| 25 | Ki-CDPM克里金信息扩散降尺度法.md | KCDP | 克里金+扩散 |
| 26 | BSMFM贝叶斯多源融合模型法.md | BSFM | 贝叶斯多源融合 |
| 27 | Kriging伪标签增强法.md | KPL | 克里金伪标签 |
| 28 | V1_VNA方法.md | VNA | Voronoi邻域平均 |
| 29 | V1_eVNA方法.md | eVNA | 乘性偏差校正 |
| 30 | V1_aVNA方法.md | aVNA | 加性偏差校正 |
| 31 | **RF-Kriging随机森林克里金残差校正法.md** | RF-Kriging | 随机森林+克里金残差 |
| 32 | **MLE-OI最大似然最优插值法.md** | MLE-OI | MLE最优插值 |

## 创新方法指令（共29个）

| 序号 | 文件名 | 方法简称 | 核心创新点 |
|-----|--------|---------|-----------|
| 1 | Innovation_HybridEAVNA.md | HybridEAVNA | eVNA+aVNA混合 |
| 2 | Innovation_ResidualKriging.md | ResidualKriging | 自适应变异函数残差克里金 |
| 3 | Innovation_PDEICNN.md | PDEICNN | PDE硬约束凸神经网络 |
| 4 | Innovation_PolyGPRAdapt.md | PolyGPRAdapt | 大气稳定度自适应GPR |
| 5 | Innovation_ConservativeTransport.md | ConservativeTransport | 质量守恒传输映射 |
| 6 | Innovation_GradientAnisotropicKriging.md | GradAnisoKriging | 梯度各向异性克里金 |
| 7 | Innovation_LocalKernelGPR.md | LocalKernelGPR | 局部带宽GPR |
| 8 | Innovation_SpatialQuantileMapping.md | SpatialQM | 空间分位数映射 |
| 9 | Innovation_多尺度稳定度自适应克里金.md | MSAdaptKriging | 多尺度自适应 |
| 10 | Innovation_时空残差共克里金.md | STCoKriging | 时空共克里金 |
| 11 | Innovation_多尺度残差克里金.md | MSResKriging | 多尺度残差 |
| 12 | Innovation_鲁棒残差克里金.md | RobustResKriging | 鲁棒残差克里金 |
| 13 | Innovation_多项式样条克里金.md | PolySplineKriging | 多项式样条 |
| 14 | Innovation_CMAQ梯度各向异性克里金.md | CMAQGradKriging | CMAQ梯度各向异性 |
| 15 | Innovation_ConcentrationStratifiedPolyRK.md | CSPolyRK | 浓度分层多项式RK |
| 16 | Innovation_HeteroscedasticGPRPolyRK.md | HGP-RK | 异方差GPR多项式RK |
| 17 | Innovation_MultiKernelGPRPolyRK.md | MKGP-RK | 多核GPR多项式RK |
| 18 | Innovation_CSP_RK_AdaptiveThreshold.md | CSP-RK-AT | 自适应阈值CSP-RK |
| 19 | Innovation_CSP_RK_Interaction.md | CSP-RK-Int | 交互项CSP-RK |
| 20 | Innovation_CSP_RK_HybridLayerGPR.md | CSP-RK-HLG | 混合层GPR CSP-RK |
| 21 | Innovation_SPIN_GraphKernel_Kriging.md | SPIN-GKK | 图核克里金 |
| 22 | Innovation_BayesianMultisourceFusion.md | BMF | 贝叶斯多源融合SPDE |
| 23 | Innovation_CorrDiff_Downscaling.md | CorrDiff | 残差扩散降尺度 |
| 24 | Innovation_VG_VNA.md | VG-VNA | 变异函数几何VNA |
| 25 | Innovation_CR_ABC.md | CR-ABC | 浓度体制ABC |
| 26 | Innovation_SLOOCV_AK.md | SLOOCV-AK | 空间LOO自适应克里金 |
| 27 | PG-STGAT物理引导时空图注意力网络法.md | PG-STGAT | 物理引导图注意力 |
| 28 | VCFFM变分协方差场融合模型.md | VCFFM | 变分协方差场 |
| 29 | **Innovation_CopulaSpatialFusion.md** | CopulaFusion | Copula非高斯空间融合 |
| 30 | **Innovation_WaveletGPR.md** | WaveletGPR | 小波多尺度GPR残差 |

## 指纹库统计

| 类别 | 数量 |
|-----|------|
| 复现方法指纹 | 5个 |
| 创新方法指纹 | 7个 |
| 排除方法记录 | 2个 |
| **总计** | **12个有效指纹** |

## 本轮新增（2026-05-07）

| 类型 | 方法 | 指纹 |
|-----|------|------|
| 复现 | RF-Kriging | rf_kriging_residual_random_forest_v1 |
| 复现 | MLE-OI | mle_optimal_interpolation_bayesian_v1 |
| 创新 | CopulaSpatialFusion | copula_non_gaussian_spatial_fusion_gamma_gaussian_v1 |
| 创新 | WaveletGPR | wavelet_multiscale_gpr_residual_db4_3level_v1 |

## 方法分类汇总

| 类别 | 方法数 | 代表方法 |
|-----|-------|---------|
| 偏差校正类 | 8 | BC, QM, eVNA, aVNA, IDWB, ODI, VG-VNA, CR-ABC |
| 空间插值类 | 10 | VNA, SK, UK, OI, FC1, FC2, STK, Cokriging, KPL, KCDP |
| 贝叶斯/同化类 | 6 | BDA, BSTK, BSFM, BMF, BayesianDA, MLE-OI |
| 克里金变体类 | 8 | ResidualKriging, GradAnisoKriging, MSAdaptKriging, SLOOCV-AK等 |
| 多项式+GPR类 | 6 | PolyRK, HGP-RK, MKGP-RK, CSP-RK系列 |
| 神经网络类 | 4 | DDNet, NDAF, PG-STGAT, PDEICNN |
| 融合框架类 | 6 | OMA, SMA, MMA, EM, GF, 华北融合 |
| 统计创新类 | 4 | CopulaFusion, WaveletGPR, VCFFM, CorrDiff |

---
更新时间: 2026-05-07
