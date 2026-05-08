# 创新方法笔记 (innovation_note.md)
生成时间: 2026-04-09
最后更新: 2026-05-07

## 创新思路

### 创新1：PhysicICNN-PDE（物理信息凸神经网络）
**核心思路**：用神经网络学习CMAQ偏差，但通过硬PDE（平流-扩散方程）约束保证物理一致性，而非数据驱动的权重集成。

**创新依据**：
- CMAQ偏差不是随机噪声，而是由大气物理规律主导
- 平流-扩散PDE提供了偏差的物理先验
- ICNN凸性结构防止非物理震荡

**风险假设**：
- D（扩散系数）和S（源项）可能过度拟合，需要正则化
- 气象场分辨率必须与CMAQ匹配
- PDE参数在复杂地形可能不稳定

**验证计划**：
- 十折CV对比Innovation_ResidualKriging（固定变异函数）
- 检验扩散系数D是否符合大气物理量级（0.1-10 m²/s）

### 创新2：PolyGPR-Adapt（大气稳定度自适应多项式-GPR融合）
**核心思路**：两步走——（1）多项式校正CMAQ均值偏差（解析解，无权重学习）；（2）GPR建模残差空间结构，变异函数参数根据Pasquill-Gifford稳定度等级自适应。

**创新依据**：
- 大气稳定度直接决定扩散率，进而决定空间相关长度
- 不稳定大气（A/B类）→大相关长度，污染传输远
- 稳定大气（E/F类）→小相关长度，污染局地累积
- 现有残差克里金方法使用固定变异函数，无法捕捉稳定度变化

**风险假设**：
- 稳定度分类依赖气象数据质量
- 对于极端污染事件（如重霾），残差可能超出GPR假设的高斯分布
- 多项式阶数需要交叉验证确定（2或3阶）

**验证计划**：
- 十折CV-RMSE对比ResidualKriging（固定参数）和GenFriberg
- 分稳定度类别统计校正误差

### 创新3：ConservativeTransport（质量守恒传输映射）
**核心思路**：半拉格朗日平流传输CMAQ场保持总质量，扩散修正项用固定权重插值。无神经网络，无权重学习。

**创新依据**：
- CMAQ作为平流-扩散方程的解，传输结构应被保留而非重新建模
- 质量守恒约束保证物理一致性
- 半拉格朗日法保证计算效率

**风险假设**：
- 静稳风场（<1m/s）下平流传输退化
- 气象场质量影响传输算子精度

**验证计划**：
- 十折CV-RMSE对比ResidualKriging
- 质量守恒检验：融合场与CMAQ总质量偏差 < 0.1%

### 创新4：CopulaSpatialFusion（Copula非高斯空间融合）【2026-05-07新增】
**核心思路**：使用Copula函数建模CMAQ与监测值的联合分布，显式处理PM2.5数据的非高斯特性（右偏、重尾）。将边际分布与依赖结构解耦，允许使用Gamma边际分布，同时通过Gaussian Copula捕获非线性依赖。

**创新依据**：
- PM2.5浓度呈对数正态/Gamma分布，非高斯
- 现有克里金/GPR方法均假设高斯残差，高浓度区建模不准确
- Copula将边际分布与依赖结构分离，理论框架更灵活
- 条件期望提供天然的不确定性量化

**风险假设**：
- Gamma分布拟合质量依赖样本量
- Gaussian Copula无法捕获尾部依赖（极端污染事件）
- 若CMAQ-监测近似线性关系，Copula退化为普通回归

**验证计划**：
- 十折CV对比PolyRK、AdvancedRK
- 分浓度区间统计：低/中/高浓度分别评估
- 检验Gamma分布拟合优度（KS检验p值）

### 创新5：WaveletGPR（小波多尺度GPR残差融合）【2026-05-07新增】
**核心思路**：使用离散小波变换将CMAQ残差分解为多个空间尺度，对每个尺度独立GPR建模后重构。大尺度捕获区域传输偏差，小尺度捕获局地效应。

**创新依据**：
- 大气过程具有多尺度特性（天气尺度~100km、城市尺度~20km、局地尺度~5km）
- 现有GPR使用单一尺度核函数，无法同时捕获多尺度结构
- 小波分解天然正交，避免尺度间干扰
- 各尺度独立优化GPR超参数，更灵活

**风险假设**：
- IDW网格化可能丢失局地细节
- 小波边界效应可能产生伪影
- 细节尺度GPR可能过拟合噪声
- 计算量较大（需对每个尺度独立训练GPR）

**验证计划**：
- 十折CV对比PolyRK、AdvancedRK
- 尺度贡献分析：各尺度对总残差的方差贡献比
- 敏感性测试：分解层数J=2,3,4的影响

## 排除方法分析

### MSEF（Multi-Scale Ensemble Fusion）- 排除
**排除原因**：使用十折交叉验证学习β1、β2、β3三个权重，满足"使用线性回归/ Ridge/Lasso学习权重"的排除条件。

**分析**：
- 虽然MSEF结合了三种方法（eVNA、GMOS、Downscaler），但权重通过网格搜索优化
- 权重不随空间位置变化，只有一个全局最优β
- 相比之下，Innovation_ResidualKriging的克里金权重由变异函数物理决定，无需学习

**替代方案**：PolyGPR-Adapt提供了更物理化的多方法融合思路（多项式+自适应GPR），无权重学习

### Stacking Ensemble - 排除
**排除原因**：使用Ridge回归作为元学习器，满足"使用Ridge学习权重"的排除条件。

## 复现方法分析

### V1_DDNet（双深度神经网络）【已完成】
**核心**：PredNet预报 + DANet偏差校正
**适配**：CMAQ作为预报，监测作为真值
**指纹**：ddnet_v1_prednet_danet_dual_system

### V1_BayesianSTK（贝叶斯时空克里金）【已完成】
**核心**：时空随机场 + MCMC推断 + 后验预测
**适配**：CMAQ作为协变量，监测作为观测
**指纹**：bayesian_stk_spatiotemporal_kriging_mcmc

### V1_NeuroDDAF（神经动态扩散平流场）【已完成】
**核心**：平流-扩散PDE + GRU-GAT + 谱域求解 + 证据融合
**适配**：CMAQ作为物理初始猜，神经网络修正偏差
**指纹**：neuroddaf_v1_physics_informed_diffusion_advection

### RF-Kriging（随机森林-克里金残差校正）【2026-05-07新增】
**核心**：随机森林学习CMAQ→监测非线性映射 + 克里金插值校正残差空间结构
**适配**：CMAQ+坐标作为RF特征，监测作为目标
**指纹**：rf_kriging_residual_random_forest_v1
**特点**：两步法，RF处理非线性偏差，克里金处理空间残差

### MLE-OI（最大似然最优插值）【2026-05-07新增】
**核心**：贝叶斯最优插值框架，CMAQ为背景场，监测为观测，MLE估计误差协方差参数
**适配**：CMAQ作为背景场（先验），监测作为观测更新
**指纹**：mle_optimal_interpolation_bayesian_v1
**特点**：经典数据同化方法，权重由误差协方差比值物理决定

## 指纹重复检查

| 方法指纹 | 状态 |
|---------|------|
| hybrid_evna_avna_fingerprint | 唯一 |
| residual_kriging_adaptive_variogram_v1 | 唯一 |
| ddnet_v1_prednet_danet_dual_system | 唯一 |
| physicicnn_pde_hard_constraint_v1 | 唯一 |
| polygpr_adapt_atmospheric_stability_v1 | 唯一 |
| bayesian_stk_spatiotemporal_kriging_mcmc | 唯一 |
| neuroddaf_v1_physics_informed_diffusion_advection | 唯一 |
| conservative_transport_mass_balance_v1 | 唯一 |
| rf_kriging_residual_random_forest_v1 | 唯一（新增） |
| mle_optimal_interpolation_bayesian_v1 | 唯一（新增） |
| copula_non_gaussian_spatial_fusion_gamma_gaussian_v1 | 唯一（新增） |
| wavelet_multiscale_gpr_residual_db4_3level_v1 | 唯一（新增） |

## 指纹库统计

| 类别 | 数量 |
|-----|------|
| 复现方法指纹 | 5个（V1_DDNet, V1_BayesianSTK, V1_NeuroDDAF, RF-Kriging, MLE-OI） |
| 创新方法指纹 | 7个（PDEICNN, PolyGPRAdapt, HybridEAVNA, ResidualKriging, ConservativeTransport, CopulaSpatialFusion, WaveletGPR） |
| 排除方法记录 | 2个（MSEF, Stacking） |
| **总计** | **12个有效指纹** |

## 本轮新增（2026-05-07）

- 复现方法：2个（RF-Kriging, MLE-OI）
- 创新方法：2个（CopulaSpatialFusion, WaveletGPR）
- 新增指纹：4个
