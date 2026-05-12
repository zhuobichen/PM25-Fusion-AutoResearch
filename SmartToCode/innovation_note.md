# 创新方法笔记 (innovation_note.md)
生成时间: 2026-04-09
最后更新: 2026-05-08

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

### 创新6：BMA-Fusion（贝叶斯模型平均融合）【2026-05-08第二批】
**核心思路**：使用贝叶斯后验模型概率（而非Ridge/Lasso回归权重）组合多个基础方法。后验概率通过BIC近似模型证据计算，反映每个方法的预测可信度。

**创新依据**：
- BMA是经典贝叶斯组合方法，后验概率有明确物理意义（方法可信度）
- 不使用权重学习（Ridge/Lasso），而是概率推断
- 自动提供不确定性估计（预测方差）
- 与Stacking/SuperEnsemble不同：BMA权重来自贝叶斯定理，非回归优化

**风险假设**：
- 基础方法高度相关时，BMA权重可能退化为近似均匀分布
- BIC近似可能不够精确（尤其小样本）
- 计算开销：需要对每个基础方法进行交叉验证

**验证计划**：
- 十折CV对比AdvancedRK（当前最优）
- 后验权重分布分析：是否集中在最优方法上
- 不确定性校准：预测区间覆盖率

### 创新7：TGK（传输引导核融合）【2026-05-08第二批】
**核心思路**：利用CMAQ梯度场构建各向异性空间核——沿大气传输方向相关距离长，垂直方向短。先用二次多项式校正CMAQ系统偏差，再用传输引导核GPR建模残差。

**创新依据**：
- 大气污染传输具有方向性（风向主导）
- 现有方法（克里金、GPR）使用各向同性核，忽略方向性
- CMAQ梯度场近似传输方向（梯度方向≈传输方向）
- 高梯度区→各向异性强→沿传输方向相关距离更长

**风险假设**：
- CMAQ梯度可能有数值噪声（需平滑处理）
- 梯度方向不总是等于传输方向（如复杂地形）
- 各向异性参数α需通过交叉验证选择

**验证计划**：
- 十折CV对比各向同性GPR和AdvancedRK
- 梯度强度与各向异性效果相关性分析
- 参数α敏感性测试

### 创新8：RDMK（残差分布匹配克里金）【2026-05-08第三批】
**核心思路**：针对PM2.5残差的非高斯特性（右偏），使用参数化Gamma分布进行残差分布匹配，将非高斯残差变换为高斯变量后进行克里金插值，再反变换回原始尺度。

**创新依据**：
- PM2.5浓度呈右偏分布（Gamma/对数正态），残差同样具有偏态
- 现有克里金/GPR方法假设残差服从高斯分布，高浓度区建模不准确
- 参数化分布匹配比经验分位数映射更稳健（参数少、可外推）
- 与Copula方法相比更简单（仅处理边际分布，不建模联合依赖结构）

**风险假设**：
- Gamma分布拟合质量依赖残差的分布形态
- 小样本下Gamma分布MLE可能不稳定
- 边界效应：极低/极高浓度区的分布匹配可能不稳定
- 变换后的高斯性需验证（Shapiro-Wilk检验）

**验证计划**：
- 十折CV对比AdvancedRK（当前最优，R²=0.916）
- 分浓度区间评估：低（<35）、中（35-75）、高（>75）μg/m³
- 残差分布检验：Gamma拟合KS检验p值
- 变换后高斯性检验：Shapiro-Wilk检验p值

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

### Cokriging（共克里金）【2026-05-08第二批】
**核心**：主变量（监测PM2.5）+辅助变量（CMAQ）的互协方差联合插值
**适配**：监测站PM2.5为主变量，CMAQ为辅助变量
**指纹**：cokriging_multivariate_joint_interpolation_v1
**特点**：利用主辅变量空间互相关，理论基础扎实

### CensoredExceedances（贝叶斯截断阈值融合）【2026-05-08第三批】
**核心**：贝叶斯分层模型 + 截断似然（处理检测限） + GPD尾部建模 + AR(1)时间结构
**适配**：CMAQ替代EAC4再分析数据，MLE替代MCMC
**指纹**：bayesian_censored_exceedances_gpd_ar1_v1
**特点**：处理PM2.5数据截断问题，GPD建模极端值

### MSF-NNG（多源最近邻网格融合）【2026-05-08第三批】
**核心**：Cressman插值 + 最近邻网格匹配 + 固定权重融合
**适配**：监测站替代"大站"，CMAQ替代"模型数据"
**指纹**：msf_nng_cressman_nearest_neighbor_fusion_v1
**特点**：简单确定性方法，固定权重α=0.7

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
| rf_kriging_residual_random_forest_v1 | 唯一 |
| mle_optimal_interpolation_bayesian_v1 | 唯一 |
| copula_non_gaussian_spatial_fusion_gamma_gaussian_v1 | 唯一 |
| wavelet_multiscale_gpr_residual_db4_3level_v1 | 唯一 |
| cokriging_multivariate_joint_interpolation_v1 | 唯一 |
| bayesian_model_averaging_fusion_bic_evidence_v1 | 唯一 |
| transport_guided_anisotropic_kernel_cmaq_gradient_v1 | 唯一 |
| bayesian_censored_exceedances_gpd_ar1_v1 | 唯一（新增） |
| mspatiotemporal_nearest_neighbor_grids_msf_nng | 唯一（新增） |
| residual_distribution_matching_kriging_gamma_gaussian_v1 | 唯一（新增） |

## 指纹库统计

| 类别 | 数量 |
|-----|------|
| 复现方法指纹 | 9个（V1_DDNet, V1_BayesianSTK, V1_NeuroDDAF, RF-Kriging, MLE-OI, Cokriging, CensoredExceedances, MSF-NNG, IDW-Bias, GWR, Gen-Friberg） |
| 创新方法指纹 | 12个（PDEICNN, PolyGPRAdapt, HybridEAVNA, ResidualKriging, ConservativeTransport, CopulaSpatialFusion, WaveletGPR, BMA-Fusion, TGK, VarioGPR-RK, HeteroGPR, RDMK） |
| 排除方法记录 | 2个（MSEF, Stacking） |
| **总计** | **23个有效指纹** |

## 本轮新增（2026-05-08 第一批）

- 复现方法：3个（IDW-Bias, GWR, Gen-Friberg）
- 创新方法：2个（VarioGPR-RK, HeteroGPR-PolyRK）
- 新增指纹：5个

## 本轮新增（2026-05-08 第二批）

- 复现方法：1个（Cokriging共克里金法）
- 创新方法：2个（BMA-Fusion, TGK）
- 新增指纹：3个

## 本轮新增（2026-05-08 第三批）

- 复现方法：2个（CensoredExceedances贝叶斯截断阈值融合法, MSF-NNG多源最近邻网格融合法）
- 创新方法：1个（RDMK残差分布匹配克里金）
- 新增指纹：3个

## 上轮新增（2026-05-07）

- 复现方法：2个（RF-Kriging, MLE-OI）
- 创新方法：2个（CopulaSpatialFusion, WaveletGPR）
- 新增指纹：4个
