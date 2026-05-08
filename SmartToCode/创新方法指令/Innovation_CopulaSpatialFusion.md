# 创新方法指令

## 方法名称
CopulaSpatialFusion - Copula非高斯空间融合法 (Copula-based Non-Gaussian Spatial Fusion)

## 创新点
使用Copula函数建模CMAQ与监测值的联合分布，显式处理PM2.5数据的非高斯特性（右偏、重尾）。现有方法（克里金、GPR）均假设高斯分布，但PM2.5浓度呈对数正态分布，高浓度区域的偏差结构与低浓度区域不同。Copula将边际分布与依赖结构解耦，允许使用任意边际分布（如Gamma、Lognormal），同时通过Copula函数捕获CMAQ-监测之间的非线性依赖关系。

## 核心公式

### 步骤1：边际分布拟合
对CMAQ值和监测值分别拟合边际分布：
$$
F_{CMAQ}(x) = \text{Gamma}(\alpha_c, \beta_c), \quad F_{obs}(y) = \text{Gamma}(\alpha_o, \beta_o)
$$
转换为均匀分布：
$$
u = F_{CMAQ}(x), \quad v = F_{obs}(y), \quad u,v \in [0,1]
$$

### 步骤2：Copula依赖建模
使用Gaussian Copula建模(u,v)的依赖结构：
$$
C_\theta(u, v) = \Phi_2\left(\Phi^{-1}(u), \Phi^{-1}(v); \rho\right)
$$
- $\Phi_2$：二元正态CDF
- $\rho$：Copula相关参数（通过MLE估计）
- $\theta = \rho$ 为Copula参数

### 步骤3：条件期望融合
给定CMAQ值 $x_0$，融合估计为条件期望：
$$
\hat{y}_{fused}(s_0) = E[Y | X = x_0] = F_{obs}^{-1}\left(E[V | U = u_0]\right)
$$
其中条件期望：
$$
E[V | U = u_0] = \Phi\left(\rho \cdot \Phi^{-1}(u_0)\right)
$$

### 步骤4：空间扩展（Copula-Kriging）
将Copula条件期望作为漂移项，残差用克里金插值：
$$
\hat{y}(s_0) = E[Y|X=x_0] + \sum_{i=1}^n \lambda_i \left(y_i - E[Y_i|X_i=x_i]\right)
$$
残差权重 $\lambda_i$ 由变异函数确定。

### 步骤5：不确定性量化
Copula天然提供条件分布：
$$
P(Y \leq y | X = x_0) = \frac{\partial C_\theta(u_0, v)}{\partial u}\bigg|_{u=u_0}
$$
可直接计算预测区间，无需额外假设。

## 关键步骤

1. **数据预处理**：
   - 提取训练集站点的CMAQ值和监测值
   - 对数据进行质量控制（去除负值、异常值）

2. **边际分布拟合**：
   - 对CMAQ值拟合Gamma分布（MLE估计α, β）
   - 对监测值拟合Gamma分布
   - 使用KS检验验证拟合优度

3. **Copula参数估计**：
   - 转换为均匀分布：$u_i = F_{CMAQ}(x_i), v_i = F_{obs}(y_i)$
   - MLE估计Copula参数ρ
   - 选择Copula类型（Gaussian / Clayton / Gumbel）

4. **空间残差建模**：
   - 计算Copula残差：$r_i = y_i - E[Y_i|X_i]$
   - 拟合残差变异函数
   - 克里金插值残差

5. **融合预测**：
   - 对测试点：$\hat{y} = E[Y|X=x_0] + \hat{r}_{kriging}$
   - 计算条件分布用于不确定性量化

## 参数清单

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| marginal_dist | str | 'gamma' | 边际分布类型 |
| copula_type | str | 'gaussian' | Copula类型 |
| variogram_model | str | 'spherical' | 残差变异函数 |
| n_neighbors | int | 12 | 克里金近邻数 |

## 创新判定分析

### 排除规则检查
- [ ] 是否使用权重学习（Ridge/Lasso等）？→ **否**
- [x] 是否有物理可解释性？→ **是**
  - Gamma分布：PM2.5浓度的物理非负性+右偏特性
  - Copula依赖：CMAQ-监测的非线性依赖结构
  - 残差克里金：空间相关性的地统计建模

### 预期提升
- R²提升来源：非高斯建模更准确地捕获高污染区偏差
- 风险假设：Gamma分布拟合可能不适用于所有地区
- 对比基准：PolyRK (R²≈0.91)，目标R²≥0.92

## 方法指纹
MD5: copula_non_gaussian_spatial_fusion_gamma_gaussian_v1

## 风险假设
1. Gamma分布拟合质量依赖样本量，站点少时可能不稳定
2. Gaussian Copula无法捕获尾部依赖（极端污染事件）
3. 若CMAQ-监测近似线性关系，Copula退化为普通回归
4. Copula参数ρ可能随季节变化，需时变建模

## 验证计划
1. 十折CV：对比PolyRK、AdvancedRK
2. 分浓度区间统计：低/中/高浓度分别评估
3. 检验Gamma分布拟合优度（KS检验p值）
4. 不确定性校验：90%预测区间覆盖率
