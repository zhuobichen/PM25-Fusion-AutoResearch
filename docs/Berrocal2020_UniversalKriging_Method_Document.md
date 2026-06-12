# Universal Kriging 融合方法数学原理

**基于 Berrocal et al. (2020) 跨范式对比中的 Universal Kriging 实现**

---

## 目录

1. 概述
2. 理论基础
3. 均值函数设定（三种场景）
4. 指数半变异函数与协方差模型
5. 两阶段参数估计
6. Universal Kriging 预测
7. 完整算法伪代码
8. 与 Berrocal 2020 中其他方法的对比
9. 与 SVCD 的关系
10. 超参数默认值
11. 参考文献

---

## 1 概述

Universal Kriging（UK，泛克里金）是地质统计学中处理非平稳空间过程的经典方法。与普通克里金（Ordinary Kriging）假设全局常数均值不同，UK 允许均值函数 $\mu(s)$ 随空间变化——这种变化由已知的协变量（如 CMAQ 输出、气象变量、土地利用数据）来解释。

Berrocal et al. (2020) 在一项跨范式方法对比中将 UK 纳入对比组，并发现 UK 在该研究中表现最佳（RMSE = 3.08 μg/m³，R = 0.87），优于 Downscaler（RMSE = 3.10）和所有机器学习方法（RMSE 3.41-3.89）。这一结果说明：**在 PM2.5 日浓度估计中，显式建模空间相关性的统计方法优于仅仅将空间坐标作为特征的机器学习方法**。

### 关键特征

- **显式空间相关性**：通过指数半变异函数建模站点间的空间依赖
- **协变量驱动的非平稳均值**：均值函数由 CMAQ + 气象/土地利用协变量线性组合构成
- **BLUP 框架**：预测是最优线性无偏的，同时提供解析预测方差
- **两阶段估计**：WLS 初估 + ML 精估，兼顾计算效率和统计效率
- **逐日独立建模**：每天单独拟合（与 Downscaler 的逐日 MCMC 类似）

---

## 2 理论基础

### 2.1 模型假设

对于日期 $t$，将 PM2.5 浓度视为空间高斯过程 $Y_t(s)$：

$$Y_t(s) \sim GP\big(\mu_t(s),\; C(s, s')\big)$$

其中：
- $\mu_t(s) = \mathbb{E}[Y_t(s)]$：空间变化的均值函数（由协变量驱动）
- $C(s, s') = \text{Cov}(Y_t(s), Y_t(s'))$：平稳协方差函数（仅依赖于站点间距离）

### 2.2 BLUP 原理

UK 的预测 $\hat{Y}_t(s_0)$ 是观测值 $Y_t(s_1), \ldots, Y_t(s_n)$ 的线性组合：

$$\hat{Y}_t(s_0) = \sum_{i=1}^{n} \lambda_i Y_t(s_i)$$

权重向量 $\boldsymbol{\lambda}$ 通过最小化预测均方误差求得，同时满足无偏约束：

$$\boldsymbol{\lambda} = \arg\min_{\tilde{\boldsymbol{\lambda}}} \mathbb{E}\left[\left(Y_t(s_0) - \sum_{i=1}^{n} \tilde{\lambda}_i Y_t(s_i)\right)^2\right] \quad \text{s.t.} \quad \mathbb{E}\left[\sum_{i=1}^{n} \tilde{\lambda}_i Y_t(s_i)\right] = \mathbb{E}[Y_t(s_0)]$$

### 2.3 与普通克里金的区别

| | Ordinary Kriging | Universal Kriging |
|---|---|---|
| 均值假设 | $\mu(s) = \mu$（全局常数） | $\mu(s) = \mathbf{x}(s)^\top\boldsymbol{\beta}$（协变量线性组合） |
| 适用场景 | 局部平稳区域 | 大尺度非平稳区域（如全国 PM2.5） |
| 参数数量 | 3（块金+基台+变程） | 3 + $p$（$p$ 个协变量系数） |
| 预测精度 | 局部最优 | 全局更优（当协变量有解释力时） |

---

## 3 均值函数设定

Berrocal et al. (2020) 测试了三种 UK 均值设定：

### 场景 1：仅 CMAQ（UK-CMAQ）★ 最佳

$$\mu_t(s) = \beta_{0,t} + \beta_{1,t} \cdot Z_t(s)$$

其中 $Z_t(s)$ 是站点 $s$ 所在 CMAQ 网格的 PM2.5 模拟值。该设定在五种对比设定中取得了最优 RMSE（3.08 μg/m³）。

### 场景 2：仅协变量（UK-Covs）

$$\mu_t(s) = \mathbf{X}_t(s)^\top \boldsymbol{\beta}_t$$

其中 $\mathbf{X}_t(s)$ 包含 11 个气象和土地利用变量（见表 1），**不含** CMAQ 输出。RMSE = 3.25 μg/m³。

**表 1：11 个选定的协变量**（Berrocal 2020, Table 1）

| 类别 | 变量 |
|------|------|
| 气象 | 温度、相对湿度、风速、降水、大气压力、边界层高度等 |
| 土地利用 | 人口密度、道路长度、海拔、植被指数（NDVI）等 |

### 场景 3：CMAQ + 协变量（UK-CMAQ+Covs）

$$\mu_t(s) = \mathbf{X}_t(s)^\top \boldsymbol{\beta}_t + \beta_{1,t} \cdot Z_t(s)$$

RMSE = 3.15 μg/m³。加入协变量后反而**略差于**仅 CMAQ 的版本——说明 CMAQ 已经包含了协变量中的大部分空间信息，额外协变量引入了轻微过拟合。

---

## 4 指数半变异函数与协方差模型

### 4.1 半变异函数

空间相关性通过半变异函数 $\gamma(d)$ 刻画——距离为 $d$ 的两个站点，其观测值差异的期望方差之半：

$$\gamma(d) = \frac{1}{2}\mathbb{E}\left[(Y(s) - Y(s'))^2\right], \quad d = \|s - s'\|$$

Berrocal et al. (2020) 采用**指数半变异函数**：

$$\gamma(d) = \tau^2 + \sigma^2 \cdot \left[1 - \exp\left(-\frac{d}{\phi}\right)\right]$$

### 4.2 参数物理解释

| 参数 | 符号 | 含义 |
|------|------|------|
| 块金值 (nugget) | $\tau^2$ | 微观尺度变异 + 测量误差；$d \to 0$ 时 $\gamma(d) \to \tau^2$ |
| 偏基台值 (partial sill) | $\sigma^2$ | 空间结构方差；$\gamma(d) \to \tau^2 + \sigma^2$ 当 $d \to \infty$ |
| 变程 (range) | $\phi$ | 空间相关消失的速率；有效变程 ≈ $3\phi$（相关性降至 ~5%） |

### 4.3 对应的协方差函数

半变异函数与协方差函数的关系为（二阶平稳假设下）：

$$C(d) = \sigma^2 + \tau^2 - \gamma(d) = \sigma^2 \cdot \exp\left(-\frac{d}{\phi}\right)$$

即：

$$C(d) = \begin{cases} \sigma^2 + \tau^2, & d = 0 \\ \sigma^2 \cdot \exp(-d/\phi), & d > 0 \end{cases}$$

**注**：Berrocal 2020 的实现假设空间协方差参数在时间上恒定（$\sigma^2, \phi, \tau^2$ 不随时间变化），这与 Downscaler 允许参数逐日变化不同。

---

## 5 两阶段参数估计

Berrocal et al. (2020) 采用两阶段法，使用 R 包 `gstat` 和 `geoR`。

### 5.1 Stage 1：WLS 初估（gstat）

**步骤**：对每日数据，先用 OLS 拟合均值函数 $\mu_t(s)$，提取残差：

$$r_t(s_i) = Y_t(s_i) - \hat{\mu}_t(s_i)$$

对**所有天**的残差合并，计算经验半变异函数：

$$\hat{\gamma}(d_k) = \frac{1}{2|N(d_k)|} \sum_{(i,j) \in N(d_k)} \left(r(s_i) - r(s_j)\right)^2$$

其中 $N(d_k)$ 是距离区间 $[d_k - \Delta, d_k + \Delta]$ 内的站点对集合。

用加权最小二乘法（WLS）将指数模型 $\gamma(d) = \tau^2 + \sigma^2[1 - \exp(-d/\phi)]$ 拟合到经验半变异函数点。权重与每个距离区间内的站点对数量成比例（更多的站点对 → 更可靠的估计 → 更高的权重）。

**输出**：WLS 协方差参数估计 $\hat{\tau}^2_{WLS}, \hat{\sigma}^2_{WLS}, \hat{\phi}_{WLS}$。

### 5.2 Stage 2：ML 精估（geoR）

以 WLS 估计作为初始值，用**最大似然**（ML）迭代优化所有参数，包括协方差参数和逐日回归系数：

$$\{\hat{\tau}^2, \hat{\sigma}^2, \hat{\phi}\} \cup \{\hat{\boldsymbol{\beta}}_t\}_{t=1}^{T} = \arg\max \prod_{t=1}^{T} \mathcal{L}(\boldsymbol{\beta}_t, \tau^2, \sigma^2, \phi \mid \mathbf{Y}_t)$$

似然函数基于多元正态假设：

$$\mathbf{Y}_t \sim N\big(\mathbf{X}_t\boldsymbol{\beta}_t,\; \sigma^2\mathbf{R}(\phi) + \tau^2\mathbf{I}\big)$$

其中 $\mathbf{R}(\phi)_{ij} = \exp(-d_{ij}/\phi)$ 是相关矩阵。

**为什么两阶段而不直接 ML？** ML 对初始值敏感——尤其是在 $n$ 大且 $\phi$ 与 $\sigma^2$ 之间存在 trade-off 时（Zhang 2004）。WLS 提供一个可靠的起点，减少 ML 收敛到局部最优的风险。

---

## 6 Universal Kriging 预测

### 6.1 预测均值（BLUP）

给定参数估计 $\hat{\tau}^2, \hat{\sigma}^2, \hat{\phi}, \hat{\boldsymbol{\beta}}_t$，预测点 $s_0$ 的 UK 预测为：

$$\hat{Y}_t(s_0) = \mathbf{x}(s_0)^\top \hat{\boldsymbol{\beta}}_t + \mathbf{c}(s_0)^\top \mathbf{V}^{-1}\big(\mathbf{Y}_t - \mathbf{X}\hat{\boldsymbol{\beta}}_t\big)$$

其中：
- $\mathbf{x}(s_0)$：预测点的协变量向量（$p \times 1$）
- $\mathbf{c}(s_0)$：$s_0$ 与各训练点的协方差向量（$n \times 1$），$[\mathbf{c}(s_0)]_i = \hat{\sigma}^2 \exp(-\|s_0 - s_i\|/\hat{\phi})$
- $\mathbf{V}$：训练点间的协方差矩阵（$n \times n$），$V_{ij} = \hat{\sigma}^2 \exp(-d_{ij}/\hat{\phi}) + \hat{\tau}^2\delta_{ij}$
- $\mathbf{X}$：训练点的设计矩阵（$n \times p$）

**注**：此公式与 SVCD 的 BLUP 结构完全一致——第一项是均值函数的贡献，第二项是空间残差的加权修正。

### 6.2 预测方差

$$\hat{v}_t(s_0) = \hat{\sigma}^2 + \hat{\tau}^2 - \mathbf{c}(s_0)^\top \mathbf{V}^{-1} \mathbf{c}(s_0) + \boldsymbol{\delta}^\top (\mathbf{X}^\top\mathbf{V}^{-1}\mathbf{X})^{-1} \boldsymbol{\delta}$$

其中 $\boldsymbol{\delta} = \mathbf{x}(s_0) - \mathbf{X}^\top\mathbf{V}^{-1}\mathbf{c}(s_0)$。

三项含义：
1. $\hat{\sigma}^2 + \hat{\tau}^2$：先验方差（无任何观测时的预测不确定性）
2. $-\mathbf{c}^\top\mathbf{V}^{-1}\mathbf{c}$：克里金方差缩减（邻近观测带来的信息增益）
3. $+\boldsymbol{\delta}^\top(\cdots)^{-1}\boldsymbol{\delta}$：固定效应 $\boldsymbol{\beta}$ 的估计不确定性传播

### 6.3 预测不确定性

由于假设 $Y_t(s)$ 是高斯过程，UK 预测自然地附带预测方差 $\hat{v}_t(s)$。95% 预测区间为：

$$\hat{Y}_t(s_0) \pm 1.96 \cdot \sqrt{\hat{v}_t(s_0)}$$

Berrocal et al. (2020) 验证表明 UK 的 95% CI 覆盖率接近名义值（0.93-0.95），不确定性量化可靠。

---

## 7 完整算法伪代码

```
Algorithm: Universal Kriging (Berrocal 2020 实现)

Require: 监测数据 {Y_t(s_i)}, 协变量 {X_t(s_i), Z_t(s_i)}, CMAQ 网格, 预测点 s_0
Ensure: PM2.5 预测 Ŷ_t(s_0) 及预测方差 v̂_t(s_0)

Phase 1: 变量选择与预处理
  for 5折交叉验证 do
    用训练折做 best subset regression
    选 RMSE 最优的变量组合
  end for
  确定 11 个气象/土地利用协变量

Phase 2: 两阶段参数估计
  for each 均值设定 (CMAQ / Covs / CMAQ+Covs) do
    for each 日期 t = 1..365 do
      用 OLS 拟合 μ_t(s) = Xβ̂_t → 提取残差 r_t(s_i)
    end for
    合并全年残差 → 计算经验半变异函数 γ̂(d_k)
    用 gstat(WLS) 拟合指数半变异函数 → θ̂_WLS = (τ², σ², φ)
    以 θ̂_WLS 为初值，用 geoR(ML) 迭代优化 → θ̂_ML, {β̂_t}
  end for

Phase 3: UK 预测
  for each 日期 t do
    构建 V = σ̂²·R(φ̂) + τ̂²·I  {n×n 协方差矩阵}
    V = Cholesky(V)                     {O(n³) 分解}
    计算 V⁻¹(Y_t - Xβ̂_t)               {反代求解}

    for each 预测点 s_0 do
      x_0 ← [协变量在 s_0 处的值]
      c_0 ← [σ̂²·exp(-‖s_0−s_i‖/φ̂)]     {n×1 交叉协方差}
      Ŷ_t(s_0) ← x_0ᵀβ̂_t + c_0ᵀV⁻¹(Y_t − Xβ̂_t)
      v̂_t(s_0) ← σ̂²+τ̂² − c_0ᵀV⁻¹c_0 + δᵀ(XᵀV⁻¹X)⁻¹δ
    end for
  end for

  return {Ŷ_t(s_0), v̂_t(s_0)}
```

---

## 8 与 Berrocal 2020 中其他方法的对比

### 8.1 UK 三种设定的对比

| 设定 | 协变量 | RMSE | MAD | R | 95% CI Coverage |
|------|--------|:---:|:---:|:---:|:---:|
| UK-CMAQ | CMAQ only | **3.08** | 1.90 | **0.87** | 0.95 |
| UK-Covs | 11 协变量 | 3.25 | 1.79 | 0.85 | 0.93 |
| UK-CMAQ+Covs | CMAQ + 11协变量 | 3.15 | 1.76 | 0.86 | 0.93 |

**发现**：仅用 CMAQ 作为协变量的 UK 性能最优，加入额外的 11 个协变量反而略微降低了精度。这说明：
- CMAQ 输出已经包含了大部分协变量所能提供的空间信息
- 额外的协变量可能引入了轻微的过拟合（$p=12 \to 13$，增加了 1 个自由参数）

### 8.2 UK vs 其他方法

| 方法 | RMSE | MAD | R | 优缺点 |
|------|:---:|:---:|:---:|------|
| **UK (CMAQ)** | **3.08** | 1.90 | **0.87** | **总体最优**，空间 + 协变量联合建模 |
| Downscaler | 3.10 | **1.70** | **0.87** | MAD 最优，但 RMSE 略逊，且更慢（MCMC） |
| IDW | 3.39 | 1.96 | 0.84 | 简单快速，但无协变量利用 + 无不确定性 |
| Random Forest | 3.41 | 2.09 | 0.84 | 非线性能力强，但未建模空间相关 |
| Neural Network | 3.89 | 2.45 | 0.79 | 在所有方法中表现最差（该任务上） |

### 8.3 UK 的优势

1. **最优 RMSE**：在所有 7 种方法中预测误差最低
2. **可靠的不确定性**：95% CI 覆盖率达 0.95，接近名义值
3. **计算效率**：WLS + ML 两阶段法比 Downscaler 的 MCMC 更快（无需 10,000 次采样）
4. **协变量可灵活组合**：三种均值设定可适配不同数据丰富度
5. **理论完备**：BLUP 框架下的预测方差是解析的，无需 Monte Carlo

### 8.4 UK 的局限

1. **线性均值假设**：$\mu_t(s) = \mathbf{x}(s)^\top\boldsymbol{\beta}$ 是线性形式，无法捕获 CMAQ 与 PM2.5 之间的非线性关系
2. **平稳协方差**：假设协方差仅依赖距离（各向同性），全国尺度下可能不成立（东西部协方差结构不同）
3. **无时间动态**：逐日独立建模，前一天的信息不被利用
4. **大 $n$ 时 Cholesky 瓶颈**：$O(n^3)$ 分解在 $n > 2000$ 时变慢
5. **对极值敏感**：高斯假设下，高 PM2.5 事件的预测可能偏保守

---

## 9 与 SVCD 的关系

### 9.1 结构对比

UK 和 SVCD 共享同一个数学核心——**BLUP 预测公式**：

$$\hat{Y}(s_0) = \mathbf{x}_0^\top \hat{\boldsymbol{\beta}} + \mathbf{c}^\top\mathbf{V}^{-1}(\mathbf{Y} - \mathbf{X}\hat{\boldsymbol{\beta}})$$

| 组件 | Universal Kriging | SVCD |
|------|:---:|:---:|
| **均值函数 $\mathbf{X}\boldsymbol{\beta}$** | 协变量线性组合 | $\beta_0 + \beta_1\tilde{U}$ |
| **协方差 $\mathbf{V}$** | $\sigma^2 \mathbf{R}(\phi) + \tau^2\mathbf{I}$ | $\boldsymbol{\Sigma}_0 + (\tilde{\mathbf{u}}\tilde{\mathbf{u}}^\top)\odot\boldsymbol{\Sigma}_1 + \tau^2\mathbf{I}$ |
| **空间场 $w(s)$** | 隐式（通过残差） | **显式双 GP** ($w_0 + w_1\tilde{U}$) |
| **CMAQ 利用方式** | 均值函数中的协变量 | 既在均值函数中，又通过 $w_1$ 调节空间协方差 |
| **参数估计** | WLS + ML | REML (L-BFGS-B) |

### 9.2 关键区别

SVCD 的 $\mathbf{V}$ 矩阵比 UK 多了一项 $(\tilde{\mathbf{u}}\tilde{\mathbf{u}}^\top) \odot \boldsymbol{\Sigma}_1$，这是 $w_1(s)\tilde{U}(s)$ 项边际化后的贡献。UK 的协方差结构与 CMAQ 值无关——两个站点之间的空间相关性纯粹由距离决定；SVCD 的协方差**同时**取决于距离和 CMAQ 值——两个站点如果 CMAQ 值都很大，它们通过斜率场的耦合更强（Hadamard 积的含义）。

这一差异的实践效果：Berrocal 2020 中 UK 与 Downscaler RMSE 分别为 3.08 和 3.10，差异仅 0.02 μg/m³。说明**当 CMAQ 的空间模式本身已经足够稳定时，让协方差依赖于 CMAQ 的边际增益很小**。

---

## 10 超参数默认值

| 参数 | 符号 | 估计方式 | 典型值范围 |
|------|------|:---:|------|
| 块金值 (nugget) | $\tau^2$ | WLS → ML | ~1-5 (μg/m³)² |
| 偏基台值 (partial sill) | $\sigma^2$ | WLS → ML | ~5-15 (μg/m³)² |
| 变程 (range) | $\phi$ | WLS → ML | ~50-300 km |
| 协变量系数 | $\boldsymbol{\beta}_t$ | ML (逐日) | 取决于协变量 |

**注**：Berrocal 2020 的实现中，$\tau^2, \sigma^2, \phi$ 在时间上恒定（全年共享），$\boldsymbol{\beta}_t$ 逐日独立估计。

---

## 参考文献

1. Berrocal, V. J., Guan, Y., Muyskens, A., Wang, H., Reich, B. J., Mulholland, J. A., & Chang, H. H. (2020). A comparison of statistical and machine learning methods for creating national daily maps of ambient PM2.5 concentration. *Atmospheric Environment*, 222, 117130.

2. Cressie, N. (1993). *Statistics for Spatial Data* (Revised Edition). Wiley.

3. Ribeiro Jr, P. J., & Diggle, P. J. (2018). geoR: Analysis of Geostatistical Data. R package.

4. Graler, B., Pebesma, E., & Heuvelink, G. (2016). Spatio-temporal interpolation using gstat. *The R Journal*, 8(1), 204-218.

5. Zhang, H. (2004). Inconsistent estimation and asymptotically equal interpolations in model-based geostatistics. *Journal of the American Statistical Association*, 99(465), 250-261.
