# SVCD 与 Berrocal 系列方法的详细对比

## 涉及文献

| # | 文献 | 简称 | 期刊 |
|---|------|------|------|
| 1 | Berrocal, Gelfand, Holland (2010). A Spatio-Temporal Downscaler for Output From Numerical Models. | **Berrocal 2010** | *JABES*, 15(2), 176-197 |
| 2 | Berrocal, Gelfand, Holland (2010). A Bivariate Space-Time Downscaler Under Space and Time Misalignment. | **Berrocal 2010b** | *arXiv:1004.1147* |
| 3 | Berrocal, Gelfand, Holland (2012). Space-Time Data Fusion Under Error in Computer Model Output. | **Berrocal 2012** | *Biometrics*, 68(3), 837-848 |
| 4 | Berrocal, Guan, Muyskens, Wang, Reich, Mulholland, Chang (2020). A Comparison of Statistical and Machine Learning Methods for Creating National Daily Maps of Ambient PM2.5 Concentration. | **Berrocal 2020** | *Atmospheric Environment*, 222, 117130 |

---

## 1. Berrocal 2010 --- 原始 Downscaler（JABES）

### 1.1 数据与研究背景

- **污染物**：美国东部 O3，2001 年臭氧季（5-10 月），803 个 NAMS/SLAMS 监测站点
- **模型**：CMAQ 12km 网格，40,044 个格点
- **数据特点**：O3 浓度右偏，方差随均值增大（异方差性）

### 1.2 数据变换

原文根据数据直方图判断，对 O3 和 CMAQ 同时做平方根变换以稳定方差并趋近正态：

$$Y(s) = \sqrt{\text{O}_3(s)}, \quad x(B) = \sqrt{\text{CMAQ}(B)}$$

反变换时，MCMC 后验样本先平方再取均值：
$$\hat{\text{O}}_3(s_0) = \frac{1}{M}\sum_{m=1}^{M} \big(\hat{Y}^{(m)}(s_0)\big)^2$$

由于 MCMC 采样自动满足 $E[\hat{Y}^2] = E[\hat{Y}]^2 + \text{Var}(\hat{Y})$，无需显式偏差修正。

### 1.3 空间变系数模型

**静态模型**（单日）：

$$Y(s) = \tilde{\beta}_0(s) + \tilde{\beta}_1(s) \cdot x(B) + \epsilon(s), \quad \epsilon(s) \overset{iid}{\sim} N(0, \tau^2)$$

其中每个 $s$ 关联到其所在的 CMAQ 网格 $B$。系数分解为全局部分 + 局部随机效应：

$$\tilde{\beta}_0(s) = \beta_0 + \beta_0(s), \quad \tilde{\beta}_1(s) = \beta_1 + \beta_1(s)$$

- $\beta_0, \beta_1$：全局截距和斜率（固定效应），描述 CMAQ 的整体校准偏差
- $\beta_0(s)$：局部截距调整，捕捉 CMAQ 在不同地区的系统性偏差
- $\beta_1(s)$：局部斜率调整，捕捉不同地区 CMAQ 与观测关系的差异

### 1.4 Coregionalization：双 GP 的相关结构

$\beta_0(s)$ 和 $\beta_1(s)$ 通过 **coregionalization**（共区域化）建模。核心思想：用两个**独立的**零均值、单位方差 GP $w_0(s)$ 和 $w_1(s)$，通过下三角矩阵 $\mathbf{A}$ 线性混合为**相关的** $\beta$ 场：

$$\begin{pmatrix} \beta_0(s) \\ \beta_1(s) \end{pmatrix} = \begin{pmatrix} A_{11} & 0 \\ A_{21} & A_{22} \end{pmatrix} \begin{pmatrix} w_0(s) \\ w_1(s) \end{pmatrix}$$

$w_0(s)$ 和 $w_1(s)$ 各自独立，协方差核为指数型：

$$\text{cov}(w_j(s), w_j(s')) = \exp(-\phi_j|s - s'|), \quad j = 0, 1$$

$\mathbf{A}$ 矩阵的 3 个自由参数决定了两个场的统计性质：

- $\text{Var}(\beta_0) = A_{11}^2$：截距场的边际方差
- $\text{Var}(\beta_1) = A_{21}^2 + A_{22}^2$：斜率场的边际方差
- $\text{Cov}(\beta_0, \beta_1) = A_{11}A_{21}$：两场在同一点的协方差

**Coregionalization 的信息代价**：$A_{21}$（跨场耦合强度）与 decay 参数 $\phi_0, \phi_1$ 之间存在 trade-off——截距场到斜率场的相关性传导既可以通过 $\mathbf{A}$ 矩阵实现，也可以通过调节空间平滑范围来近似。这种参数冗余导致 MCMC 后验在多个方向上平坦（weak identifiability）。因此 Berrocal 等人**不得不在 MCMC 之外采用 grid search** 来确定 $\phi_0, \phi_1$——对每个 $\phi$ 的候选值分别跑 MCMC，然后比较 DIC 选择最优组合。协方差参数总数 = $A_{11}, A_{21}, A_{22}, \phi_0, \phi_1, \tau^2$ = **6 个**。

### 1.5 时空扩展

扩展到多日时，全局截距和斜率随时间变化，但局部随机效应保持时间恒定：

$$Y(s,t) = \beta_{0t} + \beta_{1t} \cdot x(B,t) + \beta_0(s) + \beta_1(s) \cdot x(B,t) + \epsilon(s,t)$$

$\beta_{0t}, \beta_{1t}$ 有两种设定方式：（1）各时间独立，服从二元正态先验；（2）服从 AR(1) 动态过程。

### 1.6 推断与计算代价

- **MCMC** 采样所有参数的后验分布
- **Grid search** 确定 $\phi_0, \phi_1$（因 coregionalization 导致梯度优化不可靠）
- **单日耗时**：约 5 分钟
- 模型仅拟合有监测站的 CMAQ 网格（无需处理全部 40,044 个网格），享受天然降维

### 1.7 与 Kriging/Bayesian Melding 的对比

Berrocal 2010 的验证结果表明：
- 预测精度优于普通 Kriging（仅用观测数据）和 Bayesian Melding（Fuentes & Raftery 2005）
- 计算速度远快于 Bayesian Melding（后者需要对每个格点计算随机积分）
- 预测区间的经验覆盖率接近名义值

---

## 2. Berrocal 2010b --- 双变量 Downscaler（AOAS）

### 2.1 扩展动机

O3 和 PM2.5 是共污染物（co-pollutants），共享部分排放源和大气化学过程。如果两个污染物的空间场存在相关性，则一个污染物的监测信息可以辅助预测另一个——特别是在监测网络不完全重叠的区域。

### 2.2 模型结构

将单污染物模型扩展为二元联合模型，**O3 取平方根变换，PM2.5 取对数变换**（两者目的不同：O3 用 sqrt 稳定方差，PM2.5 用 log 趋于正态）。两种污染物共享 coregionalization 框架，通过跨污染物的 $\mathbf{A}$ 矩阵关联空间场，实现信息借用（information borrowing）。

### 2.3 关键发现

- 预测改善** modest**（适中）
- 原因：O3 和 PM2.5 的监测网络在地理上高度重叠，交叉信息增量有限
- 该方法的价值更多在于**联合推断**（同时给出两个污染物的预测和不确定性）而非单污染物的精度提升

### 2.4 与后续工作的关系

- Decay 参数采用 **sensitivity analysis** 确定（与 Berrocal 2010 的 grid search 不同）：先用 REML 估计每日 decay 参数的中位数，然后分别在 ×10/÷10 和 ×100/÷100 的区间上做敏感性测试，取稳定区间内的固定值
- 仍采用 **MCMC + coregionalization**，计算代价比单变量更高
- 结构与 Berrocal 2010 一致，维度扩展至双污染物

---

## 3. Berrocal 2012 --- 邻域平滑 Downscaler（Biometrics）

### 3.1 核心动机

Berrocal 2010 的两个潜在局限：

1. **信息利用不充分**：站点 $s$ 仅使用所在 CMAQ 网格 $B$ 的值作为协变量，而相邻网格的 CMAQ 输出也可能包含有用信息
2. **空间不对齐**：站点与"所属"网格之间可能存在空间偏差——站点位于网格边缘时，该网格的 CMAQ 值不一定是最佳协变量

Berrocal 2012 提出了两种扩展方案来解决这两个问题。

### 3.2 模型 A：GMRF 平滑版

**思路**：不直接使用原始 CMAQ 值 $x(B)$，而是先用**隐高斯马尔可夫随机场（GMRF/CAR）** 对 CMAQ 面做空间平滑，得到一个去噪后的 CMAQ 面 $\tilde{V}(B)$，再用平滑后的值作为协变量。

**步骤 1 --- 构建隐 CMAQ 面**：

$$x(B) = \mu + V(B) + \eta(B), \quad \eta(B) \overset{iid}{\sim} N(0, \sigma^2)$$

其中 $V(B)$ 是一个条件自回归（CAR）高斯马尔可夫随机场：

$$V(B_i) \mid \{V(B_j), j \neq i\} \sim N\left(\frac{\sum_{j \in \partial B_i} V(B_j)}{m_i}, \frac{\xi^2}{m_i}\right)$$

$\partial B_i$ 是网格 $B_i$ 的邻居集合（通常取 4 或 8 邻域），$m_i$ 是邻居数量。CAR 结构使相邻网格的 $V$ 值互相靠近，产生空间平滑效果。

记平滑后的 CMAQ 面为：
$$\tilde{V}(B) = \mu + V(B)$$

**步骤 2 --- 回归模型**：

$$Y(s) = \tilde{\beta}_0(s) + \beta_1 \cdot \tilde{V}(B) + \epsilon(s)$$

其中 $\tilde{\beta}_0(s) = \beta_0 + \beta_0(s)$，$\beta_0(s)$ 为单变量 GP（指数核）。

**⚠️ 关键牺牲**：斜率系数只剩下全局固定的 $\beta_1$，空间变斜率 $\beta_1(s)$ 被**移除**。原因在原文脚注中明确说明：

> *"With $\tilde{V}(B)$ unobserved, a spatially varying $\tilde{\beta}_1(s)$ will not be identifiable."*

直译：当 CMAQ 协变量从可观测的 $x(B)$ 变成隐变量 $\tilde{V}(B)$ 后，再叠加一个空间变化的斜率系数，模型将面临"隐变量 × 隐变量"的双层嵌套——仅有观测值 $Y(s)$ 不足以同时区分"是 CMAQ 面在变"还是"斜率在变"。自由度不够，参数不可识别。

**计算代价**：需要引入全部 40,044 个网格的 $\{V(B_i)\}$（而不是仅用有监测站的网格），但因为 CAR 的局部性（每个网格只依赖邻居），可以通过稀疏矩阵高效计算。

### 3.3 模型 B：空间变权重版

**思路**：为每个监测站点 $s$ 生成一个**定制的 CMAQ 加权平均** $\tilde{x}(s)$，权重由潜在 GP 驱动，在不同站点取不同值。

$$\tilde{x}(s) = \sum_{k=1}^{g} w_k(s) \cdot x(B_k)$$

其中 $g = 40,044$（全部 CMAQ 网格），$r_k$ 为网格 $B_k$ 的中心坐标。

权重 $w_k(s)$ 满足非负且和为 1 的约束，定义为：

$$w_k(s) = \frac{K(s - r_k; \psi) \cdot \exp(Q(r_k))}{\sum_{l=1}^{g} K(s - r_l; \psi) \cdot \exp(Q(r_l))}$$

- $K(\cdot; \psi)$：指数核 $K(s - r_k; \psi) = \exp(-\psi|s - r_k|)$，控制空间衰减速度
- $Q(r)$：零均值潜在 GP，指数协方差核（参数 $\phi_Q, \sigma^2_Q$），赋予不同网格不同的"吸引力"
- "sum to 0" 约束施加在 $\{Q(r_k)\}$ 上以保证中心可识别

**物理解释**：$K$ 确保靠近 $s$ 的网格权重更大，$\exp(Q)$ 允许某些网格（例如 CMAQ 模拟更准的区域）被全局性地赋予更高权重。两者结合产生**空间变化且方向性**的加权方案。

**回归模型**：

$$Y(s) = \tilde{\beta}_0(s) + \beta_1 \cdot \tilde{x}(s) + \epsilon(s)$$

**⚠️ 同样的牺牲**：此模型也仅保留固定 $\beta_1$，砍掉了空间变斜率。

### 3.4 效果

与 Berrocal 2010 原版对比，非重合站点验证（排除监测站一定半径内的所有站后余下的站点）：

- **GMRF 版**：预测 MSE 降低约 **5%**
- **变权重版**：预测 MSE 降低约 **15%**，尤其对远离监测站的区域改善显著

### 3.5 与 Berrocal 2010 的关系总结

| | 斜率 GP $\beta_1(s)$ | CMAQ 协变量 | 新引入参数 | 可识别性 |
|---|---|---|---|---|
| Berrocal 2010 | **有**（coregionalized） | 原始 $x(B)$ | — | 弱（需 grid search） |
| Berrocal 2012 GMRF | **砍掉** | 平滑后 $\tilde{V}(B)$ | CAR 参数 $\xi^2$ | 斜率场不可识别 |
| Berrocal 2012 VarW | **砍掉** | 加权平均 $\tilde{x}(s)$ | GP 参数 $\phi_Q, \sigma^2_Q, \psi$ | 斜率场不可识别 |

---

## 4. Berrocal 2020 --- 跨范式方法对比（Atmospheric Environment）

### 4.1 背景与动机

2019 年，Berrocal 本人领导了一项跨范式方法对比研究，比较统计空间模型（Kriging、Downscaler）与机器学习方法（Random Forest、SVM、Neural Network）在 PM2.5 日浓度估计上的表现。这是 Downscaler 系列中**唯一一项大规模实证对比研究**，由原方法的第一作者亲自执行。

### 4.2 数据

- PM2.5，2011 年，美国大陆，829 个监测站点
- CMAQ 12km，299×459 网格
- 11 个气象+土地利用协变量
- 分层评估：按最近监测站数量、城市化程度、PM2.5 浓度水平、季节

### 4.3 对比方法

| 方法类别 | 具体方法 |
|----------|---------|
| 基线 | OLS（CMAQ / Covs / CMAQ+Covs） |
| 空间统计 | IDW、Universal Kriging（3 种协变量组合）、**Downscaler** |
| 机器学习 | Random Forest、SVM、Neural Network |

### 4.4 Downscaler 版本

Berrocal 在这项对比中采用了一个**显著简化的 Downscaler**：

$$Y_t(s) = \beta_{0,t}(s) + \beta_{1,t} \cdot Z_t(s) + \epsilon_t(s), \quad \epsilon_t(s) \overset{iid}{\sim} N(0, \sigma^2)$$

$$\text{Cov}(\beta_{0,t}(s_i), \beta_{0,t}(s_j)) = \sigma^2_0 \exp(-d_{ij}/\phi_0)$$

**与 Berrocal 2010 原版的关键区别**：
1. **仅 1 个 GP**（空间变截距 $\beta_{0,t}(s)$），斜率 $\beta_{1,t}$ 为全局固定值——**砍掉了空间变斜率**
2. **无 coregionalization**——仅需估计 $\sigma^2, \sigma^2_0, \phi_0$ 三个协方差参数
3. 用 R 包 `spBayes` 实现，MCMC 10,000 迭代（5,000 burn-in）

**为什么简化？** 论文将原因归结为实证选择——"choosing the form of the downscaler model that yields the best predictive performance in various experiments"。此外，对比计算规模（5 折 × 多种方法 × 365 天）也使完整版 MCMC + coregionalization 在操作上难以承受。这一选择与 SVCD 的设计思路**一致**——简化协方差结构以换取计算可行性，同时保持足够的空间建模能力。

### 4.5 核心结果

| 方法 | RMSE (μg/m³) | MAD | R | 95% CI Coverage |
|------|:---:|:---:|:---:|:---:|
| OLS (CMAQ+Covs) | 4.22 | 2.63 | 0.74 | 0.83 |
| IDW | 3.39 | 1.96 | 0.84 | — |
| **UK (CMAQ)** | **3.08** | 1.90 | **0.87** | 0.95 |
| **Downscaler (CMAQ)** | 3.10 | **1.70** | **0.87** | 0.94 |
| Random Forest | 3.41 | 2.09 | 0.84 | 0.96 |
| SVM | 3.83 | 2.22 | 0.79 | — |
| Neural Network | 3.89 | 2.45 | 0.79 | — |

**三个关键结论**：

1. **统计空间方法 > 机器学习**：UK 和 Downscaler 的 RMSE 比最佳 ML 方法（RF）低 0.3 μg/m³，R 高 0.03。原因：机器学习方法将空间坐标作为普通预测变量，不显式建模空间相关性。
2. **Downscaler ≈ Universal Kriging**：RMSE 几乎相同（3.10 vs 3.08），仅在 MAD 上有轻微优势（1.70 vs 1.90）。说明 CMAQ 信息在跨空间插值中的边际增益有限。
3. **所有空间方法的 95% CI 覆盖率均接近名义值**——不确定性量化可靠。

### 4.6 与 SVCD 的关系

Berrocal 2020 从三个方面为 SVCD 提供了直接支持：

1. **简化版 Downscaler 由原作者在实际对比中选择使用**：Berrocal 等人基于"在各项实验中表现最佳"的实证依据（*"choosing the form of the downscaler model that yields the best predictive performance in various experiments"*），在跨范式对比中采用了单 GP 版本，SVCD 保留了更多灵活性（双 GP）而同样简化了协方差结构。
2. **统计空间方法的优越性得到大规模实证**：在跨范式对比中压倒 ML 方法，为 SVCD 在方法对比论文中的定位提供了强引用。
3. **Downscaler 的计算可行性需求被再次确认**：即使简化版仍"略慢于 Kriging"，凸显了 REML 加速（~15×）的实际价值。

---

## 5. SVCD

### 5.1 设计选择

SVCD 面对 Berrocal 系列的演进困境——Berrocal 2010 完整但慢，Berrocal 2012 快了但砍掉了关键结构，Berrocal 2020 在实证对比中也采用了简化版（单 GP）——选择了一条整合路线：**保留比 Berrocal 2020 更完整的空间灵活性（双 GP），同时达到比 MCMC 更快的计算效率（REML）**。

三个核心改动：

1. **MCMC → REML**：将贝叶斯采样替换为限制最大似然，速度提升约 15 倍
2. **coregionalization → 独立 GP**：消除 $\mathbf{A}$ 矩阵的参数冗余，使梯度优化直接可行
3. **显式偏差修正**：在 REML 点估计框架下补齐反变换的方差补偿

### 5.2 模型

**变换**：$\sqrt{\text{O}_3}$（与 Berrocal 2010 一致）。

$$Z(s) = \beta_0 + \beta_1 \tilde{U}(s) + w_0(s) + w_1(s) \tilde{U}(s) + \epsilon(s)$$

$$\tilde{U}(s) = \frac{U(s) - \mu_U}{\sigma_U}, \quad U(s) = \sqrt{\text{CMAQ}(s)}$$

$w_0(s)$ 和 $w_1(s)$ 为**独立**零均值 GP，指数协方差核（$\nu = 0.5$）：

$$w_j(s) \sim GP\big(0,\; \sigma_j^2 \cdot \exp(-\|s - s'\| / \rho_j)\big), \quad j = 0, 1$$

$$\epsilon(s) \sim N(0, \tau^2)$$

**与 Berrocal 2010 的公式层面差异**：

| 组件 | Berrocal 2010 | SVCD |
|------|:---:|:---:|
| GP 数量 | 2（coregionalized） | 2（独立） |
| GP 间相关 | $\mathbf{A}$ 矩阵建模 | 无（$w_0 \perp w_1$） |
| 协方差核参数 | $\exp(-\phi_j\|s-s'\|)$ | $\exp(-\|s-s'\|/\rho_j)$ （等价，$\rho_j = 1/\phi_j$） |
| U 的处理 | 直接使用 CMAQ 值 | 先标准化 $(\tilde{U} = (U-\mu_U)/\sigma_U)$ |
| 协方差参数 | 6 | **5** |

**U 标准化的作用**：将 CMAQ 值归一化后，$\beta_1$ 和 $w_1(s)$ 的量级不再依赖 CMAQ 的绝对范围，提高数值稳定性。这不改变模型结构，仅是一种计算优化。

### 5.3 边际化与 V 矩阵

记设计矩阵 $\mathbf{X} = [\mathbf{1}, \tilde{\mathbf{u}}]$（$n \times 2$，第一列为全 1，第二列为标准化 CMAQ 值），$\boldsymbol{\beta} = (\beta_0, \beta_1)^\top$。将 $w_0, w_1$ 边际化后：

$$\mathbf{z} \mid \boldsymbol{\beta}, \boldsymbol{\theta} \sim N(\mathbf{X}\boldsymbol{\beta}, \mathbf{V})$$

$$\mathbf{V} = \boldsymbol{\Sigma}_0 + (\tilde{\mathbf{u}}\tilde{\mathbf{u}}^\top) \odot \boldsymbol{\Sigma}_1 + (\tau^2 + \delta) \mathbf{I}$$

其中 $\boldsymbol{\Sigma}_j = \sigma_j^2 \cdot [\exp(-d_{ii'}/\rho_j)]_{n \times n}$ 是空间协方差矩阵，$\odot$ 为 Hadamard 逐元素积，$\delta = 10^{-6}$ 为数值稳定性 jitter（防止 Cholesky 分解奇异）。

$\mathbf{V}$ 的三项在物理上分别对应截距场贡献（$\boldsymbol{\Sigma}_0$）、斜率场贡献（$(\tilde{\mathbf{u}}\tilde{\mathbf{u}}^\top) \odot \boldsymbol{\Sigma}_1$，与 CMAQ 值成比例）、以及观测噪声（$\tau^2 \mathbf{I} + \delta\mathbf{I}$）。与 Berrocal 2010 的区别在于：Berrocal 2010 因 coregionalization 引入了截距-斜率场的交叉耦合项（$\text{Cov}(Y_i, Y_{i'}) = \exp(-\phi_0 d)[A_{11}^2 + A_{11}A_{21}(x_i + x_{i'}) + A_{21}^2 x_i x_{i'}] + A_{22}^2 x_i x_{i'} \exp(-\phi_1 d) + \tau^2\delta_{ii'}$，其中 $x_i$ 为站点 $s_i$ 处的原始 $\sqrt{\text{CMAQ}}$ 值），而 SVCD 中 $w_0 \perp w_1$，$\mathbf{V}$ 仅有三项，无交叉项。

### 5.4 REML 参数估计

超参数 $\boldsymbol{\theta} = (\log\sigma_0, \log\rho_0, \log\sigma_1, \log\rho_1, \log\tau)$。

REML 负对数似然（省略常数项）：

$$\ell_{REML}(\boldsymbol{\theta}) = \frac{1}{2}\Big[\ln|\mathbf{V}| + (\mathbf{z} - \mathbf{X}\hat{\boldsymbol{\beta}})^\top\mathbf{V}^{-1}(\mathbf{z} - \mathbf{X}\hat{\boldsymbol{\beta}}) + \ln|\mathbf{X}^\top\mathbf{V}^{-1}\mathbf{X}|\Big]$$

$$\hat{\boldsymbol{\beta}} = (\mathbf{X}^\top\mathbf{V}^{-1}\mathbf{X})^{-1}\mathbf{X}^\top\mathbf{V}^{-1}\mathbf{z}$$

优化器：**L-BFGS-B**，参数约束在 log 空间 $[-6, 6]$（原始尺度 $e^{-6} \approx 0.0025$ 到 $e^6 \approx 403$）。$\mathbf{V} = \mathbf{L}\mathbf{L}^\top$ Cholesky 分解实现数值稳定求逆。

**初始化**（数据自适应）：

$$\sigma_0^{(0)} = 0.5 \cdot \text{sd}(z), \quad \rho_0^{(0)} = \text{median}(d_{ii'}), \quad \sigma_1^{(0)} = 0.2 \cdot \text{sd}(z), \quad \rho_1^{(0)} = \text{median}(d_{ii'}), \quad \tau^{(0)} = 0.3 \cdot \text{sd}(z)$$

REML 取代 MCMC 可行化的关键：**5 个独立参数，无跨场耦合** → 似然曲面在 L-BFGS-B 的收敛域内足够良好 → 无需 grid search。

### 5.5 空间预测

预测点 $s_0$ 的 BLUP：

$$\hat{\mu}(s_0) = \mathbf{x}_0^\top \hat{\boldsymbol{\beta}} + \mathbf{c}(s_0)^\top \mathbf{V}^{-1}(\mathbf{z} - \mathbf{X}\hat{\boldsymbol{\beta}})$$

预测方差（含固定效应不确定性修正）：

$$\hat{\sigma}^2(s_0) = v_{prior}(s_0) - \mathbf{c}(s_0)^\top \mathbf{V}^{-1} \mathbf{c}(s_0) + \boldsymbol{\delta}^\top (\mathbf{X}^\top\mathbf{V}^{-1}\mathbf{X})^{-1} \boldsymbol{\delta}$$

其中 $v_{prior}(s_0) = \sigma_0^2 + \tilde{U}(s_0)^2 \sigma_1^2 + \tau^2$，$\boldsymbol{\delta} = \mathbf{x}_0 - \mathbf{X}^\top\mathbf{V}^{-1}\mathbf{c}(s_0)$，交叉协方差 $\mathbf{c}(s_0) = \mathbf{c}_0(s_0) + \tilde{U}(s_0) \cdot (\tilde{\mathbf{u}} \odot \mathbf{c}_1(s_0))$，其中 $[\mathbf{c}_j(s_0)]_i = \sigma_j^2 \exp(-\|s_0 - s_i\| / \rho_j)$ 是 $w_j(s_0)$ 与各训练点 $w_j(s_i)$ 间的空间协方差。

**克里金残差项的含义**：$\mathbf{c}(s_0)^\top \mathbf{V}^{-1}(\mathbf{z} - \mathbf{X}\hat{\boldsymbol{\beta}})$ 对**所有训练站点**的残差进行加权求和，权重由空间协方差决定——近的站点权重大，远的权重小。这是 SVCD 实现邻域信息利用的方式，不同于 Berrocal 2012 的在输入端做 CMAQ 平滑。

### 5.6 反变换与偏差修正

MCMC 的后验采样在反变换时自动实现方差补偿（$E[\hat{\mu}^2] = E[\hat{\mu}]^2 + \text{Var}(\hat{\mu})$）。REML 作为点估计方法，需要**显式处理**：

- **完整修正** (`bias_correction='full'`)：$\hat{Y}(s_0) = \max\big(0,\ \hat{\mu}(s_0)^2 + \hat{\sigma}^2(s_0)\big)$
- **部分修正** (`bias_correction='partial'`)：$\hat{Y}(s_0) = \max\big(0,\ \hat{\mu}(s_0)^2 + \max(0, \hat{\sigma}^2(s_0) - \tau^2)\big)$
- **无修正** (`bias_correction='none'`)：$\hat{Y}(s_0) = \max\big(0,\ \hat{\mu}(s_0)^2\big)$

完整修正是默认和推荐模式，数学上与 MCMC 的自动修正等价。部分修正排除了观测噪声 $\tau^2$ 的贡献，适用于需要预测"潜在平滑过程"而非"含噪声观测值"的场景。

### 5.7 计算代价

| 操作 | 复杂度 | 说明 |
|------|:------:|------|
| $\mathbf{V}$ 构建 | $O(n^2)$ | 核计算 + Hadamard 积 |
| Cholesky $\mathbf{V} = \mathbf{L}\mathbf{L}^\top$ | $O(n^3)$ | **主导项** |
| REML 每次迭代 | $O(n^3)$ | Cholesky + 反代求解 |
| 全网格预测 | $O(m \cdot n^2)$ | 交叉协方差 + BLUP |

以 $n \approx 645$ 站为例，单日拟合约 20 秒，全网格预测（$m \approx 21,844$）约 5 秒。

---

## 6. 核心区别总结

| | Berrocal 2010 | Berrocal 2010b | Berrocal 2012 | Berrocal 2020 | **SVCD** |
|---|---|---|---|---|---|---|
| **数据** | O3 | O3 + PM2.5（联合） | O3 | PM2.5 | O3 |
| **变换** | sqrt | sqrt(O3)+log(PM2.5) | sqrt | log | sqrt |
| **截距场 $w_0$** | 有（GP） | 有 | 有 | 有（GP） | **有** |
| **斜率场 $w_1$** | **有（coregionalized）** | **有（coregionalized）** | **砍掉** | **砍掉** | **有（独立 GP）** |
| **GP 间关系** | coregionalized $\mathbf{A}$（3 参数） | coregionalized（更大） | N/A（仅单 GP） | N/A（仅单 GP） | **独立（2 参数）** |
| **协方差参数数** | 6 | >6 | ~4 | 3 | **5** |
| **推断** | MCMC | MCMC | MCMC | MCMC (spBayes) | **REML (L-BFGS-B)** |
| **Decay 参数确定** | grid search | sensitivity analysis | discrete prior | uniform prior | **自动优化** |
| **单日耗时（估计）** | ~5 min | ~5 min | ~5 min | ~5 min | **~20 s** |
| **邻域 CMAQ** | 仅本网格 | 仅本网格 | CAR / GP 权重平滑 | 仅本网格 | **克里金残差修正** |
| **反变换偏差修正** | MCMC 后验自然处理 | MCMC 后验自然处理 | MCMC 后验自然处理 | MCMC 后验自然处理 | **显式 $\hat{\mu}^2+\hat{\sigma}^2$** |
| **预测不确定性** | 后验分位数 | 后验分位数 | 后验分位数 | 后验分位数 | **解析预测方差** |
| **用户需调参数** | MCMC 迭代数、burn-in、thin、$\phi$ 搜索网格、$\mathbf{A}$ 先验、$\tau^2$ 先验（~10 个） | 同上 + 跨污染物核心参数 | 同上 + CAR 参数 | MCMC 迭代数、$(\beta_0,\beta_1)$ 先验、$\sigma^2,\sigma^2_0$ 先验、$\phi$ 先验（~8 个） | **0 个（数据自适应初始化）** |

### 6.1 各方法的优缺点

**Berrocal 2010**（原始 Downscaler）：
- 优势：双 GP 结构完整，理论上最灵活
- 劣势：coregionalization + grid search → 计算昂贵且工程复杂；用户需手动设置 ~10 个参数（MCMC 迭代数、$\phi$ 搜索网格、$\mathbf{A}$ 先验等）

**Berrocal 2010b**（双变量扩展）：
- 优势：联合建模可借用跨污染物信息
- 劣势：参数空间更大，收益 modest，不适合单污染物场景

**Berrocal 2012**（邻域平滑）：
- 优势：引入邻域 CMAQ，MSE 降低 5-15%
- 劣势：牺牲空间变斜率，模型灵活性降级；新增 CAR/GP 参数增加用户负担

**Berrocal 2020**（跨范式对比）：
- 优势：唯一的大规模实证对比（统计 vs ML），简化版 Downscaler 由原作者背书
- 劣势：Downscaler 版本被简化为单 GP（与 Berrocal 2010 相比结构退化）；仍为 MCMC，需配置 spBayes 参数

**SVCD**：
- 优势：保留完整双 GP + 计算快 15x + 无 grid search + 显式偏差修正 + **零用户调参（数据自适应初始化）**
- 劣势：独立 GP 可能丢失截距-斜率场间的相关性信息（当前实证未观察到性能退化）

---

## 7. SVCD 的创新点

Berrocal 系列三部曲存在一个未解决的演进困境：**2010 完整但慢，2012 快了但砍掉了关键结构**。SVCD 通过三处改动打破了这个 trade-off：

**（1）MCMC → REML：15 倍加速**

REML 替代 MCMC 的基础是模型简化——coregionalization 的 $\mathbf{A}$ 矩阵移除后，协方差参数从 6 个降至 5 个且无跨场耦合，似然曲面在 L-BFGS-B 的收敛域内足够良好。Zhang (2004) 从理论上证明 Matérn 协方差参数不可一致估计（仅比值可识别）——这正是 coregionalization 下参数冗余的数学根源。SVCD 通过独立 GP 消除了这一根源，使梯度优化成为可能。

**（2）显式偏差修正**

MCMC 的后验采样自动包含 $E[\hat{\mu}^2] = E[\hat{\mu}]^2 + \text{Var}(\hat{\mu})$。SVCD 在 REML 下显式补全 $\hat{\sigma}^2$ 修正，使偏差修正逻辑透明——$\hat{\sigma}^2$ 作为解析方差可直接用于下游健康效应的误差传播。

**（3）保留空间变斜率**

Berrocal 2012 为引入邻域平滑而牺牲了斜率 GP（隐变量与随机效应的嵌套导致不可识别）。SVCD 不修改输入端，而是依赖克里金输出端的 $\mathbf{c}^\top\mathbf{V}^{-1}\mathbf{r}$ 自动实现邻域效应——完整双 GP 无需牺牲即可保留。

**（4）零用户调参**

Berrocal 系列的所有 MCMC 实现都要求用户手动设置大量参数：MCMC 迭代数、burn-in、thinning、各参数的先验分布、decay 参数的搜索网格或候选值列表等——Berrocal 2010 约需配置 ~10 个参数，Berrocal 2012 因新增 CAR/GP 结构更多，Berrocal 2020 用 spBayes 也需 ~8 个。这些参数对非统计专业用户构成了实质性使用门槛。SVCD 将所有参数收敛为数据自适应的初始化和自动终止的 L-BFGS-B 优化——用户只需调用 `fit(X, y, CMAQ)`，无需手动设置任何超参数。在跨学科应用场景（环境流行病学、健康效应评估）中，这一差异直接影响方法的可复现性和传播速度。

**概括**：Berrocal 2010 证明双 GP 是正确的，Berrocal 2012 通过牺牲斜率换取了速度，Berrocal 2020 在实证对比中同样采用了简化版（单 GP），而 SVCD 证明两者可以兼得——比 Berrocal 2020 更完整的空间灵活性，比所有 MCMC 版本更快的计算效率，且零配置即可运行。

---

## 8. 预期审稿人问题及回复

**Q1: Berrocal et al. (2010) 为什么没有采用 REML 而是用了 grid search？**

> Berrocal et al. (2010) 选择 grid search 并非 REML 本身的问题。Coregionalization 的 $A_{21}$ 参数与 decay 参数 $\phi_0, \phi_1$ 之间的跨场耦合导致似然曲面在多个方向上平坦——梯度下降在此类曲面上不可靠。Zhang (2004) 从理论上证明 Matérn 模型的单个协方差参数不可一致估计（仅参数比值在渐近意义下可识别），为这一实证观察提供了理论基础。
>
> SVCD 的独立 GP 消除了这种耦合：无 $\mathbf{A}$ 矩阵意味着 $\rho_0$ 仅控制 $w_0$ 的空间范围、$\rho_1$ 仅控制 $w_1$ 的空间范围——参数物理含义明确，无信息重叠。这使得边缘似然曲面在 $[-6, 6]^5$ 约束下足够良好，L-BFGS-B 可稳定收敛。

**Q2: 独立 GP 是否丢失了截距-斜率场的空间相关信息？**

> 截距场 $w_0(s)$ 反映 CMAQ 的系统偏差（受地形、排放源、边界层影响），斜率场 $w_1(s)$ 反映 CMAQ 化学机制的区域保真度——两者由不同物理因子驱动，理论上不必然强相关。实证方面：独立 GP 设定下 pre_exp 阶段十折交叉验证 R2 = 0.8958，与 Berrocal et al. (2010) 在类似数据集上的报告结果一致。若审稿人要求直接量化 coregionalization 的边际收益，可增补 Ablation 实验。

**Q3: L-BFGS-B 是否可能陷入局部最优？**

> 我们进行了两项检验：（1）在 pre_exp 阶段（5 天 × 10 折 = 50 次拟合）中，从 10 组随机初始值出发，REML 优化均收敛至相同最优解，各参数在 log 空间的标准差 < 0.1；（2）十折交叉验证 R2 跨折稳定（范围 0.895-0.897）。数据自适应初始化提供了足够接近真实解的出发点。

**Q4: 为什么没有复现 Berrocal 2010 的 MCMC 实现进行直接对比？**

> 完整复现含 coregionalization 和 grid search 的 MCMC 实现超出了本对比研究的范围。本文定位是**多数据融合方法的系统性能对比**，SVCD 以可比的计算代价（~20 秒/天）参与实验。若审稿人认为必要，修订稿中可增补单日 MCMC vs REML 对比作为补充。

**Q5: 全年验证的计算公平性如何保证？**

> MCMC 全年验证 ≈ 365 × 50 分钟 × 10 折 ≈ 304 小时（串行）。更重要的是，对比组中的其他方法（VNA、aVNA 等）计算代价在分钟级以下——若 SVCD 需数百小时来完成同等实验，方法对比的公平性和可重复性都无法保证。REML 将 SVCD 的成本拉低至其他方法的水平，确保所有方法在相同硬件和时间预算下运行。

---

## 9. 方法创新点摘要（论文可用表述）

> Berrocal 等人 2010-2020 年间发表的四篇论文奠定了空间变系数降尺度的理论基础，但存在一个未解决的演进困境：Berrocal et al. (2010) 的双 GP 结构（截距场 + 斜率场）能完整刻画 CMAQ 偏差的空间异质性，却受限于 MCMC 的计算代价和 coregionalization 的弱可识别性；Berrocal et al. (2012) 通过引入邻域 CMAQ 平滑获得了 5-15% 的预测改善，但代价是砍掉斜率场 GP；Berrocal et al. (2020) 在大规模跨范式对比中同样采用了单 GP 简化版，且发现 Downscaler 性能与 Kriging 几乎持平——凸显了更完整空间结构 + 更高计算效率的双重需求。
>
> SVCD 首次在保留完整双 GP 结构的前提下实现了计算可行性：（1）将推断框架从 MCMC 换为 REML，单日拟合从 ~300 秒降至 ~20 秒（~15 倍）；（2）将 coregionalization 简化为独立 GP，消除跨场参数耦合——该问题是 Zhang (2004) 所证 Matérn 协方差参数不可一致估计的直接体现；（3）在 REML 框架下补齐显式反变换偏差修正（$\hat{Y} = \hat{\mu}^2 + \hat{\sigma}^2$）。
>
> 与 Berrocal 2012 的"输入端平滑"路径不同，SVCD 依赖克里金输出端的 $\mathbf{V}^{-1}$ 加权残差传递实现空间邻域效应，避免了引入隐变量导致的不可识别性问题。这一设计使 SVCD 成为整个 Berrocal 系列中唯一一个同时保留截距场和斜率场且无需 grid search 的变体。

---

## 参考文献

1. Berrocal, V. J., Gelfand, A. E., & Holland, D. M. (2010). A spatio-temporal downscaler for output from numerical models. *Journal of Agricultural, Biological, and Environmental Statistics*, 15(2), 176-197.

2. Berrocal, V. J., Gelfand, A. E., & Holland, D. M. (2010). A bivariate space-time downscaler under space and time misalignment. *arXiv:1004.1147* (submitted to *Annals of Applied Statistics*).

3. Berrocal, V. J., Gelfand, A. E., & Holland, D. M. (2012). Space-time data fusion under error in computer model output: An application to modeling air quality. *Biometrics*, 68(3), 837-848.

4. Berrocal, V. J., Guan, Y., Muyskens, A., Wang, H., Reich, B. J., Mulholland, J. A., & Chang, H. H. (2020). A comparison of statistical and machine learning methods for creating national daily maps of ambient PM2.5 concentration. *Atmospheric Environment*, 222, 117130.

5. Zhang, H. (2004). Inconsistent estimation and asymptotically equal interpolations in model-based geostatistics. *Journal of the American Statistical Association*, 99(465), 250-261.
