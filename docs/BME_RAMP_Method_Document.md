# BME-RAMP 贝叶斯最大熵融合方法数学原理

**基于 Xu et al. (2016) Bayesian Maximum Entropy + RAMP 区域化模型性能评估框架**
**2026-06-08**

---

## 目录

1. 概述
2. 数据变换
3. BME 估计方法论
4. RAMP 软数据构建
5. Offset 偏移量分析
6. 时空协方差模型
7. 空间预测
8. 完整算法伪代码
9. 与其他方法的对比
10. 超参数默认值
11. 参考文献

---

## 1 概述

BME-RAMP（Bayesian Maximum Entropy with Regionalized Air Quality Model Performance）是一种现代化的时空数据融合方法，将化学传输模型（CTM/CAMx）的粗分辨率预测与稀疏的地面监测站点观测数据进行融合，生成高分辨率的臭氧（O$_3$）或 PM$_{2.5}$ 浓度估计。

该方法由 Xu et al. (2016) 在 *Environmental Science & Technology* 上发表，是对 de Nazelle et al. (2010) BME 融合框架的重要扩展，核心改进包括：

1. **BME 框架**：基于最大熵原理和贝叶斯条件化，能够融合非线性、非高斯知识源
2. **RAMP 方法**：区域化空气质量模型性能评估，首次考虑 CTM 模型性能的时空变异性
3. **非参数软数据**：不预设观测-预测之间的参数化关系（如线性、同方差），完全保留非线性和异方差性
4. **Offset 变换**：通过核平滑估计确定性趋势，将非平稳场转化为平稳随机场
5. **解析不确定性**：直接给出预测值的后验均值和方差

**方法来源**：

- **BME 理论**：Christakos (1990, 2000) 现代地统计学的 "知识处理" 框架
- **CAMP 前身**：de Nazelle et al. (2010) 恒定模型性能假设（仅适用于小区域、短时段）
- **RAMP 创新**：Xu et al. (2016) 将 CAMP 扩展为 RAMP，允许模型性能在空间和时间上变化

**应用场景**：

- 美国大陆尺度 (continental U.S.) 臭氧日浓度估计
- 两种时间指标：DM8A（日最大 8 小时平均）和 D24A（日 24 小时平均）
- 可扩展至 PM$_{2.5}$ 等其他标准空气污染物

---

## 2 数据变换

### 2.1 前向 Offset 变换

设 $Z(p) = Z(\mathbf{s}, t)$ 为表示日臭氧浓度的时空随机场（S/TRF），其中 $p = (\mathbf{s}, t)$ 为时空坐标。

将 $Z(p)$ 分解为确定性趋势（offset）与平稳随机场之和：

$$Z(p) = X(p) + o_Z(p) \quad \text{(1)}$$

其中：
- $o_Z(p)$：offset 偏移量，是时空坐标 $p$ 的确定性函数，可无误差计算
- $X(p)$：平稳/均匀时空随机场，表示去除 offset 后的变率和不确定性

对观测数据 $z_o$ 在位置 $p_o$ 进行变换：

$$x_o = z_o - o_Z(p_o) \quad \text{(2)}$$

### 2.2 Offset 的计算

Offset $o_Z(p_i)$ 通过对周围观测数据的指数核平滑计算：

$$o_Z(p_i) = \frac{\sum_{j=1}^{N} w_j z_j}{\sum_{j=1}^{N} w_j} \quad \text{(3)}$$

其中 $z_j$ 为 $p_i$ 邻域内的观测值，核平滑权重为：

$$w_j = \exp\left(-\frac{\|\mathbf{s}_i - \mathbf{s}_j\|}{a_r}\right) \cdot \exp\left(-\frac{|t_i - t_j|}{a_t}\right) \quad \text{(4)}$$

- $a_r$：空间平滑范围（km）
- $a_t$：时间平滑范围（day）

**最优 offset 选择**：$a_r = 50$ km，$a_t = 10$ 天，在低方差（最小化估计误差）和高自相关（邻域信息充分）之间取得平衡。

### 2.3 反变换

预测完成后，从 offset-removed 尺度回到原始浓度尺度：

$$\hat{z}_k = \hat{x}_k + o_Z(p_k) \quad \text{(5)}$$

其中 $\hat{x}_k$ 为 BME 对 $X(p_k)$ 的估计值。

---

## 3 BME 估计方法论

### 3.1 知识库定义

BME 框架将知识分为两类：

**G-KB（General Knowledge Base，一般知识库）**：

$$G = \{m_X(p), c_X(p, p')\}$$

- $m_X(p) = E[X]$：均值函数，描述一致性趋势
- $c_X(p, p') = E[(X(p) - m_X(p))(X(p') - m_X(p'))]$：协方差函数，描述时空依赖结构

**S-KB（Site-specific Knowledge Base，站点知识库）**：

$$S = \{x_o, f_S(\mathbf{x}_m)\}$$

- $x_o$：hard data（硬数据），观测站点处的精确值
- $f_S(\mathbf{x}_m)$：soft data（软数据），CTM 预测点处的概率密度函数

### 3.2 BME 三步骤

**步骤 1 — 先验 PDF**：利用最大熵原理处理 G-KB，得到先验概率密度函数 $f_G$。当 G-KB 仅包含均值和协方差时，$f_G$ 为高斯 PDF。

**步骤 2 — 后验 PDF**：利用贝叶斯条件化规则整合 S-KB：

$$f_K(x_k) = A^{-1} \int f_S(\mathbf{x}_m) \cdot f_G(\mathbf{x}_{map}) \, d\mathbf{x}_m \quad \text{(6)}$$

其中 $\mathbf{x}_{map} = (x_k, \mathbf{x}_o, \mathbf{x}_m)$ 为映射点的全体值，$A$ 为归一化常数。

**步骤 3 — 估计**：基于 BME 后验 PDF 计算时空估计值（如后验均值、后验方差）。

### 3.3 硬数据与软数据

**硬数据**：监测站点的观测值 $x_o$，视为无误差代理（hard data, error-free proxy）。

**软数据**：CTM 计算节点处的 offset-removed 值 $x_m$，以 PDF $f_S(\mathbf{x}_m)$ 描述其不确定性：

$$f_S(\mathbf{x}_m) = \prod_{i=1}^{n_m} f(x_i | \tilde{x}_i, p_i) \quad \text{(7)}$$

其中 $\tilde{x}_i$ 为 offset-removed CTM 预测值，$f(x_i | \tilde{x}_i, p_i)$ 刻画 CTM 在时空点 $p_i$ 处的预测性能。

### 3.4 CAMP vs RAMP 的关键区别

**CAMP（Constant Air quality Model Performance）**：参数 $λ_1, λ_2$ 仅随 CTM 预测值变化，不随空间/时间变化：

$$f(z_i | \tilde{z}_i) = \Phi(z_i; λ_1(\tilde{z}_i), λ_2(\tilde{z}_i)) \quad \text{(8)}$$

**RAMP（Regionalized Air quality Model Performance）**：参数 $λ_1, λ_2$ 同时随 CTM 预测值和时空坐标变化：

$$f(z_i | \tilde{z}_i, p_i) = \Phi(z_i; λ_1(\tilde{z}_i, p_i), λ_2(\tilde{z}_i, p_i)) \quad \text{(9)}$$

其中 $\Phi$ 为零以下截断的正态分布：
- $λ_1(\tilde{z}_i, p_i)$：期望值（bias-corrected 预测值）
- $λ_2(\tilde{z}_i, p_i)$：方差（bias-corrected 预测的不确定性）

---

## 4 RAMP 软数据构建

### 4.1 第一阶段：站点级分层统计

对每个监测站点 $s_n$ 和时间 $t$：

**配对池化**：收集时间窗口 $ΔT = 120$ 天内所有观测-预测对 $(z_j, \tilde{z}_j)$。

**十分层**：按预测值 $\tilde{z}_j$ 排序，等分为 10 个分位区间（bins）。

**逐层统计**：对第 $b$ 层（预测均值为 $\tilde{z}_b$），计算：

$$\hat{\lambda}_1(\tilde{z}_b, s_n, t) = \frac{1}{n_0(\tilde{z}_b, s_n, t)} \sum_{j: \tilde{z}_j \in \text{bin}_b} z_j(s_n, t) \quad \text{(10)}$$

$$\hat{\lambda}_2(\tilde{z}_b, s_n, t) = \frac{1}{n_0(\tilde{z}_b, s_n, t)} \sum_{j: \tilde{z}_j \in \text{bin}_b} \big(z_j(s_n, t) - \hat{\lambda}_1(\tilde{z}_b, s_n, t)\big)^2 \quad \text{(11)}$$

其中 $n_0(\tilde{z}_b, s_n, t)$ 为第 $b$ 层中的配对数量。

**ΔT = 120 天的选择理由**：在数据充足性（大窗口样本多）和季节特异性（小窗口更能反映当前季节的模型性能特征）之间取得平衡。

### 4.2 第二阶段：空间插值到 CTM 网格

对每个 CTM 计算节点 $p_i = (s_i, t_i)$：

**步骤 A — 预测值维度插值**：对每个站点 $s_n$，在 10 个 bin 点之间进行线性插值/外推，获得连续预测值 $\tilde{z}_i$ 对应的期望和方差：

$$\hat{\lambda}_1(\tilde{z}_i, s_n, t_i), \quad \hat{\lambda}_2(\tilde{z}_i, s_n, t_i)$$

**步骤 B — 空间维度插值**：将各站点的结果空间插值到 CTM 节点：

$$\hat{\lambda}_k(\tilde{z}_i, p_i) = \frac{\sum_{n=1}^{N} w(s_i, s_n) \cdot \hat{\lambda}_k(\tilde{z}_i, s_n, t_i)}{\sum_{n=1}^{N} w(s_i, s_n)}, \quad k = 1, 2 \quad \text{(12)}$$

空间权重为反距离：

$$w(s_i, s_n) = \frac{1}{\text{dist}(s_i, s_n)} \quad \text{(13)}$$

其中 $n = 1, \ldots, N$ 为距离 $s_i$ 最近的 $N$ 个监测站点。

**物理含义**：

- $\tilde{z}_i - \hat{\lambda}_1(\tilde{z}_i, p_i)$：CTM 在 $p_i$ 处的**系统偏差**（bias）
- $\hat{\lambda}_2(\tilde{z}_i, p_i)$：CTM 在 $p_i$ 处的**不精确度**（imprecision/variance）

**非参数优势**：RAMP 不预设 $z$ 与 $\tilde{z}$ 之间的任何函数关系（不假设线性、不假设同方差），地理和时间上的非线性、非齐性关系被自动捕获。

**注意**：RAMP 输出的 $\hat{\lambda}_1, \hat{\lambda}_2$ 定义于原始浓度空间。在实际 BME 融合中，需通过 offset 关系 $x = z - o_Z(p)$ 将软数据变换到 offset-removed 空间：$f(x_i | \tilde{x}_i, p_i) = \Phi(x_i; \hat{\lambda}_1 - o_Z(p_i), \hat{\lambda}_2)$。

---

## 5 Offset 偏移量分析

### 5.1 定义与目的

Offset 变换将原始臭氧数据 $Z(p)$ 分解为平滑趋势 $o_Z(p)$ 和平稳残差 $X(p)$。其目的是：

1. 去除大尺度时空趋势，使 $X(p)$ 满足平稳性假设
2. 降低变换后数据的方差，最小化估计误差
3. 保留足够的时空自相关，确保邻域信息充分

### 5.2 核平滑参数

基于方差-自相关平衡准则选择最优参数：

| 参数 | 符号 | 最优值 | 说明 |
|------|------|--------|------|
| 空间范围 | $a_r$ | 50 km | 指数核空间平滑范围 |
| 时间范围 | $a_t$ | 10 day | 指数核时间平滑范围 |

### 5.3 与均值函数的关系

在 G-KB 中，$m_X(p)$ 为 offset-removed 数据 $x_o$ 的均值。理想情况下，$m_X(p) \approx 0$（$X(p)$ 为零均值平稳场）。

---

## 6 时空协方差模型

### 6.1 实验协方差计算

从变换后的观测数据 $x_o = z_o - o_Z(p_o)$ 计算实验协方差。

对空间滞后 $r$ 和时间滞后 $\tau$：

$$\hat{c}_X(r, \tau) = \frac{1}{N(r, \tau)} \sum_{j=1}^{N(r, \tau)} (x_{\text{head},j} - m_X)(x_{\text{tail},j} - m_X) \quad \text{(14)}$$

其中 $N(r, \tau)$ 为空间滞后 $r$、时间滞后 $\tau$ 处的数据对数量，$m_X$ 为 $x_o$ 的均值。

实际中分别计算和绘制 $\hat{c}_X(r, 0)$（空间协方差）和 $\hat{c}_X(0, \tau)$（时间协方差）。

### 6.2 三结构指数协方差模型

采用三结构（3-structured）指数协方差模型：

$$c_X(r, \tau) = C_0 \cdot \left[ \exp\left(-\frac{3r}{a_{r1}}\right) \cdot \exp\left(-\frac{3\tau}{a_{t1}}\right) + (1 - \alpha) \cdot \exp\left(-\frac{3r}{a_{r2}}\right) \cdot \exp\left(-\frac{3\tau}{a_{t2}}\right) + \beta \cdot \exp\left(-\frac{3r}{a_{r3}}\right) \cdot \exp\left(-\frac{3\tau}{a_{t3}}\right) \right] \quad \text{(15)}$$

**物理解释**：三结构分别对应小尺度（短程相关性）、中尺度和大尺度（长程相关性）的时空变异成分。

### 6.3 待估协方差参数

| 参数 | 符号 | 说明 |
|------|------|------|
| 基础方差 | $C_0$ | 第一结构的偏基台值 |
| 结构权重 | $\alpha, \beta$ | 第二、三结构相对于第一结构的方差比例 |
| 空间范围 1/2/3 | $a_{r1}, a_{r2}, a_{r3}$ | 不同尺度的空间相关范围 |
| 时间范围 1/2/3 | $a_{t1}, a_{t2}, a_{t3}$ | 不同尺度的时间相关范围 |
| 总基台值 | $C_0 \cdot (2 - \alpha + \beta)$ | 总方差 (total sill) |

---

## 7 空间预测

### 7.1 BME 后验估计

给定硬数据 $x_o$ 和软数据 PDF $f_S(x_m)$，BME 后验 PDF $f_K(x_k)$ 在估计点 $p_k$ 处的值由式 (6) 给出。

在实际实现中（G-KB 为高斯型），BME 简化为带有软数据的克里金扩展。

### 7.2 估计值计算

**后验均值**（点估计）：

$$\hat{x}_k = E_{f_K}[X(p_k)] = \int x_k \cdot f_K(x_k) \, dx_k \quad \text{(16)}$$

**后验方差**（不确定性）：

$$\hat{\sigma}_k^2 = \text{Var}_{f_K}[X(p_k)] = \int (x_k - \hat{x}_k)^2 \cdot f_K(x_k) \, dx_k \quad \text{(17)}$$

### 7.3 最终预测

将 offset 加回，使用式 (5) 反变换得到最终预测值 $\hat{z}_k$，同时输出后验标准差 $\sqrt{\hat{\sigma}_k^2}$ 作为不确定性度量。

### 7.4 三种估计场景

| 场景 | 数据源 | 特点 |
|------|--------|------|
| **OBS** | 仅观测硬数据 | 克里金的线性限制情况（BME 的退化形式） |
| **CAMP** | 观测 + CTM（恒定性能软数据） | de Nazelle et al. 的原始方法 |
| **RAMP** | 观测 + CTM（区域化性能软数据） | Xu et al. 的改进方法，本文核心 |

### 7.5 验证方法

使用留一交叉验证（noncollocated validation）：

- 对于每个观测点 $p_j$，排除验证半径 $r_v$ 内的所有观测数据
- 用剩余数据重新估计 $\hat{z}_j^*(r_v)$
- 计算验证误差 $e_j^*(r_v) = \hat{z}_j^*(r_v) - z_j$

**评价指标**：

$$RMSE(r_v) = \sqrt{\frac{1}{n}\sum_{j}(e_j^*)^2} \quad \text{(18)}$$

$$R^2(r_v) = \text{Spearman's } \rho^2(\mathbf{z}^*, \mathbf{z}) \quad \text{(19)}$$

**$R^2$ 百分比改进**（相对于 OBS）：

$$PC_{R^2}^{OBS \to METHOD}(r_v) = 100 \times \frac{R^2_{METHOD}(r_v) - R^2_{OBS}(r_v)}{R^2_{OBS}(r_v)} \quad \text{(20)}$$

---

## 8 完整算法伪代码

```
Algorithm: BME-RAMP 融合算法

Require: 监测站坐标 S, 观测值 z, CTM 网格坐标 S_grid, CTM 预测值 z~(tilde)
Ensure: 预测浓度 ^z_k 及后验方差 ^sigma2_k

Phase 1: Offset 计算
  for each 估计点 p_k = (s_k, t_k) do
    o_Z(p_k) <- 核加权平均周围观测值 (式 3-4, a_r=50km, a_t=10d)
  end for
  for each 观测点 p_j do
    x_j <- z_j - o_Z(p_j)                    {Offset 变换}
  end for

Phase 2: 协方差建模
  计算实验协方差 ^c_X(r,0) 和 ^c_X(0,tau) (式 14)
  拟合三结构指数协方差模型 (式 15)

Phase 3: RAMP 软数据构建
  for each CTM 网格点 p_i = (s_i, t_i) do
    for each 最近监测站 s_n (n=1,...,N) do
      收集 |t - t_i| <= 120天 的 (z_j, ~z_j) 对
      按 ~z 十分层, 计算 ^lambda1(~z_b, s_n, t_i) 和 ^lambda2(~z_b, s_n, t_i) (式 10-11)
      线性插值获得 ^lambda1(~z_i, s_n, t_i) 和 ^lambda2(~z_i, s_n, t_i)
    end for
    反距离空间插值获得 ^lambda1(~z_i, p_i) 和 ^lambda2(~z_i, p_i) (式 12)
    构造软数据 PDF: f_S(x_i) = Phi(x_i; ^lambda1, ^lambda2)
  end for

Phase 4: BME 估计
  for each 估计点 p_k do
    ~x_k <- ~z_k - o_Z(p_k)                  {CTM 值 offset 变换}
    计算 BME 后验 PDF f_K(x_k) (式 6)
    ^x_k <- 后验均值 (式 16)
    ^sigma2_k <- 后验方差 (式 17)
    ^z_k <- ^x_k + o_Z(p_k)                  {反变换}
  end for

  return ^z_k, ^sigma2_k
```

---

## 9 与其他方法的对比

### 9.1 方法对比

| 特征 | BME-RAMP | BME-CAMP | Cokriging | SVCD | Downscaler |
|------|----------|----------|-----------|------|------------|
| 理论框架 | 贝叶斯最大熵 | 贝叶斯最大熵 | 线性地质统计 | 空间变系数 + REML | 贝叶斯 MCMC |
| 回归模型 | 非参数 (无预设关系) | 非参数 (无预设关系) | 参数化线性 | 空间变系数线性 | 全局系数线性 |
| 异方差性 | 完全捕获 | 部分捕获 | 假设同方差 | 通过 Log 变换缓解 | 假设同方差 |
| 非线性 | 完全捕获 | 部分捕获 | 无法处理 | 线性 + Log 变换 | 线性 |
| 空间核 | 三结构指数 | 三结构指数 | 指数 | Mat\'ern ($\nu$=1.5) | 指数核 |
| 参数估计 | 协方差拟合 | 协方差拟合 | 协方差拟合 | REML (L-BFGS-B) | MCMC (2500 迭代) |
| 数据变换 | Offset 去趋势 | Offset 去趋势 | 无 | Log + 偏差修正 | 无变换 |
| 时间信息 | 利用 (时空协方差) | 利用 (时空协方差) | 仅空间 | 仅空间 (逐日独立) | 仅空间 (逐日独立) |
| 模型性能区域化 | RAMP (时空变化) | CAMP (恒定) | 无 | 无 (空间变系数间接) | 无 |
| 不确定性 | 解析后验方差 | 解析后验方差 | 克里金方差 | 解析方差 + Delta Method | 后验样本 |
| 训练复杂度 | O(N_pair + n³) | O(N_pair + n³) | O(n³) | O(N_iter·n³) | O(N_MCMC·n³) |
| 适用尺度 | 大陆级时空 | 区域级时空 | 区域级空间 | 区域级空间 | 区域级空间 |

### 9.2 优势

1. **非参数关系建模**：不预设观测与 CTM 预测之间的函数形式，自动捕获非线性和异方差性，这是对 Berrocal et al. (2010)、McMillan et al. (2010) 等参数化方法的根本性超越。

2. **区域化模型性能**：RAMP 方法能体现 CTM 在不同地理位置、不同季节的差异性性能（如东西海岸城市过预测偏差高、中部低），CAMP 假设恒定性能会对此过度修正。

3. **时空一体化**：三维（经度 × 纬度 × 时间）而非逐日独立建模，能利用时间维度信息提高估计稳定性和连续性。

4. **硬-软数据统一框架**：BME 框架天然支持将监测数据（硬数据，确定性）和 CTM 输出（软数据，概率分布）统一处理，权重由不确定性自动决定。

5. **计算效率**：RAMP 基于配对数据的简单分层统计 + 确定性插值，无 MCMC 迭代、无 burn-in/thinning 问题，可在并行机上高效部署。

6. **远距离估计增强**：验证半径 108 km 处 R² 改进达 2.9%（DM8A）和 5.9%（D24A），证明该方法对远离监测站的区域尤为有效。

### 9.3 局限性

1. **协方差平稳假设**：$X(p)$ 被假设为平稳/均匀 S/TRF，实际中残差可能仍含非平稳成分。

2. **Offset 依赖核带宽**：核平滑参数 $(a_r, a_t)$ 需要针对具体研究区域调优，不同区域最优值可能不同。

3. **RAMP 分层数固定**：10 个分位区间为固定设置，数据稀疏场景可能不足或过拟合。

4. **ΔT = 120 天窗口**：平衡数据量与季节特异性的折中，极端季节性变化时可能不敏感。

5. **三结构协方差**：模型形式（指数核、结构化）为半经验选择，需人工拟合而非自动优化。

6. **计算存储**：时空协方差矩阵为 $n_{obs} \times n_{obs}$（n 为时空观测点数），大规模时需稀疏近似。

---

## 10 超参数默认值

| 参数 | 默认值 | 说明 |
|------|--------|------|
| $a_r$ | 50 km | Offset 空间核平滑范围 |
| $a_t$ | 10 day | Offset 时间核平滑范围 |
| ΔT | 120 day | RAMP 配对池化时间窗口 |
| $N_{bins}$ | 10 | RAMP 分层数（等分位数） |
| $N$ | 最近站点数 | 空间插值所用近邻站点数 |
| $r_v$ | {0, 36, 72, 108} km | 验证半径集合 |
| 协方差类型 | 三结构指数 | 时空协方差模型类型 |
| 截断分布 | 零以下截断正态 | 软数据 PDF 形式 |
| 偏移量类型 | 指数核平滑 | Offset 计算方法 |

---

## 参考文献

1. Xu, Y., Serre, M. L., Reyes, J., & Vizuete, W. (2016). Bayesian Maximum Entropy Integration of Ozone Observations and Model Predictions: A National Application. *Environmental Science & Technology*, 50(8), 4393–4400.

2. de Nazelle, A., Arunachalam, S., & Serre, M. L. (2010). Bayesian Maximum Entropy Integration of Ozone Observations and Model Predictions: An Application for Attainment Demonstration in North Carolina. *Environmental Science & Technology*, 44(15), 5707–5713.

3. Christakos, G. (2000). *Modern Spatiotemporal Geostatistics*. Oxford University Press.

4. Christakos, G. (1990). A Bayesian/maximum-entropy view to the spatial estimation problem. *Mathematical Geology*, 22(7), 763–777.

5. Serre, M. L., & Christakos, G. (1999). Modern geostatistics: computational BME analysis in the light of uncertain physical knowledge — the Equus Beds study. *Stochastic Environmental Research and Risk Assessment*, 13(1–2), 1–26.

6. Berrocal, V. J., Gelfand, A. E., & Holland, D. M. (2010). A spatio-temporal downscaler for output from numerical models. *Annals of Applied Statistics*, 4(4), 1820–1848.

7. McMillan, N. J., Holland, D. M., Morara, M., & Feng, J. (2010). Combining numerical model output and particulate data using Bayesian space-time modeling. *Environmetrics*, 21(1), 48–63.

8. Fuentes, M., & Raftery, A. E. (2005). Model Evaluation and Spatial Interpolation by Bayesian Combination of Observations with Outputs from Numerical Models. *Biometrics*, 61(1), 36–45.

9. Akita, Y., Chen, J.-C., & Serre, M. L. (2012). The moving-window Bayesian maximum entropy framework: estimation of PM$_{2.5}$ yearly average concentration across the contiguous United States. *Journal of Exposure Science and Environmental Epidemiology*, 22(5), 496–501.

10. Reyes, J. M., & Serre, M. L. (2014). An LUR/BME Framework to Estimate PM$_{2.5}$ Explained by on Road Mobile and Stationary Sources. *Environmental Science & Technology*, 48(3), 1736–1744.
