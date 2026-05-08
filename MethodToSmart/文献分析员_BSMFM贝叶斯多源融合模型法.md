# 【可执行方法规范】

## 方法名称
贝叶斯多源融合模型 (Bayesian Multisource Fusion Model, BSMFM)

## 文献来源
- 论文标题：A Bayesian Multisource Fusion Model for Spatiotemporal PM2.5 in an Urban Setting
- 作者：Abi I. Riley, Marta Blangiardo, Frédéric B. Piel, Andrew Beddows, Sean Beevers, Gary W. Fuller, Paul Agnew, Monica Pirani
- 年份：2025年

## 核心公式

### 1. 观测模型
$$
Y(s,t) = \eta(s,t) + \varepsilon(s,t), \quad \varepsilon(s,t) \sim N(0, \sigma_\varepsilon^2)
$$
其中 $Y(s,t)$ 为位置 $s$ 和时间 $t$ 的对数浓度，$\eta(s,t)$ 为潜在真值，$\varepsilon$ 为测量误差。

### 2. 潜在过程分解
$$
\eta(s,t) = \mu(s,t) + \omega(s,t)
$$
其中 $\mu(s,t)$ 为大尺度协变量效应，$\omega(s,t)$ 为时空潜在过程。

### 3. 大尺度效应（基线模型）
$$
\mu(s,t) = \beta X(s,t)
$$

### 4. 空间变化系数（SVC）模型
$$
\mu(s,t) = \beta_k(s) X_k(s,t) + \beta X(s,t)
$$
其中 $\beta_k(s) \sim N(0, \sigma_\beta^2 \Sigma)$，允许协变量的边际效应在空间上非平稳。

### 5. 时间变化系数（TVC）模型
$$
\mu(s,t) = \beta_k(t) X_k(s,t) + \beta X(s,t)
$$
TVC可采用三种时间结构：
- **IID**: $\beta_k(t) \sim N(0, \sigma_\beta^2)$
- **AR(1)**: $\beta_k(t) = \phi \beta_k(t-1) + \epsilon_t, \quad \epsilon_t \sim N(0, \sigma_\beta^2)$
- **RW1**: $\beta_k(t) - \beta_k(t-1) \sim N(0, \sigma_\beta^2)$

### 6. 时空潜在过程（SPDE-INLA）
$$
\omega(s,t) = a \omega(s,t-1) + u(s,t)
$$
其中 $u(s,t)$ 为高斯随机场（GRF），通过SPDE方法近似：
$$
(\kappa^2 - \Delta)^{\alpha/2} u(s) = W(s), \quad \alpha > d/2
$$
其解为Matérn协方差函数。

### 7. 预测分布
$$
f(Y(s^*,t)|Y) = \int f(Y(s^*,t)|\theta) f(\theta|Y(s,t)) d\theta
$$

### 8. NDVI计算
$$
\text{NDVI} = \frac{\text{NIR} - \text{Red}}{\text{NIR} + \text{Red}}
$$

## 参数清单

| 参数名 | 类型 | 默认值 | 说明 |
|-------|------|-------|------|
| $\sigma_\varepsilon^2$ | float | - | 测量误差方差（nugget） |
| $\sigma_\omega^2$ | float | - | 潜在过程方差 |
| $\rho$ | float | 20km | 空间相关范围（PC先验） |
| $a$ | float | 0.95 | AR(1)自相关系数 |
| $\alpha$ | float | 3/2 | SPDE平滑参数 |
| max_edge_inner | float | 2.5km | 内网格最大边长 |
| max_edge_outer | float | 4km | 外网格最大边长 |

## 实现步骤

### 数据准备
1. 收集地面监测站PM2.5数据（44个站点，2014-2019年月均数据）
2. 获取卫星AOD产品（NASA MAIAC MCD19A2）
3. 获取污染气候图模型（PCM）输出（1km×1km分辨率）
4. 获取英国空气质量再分析（AQR）数据（0.1°分辨率）
5. 计算归一化植被指数（NDVI）

### 模型构建
1. **基础框架**：定义对数变换后的观测模型 $Y(s,t) = \eta(s,t) + \varepsilon(s,t)$
2. **大尺度效应**：$\mu(s,t) = \beta X(s,t)$，包含PCM、AQR、背景指示符、环境变量
3. **潜在时空过程**：使用SPDE-INLA方法建模 $\omega(s,t)$
4. **变化系数**：引入SVC和/或TVC以捕捉非平稳性

### 推断
1. 使用INLA（集成嵌套拉普拉斯近似）进行贝叶斯推断
2. 构建三角网格（内网最大边长2.5km，外网4km）
3. 设置PC先验：$P(\rho > 20\text{km}) = 0.01$，$P(\sigma > 0.1) = 0.1$
4. 计算后验预测分布

### 预测
1. 对1km×1km网格上的新位置进行预测
2. 获取完整后验预测分布（用于不确定性量化）
3. 计算阈值超标概率

## 方法特点

1. **多源数据融合**：整合监测数据、卫星数据、模型输出、植被指数
2. **变化系数**：SVC和TVC允许效应在空间和时间上非平稳
3. **SPDE-INLA方法**：高效处理大尺度时空高斯随机场
4. **不确定性量化**：提供完整后验预测分布和95%可信区间
5. **尺度对齐**：通过升尺度方法解决空间错位问题

## 性能指标

| 指标 | 数值 |
|-----|------|
| R²（模型拟合） | 0.941-0.943 |
| PMCC | 185-195 |
| 时间交叉验证R² | 0.71-0.72 |
| 空间交叉验证R² | 0.68-0.70 |
| 95%CI覆盖率 | 0.89-0.91 |

## 应用场景

- 城市尺度PM2.5高分辨率制图（1km）
- 多源空气质量数据融合
- 不确定性量化与超标概率预测
- 流行病学暴露评估

### 数据规格

#### 输入
| 数据 | 格式 | 维度 | 单位 |
|-----|------|-----|------|
| 地面监测PM2.5 | array | (n_stations, n_times) | μg/m³ |
| 卫星AOD | array | (n_grids, n_times) | - |
| PCM模型输出 | array | (1km, 1km) | μg/m³ |
| AQR再分析 | array | (0.1°, 0.1°) | μg/m³ |
| NDVI | array | (n_grids,) | - |

#### 输出
| 数据 | 格式 | 单位 |
|-----|------|------|
| PM2.5预测分布 | array | μg/m³ |

## 方法指纹
MD5: bayesian_multisource_fusion_model_bsmfm

## 随机性
- [x] 是（贝叶斯推断带随机采样）
