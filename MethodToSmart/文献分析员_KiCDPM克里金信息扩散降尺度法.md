# 【可执行方法规范】

## 方法名称
克里金信息扩散概率模型 (Kriging-informed Conditional Diffusion Probabilistic Model, Ki-CDPM)

## 文献来源
- 论文标题：Towards Kriging-informed Conditional Diffusion for Regional Sea-Level Data Downscaling
- 作者：Subhankar Ghosh, Arun Sharma, Jayant Gupta, Aneesh Subramanian, Shashi Shekhar
- 年份：2024年
- 会议：SIGSPATIAL '24

## 核心公式

### 1. 条件扩散模型框架
扩散模型通过去噪过程学习条件分布 $p(y|x)$，其中 $x$ 为低分辨率输入，$y$ 为高分辨率目标。

前向过程（加噪）：
$$
q(y_t | y_{t-1}) = \mathcal{N}(y_t; \sqrt{1-\beta_t} y_{t-1}, \beta_t I)
$$

反向过程（去噪）：
$$
p_\theta(y_{t-1} | y_t, x) = \mathcal{N}(\mu_\theta(y_t, x, t), \sigma_t^2 I)
$$

### 2. Kriging约束项
将空间克里金插值的协方差结构融入扩散模型的注意力机制：

空间协方差矩阵（基于变异函数）：
$$
\Sigma_{ij} = \gamma(h_{ij}) = \begin{cases} 0 & h = 0 \\ C(0) - C(h) & h > 0 \end{cases}
$$

其中 $h_{ij}$ 是两点间的距离，$\gamma(h)$ 是变异函数。

### 3. Kriging条件均值
对于低分辨率网格点 $x_i$，Kriging预测：
$$
\hat{y}_{Krig}(x_i) = \sum_{j=1}^{n} \lambda_j C(x_i, x_j)
$$
其中 $\lambda_j$ 是Kriging权重，$C$ 是协方差函数。

### 4. 条件引导机制
将Kriging信息编码为扩散模型的条件输入：
$$
\epsilon_\theta(y_t, x, t, K_{spatial}) = \epsilon_\theta(y_t, x, t) + w \cdot \nabla_{y_t} \log p(K_{spatial}|y_t)
$$

## 参数清单

| 参数名 | 类型 | 默认值 | 说明 |
|-------|------|-------|------|
| diffusion_steps | int | 1000 | 扩散模型时间步数 |
| beta_schedule | str | linear | 噪声调度方案 |
| spatial_kernel | str | gaussian | 空间核函数类型 |
| range_param | float | - | 变异函数变程参数 |
| nugget | float | 0 | 变异函数块金效应 |
| guidance_weight w | float | 0.1-0.5 | Kriging引导权重 |
| backbone | str | U-Net | 扩散模型骨干网络 |

## 数据规格

| 数据 | 格式 | 说明 |
|-----|------|-----|
| 低分辨率输入 | 2D grid | GCM输出或CMAQ 25km |
| 高分辨率目标 | 2D grid | 观测或精细化参考 |
| 空间掩码 | binary | 陆地/海洋或城市/郊区 |
| 协方差结构 | matrix | 从训练数据学习 |

## 实现步骤

### 阶段1：空间协方差结构学习
1. 使用训练数据估计变异函数参数 $\gamma(h)$
2. 构建空间协方差矩阵 $\Sigma$
3. 计算Kriging权重矩阵 $K$

### 阶段2：条件扩散模型构建
1. 构建U-Net骨干网络
2. 编码Kriging空间结构为额外条件通道
3. 设计时空注意力机制融入Kriging核

### 阶段3：训练与推理
1. 条件扩散训练：$p_\theta(y|x, K)$
2. 多步去噪生成高分辨率场
3. 可选：多次采样产生集合预报

## 方法特点

1. **Kriging集成**：将地统计学空间插值的理论保证融入深度生成模型
2. **不确定性量化**：扩散模型的随机性自然产生概率预报
3. **空间一致性**：保证生成的高分辨率场与低分辨率输入在空间上一致
4. **泛化能力强**：可应用于气象、生态、城市规划等多领域

## 可复现性评估

- **数据需求**：需要成对的低-高分辨率训练数据
- **计算成本**：中等（需要学习空间协方差结构）
- **代码可用性**：暂无开源代码
- **方法创新度**：高（首次将Kriging与扩散模型结合）

## 物理可解释性

- Kriging核编码了空间相关性的物理结构
- 扩散模型学习残差的随机性
- 条件引导确保空间一致性



## 方法指纹
MD5: kriging_informed_conditional_diffusion_pm

## 应用场景

- CMAQ 25km→3km统计降尺度
- 卫星产品与地面观测的空间融合
- 多尺度空气质量预测不确定性估计
