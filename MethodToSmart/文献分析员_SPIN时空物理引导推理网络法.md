# 【可执行方法规范】

## 方法名称
时空物理引导推理网络 (Spatiotemporal Physics-Guided Inference Network, SPIN)

## 文献来源
- 论文标题：Physics-Guided Inductive Spatiotemporal Kriging for PM2.5 with Satellite Gradient Constraints
- 作者：Shuo Wang, Mengfan Teng, Yun Cheng, Lothar Thiele, Olga Saukh, Shuangshuang He, Yuanting Zhang, Jiang Zhang, Gangfeng Zhang, Xingyuan Yuan, Jingfang Fan
- 年份：2025年

## 核心公式

### 1. 地理空间图 (Geospatial Graph)
$$
A^S_{ij} = \begin{cases} \exp\left(-\frac{\text{dist}(v_i, v_j)^2}{\sigma^2}\right) & \text{if } \text{dist}(v_i, v_j) < \xi \\ 0 & \text{otherwise} \end{cases}
$$
其中 $\xi$ 为距离阈值，$\sigma^2$ 为距离方差。

### 2. 扩散核 (Diffusion Kernel)
$$
\tilde{A}^D = D_S^{-1/2} A_S D_S^{-1/2}
$$
其中 $D_S$ 为 $A_S$ 的度矩阵。该核模拟污染物的各向同性扩散过程。

### 3. 平流核 (Advection Kernel)
$$
A^A_{ij} = d_{ij} \cdot \max(0, |\vec{v}| \cos(\alpha)) \cdot \mathbb{1}_{d_{ij} < \xi}
$$
其中 $\alpha$ 为风向与节点连接向量的夹角，$|\vec{v}|$ 为风速。该核模拟风驱动下的各向异性传输。

### 4. 传播层更新
$$
H^{(l)} = \sigma\left(\tilde{A}^D H^{(l-1)} + A^A H^{(l-1)} W^{(l)}\right)
$$
其中 $W^{(l)}$ 为可学习权重矩阵，$\sigma$ 为激活函数。

### 5. 推理输出
$$
\hat{X}_i^t = \text{MLP}(H_i^{(L)}, t)
$$

### 6. 复合损失函数
$$
\mathcal{L} = \mathcal{L}_{\text{infer}} + \lambda_1 \mathcal{L}_{\text{init}} + \lambda_2 \mathcal{L}_{\text{AOD}}
$$

其中：
- $\mathcal{L}_{\text{infer}} = \sum_{t=1}^T \sum_{i \in V_{\text{target}}} |\hat{X}_i^t - X_i^t|$ （推理损失，L1误差）
- $\mathcal{L}_{\text{init}} = \sum_{t=1}^T \sum_{i \in V_{\text{target}}} |\text{MLP}(H_i^{(0)}, t) - X_i^t|$ （初始化损失）
- $\mathcal{L}_{\text{AOD}}$ 为掩码AOD空间梯度损失

## 参数清单

| 参数名 | 类型 | 默认值 | 说明 |
|-------|------|-------|------|
| distance_threshold ($\xi$) | float | 200km | 地理空间图构建的距离阈值 |
| distance_variance ($\sigma^2$) | float | - | 高斯核距离方差 |
| masking_ratio | float | 0.5 | 归纳节点掩码比例（训练时随机掩码50%站点） |
| $\lambda_1$ | float | - | 初始化损失权重 |
| $\lambda_2$ | float | - | AOD梯度损失权重 |
| TCN层数 | int | - | 时间卷积网络层数 |
| GNN传播层数 | int | - | 图神经网络传播层数 |
| hidden_dim | int | - | 隐藏层维度 |
| learning_rate | float | - | 学习率 |
| batch_size | int | - | 批大小 |

## 数据规格

### 输入
| 数据 | 格式 | 维度 | 单位 |
|-----|------|-----|------|
| 监测站点PM2.5 | array | (n_stations,) | μg/m³ |
| 气象特征（温度、风速、降水等） | array | (n_stations, n_meteo) | - |
| 排放清单（NOx、SO2等） | array | (n_stations, n_emiss) | - |
| 站点坐标 | array | (n_stations, 2) | 度 |
| 卫星AOD（梯度约束用） | array | (n_grids, n_times) | - |

### 输出
| 数据 | 格式 | 单位 |
|-----|------|------|
| PM2.5预测值 | array | μg/m³ |

## 实现步骤

### 数据准备
1. 收集地面监测站PM2.5数据（目标变量）
2. 收集气象数据（温度、风速、降水、边界层高度等）作为输入特征
3. 收集排放清单数据（NOx、SO2、NH3、VOC、PM2.5、PM10）
4. 获取卫星AOD产品作为空间梯度约束（不作为直接输入）

### 模型构建
1. **时间特征编码（TCN）**：
   - 对每个节点的时间序列（气象+排放）应用时间卷积网络
   - 输出初始隐表示 $H^{(0)}_i = \text{TCN}([P_i, Q_i])$

2. **物理引导空时传播**：
   - 构建地理空间图 $A^S$：基于节点间距离的高斯核
   - 构建扩散图 $\tilde{A}^D$：$A^S$ 的对称归一化拉普拉斯
   - 构建平流图 $A^A$：基于风场投影的有向加权图
   - 并行传播：$H^{(l)} = \sigma(\tilde{A}^D H^{(l-1)} + A^A H^{(l-1)} W^{(l)})$

3. **推理输出**：
   - 对最终隐表示 $H^{(L)}$ 应用共享MLP生成PM2.5预测

### 训练策略
1. **归纳节点掩码**：随机掩码50%观测站点为"未知"目标节点
2. **复合损失优化**：
   - 主损失：掩码节点的L1推理误差
   - 辅助损失1：TCN初始化损失（确保无邻居信息时也能生成合理基线）
   - 辅助损失2：掩码AOD空间梯度损失（使预测的空间差异与AOD梯度一致）
3. **AOD作为梯度约束而非直接输入**：避免云覆盖等导致的缺失问题

### 推理阶段
1. 对未观测站点（站点推理）或网格单元（网格推理）进行预测
2. 利用观测站点的信息通过物理引导图网络传播
3. 结合局部气象和排放特征生成最终预测

## 方法特点

1. **物理引导**：显式建模大气扩散（各向同性）和平流（各向异性）过程
2. **归纳学习**：可泛化到训练期间未观测的新位置
3. **AOD鲁棒性**：将AOD作为空间梯度约束而非直接输入，避免云覆盖缺失问题
4. **多图融合**：地理空间图、扩散图、平流图分别编码不同物理机制

## 性能指标

| 指标 | 数值 |
|-----|------|
| 年均MAE | 9.52 µg/m³ |
| 冬季MAE | 15.09 µg/m³ |
| 夏季MAE | 7.65 µg/m³ |
| 相对基线提升 | 25.2% |

## 方法指纹
MD5: spin_physics_guided_spatiotemporal_kriging

## 随机性
- [x] 是（训练过程随机掩码节点，存在随机初始化）

## 应用场景

- 稀疏监测网络下的PM2.5插值
- 无观测区域的高分辨率PM2.5制图
- 全天候（不受云覆盖影响）的空气质量推理
