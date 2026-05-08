# 【可执行方法规范】

## 方法名称
空间反向传播神经网络 (Spatial Back Propagation Neural Network, S-BPNN)

## 文献来源
- 论文标题：Estimation of PM2.5 Concentrations in China Using a Spatial Back Propagation Neural Network
- 作者：Scientific Reports Authors
- 年份：2019年
- 来源：Nature Scientific Reports

## 核心公式

### 1. BP神经网络前向传播
$$
h^{(l)} = \sigma(W^{(l)} h^{(l-1)} + b^{(l)})
$$

### 2. 反向传播梯度
$$
\frac{\partial L}{\partial W^{(l)}} = \delta^{(l)} (h^{(l-1)})^T
$$

### 3. PCA降维
$$
Z = X \cdot V_k
$$
其中 $V_k$ 为主成分载荷矩阵。

### 4. 空间特征增强
$$
x_{spatial} = [x_{raw}, \text{NDVI}, \text{elevation}, \text{landuse}, \text{population}]
$$

## 参数清单

| 参数名 | 类型 | 默认值 | 说明 |
|-------|------|-------|------|
| hidden_layers | list | [128, 64, 32] | 隐藏层结构 |
| n_components | int | 10 | PCA主成分数 |
| learning_rate | float | 0.001 | 学习率 |
| epochs | int | 200 | 训练轮数 |
| batch_size | int | 32 | 批大小 |

## 数据规格

### 输入
| 数据 | 格式 | 维度 | 单位 |
|-----|------|-----|------|
| 卫星AOD | array | (n_grids, n_days) | - |
| 气象数据 | array | (n_grids, n_days, n_vars) | - |
| NDVI | array | (n_grids,) | - |
| 高程 | array | (n_grids,) | m |
| 土地利用 | array | (n_grids,) | - |
| 人口密度 | array | (n_grids,) | 人/km² |

### 输出
| 数据 | 格式 | 单位 |
|-----|------|------|
| PM2.5估计 | array | μg/m³ |

## 实现步骤

1. **数据收集**：卫星AOD、气象、地理辅助数据
2. **PCA降维**：对高维特征进行主成分分析
3. **网络构建**：构建多层BP神经网络
4. **训练**：使用反向传播算法训练
5. **预测**：生成全国PM2.5空间分布图

## 随机性
- [x] 是（神经网络训练带随机初始化）

## 方法指纹
MD5: spatial_back_propagation_neural_network_sbpnn
