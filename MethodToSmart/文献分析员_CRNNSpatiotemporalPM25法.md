# 【可执行方法规范】

## 方法名称
CRNN-PM25 - 卷积递归神经网络法 (Convolutional Recursive Neural Network)

## 文献来源
- 论文标题: "A Novel Prediction Approach for Exploring PM2.5 Spatiotemporal Propagation Based on Convolutional Recursive Neural Networks"
- 作者: Hsing-Chung Chen, Karisma Trinanda Putra
- 机构: Asia University, China Medical University, Muhammadiyah Yogyakarta
- 日期: 2021
- arXiv: 2101.06213

## 核心公式

### 模型架构: CRNN

**卷积层（空间特征提取）:**
$$
F_{conv} = ReLU(Conv_{2D}(X, W) + b)
$$

**递归层（时间依赖建模）:**
$$
h_t = GRU(x_t, h_{t-1})
$$
或
$$
h_t = LSTM(x_t, h_{t-1})
$$

### 损失函数:
$$
L = \frac{1}{N}\sum_{i=1}^{N}(y_i - \hat{y}_i)^2
$$

## 参数清单

| 参数名 | 类型 | 默认值 | 说明 |
|-------|------|-------|------|
| n_filters | int | 64 | 卷积核数量 |
| kernel_size | int | 3 | 卷积核大小 |
| hidden_units | int | 128 | LSTM/GRU隐藏单元 |
| n_layers | int | 2 | 递归层层数 |
| seq_len | int | 168 | 输入序列长度 |

## 数据规格

### 输入
| 数据 | 格式 | 维度 |
|-----|------|-----|
| 多站点时间序列 | array | (n_stations, seq_len, n_features) |

### 输出
| 数据 | 格式 | 单位 |
|-----|------|------|
| 预测值 | array | μg/m³ |

## 实现步骤

1. **数据准备**: 收集多站点PM2.5时间序列数据
2. **特征构建**: 构建时空特征矩阵
3. **CNN构建**: 2D卷积层提取空间特征
4. **RNN构建**: LSTM/GRU层捕捉时间依赖
5. **训练**: 使用MSE损失训练端到端模型
6. **预测**: 生成PM2.5浓度预测

## 随机性
- [ ] 是（深度学习训练带随机初始化）

## 方法指纹
MD5: crnn_spatiotemporal_method

## 实现检查清单
- [ ] 核心公式已验证
- [ ] CNN层已实现
- [ ] LSTM/GRU层已实现
