# 【可执行方法规范】

## 方法名称
CNN-LSTM混合PM2.5预测法 (CNN-LSTM PM2.5 Prediction)

## 文献来源
- 论文标题：Air Quality PM2.5 Index Prediction Model Based on CNN-LSTM
- 作者：Zicheng Guo, Shuqi Zhu, Meixing Zhu, He Guandi
- 年份：2025年
- arXiv: 2508.11215

## 核心公式

### 1. CNN特征提取
$$
h_{CNN} = \text{Conv1D}(x; W_c) = \sigma(W_c * x + b_c)
$$

### 2. LSTM时序建模
$$
i_t = \sigma(W_i [h_{t-1}, x_t] + b_i)
$$
$$
f_t = \sigma(W_f [h_{t-1}, x_t] + b_f)
$$
$$
o_t = \sigma(W_o [h_{t-1}, x_t] + b_o)
$$
$$
c_t = f_t \odot c_{t-1} + i_t \odot \tanh(W_c [h_{t-1}, x_t] + b_c)
$$
$$
h_t = o_t \odot \tanh(c_t)
$$

### 3. 混合预测
$$
\hat{y} = W_{out} \cdot \text{LSTM}(\text{CNN}(x)) + b
$$

## 参数清单

| 参数名 | 类型 | 默认值 | 说明 |
|-------|------|-------|------|
| cnn_filters | int | 64 | CNN卷积核数量 |
| cnn_kernel_size | int | 3 | CNN卷积核大小 |
| lstm_units | int | 128 | LSTM隐藏单元数 |
| sequence_length | int | 24 | 输入序列长度（小时） |
| learning_rate | float | 0.001 | 学习率 |
| epochs | int | 100 | 训练轮数 |

## 数据规格

### 输入
| 数据 | 格式 | 维度 | 单位 |
|-----|------|-----|------|
| PM2.5历史序列 | array | (n_samples, seq_len, 1) | μg/m³ |
| 气象特征 | array | (n_samples, seq_len, n_features) | - |

### 输出
| 数据 | 格式 | 单位 |
|-----|------|------|
| PM2.5预测 | array | μg/m³ |

## 实现步骤

1. **数据收集**：北京工业区2010-2015年多变量数据集
2. **数据预处理**：6小时平均、归一化
3. **CNN构建**：1D卷积层提取局部空间特征
4. **LSTM构建**：捕捉时序依赖关系
5. **模型训练**：Adam优化器，MSE损失
6. **预测评估**：6小时间隔PM2.5浓度预测

## 随机性
- [x] 是（深度学习训练带随机初始化）

## 方法指纹
MD5: cnn_lstm_hybrid_pm25_prediction
