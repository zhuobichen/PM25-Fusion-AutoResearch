# 【可执行方法规范】

## 方法名称
双向LSTM时空插值法 (Bidirectional LSTM Spatiotemporal Interpolation)

## 文献来源
- 论文标题：Deep learning PM2.5 concentrations with bidirectional LSTM RNN
- 作者：Air Quality, Atmosphere & Health Authors
- 年份：2019年
- 来源：Springer

## 核心公式

### 1. LSTM单元
$$
i_t = \sigma(W_i x_t + U_i h_{t-1} + b_i)
$$
$$
f_t = \sigma(W_f x_t + U_f h_{t-1} + b_f)
$$
$$
o_t = \sigma(W_o x_t + U_o h_{t-1} + b_o)
$$
$$
c_t = f_t \odot c_{t-1} + i_t \odot \tanh(W_c x_t + b_c)
$$
$$
h_t = o_t \odot \tanh(c_t)
$$

### 2. 双向LSTM
$$
\vec{h}_t = \text{LSTM}_{forward}(x_t, \vec{h}_{t-1})
$$
$$
\overleftarrow{h}_t = \text{LSTM}_{backward}(x_t, \overleftarrow{h}_{t+1})
$$
$$
h_t = [\vec{h}_t; \overleftarrow{h}_t]
$$

### 3. 时空插值
$$
\hat{y}_{s,t} = f_{BiLSTM}(x_{s,t-lag:t+lag})
$$

## 参数清单

| 参数名 | 类型 | 默认值 | 说明 |
|-------|------|-------|------|
| lstm_units | int | 128 | LSTM隐藏单元数 |
| sequence_length | int | 30 | 输入序列长度（天） |
| n_layers | int | 2 | BiLSTM层数 |
| dropout | float | 0.2 | Dropout比例 |
| learning_rate | float | 0.001 | 学习率 |

## 数据规格

### 输入
| 数据 | 格式 | 维度 | 单位 |
|-----|------|-----|------|
| PM2.5时序 | array | (n_stations, n_days) | μg/m³ |
| 气象特征 | array | (n_stations, n_days, n_features) | - |

### 输出
| 数据 | 格式 | 单位 |
|-----|------|------|
| PM2.5插值 | array | μg/m³ |

## 实现步骤

1. **数据收集**：美国东南部PM2.5日均数据
2. **序列构建**：构建双向时间窗口序列
3. **BiLSTM构建**：双向LSTM层+全连接层
4. **训练**：使用MSE损失训练
5. **时空插值**：对缺失站点和时间进行插值

## 随机性
- [x] 是（深度学习训练带随机初始化）

## 方法指纹
MD5: bidirectional_lstm_spatiotemporal_interpolation
