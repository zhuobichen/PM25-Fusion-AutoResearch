# 【可执行方法规范】

## 方法名称
Deep-AIR混合CNN-LSTM空气质量管理框架 (Deep-AIR: Hybrid CNN-LSTM Framework for Air Quality Modeling)

## 文献来源
- 论文标题：Deep-AIR: A Hybrid CNN-LSTM Framework for Air Quality Modeling in Metropolitan Cities
- 作者：Yang Han, Qi Zhang, Victor O.K. Li, Jacqueline C.K. Lam
- 年份：2021年

## 核心公式

### 1. 残差网络单元
$$
X^{(l+1)} = X^{(l)} + F(X^{(l)})
$$
ResNet通过恒等映射解决深层网络梯度消失问题。

### 2. LSTM单元
$$
i_t = \sigma(W_i x_t + U_i h_{t-1} + b_i) \quad \text{（输入门）}
$$
$$
f_t = \sigma(W_f x_t + U_f h_{t-1} + b_f) \quad \text{（遗忘门）}
$$
$$
o_t = \sigma(W_o x_t + U_o h_{t-1} + b_o) \quad \text{（输出门）}
$$
$$
c_t = f_t \odot c_{t-1} + i_t \odot \tanh(W_c x_t + b_c) \quad \text{（细胞状态）}
$$
$$
h_t = o_t \odot \tanh(c_t) \quad \text{（隐藏状态）}
$$

### 3. 1x1卷积（通道交互）
$$
y_{k,i,j} = \sum_{c} w_{k,c} \odot x_{c,i,j} + b_k
$$
1x1卷积实现不同通道特征的线性组合，增强跨特征空间交互。

### 4. 网格划分
- 香港：1km×1km网格 → 44×60地图
- 北京：3km×3km网格 → 50×55地图

### 5. 插值公式（最近邻加权平方）
$$
q_{g,t} = \frac{\sum_{s \in N_g} q_{s,t} / d_{gs}^2}{\sum_{s \in N_g} 1 / d_{gs}^2}
$$
两步插值：先时间插值，再空间插值。

## 参数清单

| 参数名 | 类型 | 默认值 | 说明 |
|-------|------|-------|------|
| patch_size | int | 15 | 训练patch尺寸 |
| input_length (W) | int | 48 | LSTM输入历史小时数 |
| n_residual_units | int | 4 | 残差单元数量 |
| conv_kernel | int | 3 | 卷积核大小 |
| hidden_units | int | 128/256/512 | LSTM隐藏单元数 |
| n_lstm_layers | int | 1-2 | LSTM层数 |
| learning_rate | float | 10⁻⁴ | 学习率 |
| optimizer | str | SGD | 优化器 |

## 实现步骤

### 数据预处理
1. 将城市划分为网格（香港1km，北京3km）
2. 收集多源数据：
   - 空气质量：PM2.5, PM10, NO2, SO2, O3, CO
   - 气象：温度、湿度、气压、风速、风向、降水
   - 交通：拥堵水平、车速
   - 城市形态：路网密度、建筑密度/高度、街道峡谷指示
3. 两阶段插值填补缺失值：
   - 阶段1：时间维度线性插值
   - 阶段2：空间维度距离平方反比加权插值

### AirResCNN组件（空间特征提取）
1. 构建4个残差单元的深度CNN
2. 每个残差单元：2层3×3卷积 + BatchNorm + ReLU
3. 在相邻残差单元间插入1x1卷积层
4. 从n通道图像提取空间特征向量序列

### LSTM组件（时序建模）
1. 输入：CNN提取的特征序列（48小时）
2. 单向LSTM处理时序依赖
3. 取最后隐藏状态用于预测

### 双任务输出
1. **细粒度城市级估计**：无局部观测时预测全城各网格PM2.5
2. **站点级预报**：预测监测站点未来1-24小时PM2.5

### Patch训练算法
1. 对每个样本，以监测站点为中心提取N×N邻域patch
2. 估计任务：使用插值后的空气污染数据
3. 预报任务：使用观测的真实空气污染数据
4. 最小化MSE损失

## 特征重要性（显著性分析）

| 污染物 | 香港最重要特征 | 北京最重要特征 |
|-------|--------------|---------------|
| PM2.5 | 气象（湿度） | 历史NO2 |
| NO2 | 街道峡谷+路网密度 | 风速+风向 |
| SO2 | 气象+建筑密度 | 气象 |
| O3 | 历史O3+气象 | 历史O3 |

## 性能指标

### 香港
| 任务 | 准确率 |
|-----|-------|
| 细粒度城市估计 | 67.6% |
| 1小时预报 | 77.2% |
| 24小时预报 | 66.1% |

### 北京
| 任务 | 准确率 |
|-----|-------|
| 细粒度城市估计 | 65.0% |
| 1小时预报 | 75.3% |
| 24小时预报 | 63.5% |



## 数据规格

### 输入
| 数据 | 格式 | 维度 | 单位 |
|-----|------|-----|------|
| 空气质量 | array | (n_grids, n_grids, n_vars) | μg/m³ |
| 气象数据 | array | (n_grids, n_grids, n_vars) | - |
| 交通数据 | array | (n_grids, n_grids, n_vars) | - |
| 城市形态 | array | (n_grids, n_grids, n_vars) | - |

### 输出
| 数据 | 格式 | 单位 |
|-----|------|------|
| PM2.5预测 | array | μg/m³ |

## 应用场景

- 城市街道峡谷PM2.5预测
- 结合气象、气象、交通、城市形态的多源数据融合
- 高密度城市空气质量精细化制图
- 交通排放对空气质量影响评估

## 随机性
- [x] 是（深度学习训练带随机初始化）

## 方法指纹
MD5: deep_air_hybrid_cnn_lstm
