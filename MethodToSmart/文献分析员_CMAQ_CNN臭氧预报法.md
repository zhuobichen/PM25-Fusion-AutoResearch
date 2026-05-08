# 【可执行方法规范】

## 方法名称
CMAQ-CNN混合臭氧预报法 (CMAQ-CNN Hybrid Ozone Forecasting)

## 文献来源
- 论文标题：A novel CMAQ-CNN hybrid model to forecast hourly surface-ozone concentrations 14 days in advance
- 作者：Scientific Reports Authors
- 年份：2021年
- 来源：Nature Scientific Reports

## 核心公式

### 1. CNN特征提取
$$
h_{CNN} = \text{Conv2D}(X_{CMAQ}; W) = \sigma(W * X_{CMAQ} + b)
$$

### 2. 混合输入
$$
X_{hybrid} = [X_{CMAQ}, X_{WRF}, X_{history}]
$$

### 3. 预测模型
$$
\hat{O}_3(t+k) = f_{CNN}(X_{hybrid}(t))
$$

### 4. 多步预测
$$
[\hat{O}_3(t+1), ..., \hat{O}_3(t+24)] = f_{seq2seq}(X_{hybrid}(t))
$$

## 参数清单

| 参数名 | 类型 | 默认值 | 说明 |
|-------|------|-------|------|
| cnn_layers | int | 3 | CNN层数 |
| filters | list | [64, 128, 256] | 卷积核数量 |
| kernel_size | int | 3 | 卷积核大小 |
| forecast_hours | int | 336 | 预报时长（14天） |
| n_stations | int | 255 | 监测站数量 |

## 数据规格

### 输入
| 数据 | 格式 | 维度 | 单位 |
|-----|------|-----|------|
| CMAQ预报 | array | (n_grids, n_hours, n_species) | ppb |
| WRF气象 | array | (n_grids, n_hours, n_vars) | - |
| 历史O3 | array | (n_stations, 24) | ppb |

### 输出
| 数据 | 格式 | 单位 |
|-----|------|------|
| O3小时预报 | array | ppb |

## 实现步骤

1. **数据收集**：CMAQ、WRF、监测站数据
2. **特征工程**：构建时空特征矩阵
3. **CNN构建**：多层卷积+池化+全连接
4. **训练**：使用历史数据训练
5. **14天预报**：逐小时预报臭氧浓度

## 随机性
- [x] 是（深度学习训练带随机初始化）

## 方法指纹
MD5: cmaq_cnn_hybrid_ozone_forecasting
