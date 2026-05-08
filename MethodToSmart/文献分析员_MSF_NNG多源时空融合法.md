# 【可执行方法规范】

## 方法名称
多源时空最近邻网格融合法 (Multi-source Spatiotemporal Nearest Neighbor Grids, MSF-NNG)

## 文献来源
- 论文标题：An Improved Multi-source Spatiotemporal Data Fusion Model Based on the Nearest Neighbor Grids for PM2.5 Concentration Interpolation and Prediction
- 作者：Springer Chapter Authors
- 年份：2023年
- 来源：Springer

## 核心公式

### 1. Cressman插值
$$
\hat{Z}(s_0) = \sum_{i=1}^{n} w_i Z(s_i), \quad w_i = \frac{R^2 - d_i^2}{R^2 + d_i^2}
$$
其中 $R$ 为影响半径，$d_i$ 为距离。

### 2. 最近邻网格匹配
$$
N(s_0) = \arg\min_{s_j \in G} \|s_0 - s_j\|_2
$$
基于大气条件相似性找到最近邻网格。

### 3. 多源融合
$$
\hat{Z}_{fused}(s,t) = \alpha Z_{obs}(s,t) + (1-\alpha) Z_{model}(s,t)
$$

## 参数清单

| 参数名 | 类型 | 默认值 | 说明 |
|-------|------|-------|------|
| grid_size | float | 1km | 网格大小 |
| radius_R | float | 50km | Cressman影响半径 |
| n_neighbors | int | 5 | 最近邻数量 |
| alpha | float | 0.7 | 观测数据权重 |

## 数据规格

### 输入
| 数据 | 格式 | 维度 | 单位 |
|-----|------|-----|------|
| 大型站PM2.5 | array | (n_large, n_times) | μg/m³ |
| 小型站PM2.5 | array | (n_small, n_times) | μg/m³ |
| 湿度 | array | (n_stations, n_times) | % |
| 温度 | array | (n_stations, n_times) | °C |
| 风速 | array | (n_stations, n_times) | m/s |

### 输出
| 数据 | 格式 | 单位 |
|-----|------|------|
| 融合PM2.5 | array | μg/m³ |

## 实现步骤

1. **网格划分**：将城市划分为1km网格
2. **缺失值填充**：使用Cressman插值填充小型站缺失值
3. **最近邻匹配**：基于大气条件相似性找到最近邻网格
4. **多源融合**：加权融合观测和模型数据
5. **预测**：利用时空相关性进行预测

## 随机性
- [ ] 是  - [x] 否（确定性方法）

## 方法指纹
MD5: mspatiotemporal_nearest_neighbor_grids_msf_nng
