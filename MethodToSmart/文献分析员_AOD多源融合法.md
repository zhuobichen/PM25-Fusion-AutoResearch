# 【可执行方法规范】

## 方法名称
卫星多传感器AOD数据与决策级融合法 (Data/Decision Level AOD Fusion)

## 文献来源
- 论文标题：Data level and decision level fusion of satellite multi-sensor AOD retrievals for improving PM2.5 estimations, a study on Tehran
- 作者：Ali Mirzaei, Hossein Bagheri, Mehran Sattari
- 年份：2023年
- arXiv: 2302.10278

## 核心公式

### 1. 数据级融合（平均法）
$$
AOD_{fused} = \frac{1}{N} \sum_{i=1}^{N} AOD_i
$$

### 2. 数据级融合（加权平均法）
$$
AOD_{fused} = \sum_{i=1}^{N} w_i \cdot AOD_i, \quad \sum w_i = 1
$$

### 3. 决策级融合（Stacking）
$$
\hat{PM2.5} = f_{meta}(\hat{y}_1, \hat{y}_2, ..., \hat{y}_N)
$$
其中 $\hat{y}_i$ 为各传感器AOD反演的PM2.5估计。

### 4. 决策级融合（加权平均）
$$
\hat{PM2.5}_{fused} = \sum_{i=1}^{N} w_i \cdot \hat{y}_i
$$

## 参数清单

| 参数名 | 类型 | 默认值 | 说明 |
|-------|------|-------|------|
| n_sensors | int | 4 | 传感器数量（MODIS-DT, MODIS-DB, VIIRS-DT, VIIRS-DB） |
| fusion_level | str | 'decision' | 融合级别（data/decision） |
| meta_model | str | 'RF' | 元模型类型 |

## 数据规格

### 输入
| 数据 | 格式 | 维度 | 单位 |
|-----|------|-----|------|
| MODIS Dark Target AOD | array | (n_grids, n_days) | - |
| MODIS Deep Blue AOD | array | (n_grids, n_days) | - |
| VIIRS Dark Target AOD | array | (n_grids, n_days) | - |
| VIIRS Deep Blue AOD | array | (n_grids, n_days) | - |
| 地面PM2.5监测 | array | (n_stations, n_days) | μg/m³ |

### 输出
| 数据 | 格式 | 单位 |
|-----|------|------|
| 融合PM2.5估计 | array | μg/m³ |

## 实现步骤

1. **数据收集**：获取MODIS和VIIRS的AOD产品
2. **数据级融合**：平均或加权平均多源AOD
3. **PM2.5反演**：使用机器学习建立AOD-PM2.5关系
4. **决策级融合**：融合各传感器反演的PM2.5
5. **评估**：比较数据级和决策级融合效果

## 随机性
- [ ] 是  - [x] 否（确定性融合方法）

## 方法指纹
MD5: aod_multi_sensor_data_decision_fusion