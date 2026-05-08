# 【可执行方法规范】

## 方法名称
SmaAt-Krige-GNet降水临近预报法 (SmaAt-Krige-GNet Precipitation Nowcasting)

## 文献来源
- 论文标题：Integrating Weather Station Data and Radar for Precipitation Nowcasting: SmaAt-fUsion and SmaAt-Krige-GNet
- 作者：Jie Shi, Aleksej Cornelissen, Siamak Mehrkanoon
- 年份：2025年
- arXiv: 2502.16116

## 核心公式

### 1. Kriging插值
$$
\hat{Z}(s_0) = \sum_{i=1}^{n} \lambda_i Z(s_i)
$$
将站点数据插值到网格。

### 2. 双编码器架构
$$
h_{radar} = f_{enc1}(X_{radar})
$$
$$
h_{kriging} = f_{enc2}(X_{kriging})
$$

### 3. 多级融合
$$
h_{fused} = \text{Attention}(h_{radar}, h_{kriging})
$$

### 4. 预测
$$
\hat{Y}_{t+1:t+K} = f_{dec}(h_{fused})
$$

## 参数清单

| 参数名 | 类型 | 默认值 | 说明 |
|-------|------|-------|------|
| encoder_type | str | 'UNet' | 编码器类型 |
| n_stations | int | - | 气象站数量 |
| kriging_range | float | 50km | Kriging搜索半径 |
| forecast_steps | int | 6 | 预报步数 |
| fusion_method | str | 'attention' | 融合方法 |

## 数据规格

### 输入
| 数据 | 格式 | 维度 | 单位 |
|-----|------|-----|------|
| 雷达降水图 | array | (H, W, T) | mm/h |
| 气象站数据 | array | (n_stations, T, n_vars) | - |
| 站点坐标 | array | (n_stations, 2) | 度 |

### 输出
| 数据 | 格式 | 单位 |
|-----|------|------|
| 降水临近预报 | array | mm/h |

## 实现步骤

1. **数据准备**：雷达图+气象站数据
2. **Kriging插值**：站点数据→网格
3. **双编码器**：分别处理雷达和Kriging数据
4. **多级融合**：注意力机制融合
5. **解码预测**：生成降水临近预报

## 随机性
- [x] 是（深度学习训练带随机初始化）

## 方法指纹
MD5: smaat_krige_gnet_precipitation_nowcasting
