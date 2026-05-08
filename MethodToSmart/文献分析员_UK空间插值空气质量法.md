# 【可执行方法规范】

## 方法名称
英国空气质量空间插值法 (UK Spatial Interpolation for Air Quality)

## 文献来源
- 论文标题：Spatial Interpolation of Air Quality: A UK Case Study
- 作者：Springer Conference Authors
- 年份：2024年
- 来源：Springer Conference

## 核心公式

### 1. 普通克里金
$$
\hat{Z}(s_0) = \sum_{i=1}^{n} \lambda_i Z(s_i)
$$
权重满足无偏约束 $\sum \lambda_i = 1$。

### 2. 变异函数
$$
\gamma(h) = \frac{1}{2N(h)} \sum_{i=1}^{N(h)} [Z(s_i+h) - Z(s_i)]^2
$$

### 3. 协方差函数
$$
C(h) = C(0) - \gamma(h)
$$

### 4. 反距离加权
$$
\hat{Z}(s_0) = \frac{\sum_{i=1}^{n} Z(s_i)/d_i^p}{\sum_{i=1}^{n} 1/d_i^p}
$$

## 参数清单

| 参数名 | 类型 | 默认值 | 说明 |
|-------|------|-------|------|
| method | str | 'kriging' | 插值方法 |
| variogram_model | str | 'spherical' | 变异函数模型 |
| n_neighbors | int | 12 | 近邻数量 |
| power | int | 2 | IDW幂指数 |
| pollutant | str | 'PM2.5' | 污染物类型 |

## 数据规格

### 输入
| 数据 | 格式 | 维度 | 单位 |
|-----|------|-----|------|
| IoT传感器数据 | array | (n_sensors, n_times) | μg/m³ |
| 传感器坐标 | array | (n_sensors, 2) | 度 |
| 气象协变量 | array | (n_sensors, n_vars) | - |

### 输出
| 数据 | 格式 | 单位 |
|-----|------|------|
| 插值空气质量图 | array | μg/m³ |

## 实现步骤

1. **数据收集**：英国IoT传感器网络数据
2. **探索性分析**：检查数据分布和空间自相关
3. **变异函数建模**：拟合理论变异函数
4. **克里金插值**：生成连续表面
5. **交叉验证**：评估插值精度

## 随机性
- [ ] 是  - [x] 否（确定性插值方法）

## 方法指纹
MD5: uk_spatial_interpolation_air_quality
