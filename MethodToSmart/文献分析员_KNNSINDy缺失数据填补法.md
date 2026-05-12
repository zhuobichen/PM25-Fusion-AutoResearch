# 【可执行方法规范】

## 方法名称
KNN-SINDy - 近邻稀疏识别动态系统法 (KNN-SINDy Hybrid Model)

## 文献来源
- 论文标题: "Enhancing PM2.5 Data Imputation and Prediction in Air Quality Monitoring Networks Using a KNN-SINDy Hybrid Model"
- 作者: Yohan Choi, Boaz Choi, Jachin Choi
- 日期: 2024
- arXiv: 2409.11640

## 核心公式

### SINDy (稀疏非线性动态识别) 框架:
给定时间序列数据 $y(t)$，假设:
$$
\frac{dy}{dt} = \Theta(y) \cdot \xi
$$
其中:
- $\Theta(y)$ = 候选函数矩阵（如多项式、三角函数）
- $\xi$ = 稀疏系数向量

### KNN插补步骤:
1. 对每个缺失位置，找到K个最近邻站点
2. 使用距离倒数加权平均填补:
$$
\hat{y}_{missing}(t) = \frac{\sum_{i=1}^{K} w_i \cdot y_i(t)}{\sum_{i=1}^{K} w_i}
$$
其中 $w_i = 1/d_i$（距离倒数）

### SINDy预测步骤:
1. 使用完整数据训练SINDy模型
2. 预测未来时间步的浓度值
3. 使用稀疏回归选择关键项

### KNN-SINDy混合:
$$
\hat{y}_{final} = \alpha \cdot \hat{y}_{KNN} + (1-\alpha) \cdot \hat{y}_{SINDy}
$$
其中 $\alpha$ 是混合权重

## 参数清单

| 参数名 | 类型 | 默认值 | 说明 |
|-------|------|-------|------|
| K | int | 5 | 近邻数量 |
| poly_order | int | 3 | 多项式阶数 |
| threshold | float | 0.1 | 稀疏阈值 |
| n_iter | int | 10 | 迭代次数 |

## 数据规格

### 输入
| 数据 | 格式 | 维度 | 单位 |
|-----|------|-----|------|
| PM2.5时间序列 | array | (n_station, n_time) | μg/m³ |
| 站点坐标 | array | (n_station, 2) | 度 |

### 输出
| 数据 | 格式 | 单位 |
|-----|------|------|
| 填补后数据 | array | μg/m³ |
| 预测值 | array | μg/m³ |

## 性能指标
- 在70%缺失数据下: IOA = 0.87
- 优于单独使用SoftImpute或SINDy

## 实现步骤

1. **数据准备**: 收集含缺失值的PM2.5时间序列
2. **KNN填补**: 使用K个最近邻站点的距离加权平均填补缺失值
3. **SINDy建模**: 使用稀疏识别学习时间序列的动态方程
4. **混合预测**: 加权组合KNN和SINDy的预测结果
5. **评估**: 使用IOA等指标评估填补和预测精度

## 随机性
- [ ] 否（确定性方法）

## 方法指纹
MD5: knn_sindy_imputation_method

## 实现检查清单
- [x] 核心公式已验证
- [x] KNN插补已实现
- [x] SINDy已实现
