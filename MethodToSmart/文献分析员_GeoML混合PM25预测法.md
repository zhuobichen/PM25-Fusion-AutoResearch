# 【可执行方法规范】

## 方法名称
地统计与机器学习混合PM2.5预测法 (Geostatistical-Machine Learning Hybrid PM2.5 Prediction)

## 文献来源
- 论文标题：A comparison between geostatistical and machine learning models for spatio-temporal prediction of PM2.5 data
- 作者：Zeinab Mohamed, Wenlong Gong
- 年份：2025年

## 核心公式

### 1. 通用克里金 (Universal Kriging)
$$
Y(s,t) = \mu(s,t) + w(s,t) + \epsilon(s,t)
$$
其中 $\mu(s,t) = X(s,t)\beta$ 为均值，$w(s,t)$ 为时空高斯过程，$\epsilon(s,t) \sim N(0, \sigma_\epsilon^2)$ 为测量误差。

### 2. 可分指数协方差函数
$$
\Gamma(h,\tau;\tau,\theta) = \sigma_s^2 \exp(-||h||/\rho_s) \cdot \sigma_t^2 \exp(-|\tau|/\rho_t)
$$

### 3. 克里金预测
$$
\hat{Y}(s_0,t_0) = X(s_0,t_0)^T\beta_{gls} + c(\theta)^T \Sigma_y^{-1}(\theta)(Y - X\beta_{gls})
$$

### 4. 最近邻高斯过程 (NNGP)
$$
\hat{Y}(s_0,t_0) = X(s_0,t_0)^T\beta + C_{s_0,N_0} C_{N_0}^{-1}(Y_{N_0} - X\beta)
$$

### 5. 固定秩克里金 (FRK)
$$
\tilde{w}(s,t) = \sum_{r=1}^{R} \sum_{i=1}^{K_r} \phi_{ri}(s,t) w_{ri}^*
$$

### 6. 支持向量回归
$$
F(x|w) = \sum_{i=1}^{N} (\alpha_i - \alpha_i^*) K(v_i, x)
$$
其中 $K(v_i, x) = \exp(-\gamma \sum (x_{ij} - v_{ij})^2)$ 为径向基函数。

### 7. 混合模型（最优组合）
$$
\hat{Y}_{hybrid} = w_1 \cdot \hat{Y}_{NNO+Krig} + w_2 \cdot \hat{Y}_{SVR}
$$
其中 $w_1 + w_2 = 1$，$w_1, w_2 \geq 0$

### 8. PurpleAir校正公式
$$
PM_{2.5,corrected} = 0.524 \cdot PA - 0.0852 \cdot RH + 5.72
$$

## 参数清单

| 参数名 | 类型 | 默认值 | 说明 |
|-------|------|-------|------|
| m | int | 10 | NNGP近邻数量 |
| K | int | 1600 | FRK基函数数量（80空间×20时间） |
| n_trees | int | 500 | 随机森林树数量 |
| C | float | 1.0 | SVR正则化参数 |
| $\gamma$ | float | 0.01 | RBF核带宽参数 |
| $\xi$ | float | 0.1 | SVR epsilon敏感参数 |

## 实现步骤

### 数据准备
1. 收集PurpleAir传感器PM2.5数据（2019年加州1015个站点）
2. 应用Barkjohn等人(2020)的校正公式校正湿度偏差
3. 对A/B通道数据取平均得到小时值
4. 质量控制：剔除温度超出-58-140°F、湿度超出0-100%的异常值

### 特征构建（四组实验）
1. **Group 1（地统计）**：仅使用经纬度作为协变量
   - Universal Kriging (UK)
   - Nearest Neighbor Gaussian Process (NNGP)
   - Fixed Rank Kriging (FRK)

2. **Group 2（非地统计）**：仅使用经纬度
   - Regression (Reg)
   - Random Forest (RF)
   - Support Vector Regression (SVR)
   - Ensemble Neural Network (ENN)

3. **Group 3（非地统计+NNO）**：使用经纬度+最近邻观测
   - 各模型添加10个最近邻观测值作为特征

4. **Group 4（非地统计+NNO+Krig）**：添加NNGP克里金预测
   - 各模型添加NNGP克里金预测作为额外特征

### 模型训练
1. 5折交叉验证，随机划分80%训练/20%测试
2. 地统计模型使用最大似然估计参数
3. 机器学习模型使用网格搜索调优超参数
4. 集成神经网络：两个网络（NN1: 512-128, NN2: 256-128-64）堆叠

### 预测与评估
1. 计算RMSE、SMAPE、MAD、相关系数
2. 计算95%预测区间覆盖率
3. 使用Moran's I检验残差空间自相关

## 方法特点

1. **系统性比较**：涵盖地统计与机器学习两大类方法
2. **混合策略有效性**：NNO+Krig特征显著提升ML模型性能
3. **最近邻观测(NNO)**：有效编码空间相关性
4. **不确定性量化**：UK、FRK、NNGP提供解析预测方差

## 性能指标

| 模型 | RMSE | SMAPE | MAD | 相关系数 | 95%覆盖率 |
|-----|------|-------|-----|---------|----------|
| UK | 0.3730 | 7.802% | 0.2630 | 0.8701 | 93% |
| NNGP | 0.4076 | 9.117% | 0.3174 | 0.7925 | 89% |
| SVR+NNO+Krig | 0.0819 | 1.330% | 0.0449 | 0.9792 | - |
| RF+NNO+Krig | 0.0989 | 1.458% | 0.0493 | 0.9694 | 96% |

## 应用场景

- 城市尺度PM2.5小时级制图
- 低成本传感器网络数据融合
- 空气质量指数填补
- 野火期间空气质量监测

## 数据规格

### 输入
| 数据 | 格式 | 维度 | 单位 |
|-----|------|-----|------|
| PurpleAir传感器PM2.5 | array | (n_stations,) | μg/m³ |
| 气象数据 | array | (n_stations, n_features) | - |
| 经纬度坐标 | array | (n_stations, 2) | 度 |

### 输出
| 数据 | 格式 | 单位 |
|-----|------|------|
| PM2.5预测 | array | μg/m³ |

## 随机性
- [ ] 是  - [x] 否（地统计方法为确定性，RF/SVR带随机性）

## 方法指纹
MD5: geostatistical_machine_learning_hybrid_geoML
