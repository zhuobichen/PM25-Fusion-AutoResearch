# 【可执行方法规范】

## 方法名称
克里金伪标签增强法 (Kriging-based Pseudo-Label Augmentation)

## 文献来源
- 论文标题：Augmenting Ground-Level PM2.5 Prediction via Kriging-Based Pseudo-Label Generation
- 作者：Lei Duan, Ziyang Jiang, David Carlson
- 年份：2024年
- arXiv: 2401.08061

## 核心公式

### 1. 普通克里金插值
$$
\hat{Z}(s_0) = \sum_{i=1}^{n} \lambda_i Z(s_i)
$$
权重 $\lambda_i$ 通过以下约束求解：
$$
\sum_{i=1}^{n} \lambda_i = 1, \quad \sum_{i=1}^{n} \lambda_i C(s_i, s_j) = C(s_0, s_j), \forall j
$$
其中 $C$ 是协方差函数。

### 2. 变异函数建模
$$
\gamma(h) = \begin{cases} 0 & h = 0 \\ c_0 + c \cdot \text{Sph}(h/a) & h > 0 \end{cases}
$$
其中 $c_0$ 是块金值，$c$ 是拱高，$a$ 是变程。

### 3. CNN-RF混合模型
$$
\hat{y}_{CNN-RF} = f_{RF}(f_{CNN}(x; \theta_CNN); \theta_{RF})
$$
CNN提取空间特征，RF进行最终回归。

### 4. 伪标签生成策略
$$
\tilde{Z}(s_{pseudo}) = \hat{Z}_{Krig}(s_{pseudo}), \quad s_{pseudo} \in \text{卫星覆盖区域}
$$
对没有地面监测但有卫星数据的区域生成克里金伪标签。

### 5. 增强训练损失
$$
\mathcal{L}_{aug} = \mathcal{L}_{original} + \alpha \cdot \mathcal{L}_{pseudo}
$$
其中 $\mathcal{L}_{pseudo} = \sum_{s \in S_{pseudo}} |f(x(s)) - \tilde{Z}(s)|$

## 参数清单

| 参数名 | 类型 | 默认值 | 说明 |
|-------|------|-------|------|
| kriging_range | float | 200km | 克里金搜索半径 |
| min_stations | int | 10 | 克里金最少站点数 |
| nugget | float | 0.1 | 变异函数块金值 |
| sill | float | 1.0 | 变异函数拱高 |
| augmentation_ratio | float | 0.3-0.5 | 伪标签样本比例 |
| CNN_epochs | int | 50 | CNN特征提取训练轮数 |
| RF_n_estimators | int | 100 | 随机森林树数量 |

## 数据规格

| 数据 | 格式 | 说明 |
|-----|------|-----|
| 地面监测PM2.5 | CSV | 参考真值 |
| 卫星AOD | HDF/NetCDF | 卫星气溶胶产品 |
| 气象协变量 | CSV/NetCDF | 风、温、湿等 |
| 土地覆被 | GeoTIFF | 土地利用类型 |
| 人口数据 | CSV | 人口密度网格 |

## 实现步骤

### 阶段1：空间变异函数估计
1. 计算实验变异函数 $\hat{\gamma}(h)$
2. 拟合理论模型（球状模型）
3. 验证交叉验证

### 阶段2：克里金插值
1. 对观测站点进行克里金插值
2. 生成均匀网格上的插值结果
3. 计算插值标准误差

### 阶段3：伪标签生成
1. 识别有AOD覆盖但无监测的区域
2. 使用克里金模型预测PM2.5值
3. 筛选高置信度伪标签（低插值方差）

### 阶段4：CNN-RF训练
1. 使用原始标注数据训练CNN特征提取器
2. 使用原始+伪标签数据训练RF回归器
3. 评估泛化性能

## 方法特点

1. **数据增强**：利用卫星覆盖区域扩展训练集
2. **空间插值保证**：克里金提供最优线性无偏估计
3. **端到端学习**：CNN-RF混合架构
4. **物理一致性**：伪标签来自物理解释的插值

## 可复现性评估

- **数据需求**：地面监测 + 卫星AOD + 气象
- **计算成本**：低-中等
- **代码可用性**：arXiv论文提供方法描述
- **方法创新度**：中等（数据增强策略）

## 物理可解释性

- 克里金插值基于空间相关性假设
- 变异函数反映PM2.5的空间变异结构
- CNN-RF学习AOD与PM2.5的物理关联

## 应用场景

- 扩展PM2.5训练数据集
- CMAQ模型输出的空间验证
- 卫星AOD反演PM2.5的精度提升
- 无监测区域的高分辨率PM2.5估算

## 随机性
- [x] 是（随机森林训练带随机性）

## 方法指纹
MD5: kriging_pseudo_label_augmentation
