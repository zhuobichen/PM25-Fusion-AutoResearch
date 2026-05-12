# 【可执行方法规范】

## 方法名称
GWR - 地理加权回归法 (Geographically Weighted Regression)

## 文献来源
- 论文标题: "Geographically and temporally weighted neural networks for satellite-based mapping of ground-level PM2.5"
- 作者: 相关GWR研究
- arXiv: 1809.09860

## 核心公式

### 全局GWR:
$$
y_i = \beta_0 + \sum_k \beta_k x_{ik} + \epsilon_i
$$

### 地理加权回归:
$$
\hat{y}_i = \beta_0(s_i) + \sum_k \beta_k(s_i) x_{ik}
$$
其中回归系数在每个位置估计:
$$
\hat{\beta}(s_i) = (X^T W(s_i) X)^{-1} X^T W(s_i) Y
$$

### 权重函数（高斯核）:
$$
w_j(s_i) = \exp\left(-\frac{d_{ij}}{b}\right)
$$
其中 $d_{ij}$ = 距离，$b$ = 带宽参数

## 参数清单

| 参数名 | 类型 | 默认值 | 说明 |
|-------|------|-------|------|
| kernel | str | 'gaussian' | 核函数类型 |
| bandwidth | float | data-fitted | 空间带宽 |
| adaptive | bool | True | 自适应带宽 |

## 数据规格

### 输入
| 数据 | 格式 | 维度 | 单位 |
|-----|------|-----|------|
| 站点坐标 | array | (n, 2) | 度 |
| 特征（CMAQ、卫星等） | array | (n, p) | - |
| 目标观测 | array | (n,) | μg/m³ |

### 输出
| 数据 | 格式 | 单位 |
|-----|------|------|
| 空间预测场 | array | μg/m³ |
| 空间变化系数 | array | (n, p+1) |

## 实现步骤

1. **数据准备**: 收集站点坐标、协变量特征和目标观测值
2. **带宽选择**: 使用交叉验证或AIC准则选择最优带宽 $b$
3. **核权重计算**: 对每个预测位置，计算所有观测点的高斯核权重 $w_j(s_i)$
4. **局部回归**: 在每个位置求解加权最小二乘 $\hat{\beta}(s_i) = (X^T W(s_i) X)^{-1} X^T W(s_i) Y$
5. **空间预测**: 使用局部回归系数预测目标位置的PM2.5浓度
6. **系数映射**: 输出空间变化的回归系数图

## 随机性
- [x] 否（确定性方法，带宽选择为确定性优化）

## 方法指纹
MD5: gwr_geographically_weighted_regression

## 实现检查清单
- [x] 核心公式已验证
- [x] 带宽选择已实现
