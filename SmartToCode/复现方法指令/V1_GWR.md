# 复现方法指令

## 方法名称
GWR — 地理加权回归法 (Geographically Weighted Regression)

## 文献来源
- 论文: "Geographically and temporally weighted neural networks for satellite-based mapping of ground-level PM2.5"
- arXiv: 1809.09860

## 核心思路
地理加权回归（GWR）的核心思想是：回归系数不是全局固定的，而是随空间位置变化的。每个预测位置使用附近的观测数据进行局部加权最小二乘回归，权重由空间距离的核函数决定。

## 输入数据

| 数据 | 格式 | 说明 |
|------|------|------|
| 监测站坐标 | array (n, 2) | 经纬度 |
| 监测站浓度 | array (n,) | 日均PM2.5 (μg/m³) |
| CMAQ网格值 | array (lat, lon) | 对应日的CMAQ模拟浓度 |
| CMAQ网格坐标 | array (m, 2) | 网格点经纬度 |

## 输出数据

| 数据 | 格式 | 说明 |
|------|------|------|
| 融合结果 | array (lat, lon) | 融合后的PM2.5浓度场 (μg/m³) |

## 核心公式

### 全局OLS回归（基准对比）
$$
y_i = \beta_0 + \beta_1 \cdot x_i + \epsilon_i
$$
其中 $x_i$ = CMAQ值，$y_i$ = 观测值

### 地理加权回归
在每个预测位置 $s_0$，求解局部加权最小二乘：
$$
\hat{\beta}(s_0) = (X^T W(s_0) X)^{-1} X^T W(s_0) Y
$$

其中 $W(s_0)$ = 对角权重矩阵，对角元素：
$$
w_j(s_0) = \exp\left(-\frac{d_{j0}^2}{2b^2}\right)
$$
- $d_{j0}$ = 站点j到预测位置 $s_0$ 的距离
- $b$ = 带宽参数（控制空间影响范围）

### 预测
$$
\hat{y}(s_0) = \hat{\beta}_0(s_0) + \hat{\beta}_1(s_0) \cdot CTM(s_0)
$$

### 带宽选择
使用AICc（修正Akaike信息准则）选择最优带宽：
$$
AICc = 2n\ln(\hat{\sigma}) + n\ln(2\pi) + \frac{n + tr(S)}{n - 2 - tr(S)}
$$
其中 $S$ = hat matrix，$tr(S)$ = 有效参数数

## 关键步骤

1. **数据准备**: 读取站点坐标、观测浓度、CMAQ网格值
2. **构建设计矩阵**: $X = [1, CTM]$（截距 + CMAQ值）
3. **带宽搜索**: 在候选带宽范围内，用AICc准则选择最优带宽b
4. **逐网格预测**: 对每个网格点 $s_0$：
   a. 计算到所有站点的距离 $d_{j0}$
   b. 计算高斯核权重 $w_j = \exp(-d_{j0}^2 / 2b^2)$
   c. 求解局部加权最小二乘 $\hat{\beta}(s_0)$
   d. 预测 $\hat{y}(s_0) = \hat{\beta}_0 + \hat{\beta}_1 \cdot CTM(s_0)$
5. **非负约束**: 取 max(预测值, 0)

## 参数清单

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| kernel | str | 'gaussian' | 核函数类型：'gaussian', 'bisquare' |
| bandwidth_search | str | 'aic' | 带宽选择准则：'aic', 'cv' |
| bandwidth_min | float | 0.1 | 最小带宽（度） |
| bandwidth_max | float | 10.0 | 最大带宽（度） |
| bandwidth_n | int | 50 | 带宽搜索点数 |
| fix_n_neighbors | int | None | 固定近邻数模式（自适应带宽） |

## 适配要点

- 高维网格点计算量大，建议使用KDTree加速距离计算
- 带宽选择可使用Golden Section搜索提高效率
- 对于n<10的局部回归，退化为全局OLS
- 十折验证：训练用9折站点，预测1折站点坐标
- 确定性方法（无随机性）

## 方法指纹
```
gwr_geographically_weighted_regression_local_v1
```
