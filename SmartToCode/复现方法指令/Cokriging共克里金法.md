# 复现方法指令

## 方法名称
Cokriging - 共克里金法 (Cokriging for Multi-variable Spatial Prediction)

## 文献来源
- 论文标题: "A cokriging based approach to reconstruct air pollution maps" (Singh et al., 2011)
- 方法: 使用主变量（监测PM2.5）和辅助变量（CMAQ模拟）的空间互相关性进行联合插值

## 方法概述

共克里金是普通克里金的多变量扩展。核心思想：当两个变量存在空间互相关时，利用辅助变量（CMAQ）可以提高主变量（PM2.5监测）的插值精度。

与普通克里金的区别：
- 普通克里金：仅使用监测站数据
- 共克里金：同时使用监测站数据 + CMAQ网格数据

## 输入数据

| 数据 | 格式 | 维度 | 单位 | 说明 |
|-----|------|-----|------|------|
| 监测站PM2.5 | array | (n_obs,) | μg/m³ | 主变量 |
| CMAQ模拟值 | array | (n_grid,) | μg/m³ | 辅助变量 |
| 监测站坐标 | array | (n_obs, 2) | 度 | lon, lat |
| CMAQ网格坐标 | array | (n_grid, 2) | 度 | lon, lat |
| 预测目标坐标 | array | (n_pred, 2) | 度 | 待预测位置 |

## 输出数据

| 数据 | 格式 | 单位 | 说明 |
|-----|------|------|------|
| 融合预测值 | array | (n_pred,) | μg/m³ | 共克里金估计 |
| 预测方差 | array | (n_pred,) | (μg/m³)² | 估计不确定性 |

## 核心公式

### 1. 互协方差函数
$$
C_{UV}(h) = Cov(U(s), V(s+h))
$$
其中 $U$ = 监测PM2.5（主变量），$V$ = CMAQ模拟（辅助变量），$h$ = 空间距离。

### 2. 交叉变异函数
$$
\gamma_{UV}(h) = \frac{1}{2} Var(U(s) - V(s+h))
$$

### 3. 共克里金估计
$$
\hat{U}(s_0) = \sum_{i=1}^{n} \lambda_i^U U(s_i) + \sum_{j=1}^{m} \lambda_j^V V(s_j)
$$

### 4. 无偏约束
$$
\sum_{i=1}^{n} \lambda_i^U = 1, \quad \sum_{j=1}^{m} \lambda_j^V = 0
$$

### 5. 共克里金方程组（矩阵形式）
$$
\begin{bmatrix}
C_{UU} & C_{UV} & 1 & 0 \\
C_{VU} & C_{VV} & 0 & 1 \\
1^T & 0^T & 0 & 0 \\
0^T & 1^T & 0 & 0
\end{bmatrix}
\begin{bmatrix}
\lambda^U \\
\lambda^V \\
\mu_1 \\
\mu_2
\end{bmatrix}
=
\begin{bmatrix}
C_{UU}(s_0) \\
C_{VV}(s_0) \\
1 \\
0
\end{bmatrix}
$$

其中：
- $C_{UU}$: 主变量自身协方差矩阵 (n×n)
- $C_{UV}$: 主辅变量互协方差矩阵 (n×m)
- $C_{VV}$: 辅助变量自身协方差矩阵 (m×m)
- $\mu_1, \mu_2$: 拉格朗日乘子

### 6. 预测方差
$$
\sigma^2(s_0) = C_{UU}(0) - \sum_{i=1}^{n} \lambda_i^U C_{UU}(s_i - s_0) - \sum_{j=1}^{m} \lambda_j^V C_{UV}(s_j - s_0)
$$

## 关键步骤

### Step 1: 变异函数拟合
对主变量（监测PM2.5）拟合变异函数：
$$
\gamma_{UU}(h) = c_0 + c_1 \left(1 - \exp\left(-\frac{h}{a}\right)\right)
$$
- $c_0$: 块金效应（nugget）
- $c_1$: 基台值（sill）
- $a$: 变程（range）

### Step 2: 交叉变异函数拟合
对主辅变量的交叉变异函数：
$$
\gamma_{UV}(h) = c_0^{UV} + c_1^{UV} \left(1 - \exp\left(-\frac{h}{a^{UV}}\right)\right)
$$

### Step 3: 构建协方差矩阵
根据拟合的变异函数，计算所有站点对之间的协方差：
$$
C_{UU}(h) = C_{UU}(0) - \gamma_{UU}(h)
$$
$$
C_{UV}(h) = C_{UV}(0) - \gamma_{UV}(h)
$$

### Step 4: 求解共克里金方程组
对每个预测位置 $s_0$，解线性方程组获得权重 $\lambda^U, \lambda^V$。

### Step 5: 预测与不确定性估计
$$
\hat{U}(s_0) = \sum_{i=1}^{n} \lambda_i^U U(s_i) + \sum_{j=1}^{m} \lambda_j^V V(s_j)
$$

## 参数清单

| 参数名 | 类型 | 默认值 | 说明 |
|-------|------|-------|------|
| variogram_model | str | 'exponential' | 变异函数模型（exponential/spherical/gaussian） |
| cross_variogram_model | str | 'exponential' | 交叉变异函数模型 |
| n_neighbors_obs | int | 15 | 主变量近邻数量 |
| n_neighbors_cmaq | int | 30 | 辅助变量近邻数量 |
| nugget_U | float | auto | 主变量块金效应 |
| sill_U | float | auto | 主变量基台值 |
| range_U | float | auto | 主变量变程(km) |
| nugget_UV | float | auto | 交叉块金效应 |
| sill_UV | float | auto | 交叉基台值 |
| range_UV | float | auto | 交叉变程(km) |

## 适配系统格式

### 十折验证模式（标准模式）
```python
# 输入
train_monitor = fold_data['train_monitor']   # (n_train,) 监测PM2.5
train_cmaq = fold_data['train_cmaq']         # (n_train,) CMAQ在监测站的值
train_coords = fold_data['train_coords']     # (n_train, 2) 监测站坐标
test_cmaq = fold_data['test_cmaq']           # (n_test,) CMAQ在测试站的值
test_coords = fold_data['test_coords']       # (n_test, 2) 测试站坐标

# 共克里金预测
# 主变量: train_monitor at train_coords
# 辅助变量: train_cmaq at train_coords + test_cmaq at test_coords
pred = cokriging_predict(
    U_obs=train_monitor,          # 主变量观测
    V_obs=train_cmaq,             # 辅助变量在观测点的值
    V_pred=test_cmaq,             # 辅助变量在预测点的值
    coords_obs=train_coords,      # 观测坐标
    coords_pred=test_coords       # 预测坐标
)
```

### 特殊模式（Downscaler兼容）
```python
# 输入: 全域CMAQ + 监测数据
# 输出: 全域融合结果
all_cmaq = cmaq_grid.flatten()           # 全网格CMAQ
all_coords = cmaq_grid_coords             # 全网格坐标
pred_grid = cokriging_predict(
    U_obs=train_monitor,
    V_obs=train_cmaq_at_stations,
    V_pred=all_cmaq,
    coords_obs=train_coords,
    coords_pred=all_coords
)
```

## 随机性
- [x] 否（确定性方法，变异函数拟合和方程组求解均为确定性过程）

## 方法指纹
MD5: cokriging_multivariate_joint_interpolation_v1

## 实现检查清单
- [ ] 变异函数拟合（主变量 + 交叉）
- [ ] 共克里金方程组构建
- [ ] 权重求解（带无偏约束）
- [ ] 预测方差计算
- [ ] 十折验证适配
- [ ] 大规模网格预测优化（分块计算）

## 预期性能

基于文献和方法特性：
- 预期R²: 0.88-0.91（优于普通克里金，因为利用了CMAQ辅助信息）
- 优势：利用主辅变量空间互相关，理论基础扎实
- 风险：交叉变异函数拟合可能不稳定；大规模计算开销较大

## 创新排除判定
- [ ] 是否使用权重学习（Ridge/Lasso等）？→ 否
- [x] 是否物理可解释？→ 是（地统计理论，空间互相关）
- 结论：**复现方法**，非创新方法
