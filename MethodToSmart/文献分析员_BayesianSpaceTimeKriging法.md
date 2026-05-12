# 【可执行方法规范】

## 方法名称
Bayesian-STK - 贝叶斯时空克里金法 (Bayesian Spatiotemporal Kriging)

## 文献来源
- 论文标题: 相关贝叶斯时空建模研究 (McMillan et al., 2009; Lindstrom et al., 2014)
- 方法: 贝叶斯方法结合数值模型输出和观测数据

## 核心公式

### 时空随机场模型:
$$
Y(s,t) = X(s,t)\beta + U(s,t) + \epsilon(s,t)
$$
其中:
- $X(s,t)$ = 时空协变量
- $U(s,t)$ = 时空随机场
- $\epsilon(s,t)$ = 观测误差

### 时空协方差函数:
$$
Cov(U(s_i,t_i), U(s_j,t_j)) = \sigma^2 \exp\left(-\frac{d_{ij}}{\rho_s} - \frac{|t_i-t_j|}{\rho_t}\right)
$$

### 贝叶斯推断:
$$
p(\beta, \sigma^2, \rho_s, \rho_t | Y) \propto p(Y | \beta, \sigma^2) \cdot p(\beta) \cdot p(\sigma^2) \cdot p(\rho_s) \cdot p(\rho_t)
$$

## 参数清单

| 参数名 | 类型 | 默认值 | 说明 |
|-------|------|-------|------|
| spatial_range | float | data-fitted | 空间相关尺度 |
| temporal_range | float | data-fitted | 时间相关尺度 |
| n_iter | int | 5000 | MCMC迭代次数 |

## 数据规格

### 输入
| 数据 | 格式 | 维度 | 单位 |
|-----|------|-----|------|
| 监测站点观测 | array | (n_obs, n_time) | μg/m³ |
| 时空协变量 | array | (n_obs, n_time, n_cov) | - |
| 站点坐标 | array | (n_obs, 2) | 度 |

### 输出
| 数据 | 格式 | 单位 |
|-----|------|------|
| 融合时空场 | array | μg/m³ |
| 后验不确定性 | array | μg/m³ |

## 实现步骤

1. **数据准备**: 收集时空观测数据和协变量
2. **先验设定**: 指定空间范围 $\rho_s$、时间范围 $\rho_t$、方差 $\sigma^2$ 的先验分布
3. **MCMC采样**: 使用MCMC从后验分布 $p(\beta, \sigma^2, \rho_s, \rho_t | Y)$ 采样
4. **时空预测**: 基于后验样本进行时空克里金预测
5. **不确定性估计**: 使用后验样本计算预测区间

## 随机性
- [x] 是（MCMC贝叶斯推断带随机采样）

## 方法指纹
MD5: bayesian_st_kriging_method

## 实现检查清单
- [ ] 核心公式已验证
- [ ] MCMC采样已实现
