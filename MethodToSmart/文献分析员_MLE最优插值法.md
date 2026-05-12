# 【可执行方法规范】

## 方法名称
MLE-OI - 最大似然估计最优插值法 (Maximum Likelihood Optimal Interpolation)

## 文献来源
- 论文标题: "Model evaluation and spatial interpolation by Bayesian combination of observations with numerical models" (Fuentes and Raftery, 2005)
- 方法: 贝叶斯组合观测与数值模型输出

## 核心公式

### 简化最优插值:
$$
\hat{y}_{OI}(s_0) = \mathbf{x}(s_0)^T \beta + \sum_{i=1}^{n} w_i (y_i - \mathbf{x}(s_i)^T \beta)
$$

### 权重确定（最小化估计方差）:
$$
\mathbf{w} = (\mathbf{X}^T \mathbf{R}^{-1} \mathbf{X} + \lambda \mathbf{I})^{-1} \mathbf{X}^T \mathbf{R}^{-1}
$$

### 偏差校正形式:
$$
\hat{y}_{final}(s) = (1 - k) \cdot CMAQ(s) + k \cdot \hat{y}_{OI}(s)
$$
其中k是融合权重

## 参数清单

| 参数名 | 类型 | 默认值 | 说明 |
|-------|------|-------|------|
| lambda | float | 1.0 | 正则化参数 |
| correlation_scale | float | data-fitted | 空间相关尺度 |

## 数据规格

### 输入
| 数据 | 格式 | 维度 | 单位 |
|-----|------|-----|------|
| 监测站点观测 | array | (n_obs,) | μg/m³ |
| CMAQ网格值 | array | (n_grid,) | μg/m³ |
| 站点坐标 | array | (n_obs, 2) | 度 |
| 网格坐标 | array | (n_grid, 2) | 度 |

### 输出
| 数据 | 格式 | 单位 |
|-----|------|------|
| 融合网格场 | array | μg/m³ |

## 实现步骤

1. **数据准备**: 收集观测数据和CMAQ模型输出
2. **空间相关性建模**: 估计观测误差协方差矩阵 R
3. **权重计算**: 通过 $\mathbf{w} = (X^T R^{-1} X + \lambda I)^{-1} X^T R^{-1}$ 计算最优权重
4. **最优插值**: 对目标网格点进行加权融合
5. **偏差校正**: 使用 $(1-k) \cdot CMAQ + k \cdot OI$ 进行最终融合

## 随机性
- [x] 否（确定性方法，MLE参数估计为优化过程）

## 方法指纹
MD5: mle_optimal_interpolation_method

## 实现检查清单
- [ ] 核心公式已验证
- [ ] 权重计算已实现
