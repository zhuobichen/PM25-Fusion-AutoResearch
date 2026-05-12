# 【可执行方法规范】

## 方法名称
RF-Kriging - 随机森林-克里金残差校正法 (Random Forest with Kriging Residual Correction)

## 文献来源
- 论文标题: "A three-step method to fuse satellite, CMAQ, and observation data" (Xue et al.)
- 方法: 随机森林预测 + 克里金插值残差校正

## 核心公式

### 步骤1: 随机森林预测:
$$
\hat{y}_{RF}(s) = RF(X(s))
$$
其中 $X(s)$ = 特征向量（CMAQ、AOD、气象等）

### 步骤2: 计算残差:
$$
r(s_i) = y_{obs}(s_i) - \hat{y}_{RF}(s_i)
$$

### 步骤3: 克里金插值残差:
$$
\hat{r}(s_0) = \sum_{i=1}^{n} \lambda_i r(s_i)
$$
权重 $\lambda_i$ 通过克里金方程确定

### 步骤4: 最终预测:
$$
\hat{y}_{final}(s_0) = \hat{y}_{RF}(s_0) + \hat{r}(s_0)
$$

## 参数清单

| 参数名 | 类型 | 默认值 | 说明 |
|-------|------|-------|------|
| n_estimators | int | 100 | 决策树数量 |
| max_depth | int | 10 | 最大深度 |
| variogram_model | str | 'spherical' | 半变异函数模型 |

## 数据规格

### 输入
| 数据 | 格式 | 说明 |
|-----|------|------|
| CMAQ/卫星数据 | array | 特征 |
| 观测数据 | array | 目标值 |

### 输出
| 数据 | 格式 | 单位 |
|-----|------|------|
| 融合预测 | array | μg/m³ |

## 实现步骤

1. **数据准备**: 收集CMAQ/卫星特征和观测数据
2. **随机森林训练**: 使用特征X训练随机森林回归模型
3. **残差计算**: 在站点计算观测值与RF预测的残差
4. **变异函数拟合**: 对残差拟合半变异函数模型
5. **克里金插值**: 对残差进行克里金插值
6. **融合预测**: 将RF预测与克里金残差相加

## 随机性
- [ ] 是（随机森林训练带随机初始化）

## 方法指纹
MD5: rf_kriging_residual_method

## 实现检查清单
- [ ] 随机森林已实现
- [ ] 克里金残差校正已实现
