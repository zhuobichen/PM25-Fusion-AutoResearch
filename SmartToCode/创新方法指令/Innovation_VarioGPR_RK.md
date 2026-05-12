# 创新方法指令

## 方法名称
VarioGPR-RK — 变异函数引导的GPR残差克里金 (Variogram-Guided GPR Residual Kriging)

## 【创新点】

**核心创新：** 用经验变异函数的空间相关结构直接引导GPR核函数设计，实现数据驱动的核参数自适应。

**与PolyRK的区别：**
- PolyRK使用固定RBF核（length_scale需手动调参）
- VarioGPR-RK从变异函数自动提取range→GPR length_scale，nugget→noise_level
- 物理意义：变异函数描述了大气污染物的空间相关结构，用它来指导空间插值核设计是物理一致的

**与SLOOCV_AK的区别：**
- SLOOCV_AK使用逐站点留一交叉验证选择带宽（计算量大）
- VarioGPR-RK用变异函数一次拟合获取全局空间结构参数（计算效率高）

## 核心思路

三阶段融合：①多项式偏差校正 → ②变异函数分析提取空间结构 → ③以变异函数参数引导GPR建模残差

## 输入数据

| 数据 | 格式 | 说明 |
|------|------|------|
| 监测站坐标 | array (n, 2) | 经纬度 |
| 监测站浓度 | array (n,) | 日均PM2.5 (μg/m³) |
| CMAQ站点值 | array (n,) | 站点处CMAQ模拟值 |
| CMAQ网格坐标 | array (m, 2) | 网格点经纬度 |
| CMAQ网格值 | array (m,) | 网格点CMAQ模拟值 |

## 输出数据

| 数据 | 格式 | 说明 |
|------|------|------|
| 融合结果 | array (m,) | 融合后的PM2.5浓度场 (μg/m³) |

## 核心公式

### 阶段1: 多项式偏差校正

建立OBS~CMAQ的多项式回归关系：
$$
\hat{y}_{poly}(s) = \beta_0 + \beta_1 \cdot CTM(s) + \beta_2 \cdot CTM(s)^2
$$

用OLS拟合 $\beta_0, \beta_1, \beta_2$，得到网格点的多项式校正值 $\hat{y}_{poly}$。

### 阶段2: 经验变异函数分析

计算残差 $r_i = OBS_i - \hat{y}_{poly}(s_i)$，然后拟合经验变异函数：

**实验变异函数**：
$$
\hat{\gamma}(h) = \frac{1}{2N(h)} \sum_{(i,j):d_{ij}=h} (r_i - r_j)^2
$$

**拟合指数模型**：
$$
\gamma(h) = c_0 + c_1 \left[1 - \exp\left(-\frac{h}{a}\right)\right]
$$

提取三个关键参数：
- **Nugget（块金）** $c_0$ → 测量噪声/微尺度变异
- **Sill（基台）** $c_0 + c_1$ → 总空间变异
- **Range（变程）** $a$ → 空间相关距离

### 阶段3: 变异函数引导的GPR残差建模

**GPR核函数设计**：
$$
k(s_i, s_j) = \sigma_f^2 \cdot \mathcal{M}_{\nu}\left(\frac{\sqrt{2\nu} \cdot d_{ij}}{l}\right) + \sigma_n^2 \cdot \delta_{ij}
$$

**变异函数→GPR参数映射**：
| 变异函数参数 | GPR参数 | 映射关系 |
|-------------|---------|----------|
| Range $a$ | length_scale $l$ | $l = a$ |
| Sill $c_0+c_1$ | signal variance $\sigma_f^2$ | $\sigma_f^2 = c_0 + c_1$ |
| Nugget $c_0$ | noise variance $\sigma_n^2$ | $\sigma_n^2 = c_0$ |
| Matérn阶 $\nu$ | smoothness | $\nu = 1.5$（默认） |

**GPR预测**：
$$
\hat{r}(s_*) = K(s_*, S) [K(S,S) + \sigma_n^2 I]^{-1} \mathbf{r}
$$

### 最终融合
$$
FC(s) = \hat{y}_{poly}(s) + \hat{r}(s)
$$

## 关键步骤

1. **多项式回归**: OLS拟合 OBS ~ β₀ + β₁·CMAQ + β₂·CMAQ²
2. **残差计算**: r = OBS - poly_predict
3. **变异函数拟合**: 计算实验变异函数，拟合指数模型，提取 nugget/sill/range
4. **GPR核设计**: 用变异函数参数初始化GPR核（Matérn 1.5）
5. **GPR训练**: 最化 log marginal likelihood 微调核参数
6. **GPR预测**: 在网格点预测残差
7. **融合**: FC = poly_pred + gpr_residual
8. **非负约束**: max(FC, 0)

## 参数清单

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| poly_degree | int | 2 | 多项式阶数（1=线性, 2=二次） |
| variogram_model | str | 'exponential' | 变异函数模型类型 |
| variogram_n_lags | int | 15 | 变异函数滞后距离数 |
| matern_nu | float | 1.5 | Matérn核光滑度参数 |
| optimize_kernel | bool | True | 是否用MLE微调GPR核参数 |
| normalize | bool | True | 是否标准化输入坐标 |

## 创新判定预期

| 指标 | 阈值 | 预期 | 依据 |
|------|------|------|------|
| R² | ≥ 0.9134 (stage1) | 0.915-0.925 | 变异函数引导的核设计比固定核更贴合实际空间结构 |
| RMSE | ≤ 15.51 | 14.5-15.5 | 自适应核减少过拟合/欠拟合 |
| \|MB\| | ≤ 0.15 | < 0.15 | 多项式校正保留系统偏差校正能力 |

## 风险假设

1. **变异函数拟合不稳定**: 样本量少时（n<30），变异函数拟合可能不稳定 → 缓解：使用交叉验证验证变异函数参数
2. **MLE优化陷入局部最优**: GPR核参数MLE优化可能收敛到局部最优 → 缓解：用变异函数参数作为初始化（好的起点）
3. **计算量**: 变异函数计算+GPR训练双重开销 → 缓解：变异函数计算O(n²)，GPR训练O(n³)，n=站点数通常<200

## 方法指纹
```
variogram_guided_gpr_matern_residual_kriging_v1
```

## 物理可解释性

- **多项式校正**：修正CMAQ系统偏差（排放清单、化学机制误差）
- **变异函数Nugget**：反映测量噪声和微尺度变异（<站点间距的变异）
- **变异函数Range**：反映大气污染物空间相关距离（典型值50-200km）
- **变异函数Sill**：反映空间变异总量
- **GPR残差**：捕获多项式未解释的局部空间变异
