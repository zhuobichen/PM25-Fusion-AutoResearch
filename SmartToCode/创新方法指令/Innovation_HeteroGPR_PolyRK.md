# 创新方法指令

## 方法名称
HeteroGPR-PolyRK — 异方差GPR多项式残差克里金 (Heteroscedastic GPR Polynomial Residual Kriging)

## 【创新点】

**核心创新：** 引入空间异方差GPR建模残差，用辅助GP学习噪声场的空间变化方差，而非假设同方差噪声。

**物理动机：**
- 城区站点：周边排放源多、地形复杂 → 观测噪声大
- 郊区站点：排放源单一、地形平坦 → 观测噪声小
- CMAQ在复杂地形区域偏差更大 → 残差异方差

**与PolyRK的区别：**
- PolyRK假设残差噪声同方差（$\sigma_n^2 I$）
- HeteroGPR-PolyRK用第二个GP建模log(σ²(s))的空间分布

**与HeteroscedasticGPRPolyRK的区别：**
- 注册表中的HeteroscedasticGPRPolyRK是基于浓度分层的异方差（高浓度区≠低浓度区）
- 本方法基于空间位置的异方差（城区≠郊区），物理机制不同

## 核心思路

三阶段融合：①多项式偏差校正 → ②空间特征提取 → ③异方差GPR建模残差

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

$$
\hat{y}_{poly}(s) = \beta_0 + \beta_1 \cdot CTM(s) + \beta_2 \cdot CTM(s)^2
$$

OLS拟合，得到残差 $r_i = OBS_i - \hat{y}_{poly}(s_i)$。

### 阶段2: 空间特征提取

为每个站点提取空间辅助特征（用于建模方差场）：
$$
\mathbf{z}_i = [lat_i, lon_i, d_{urban,i}, elev_i]
$$
- $lat_i, lon_i$ = 经纬度
- $d_{urban,i}$ = 到最近城区中心的距离（代理变量）
- $elev_i$ = 海拔（可选）

简化版本（仅用坐标）：
$$
\mathbf{z}_i = [lat_i, lon_i]
$$

### 阶段3: 异方差GPR

**双GP结构：**

**GP1（均值GP）**：建模残差的空间趋势
$$
r(s) \sim \mathcal{GP}(0, k_{mean}(s, s'))
$$

**GP2（方差GP）**：建模log方差的空间变化
$$
\log \sigma^2(s) \sim \mathcal{GP}(\mu_v, k_{var}(z, z'))
$$

**联合似然**：
$$
p(\mathbf{r} | S, \theta) = \mathcal{N}(\mathbf{r} | 0, K_{mean} + \text{diag}(\sigma^2(\mathbf{s})))
$$

其中 $\sigma^2(s_i) = \exp(f_i)$，$f_i$ 由GP2预测。

**GP2的核函数**（Matérn 1.5）：
$$
k_{var}(z_i, z_j) = \sigma_v^2 \left(1 + \frac{\sqrt{3} d_{ij}}{l_v}\right) \exp\left(-\frac{\sqrt{3} d_{ij}}{l_v}\right)
$$

**优化目标**（近似）：
$$
\mathcal{L} = -\frac{1}{2}\mathbf{r}^T (K_{mean} + D)^{-1} \mathbf{r} - \frac{1}{2}\log|K_{mean} + D| - \frac{1}{2}\mathbf{f}^T K_{var}^{-1} \mathbf{f}
$$

其中 $D = \text{diag}(\exp(\mathbf{f}))$。

### 预测

$$
\hat{r}(s_*) = K_{mean}(s_*, S) [K_{mean}(S,S) + D]^{-1} \mathbf{r}
$$

### 最终融合
$$
FC(s) = \hat{y}_{poly}(s) + \hat{r}(s)
$$

## 关键步骤

1. **多项式回归**: OLS拟合 OBS ~ β₀ + β₁·CMAQ + β₂·CMAQ²
2. **残差计算**: r = OBS - poly_predict
3. **空间特征提取**: 计算站点坐标作为方差GP输入
4. **异方差GPR初始化**: 用同方差GPR作为初始估计
5. **迭代优化**: 交替优化GP1（均值）和GP2（方差）
6. **GPR预测**: 在网格点预测残差（使用空间变化的噪声方差）
7. **融合**: FC = poly_pred + gpr_residual
8. **非负约束**: max(FC, 0)

## 参数清单

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| poly_degree | int | 2 | 多项式阶数 |
| mean_kernel | str | 'matern1.5' | 均值GP核函数类型 |
| var_kernel | str | 'matern1.5' | 方差GP核函数类型 |
| n_optimization_restarts | int | 5 | GP超参数优化重启次数 |
| max_iter_hetero | int | 10 | 异方差迭代次数 |
| normalize | bool | True | 是否标准化输入坐标 |
| jitter | float | 1e-6 | 数值稳定项 |

## 创新判定预期

| 指标 | 阈值 | 预期 | 依据 |
|------|------|------|------|
| R² | ≥ 0.9134 (stage1) | 0.915-0.920 | 异方差建模更准确地刻画噪声空间结构 |
| RMSE | ≤ 15.51 | 14.8-15.5 | 噪声模型更准确→预测更稳定 |
| \|MB\| | ≤ 0.15 | < 0.15 | 多项式校正保留偏差校正能力 |

## 风险假设

1. **GP2训练不稳定**: 站点数少时（n<50），方差GP可能过拟合 → 缓解：使用强先验（大正则化）
2. **迭代收敛慢**: 双GP交替优化可能收敛慢 → 缓解：限制最大迭代次数，使用warm start
3. **计算量**: 双GP结构增加计算量 → 缓解：使用 inducing points 近似（如果n>200）
4. **与HeteroscedasticGPRPolyRK指纹冲突**: 需验证核心公式不同 → 已确认：空间异方差 vs 浓度异方差，机制不同

## 方法指纹
```
heteroscedastic_spatial_gpr_poly_residual_kriging_v1
```

## 物理可解释性

- **多项式校正**：修正CMAQ系统偏差
- **均值GP**：捕获残差的空间相关性（大气输送、区域污染）
- **方差GP**：捕获噪声的空间变化（城区/郊区、地形复杂度）
- **空间异方差的物理意义**：城市站点附近排放源复杂→噪声大，郊区站点→噪声小
