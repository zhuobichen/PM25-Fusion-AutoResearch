# 复现方法指令

## 方法名称
Gen-Friberg — 广义CMAQ观测融合法 (Generalized Friberg Data Fusion)

## 文献来源
- 论文: "A Generalized User-friendly Method for Fusing Observational Data and Chemical Transport Model (Gen-Friberg V1.0: GF-1)"
- 作者: Li et al.
- 期刊: Environmental Modelling and Software (2025)
- 机构: Georgia Institute of Technology

## 核心思路
三步级联融合：①年均值回归校正 + 克里金比值插值（FC1）→ ②季节余弦函数校正（FC2）→ ③基于时空相关性的加权优化融合。物理意义清晰：先修正系统偏差，再修正季节偏差，最后基于相关性结构融合。

## 输入数据

| 数据 | 格式 | 说明 |
|------|------|------|
| 监测站坐标 | array (n, 2) | 经纬度 |
| 监测站浓度 | array (n, time) | 日均PM2.5 (μg/m³) |
| CMAQ网格值 | array (lat, lon, time) | CMAQ模拟浓度 |

## 输出数据

| 数据 | 格式 | 说明 |
|------|------|------|
| 融合结果 | array (lat, lon, time) | 融合后的PM2.5浓度场 (μg/m³) |

## 核心公式

### 步骤1: 年均值校正回归

**回归模型**（自动选择线性或指数）：
$$
\overline{OBS}_m = \alpha \times CTM(s_m)^{\beta} + \epsilon
$$

- 线性模式（β=1）：$\overline{OBS}_m = \alpha \times CTM(s_m) + \epsilon$
- 指数模式：$\overline{OBS}_m = \alpha \times CTM(s_m)^{\beta} + \epsilon$
- 使用10折交叉验证自动选择更优模式

**校正年均场**：
$$
\overline{FC}(s) = \alpha \times CTM(s)^{\beta}
$$

### 步骤1续: FC1融合（克里金比值插值）
$$
FC_1(s,t) = krig\left(\frac{OBS_m(t)}{\overline{OBS}_m}\right) \times \overline{FC}(s)
$$

- 计算站点日值/年均值的比值（归一化日变化）
- 使用克里金（指数半变异函数模型）将比值插值到全网格
- 乘以校正年均场，得到FC1

### 步骤2: 季节偏差校正

**调整CMAQ日场**：
$$
CTM_{adj}(s,t) = CTM(s,t) \times \frac{\overline{FC}(s)}{\overline{CTM}(s)}
$$

**季节校正因子**：
$$
\beta_{season}(j_t) = 1 + A \times \cos\left[\frac{2\pi}{365.25}(j_t - j_{t_{max}})\right]
$$
- $j_t$ = 年内日序号（day of year）
- $A$ = 振幅（由观测数据拟合）
- $j_{t_{max}}$ = 峰值日序号

**FC2融合**：
$$
FC_2(s,t) = CTM_{adj}(s,t) \times \beta_{season}(j_t)
$$

### 步骤3: 优化融合

**指数相关图**：
$$
R_{obs}(d) = R_{coll} \times e^{-d/r} + \epsilon
$$
- $R_{coll}$ = 协方差（块金效应以上的空间相关性）
- $r$ = 相关距离

**R1（FC1时空相关性）**：
$$
R_1(s,t) = R_{coll} \times e^{-x(s,t)/r}
$$
- $x(s,t)$ = FC1值（作为时空相关性的代理变量）

**R2（CMAQ时间相关性）**：
$$
R_2 = \frac{1}{M}\sum_{m=1}^{M} corr(OBS_m, CTM_m)
$$

**权重因子**：
$$
W(s,t) = \frac{R_1(s,t) \times (1 - R_2)}{R_1(s,t) \times (1 - R_2) + R_2 \times (1 - R_1(s,t))}
$$

**最终融合**：
$$
FC_{final}(s,t) = W(s,t) \times FC_1(s,t) + (1 - W(s,t)) \times FC_2(s_t)
$$

## 关键步骤

1. **年均值回归**: 拟合 $\overline{OBS} \sim CTM$ 关系，选择线性/指数模式
2. **校正年均场**: 用回归系数校正CMAQ全网格年均值
3. **FC1**: 计算站点日/年比值 → 克里金插值 → 乘以校正年均场
4. **FC2**: CMAQ日场 × 年均校正比 × 季节余弦因子
5. **相关图拟合**: 拟合指数相关图，获取 $R_{coll}$ 和 $r$
6. **权重计算**: 基于R1、R2计算空间变化权重W
7. **最终融合**: $FC = W \times FC_1 + (1-W) \times FC_2$

## 参数清单

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| regression_mode | str | 'auto' | 'linear', 'exponential', 'auto' |
| variogram_model | str | 'exponential' | 克里金半变异函数模型 |
| n_folds | int | 10 | 回归模式选择的交叉验证折数 |
| season_fit_method | str | 'ls' | 季节参数拟合方法：'ls'最小二乘 |
| corr_distance_max | float | 500.0 | 相关图拟合最大距离(km) |

## 适配要点

- 克里金插值可用PyKrige包
- 季节参数A和 $j_{t_{max}}$ 需从观测数据拟合
- 十折验证：训练用9折站点，预测1折站点坐标
- 全网格预测时需逐日处理
- 确定性方法（克里金有随机性，但可固定种子）

## 方法指纹
```
gen_friberg_three_step_cascade_fusion_v1
```
