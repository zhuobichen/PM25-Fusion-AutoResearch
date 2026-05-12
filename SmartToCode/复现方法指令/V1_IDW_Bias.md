# 复现方法指令

## 方法名称
IDW-Bias — 距离反比偏差加权融合法 (Inverse Distance Weighting Bias Correction)

## 文献来源
- 论文: "Application of a Fusion Method for Gas and Particle Air Pollutants between Observational Data and Chemical Transport Model Simulations Over the Contiguous United States for 2005-2014"
- 作者: Senthilkumar et al.
- 期刊: Int. J. Environ. Res. Public Health 2019, 16, 3314

## 核心思路
简单高效的空间偏差校正方法：先计算站点处观测/CMAQ的比值（归一化偏差），再用距离反比加权（IDW）将比值插值到全网格，最后乘以CMAQ原始场得到融合结果。

## 输入数据

| 数据 | 格式 | 说明 |
|------|------|------|
| 监测站坐标 | array (n, 2) | 经纬度 |
| 监测站浓度 | array (n,) | 日均PM2.5 (μg/m³) |
| CMAQ网格值 | array (lat, lon) | 对应日的CMAQ模拟浓度 |

## 输出数据

| 数据 | 格式 | 说明 |
|------|------|------|
| 融合结果 | array (lat, lon) | 融合后的PM2.5浓度场 (μg/m³) |

## 核心公式

### 步骤1: 归一化比值计算
$$
R_m = \frac{OBS_m}{\overline{CTM}_m}
$$
- $OBS_m$ = 站点m的观测浓度
- $\overline{CTM}_m$ = 站点m处的CMAQ模拟值

### 步骤2: IDW空间插值
$$
\hat{R}(s) = \frac{\sum_{i=1}^{n} w_i \cdot R_i}{\sum_{i=1}^{n} w_i}
$$
其中权重 $w_i = 1/d_i^p$，$d_i$ = 预测位置s到站点i的距离，$p$ = 距离指数（通常取2）

### 步骤3: 融合
$$
FC(s) = CTM(s) \times \hat{R}(s)
$$

### 距离退化机制（可选增强）
当预测位置距离所有站点超过阈值时，退化为CMAQ原始值：
$$
FC(s) = \begin{cases} CTM(s) \times \hat{R}(s), & \min(d_i) \leq d_{max} \\ CTM(s), & \min(d_i) > d_{max} \end{cases}
$$

## 关键步骤

1. **数据准备**: 读取站点坐标、观测浓度、CMAQ网格数据
2. **站点比值计算**: 对每个站点，计算 $R_m = OBS_m / CTM_m$（保护除零：CTM < 0.1时设R=1）
3. **网格距离计算**: 对每个网格点，计算到所有站点的Haversine距离
4. **IDW插值**: 对每个网格点，用距离反比加权插值比值场
5. **融合**: 网格CMAQ值 × 插值比值
6. **非负约束**: 融合结果取 max(FC, 0)

## 参数清单

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| power | float | 2.0 | IDW距离权重指数 |
| max_distance | float | 100.0 | 最大插值距离(km)，超出退化为CMAQ |
| min_neighbors | int | 3 | 最小近邻数（距离内不足则用最近N个） |
| eps | float | 0.1 | CTM最小阈值，防除零 |

## 适配要点

- 距离计算使用Haversine公式（球面距离）
- CMAQ数据为三维 (lat, lon, time)，逐日处理
- 十折验证模式：训练时用9折站点，预测1折站点坐标处的融合值
- 无随机性，确定性方法

## 方法指纹
```
idw_bias_correction_ratio_interpolation_v1
```
