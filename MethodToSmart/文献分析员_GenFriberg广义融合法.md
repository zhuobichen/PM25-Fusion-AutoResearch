# 【可执行方法规范】

## 方法名称
Gen-Friberg - 广义CMAQ观测融合法 (Generalized Friberg Data Fusion Method)

## 文献来源
- 论文标题: "A Generalized User-friendly Method for Fusing Observational Data and Chemical Transport Model (Gen-Friberg V1.0: GF-1)"
- 作者: Zongrun Li, Abiola S. Lawal, Bingqing Zhang, Kamal J. Maji, Pengfei Liu, Yongtao Hu, Armistead G. Russell, M. Talat Odman
- 期刊: Environmental Modelling and Software (2025)
- 机构: Georgia Institute of Technology
- GitHub: 开源代码可用

## 核心公式

### 步骤1: 年均值校正回归 (方程1-2)
$$
\overline{OBS}_m = \alpha \times CTM(s_m)^{\beta} + \epsilon \quad (1)
$$
$$
\overline{FC}(s) = \alpha_{year} \times CTM(s)^{\beta} + \epsilon \quad (2)
$$
选择线性(β=1)或指数回归，使用10折交叉验证自动选择。

### 步骤1: FC1融合 (方程3)
$$
FC_1(s,t) = krig\left(\frac{OBS_m(t)}{\overline{OBS}_m}\right) \times \overline{FC}(s) \quad (3)
$$
使用PyKrige包，指数半变异函数模型进行克里金插值。

### 步骤2: 季节偏差校正 (方程4-8)
首先调整CMAQ日场:
$$
CTM_{adj}(s,t) = CTM(s,t) \times \frac{\overline{FC}(s)}{\overline{CTM}(s)} \quad (4)
$$

计算季节比值:
$$
\beta_{season}(j_t) = 1 + A \times \cos\left[\frac{2\pi}{365.25}(j_t - j_{t_{max}})\right] \quad (7)
$$

FC2融合:
$$
FC_2(s,t) = CTM_{adj}(s,t) \times \beta_{season}(j_t) \quad (8)
$$

### 步骤3: 优化融合 (方程9-12)

指数相关图:
$$
R_{obs}(d) = R_{coll} \times e^{-d/r} + \epsilon \quad (9)
$$

R1 (FC1时空相关性):
$$
R_1(s,t) = R_{coll} \times e^{-x(s,t)/r} \quad (10)
$$

R2 (CMAQ时间相关性):
$$
R_2 = \frac{1}{M}\sum_{m=1}^{M} corr(OBS_m, CTM_m) \quad (11)
$$

权重因子:
$$
W(s,t) = \frac{R_1(s,t) \times (1 - R_2)}{R_1(s,t) \times (1 - R_2) + R_2 \times (1 - R_1(s,t))} \quad (12)
$$

最终融合:
$$
FC_{final}(s,t) = W(s,t) \times FC_1(s,t) + (1 - W(s,t)) \times FC_2(s,t)
$$

## 参数清单

| 参数名 | 类型 | 默认值 | 说明 |
|-------|------|-------|------|
| regression_mode | str | 'auto' | 'linear', 'exponential', 或 'auto' |
| variogram_model | str | 'exponential' | PyKrige半变异函数模型 |
| n_folds | int | 10 | 交叉验证折数 |
| parallel | bool | True | 是否并行计算 |

## 数据规格

### 输入
| 数据 | 格式 | 说明 |
|-----|------|------|
| 观测数据 | CSV | 站点ID, 坐标, 时间, 浓度 |
| CTM数据 | NetCDF | 网格浓度场 |

### 输出
| 数据 | 格式 | 单位 |
|-----|------|------|
| 融合浓度场 | NetCDF | μg/m³ |

## 实现优势
- 支持多种CTM: CMAQ, GEOS-Chem, WRF-Chem
- 封装为单一函数，易于使用
- 支持并行计算加速
- 自动选择回归模式
- 开源软件

## 实现步骤

1. **年均值校正**: 使用线性或指数回归建立CTM与观测年均值关系
2. **FC1融合**: 对归一化比值进行克里金插值，乘以校正年均场
3. **季节偏差校正**: 使用正弦函数拟合季节性差异
4. **FC2融合**: 将调整后CMAQ日场乘以季节校正因子
5. **优化融合**: 基于时空相关性计算权重W，加权融合FC1和FC2

## 随机性
- [ ] 是（克里金插值有随机性）

## 方法指纹
MD5: gen_friberg_fusion_method

## 实现检查清单
- [x] 核心公式已验证
- [x] PyKrige集成已实现
- [x] 并行计算已实现
