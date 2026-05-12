# 【可执行方法规范】

## 方法名称
IDW-Bias - 距离反比偏差加权融合法 (Inverse Distance Weighting Bias Correction)

## 文献来源
- 论文标题: "Application of a Fusion Method for Gas and Particle Air Pollutants between Observational Data and Chemical Transport Model Simulations Over the Contiguous United States for 2005-2014"
- 作者: Niru Senthilkumar, Mark Gilfether, Francesca Metcalf, Armistead G. Russell, James A. Mulholland, Howard H. Chang
- 期刊: Int. J. Environ. Res. Public Health 2019, 16, 3314
- 应用: 美国2005-2014年 CMAQ融合

## 核心公式

### 归一化比值计算:
$$
R_m = \frac{OBS_m}{\overline{CTM}_m}
$$
其中:
- $OBS_m$ = 站点m的观测均值
- $\overline{CTM}_m$ = 站点m对应的CMAQ年均值

### IDW空间插值:
$$
\hat{R}(s) = \frac{\sum_{i=1}^{n} w_i \cdot R_i}{\sum_{i=1}^{n} w_i}
$$
其中 $w_i = 1/d_i^p$，通常 $p = 2$

### 融合场:
$$
FC(s) = CTM(s) \times \hat{R}(s)
$$

### 带距离加权的改进:
$$
\hat{R}_{weighted}(s) = \frac{\sum_i w_i \cdot (OBS_i / CTM_i) \cdot (1/d_i)}{\sum_i w_i \cdot (1/d_i)}
$$

## 参数清单

| 参数名 | 类型 | 默认值 | 说明 |
|-------|------|-------|------|
| power | float | 2.0 | IDW距离权重指数 |
| max_distance | float | 100.0 | 最大插值距离(km) |
| min_neighbors | int | 3 | 最小近邻数 |

## 数据规格

### 输入
| 数据 | 格式 | 单位 |
|-----|------|------|
| 监测数据 | array | μg/m³ |
| CMAQ网格 | array | μg/m³ |
| 站点坐标 | array | 度 |

### 输出
| 数据 | 格式 | 单位 |
|-----|------|------|
| 融合场 | array | μg/m³ |

## 方法特点
- 简单高效
- 在观测100km范围内效果好
- 超出范围退化到CMAQ原始值

## 实现步骤

1. **数据准备**: 收集监测数据和CMAQ网格数据
2. **比值计算**: 计算每个站点的OBS/CTM归一化比值
3. **IDW插值**: 使用距离反比加权将比值插值到网格
4. **融合**: 将CMAQ值乘以插值得到的比值场
5. **输出**: 生成融合后的PM2.5浓度场

## 随机性
- [x] 否（确定性方法）

## 方法指纹
MD5: idw_bias_weighting_method

## 实现检查清单
- [x] 核心公式已验证
- [x] IDW插值已实现
