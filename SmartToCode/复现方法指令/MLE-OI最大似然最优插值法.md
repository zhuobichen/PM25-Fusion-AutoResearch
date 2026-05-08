# 复现方法指令

## 方法名称
MLE-OI - 最大似然估计最优插值法 (Maximum Likelihood Optimal Interpolation)

## 文献来源
Fuentes and Raftery (2005), "Model evaluation and spatial interpolation by Bayesian combination of observations with numerical models"
方法类别：经典数据同化 / 最优插值

## 方法概述
贝叶斯框架下的最优插值：将CMAQ作为背景场（先验），监测作为观测，通过最小化分析误差方差确定最优权重。物理可解释：权重由背景误差协方差和观测误差协方差的比值决定。

## 输入数据
- 监测站坐标：shape (n_stations, 2)
- CMAQ网格数据：shape (lat, lon, time)
- 监测站PM2.5：shape (n_stations, time)

## 输出数据
- 融合结果：shape (n_stations, time) 或 (lat, lon, time)
- 单位：μg/m³

## 核心公式

### 最优插值基本形式
$$
\hat{y}(s_0) = y_{b}(s_0) + \mathbf{k}^T \cdot (\mathbf{y}_{obs} - H \cdot \mathbf{y}_{b})
$$
- $y_b(s_0)$：CMAQ背景值（先验）
- $\mathbf{y}_{obs}$：观测向量
- $H$：观测算子（从网格到站点的插值）
- $\mathbf{k}$：最优增益向量

### 最优增益（Kalman增益）
$$
\mathbf{k} = \mathbf{B} H^T (H \mathbf{B} H^T + \mathbf{R})^{-1}
$$
- $\mathbf{B}$：背景误差协方差矩阵（CMAQ误差的空间结构）
- $\mathbf{R}$：观测误差协方差矩阵（监测站误差）

### 背景误差协方差建模
$$
B_{ij} = \sigma_b^2 \cdot \exp\left(-\frac{d_{ij}}{L_c}\right)
$$
- $\sigma_b^2$：背景误差方差（CMAQ偏差方差）
- $L_c$：空间相关长度
- $d_{ij}$：站点间距离

### 观测误差协方差
$$
R_{ij} = \sigma_o^2 \cdot \delta_{ij}
$$
假设观测误差独立，$\sigma_o^2$ 为观测误差方差。

### MLE参数估计
通过最大化观测的边际似然函数估计 $\sigma_b^2$、$\sigma_o^2$、$L_c$：
$$
\ell(\sigma_b^2, \sigma_o^2, L_c) = -\frac{1}{2} \log|\mathbf{H}\mathbf{B}\mathbf{H}^T + \mathbf{R}| - \frac{1}{2} \mathbf{d}^T (\mathbf{H}\mathbf{B}\mathbf{H}^T + \mathbf{R})^{-1} \mathbf{d}
$$
其中 $\mathbf{d} = \mathbf{y}_{obs} - H \cdot \mathbf{y}_b$ 为观测增量（innovation）。

## 关键步骤

1. **构建背景场**：CMAQ网格值作为先验估计
   - 从CMAQ NetCDF提取站点位置的网格值
   - 或使用双线性插值获取站点处CMAQ值

2. **计算观测增量**：$d_i = y_{obs,i} - y_{CMAQ,i}$

3. **MLE参数估计**：优化 $\sigma_b^2, \sigma_o^2, L_c$
   - 使用scipy.optimize.minimize
   - 目标：最小化负对数似然
   - 约束：$\sigma_b^2 > 0, \sigma_o^2 > 0, L_c > 0$

4. **构建协方差矩阵**：
   - $B$：指数协方差模型
   - $R$：对角矩阵

5. **计算最优增益**：$\mathbf{k} = \mathbf{B} H^T (H \mathbf{B} H^T + \mathbf{R})^{-1}$

6. **分析更新**：$\hat{y}(s_0) = y_{CMAQ}(s_0) + \mathbf{k}^T \mathbf{d}$

7. **网格预测**：对每个CMAQ网格点执行步骤6

## 参数清单

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| sigma_b_init | float | 5.0 | 背景误差标准差初值 (μg/m³) |
| sigma_o_init | float | 2.0 | 观测误差标准差初值 (μg/m³) |
| Lc_init | float | 50.0 | 空间相关长度初值 (km) |
| cov_model | str | 'exponential' | 背景误差协方差模型 |

## 十折验证适配

### 标准模式
```
训练集：9折站点观测 + 对应CMAQ值
  → 计算观测增量 d
  → MLE估计参数 (σ_b², σ_o², L_c)
  → 构建B和R矩阵
  → 计算增益k

预测：对1折站点
  → y_analysis = y_CMAQ + k^T * d_train
```

### 注意事项
- OI需要求解矩阵逆，站点数多时计算量大
- 建议限制近邻数（如最近20个站点）以降低计算成本
- 参数估计可能有多个局部最优，需多次初始化

## 依赖库
- scipy.optimize（MLE参数估计）
- numpy（矩阵运算）
- scipy.spatial.distance（距离矩阵）

## 实现检查清单
- [ ] CMAQ背景场构建
- [ ] 观测增量计算
- [ ] MLE参数估计（σ_b², σ_o², L_c）
- [ ] 协方差矩阵构建（B, R）
- [ ] 最优增益计算
- [ ] 分析更新
- [ ] 十折验证集成
