# 复现方法指令

## 方法名称
RF-Kriging - 随机森林-克里金残差校正法 (Random Forest with Kriging Residual Correction)

## 文献来源
Xue et al., "A three-step method to fuse satellite, CMAQ, and observation data"
方法类别：回归+空间残差插值两步法

## 方法概述
两步法：(1) 随机森林学习CMAQ→监测的非线性映射关系；(2) 克里金插值校正RF残差的空间结构。物理可解释：RF捕获CMAQ系统偏差，克里金捕获空间相关残差。

## 输入数据
- 监测站坐标：shape (n_stations, 2)，经纬度
- CMAQ网格数据：shape (lat, lon, time)，模拟PM2.5浓度
- 监测站PM2.5：shape (n_stations, time)，观测值
- 辅助特征（可选）：气象、AOD、土地利用等

## 输出数据
- 融合结果：shape (n_stations, time) 或 (lat, lon, time)
- 单位：μg/m³

## 核心公式

### 步骤1：随机森林预测
$$
\hat{y}_{RF}(s) = \frac{1}{B} \sum_{b=1}^{B} T_b(X(s))
$$
其中 $X(s) = [CMAQ(s), \text{气象}(s), \text{经纬度}(s)]$，$B$ 为决策树数量，$T_b$ 为第 $b$ 棵树的预测。

### 步骤2：计算训练残差
$$
r(s_i) = y_{obs}(s_i) - \hat{y}_{RF}(s_i), \quad i = 1, ..., n
$$

### 步骤3：变异函数拟合
对残差 $r(s_i)$ 拟合理论变异函数：
$$
\gamma(h) = c_0 + c \cdot \left[1 - \exp\left(-\frac{3h}{a}\right)\right]
$$
- $c_0$：块金效应（测量误差+微尺度变异）
- $c$：基台值（空间结构方差）
- $a$：变程（空间相关范围）

### 步骤4：克里金残差插值
$$
\hat{r}(s_0) = \sum_{i=1}^{n} \lambda_i \cdot r(s_i)
$$
权重由克里金方程组确定：
$$
\begin{bmatrix}
\gamma(s_1,s_1) & \cdots & \gamma(s_1,s_n) & 1 \\
\vdots & \ddots & \vdots & \vdots \\
\gamma(s_n,s_1) & \cdots & \gamma(s_n,s_n) & 1 \\
1 & \cdots & 1 & 0
\end{bmatrix}
\begin{bmatrix}
\lambda_1 \\ \vdots \\ \lambda_n \\ \mu
\end{bmatrix}
=
\begin{bmatrix}
\gamma(s_1,s_0) \\ \vdots \\ \gamma(s_n,s_0) \\ 1
\end{bmatrix}
$$

### 步骤5：最终融合结果
$$
\hat{y}_{final}(s_0) = \hat{y}_{RF}(s_0) + \hat{r}(s_0)
$$

## 关键步骤

1. **特征工程**：构建X(s)特征向量
   - CMAQ值（从最近网格点提取）
   - 经纬度坐标
   - 可选：气象变量（风速、温度、湿度）
   - 可选：土地利用类型编码

2. **随机森林训练**：用十折训练集训练RF模型
   - n_estimators=100, max_depth=10
   - 输入：X_train, y_train（监测值）
   - 输出：预测值 $\hat{y}_{RF}$

3. **残差计算**：$r_i = y_{obs,i} - \hat{y}_{RF,i}$

4. **变异函数拟合**：对残差拟合球形变异函数
   - 使用pykrige或skgstat
   - 参数：块金、基台、变程

5. **克里金预测**：对测试点残差进行克里金插值

6. **结果合成**：$\hat{y}_{final} = \hat{y}_{RF} + \hat{r}$

## 参数清单

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| n_estimators | int | 100 | RF决策树数量 |
| max_depth | int | 10 | RF最大深度 |
| min_samples_leaf | int | 5 | 叶节点最小样本数 |
| variogram_model | str | 'spherical' | 变异函数模型类型 |
| n_neighbors_kriging | int | 12 | 克里金近邻数 |

## 十折验证适配

### 标准模式（适用于RK-Poly等同框架）
```
训练集：9折监测数据 + 对应CMAQ网格值
特征X：CMAQ值 + 坐标
目标y：监测值
预测：1折站点的CMAQ→RF预测→残差克里金
```

### 步骤
1. 读取fold_split_table.csv获取站点折号
2. 对每折：训练RF → 计算残差 → 拟合变异函数 → 克里金预测测试点
3. 汇总10折结果，计算R²、RMSE、|MB|

## 依赖库
- sklearn.ensemble.RandomForestRegressor
- pykrige.ok.OrdinaryKriging 或 skgstat.Variogram
- numpy, pandas

## 实现检查清单
- [ ] 随机森林特征构建
- [ ] RF训练与预测
- [ ] 残差计算
- [ ] 变异函数拟合
- [ ] 克里金残差插值
- [ ] 最终结果合成
- [ ] 十折验证集成
