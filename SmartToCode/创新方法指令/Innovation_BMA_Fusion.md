# 创新方法指令

## 方法名称
BMA-Fusion - 贝叶斯模型平均融合法 (Bayesian Model Averaging for PM2.5 Fusion)

## 【创新点】
**核心创新**：使用贝叶斯后验模型概率（而非Ridge/Lasso回归权重）组合多个基础方法，实现概率最优融合。

**与排除方法的区别**：
- StackingEnsemble → 使用Ridge元学习器学习权重 → **排除**
- SuperStackingEnsemble → 使用线性回归学习权重 → **排除**
- BMA-Fusion → 使用贝叶斯后验模型概率 → **保留**（非回归权重学习，是概率推断）

## 文献来源
- 贝叶斯模型平均经典方法 (Hoeting et al., 1999; Raftery et al., 2005)
- Winker (2000) - BMA组合预测
- 应用于大气科学领域 (Sloughter et al., 2007)

## 方法概述

BMA-Fusion的核心思想：
1. 有M个基础方法（如VNA, eVNA, aVNA, RK-Poly等）
2. 每个方法的预测视为一个"模型"
3. 使用贝叶斯定理计算每个模型的后验概率
4. 最终预测 = 各方法预测的后验概率加权平均
5. 同时提供预测不确定性（后验方差）

**物理可解释性**：后验模型概率反映每个方法在历史数据上的预测能力，概率越高说明该方法越可信。这比Ridge/Lasso的回归权重更有物理意义——概率是正的且归一化，不会出现负权重。

## 输入数据

| 数据 | 格式 | 维度 | 单位 | 说明 |
|-----|------|-----|------|------|
| 基础方法预测 | list of arrays | M × (n,) | μg/m³ | M个方法的预测值 |
| 监测站观测 | array | (n,) | μg/m³ | 真实观测值 |
| 基础方法训练预测 | list of arrays | M × (n_train,) | μg/m³ | 用于计算后验概率 |

## 输出数据

| 数据 | 格式 | 单位 | 说明 |
|-----|------|------|------|
| BMA融合预测 | array | (n_pred,) | μg/m³ | 后验概率加权平均 |
| 预测方差 | array | (n_pred,) | (μg/m³)² | 不确定性估计 |
| 后验模型概率 | array | (M,) | - | 各方法的可信度 |

## 核心公式

### 1. 贝叶斯模型平均（BMA）
$$
p(y | D) = \sum_{k=1}^{M} p(y | M_k, D) \cdot p(M_k | D)
$$
其中：
- $y$: 待预测的PM2.5值
- $D$: 观测数据
- $M_k$: 第k个基础方法（模型）
- $p(M_k | D)$: 模型k的后验概率（模型权重）
- $p(y | M_k, D)$: 模型k的预测分布

### 2. 后验模型概率
$$
p(M_k | D) = \frac{p(D | M_k) \cdot p(M_k)}{\sum_{j=1}^{M} p(D | M_j) \cdot p(M_j)}
$$
其中 $p(D | M_k)$ 是模型k的边缘似然（模型证据）。

### 3. 模型证据（近似）
使用BIC近似：
$$
\log p(D | M_k) \approx \log p(D | \hat{\theta}_k, M_k) - \frac{d_k}{2} \log n
$$
其中：
- $\hat{\theta}_k$: 模型k的最大似然参数估计
- $d_k$: 模型k的参数数量
- $n$: 观测数量

### 4. BMA预测
$$
\hat{y}_{BMA} = \sum_{k=1}^{M} w_k \cdot \hat{y}_k
$$
其中 $w_k = p(M_k | D)$ 是后验模型概率。

### 5. 预测方差（不确定性）
$$
\sigma^2_{BMA} = \sum_{k=1}^{M} w_k \left(\sigma_k^2 + (\hat{y}_k - \hat{y}_{BMA})^2\right)
$$
其中 $\sigma_k^2$ 是模型k的预测方差。

### 6. 对数评分（用于模型权重更新）
$$
\text{logscore}_k = -\frac{1}{n} \sum_{i=1}^{n} \log p(y_i | \hat{y}_k, \sigma_k)
$$
预测越准，对数评分越高，后验权重越大。

## 关键步骤

### Step 1: 基础方法训练
分别训练M个基础方法（如VNA, eVNA, aVNA, RK-Poly, AdvancedRK等）。

### Step 2: 交叉验证预测
使用k折交叉验证，获取每个方法在验证集上的预测值和残差。

### Step 3: 计算模型证据
对每个方法，基于其交叉验证残差计算BIC：
$$
BIC_k = n \log(\text{MSE}_k) + d_k \log(n)
$$
$$
\log p(D | M_k) \approx -\frac{1}{2} BIC_k
$$

### Step 4: 计算后验模型概率
$$
w_k = \frac{\exp(-\frac{1}{2} BIC_k)}{\sum_{j=1}^{M} \exp(-\frac{1}{2} BIC_j)}
$$

### Step 5: BMA预测
$$
\hat{y}_{BMA}(s_0) = \sum_{k=1}^{M} w_k \cdot \hat{y}_k(s_0)
$$

### Step 6: 不确定性估计
$$
\sigma^2_{BMA}(s_0) = \sum_{k=1}^{M} w_k \left(\sigma_k^2 + (\hat{y}_k(s_0) - \hat{y}_{BMA}(s_0))^2\right)
$$

## 参数清单

| 参数名 | 类型 | 默认值 | 说明 |
|-------|------|-------|------|
| base_methods | list | ['VNA', 'eVNA', 'aVNA', 'RK-Poly'] | 基础方法列表 |
| n_methods | int | 4 | 基础方法数量 |
| cv_folds | int | 10 | 交叉验证折数 |
| prior | str | 'uniform' | 先验模型概率（uniform/informed） |
| evidence_approx | str | 'BIC' | 模型证据近似方法（BIC/AIC/CV） |
| temperature | float | 1.0 | 温度参数（控制权重集中度） |

## 适配系统格式

### 十折验证模式（标准模式）
```python
# Step 1: 在训练集上训练各基础方法
for method in base_methods:
    method.fit(train_data)

# Step 2: 交叉验证获取各方法的验证预测
cv_predictions = cross_validate(base_methods, train_data, cv=10)

# Step 3: 计算后验模型概率
posterior_probs = compute_bma_weights(cv_predictions, train_obs)

# Step 4: 对测试集进行BMA预测
test_predictions = [method.predict(test_data) for method in base_methods]
bma_pred = sum(w * pred for w, pred in zip(posterior_probs, test_predictions))

# Step 5: 计算不确定性
bma_var = sum(w * (sigma_k**2 + (pred_k - bma_pred)**2)
              for w, sigma_k, pred_k in zip(posterior_probs, sigmas, test_predictions))
```

### 创新判定验证
```python
# 与最优基准（VNA）比较
vna_pred = VNA.predict(test_data)
bma_pred = BMA.predict(test_data)

# R²提升 >= 0.01 → 主级创新
# R²提升 > 0 → 次级创新
```

## 随机性
- [x] 否（确定性方法，后验概率计算为确定性过程）

## 方法指纹
MD5: bayesian_model_averaging_fusion_bic_evidence_v1

## 创新判定

| 判定项 | 结果 |
|-------|------|
| 是否使用权重学习（Ridge/Lasso等）？ | 否（使用贝叶斯后验概率） |
| 是否物理可解释？ | 是（后验概率反映方法可信度） |
| 创新类型 | 主级创新候选 |

## 预期性能

基于BMA理论优势：
- 预期R²: 0.910-0.920（优于任何单一基础方法）
- 预期RMSE: 低于最优单一方法
- 优势：自动选择最优方法组合，提供不确定性估计
- 风险：基础方法高度相关时权重可能退化；计算开销较大

## 实现检查清单
- [ ] 基础方法接口适配
- [ ] BIC模型证据计算
- [ ] 后验模型概率归一化
- [ ] BMA预测与不确定性
- [ ] 十折验证集成
- [ ] 与AdvancedRK基准比较
