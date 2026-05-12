# 创新方法指令

## 方法名称
TGK - 传输引导核融合法 (Transport-Guided Kernel Fusion)

## 【创新点】
**核心创新**：利用CMAQ梯度场构建各向异性空间核，使空间相关性沿大气传输方向增强、垂直方向衰减，实现物理引导的空间插值。

**与已有方法的区别**：
- GradientAnisotropicKriging → 使用固定梯度各向异性 → 不动态适应
- CMAQGradKriging → 使用CMAQ梯度作为特征 → 不改变核结构
- TGK → **动态构建各向异性核**，梯度方向决定核形状 → 物理引导

## 文献来源
- 各向异性克里金理论 (Chilès & Delfiner, 2012)
- 大气传输方向与空间相关性 (Berrocal et al., 2010)
- 自适应核方法 (Silverman, 1986)

## 方法概述

传统空间插值（如克里金、IDW）使用**各向同性核**——假设空间相关性仅取决于距离，与方向无关。但在大气污染中，风向和传输方向导致空间相关性具有**方向性**。

TGK的核心思想：
1. 从CMAQ模拟中计算**浓度梯度场** $\nabla C_{CMAQ}$
2. 梯度方向 ≈ 传输方向（污染物沿梯度方向传播）
3. 构建各向异性核：沿梯度方向相关距离长，垂直方向短
4. 使用自适应核进行空间插值

**物理类比**：就像河流中的污染物——沿水流方向相关性强，垂直方向弱。CMAQ梯度场揭示了"大气河流"的方向。

## 输入数据

| 数据 | 格式 | 维度 | 单位 | 说明 |
|-----|------|-----|------|------|
| 监测站PM2.5 | array | (n_obs,) | μg/m³ | 观测值 |
| CMAQ模拟值 | array | (n_grid,) | μg/m³ | 模型输出 |
| 监测站坐标 | array | (n_obs, 2) | 度 | lon, lat |
| CMAQ网格坐标 | array | (n_grid, 2) | 度 | lon, lat |
| 预测目标坐标 | array | (n_pred, 2) | 度 | 待预测位置 |

## 输出数据

| 数据 | 格式 | 单位 | 说明 |
|-----|------|------|------|
| 融合预测值 | array | (n_pred,) | μg/m³ | TGK估计 |
| 预测方差 | array | (n_pred,) | (μg/m³)² | 不确定性 |

## 核心公式

### 1. CMAQ梯度场
$$
\nabla C(s) = \left(\frac{\partial C}{\partial x}, \frac{\partial C}{\partial y}\right)
$$
使用有限差分计算CMAQ网格上的梯度。

### 2. 梯度方向与强度
$$
\theta(s) = \arctan\left(\frac{\partial C / \partial y}{\partial C / \partial x}\right)
$$
$$
|\nabla C(s)| = \sqrt{\left(\frac{\partial C}{\partial x}\right)^2 + \left(\frac{\partial C}{\partial y}\right)^2}
$$

### 3. 各向异性距离度量
对任意两点 $s_i, s_j$，定义各向异性距离：
$$
d_A(s_i, s_j) = \sqrt{(s_i - s_j)^T A(s_i, s_j) (s_i - s_j)}
$$
其中 $A$ 是各向异性矩阵：
$$
A = R^T \begin{bmatrix} 1 & 0 \\ 0 & \lambda^2 \end{bmatrix} R
$$
- $R$ = 旋转矩阵（由梯度方向 $\theta$ 决定）
- $\lambda$ = 各向异性比（$\lambda > 1$ 时沿梯度方向相关距离更长）

### 4. 自适应各向异性比
$$
\lambda(s_i, s_j) = 1 + \alpha \cdot \min(|\nabla C(s_i)|, |\nabla C(s_j)|) / \sigma_{|\nabla C|}
$$
其中：
- $\alpha$: 各向异性强度参数
- $\sigma_{|\nabla C|}$: 梯度强度的标准差（归一化）

**物理意义**：梯度越大 → 各向异性越强 → 沿传输方向相关距离越长。

### 5. 传输引导核函数
$$
K(s_i, s_j) = \exp\left(-\frac{d_A(s_i, s_j)^2}{2\ell^2}\right)
$$
其中 $\ell$ 是长度尺度参数。

### 6. 残差建模
$$
r(s) = PM2.5_{obs}(s) - f(CMAQ(s))
$$
$$
f(C) = \beta_0 + \beta_1 C + \beta_2 C^2
$$
（二次多项式拟合CMAQ与观测的系统偏差）

### 7. TGK预测
$$
\hat{r}(s_0) = k_*^T (K + \sigma_n^2 I)^{-1} r
$$
$$
\hat{y}(s_0) = f(CMAQ(s_0)) + \hat{r}(s_0)
$$

其中：
- $k_*$: 预测点与训练点的核向量
- $K$: 训练点之间的核矩阵
- $r$: 训练残差向量
- $\sigma_n^2$: 噪声方差

## 关键步骤

### Step 1: CMAQ梯度计算
```python
# 从CMAQ网格计算梯度
grad_x, grad_y = np.gradient(cmaq_grid, dx, dy)
grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
grad_direction = np.arctan2(grad_y, grad_x)
```

### Step 2: 系统偏差建模
```python
# 二次多项式拟合CMAQ与观测的关系
from numpy.polynomial import polynomial as P
coeffs = P.polyfit(cmaq_at_stations, observations, 2)
residuals = observations - P.polyval(cmaq_at_stations, coeffs)
```

### Step 3: 各向异性核构建
```python
def anisotropic_distance(si, sj, grad_i, grad_j, alpha, ell):
    # 计算局部梯度方向
    theta = (grad_direction_at(si) + grad_direction_at(sj)) / 2
    grad_strength = min(|grad_i|, |grad_j|)

    # 各向异性比
    lam = 1 + alpha * grad_strength / grad_std

    # 旋转矩阵
    R = rotation_matrix(theta)

    # 各向异性矩阵
    A = R.T @ diag(1, lam**2) @ R

    # 各向异性距离
    diff = si - sj
    return sqrt(diff.T @ A @ diff)
```

### Step 4: 核矩阵构建与求解
```python
# 构建核矩阵
K = np.zeros((n_train, n_train))
for i in range(n_train):
    for j in range(n_train):
        K[i,j] = exp(-anisotropic_distance(si, sj)**2 / (2*ell**2))

# 添加噪声项
K_reg = K + sigma_n**2 * np.eye(n_train)

# 求解权重
weights = solve(K_reg, residuals)
```

### Step 5: 预测
```python
# 对每个预测点
k_star = compute_kernel_vector(pred_point, train_points)
residual_pred = k_star @ weights
y_pred = polynomial_pred(cmaq_pred) + residual_pred
```

## 参数清单

| 参数名 | 类型 | 默认值 | 说明 |
|-------|------|-------|------|
| alpha | float | 2.0 | 各向异性强度 |
| length_scale | float | auto | 核长度尺度（通过变异函数拟合） |
| sigma_n | float | auto | 噪声标准差 |
| poly_degree | int | 2 | 多项式偏差校正阶数 |
| grad_smooth | int | 3 | 梯度平滑窗口大小 |
| n_neighbors | int | 30 | 近邻数量（大规模计算时使用） |

## 适配系统格式

### 十折验证模式（标准模式）
```python
# 输入
train_monitor = fold_data['train_monitor']   # (n_train,)
train_cmaq = fold_data['train_cmaq']         # (n_train,)
train_coords = fold_data['train_coords']     # (n_train, 2)
test_cmaq = fold_data['test_cmaq']           # (n_test,)
test_coords = fold_data['test_coords']       # (n_test, 2)
cmaq_grid = fold_data['cmaq_grid']           # (lat, lon) 全域CMAQ
cmaq_coords = fold_data['cmaq_coords']       # (n_grid, 2)

# Step 1: 计算CMAQ梯度场
grad_x, grad_y = compute_gradient(cmaq_grid, cmaq_coords)

# Step 2: 系统偏差校正
poly_coeffs = fit_polynomial(train_cmaq, train_monitor, degree=2)
residuals = train_monitor - polynomial(train_cmaq, poly_coeffs)

# Step 3: 构建各向异性核
K = build_anisotropic_kernel(train_coords, grad_x, grad_y, alpha=2.0)

# Step 4: 求解并预测
weights = solve(K + sigma_n**2 * I, residuals)
pred_residual = kernel_vector(test_coords, train_coords) @ weights
pred = polynomial(test_cmaq, poly_coeffs) + pred_residual
```

## 随机性
- [x] 否（确定性方法，核构建和求解均为确定性过程）

## 方法指纹
MD5: transport_guided_anisotropic_kernel_cmaq_gradient_v1

## 创新判定

| 判定项 | 结果 |
|-------|------|
| 是否使用权重学习（Ridge/Lasso等）？ | 否（使用核方法求解） |
| 是否物理可解释？ | 是（大气传输方向引导核形状） |
| 创新类型 | 主级创新候选 |

## 预期性能

基于物理引导核的优势：
- 预期R²: 0.910-0.920（优于各向同性方法）
- 预期RMSE: 低于普通克里金
- 优势：物理引导、方向性相关、梯度自适应
- 风险：梯度计算可能有噪声；各向异性参数选择敏感

## 实现检查清单
- [ ] CMAQ梯度场计算
- [ ] 各向异性距离度量
- [ ] 传输引导核函数
- [ ] 多项式偏差校正
- [ ] 核矩阵构建与求解
- [ ] 大规模近邻优化
- [ ] 十折验证适配
- [ ] 与AdvancedRK基准比较
