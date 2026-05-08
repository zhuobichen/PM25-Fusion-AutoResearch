# 【可执行方法规范】

## 方法名称
物理引导扩散模型PDE降尺度法 (Physics-Guided Diffusion Model for PDE Downscaling, PGDM)

## 文献来源
- 论文标题：Generative downscaling of PDE solvers with physics-guided diffusion models
- 作者：Yulong Lu, Wuzhe Xu
- 年份：2024年
- arXiv: 2404.05009

## 核心公式

### 1. 前向扩散
$$
q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t} x_{t-1}, \beta_t I)
$$

### 2. 反向去噪
$$
p_\theta(x_{t-1} | x_t, y) = \mathcal{N}(\mu_\theta(x_t, y, t), \sigma_t^2 I)
$$
其中 $y$ 为低分辨率输入。

### 3. 物理残差损失
$$
\mathcal{L}_{physics} = \|N_\alpha(x) - f\|^2
$$
其中 $N_\alpha$ 为PDE算子，$f$ 为源项。

### 4. 总损失
$$
\mathcal{L} = \mathcal{L}_{diffusion} + \lambda_p \mathcal{L}_{physics}
$$

## 参数清单

| 参数名 | 类型 | 默认值 | 说明 |
|-------|------|-------|------|
| diffusion_steps | int | 1000 | 扩散步数 |
| lambda_physics | float | 0.1 | 物理损失权重 |
| scaling_factor | int | 8-16x | 降尺度倍数 |
| pde_type | str | 'navier-stokes' | PDE类型 |

## 数据规格

### 输入
| 数据 | 格式 | 维度 | 单位 |
|-----|------|-----|------|
| 低分辨率PDE解 | array | (H_lr, W_lr, T) | - |
| 物理参数 | array | (H_lr, W_lr, n_params) | - |

### 输出
| 数据 | 格式 | 单位 |
|-----|------|------|
| 高分辨率PDE解 | array | - |

## 实现步骤

1. **数据准备**：低-高分辨率PDE解对
2. **扩散模型构建**：条件U-Net
3. **物理约束设计**：PDE算子残差
4. **训练**：扩散+物理联合损失
5. **微调**：最小化物理偏差
6. **生成**：高分辨率PDE解

## 随机性
- [x] 是（扩散模型带随机采样）

## 方法指纹
MD5: physics_guided_diffusion_pde_downscaling
