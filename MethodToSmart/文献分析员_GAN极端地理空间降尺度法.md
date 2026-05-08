# 【可执行方法规范】

## 方法名称
生成对抗网络极端地理空间降尺度法 (GAN-based Extreme Geospatial Downscaling)

## 文献来源
- 论文标题：Generative Adversarial Models for Extreme Geospatial Downscaling
- 作者：Guiye Li, Guofeng Cao
- 年份：2024年
- arXiv: 2402.14049

## 核心公式

### 1. 条件GAN框架
$$
\min_G \max_D V(D,G) = \mathbb{E}_{y \sim p_{data}(y)}[\log D(y|x)] + \mathbb{E}_{z \sim p_z(z)}[\log(1-D(G(x,z)|x))]
$$
其中 $x$ 为低分辨率输入，$y$ 为高分辨率目标，$z$ 为随机噪声。

### 2. 生成器损失
$$
\mathcal{L}_G = \mathcal{L}_{adv} + \lambda_{L1} \mathcal{L}_{L1} + \lambda_{p} \mathcal{L}_{physics}
$$
其中 $\mathcal{L}_{L1} = \|y - G(x,z)\|_1$ 为像素级损失。

### 3. 判别器损失
$$
\mathcal{L}_D = -\mathbb{E}[\log D(y)] - \mathbb{E}[\log(1-D(G(x,z)))]
$$

### 4. 不确定性量化
通过多次采样 $z$ 生成预测集合：
$$
\{y^{(1)}, y^{(2)}, ..., y^{(n)}\} = \{G(x,z_1), G(x,z_2), ..., G(x,z_n)\}
$$

## 参数清单

| 参数名 | 类型 | 默认值 | 说明 |
|-------|------|-------|------|
| scaling_factor | int | 8-64x | 降尺度倍数 |
| latent_dim | int | 100 | 隐空间维度 |
| lambda_L1 | float | 100 | L1损失权重 |
| lambda_physics | float | 10 | 物理约束权重 |
| learning_rate | float | 2e-4 | 学习率 |
| batch_size | int | 16 | 批大小 |

## 数据规格

### 输入
| 数据 | 格式 | 维度 | 单位 |
|-----|------|-----|------|
| 低分辨率气候数据 | array | (H_lr, W_lr, C) | - |
| 地形数据 | array | (H_hr, W_hr, 1) | m |

### 输出
| 数据 | 格式 | 单位 |
|-----|------|------|
| 高分辨率预测 | array | - |

## 实现步骤

1. **数据准备**：收集成对的低-高分辨率气候数据
2. **生成器构建**：U-Net架构，输入低分辨率+噪声，输出高分辨率
3. **判别器构建**：PatchGAN判别器，判断局部区域真实性
4. **对抗训练**：交替优化生成器和判别器
5. **物理约束**：添加梯度一致性、质量守恒等物理损失
6. **概率预测**：多次采样生成不确定性估计

## 随机性
- [x] 是（GAN生成器带随机噪声输入）

## 方法指纹
MD5: gan_extreme_geospatial_downscaling
