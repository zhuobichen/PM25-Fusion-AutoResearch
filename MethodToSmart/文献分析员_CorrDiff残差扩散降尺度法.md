# 【可执行方法规范】

## 方法名称
残差校正扩散模型降尺度法 (Corrective Diffusion, CorrDiff)

## 文献来源
- 论文标题：China Regional 3km Downscaling Based on Residual Corrective Diffusion Model
- 作者：Honglu Sun, Hao Jing, Zhixiang Dai, Sa Xiao, Wei Xue, Jian Sun, Qifeng Lu
- 年份：2025年

## 核心公式

### 1. 双线性插值
$$
x_{bilinear} = bilinear(x_i)
$$
将25km低分辨率输入双线性插值到3km高分辨率网格。

### 2. 回归模型（UNet）
$$
\hat{y}_{regress} = f(x_{bilinear}; \theta)
$$
UNet回归模型直接预测高分辨率目标。

### 3.  CorrDiff两阶段预测
$$
\hat{y}_{CorrDiff} = g(x_{bilinear}, \hat{y}_{regress}, \epsilon) + \hat{y}_{regress}
$$
其中 $g$ 为扩散模型，$\epsilon \sim N(0,1)$ 为随机噪声。

### 4. 扩散模型（EDM）
Elucidated Diffusion Model通过去噪过程学习从噪声到数据的条件分布。

### 5. 网格尺寸关系
- 低分辨率（输入）：192×288（25km ERA5）
- 高分辨率（输出）：1600×2400（3km CMA-RRA）
- 区域：中国（12.25-60°N, 70-141.75°E）

### 6. 概率预测
通过多次采样不同随机噪声 $\epsilon$，生成预测集合：
$$
\{\hat{y}^{(1)}, \hat{y}^{(2)}, ..., \hat{y}^{(n)}\}
$$
可用于量化不确定性。

## 参数清单

| 参数名 | 类型 | 默认值 | 说明 |
|-------|------|-------|------|
| input_resolution | float | 25km | 低分辨率输入网格 |
| output_resolution | float | 3km | 高分辨率输出网格 |
| UNet_embedding | list | [128,256,512,512,1024] | UNet嵌入维度 |
| batch_size | int | 64（部分模型8） | 批大小 |
| epochs_regression | int | 35-120 | 回归模型训练轮数 |
| epochs_diffusion | int | 59-285 | 扩散模型训练轮数 |
| optimizer | str | Adam | 优化器 |
| learning_rate | float | - | 学习率 |

## 实现步骤

### 数据准备
1. 收集2019-2022年训练数据（每日8个时次：00/03/06/09/12/15/18/21 UTC）
2. 低分辨率输入：ERA5 25km再分析数据（192×288）
3. 高分辨率目标：CMA-RRA 3km再分析数据（1600×2400）
4. Min-Max归一化到[-1,1]区间

### 第一阶段：回归模型训练
1. 对低分辨率输入进行双线性插值到高分辨率网格
2. 构建UNet网络（6层编码器+6层解码器）
3. 采用残差学习策略：预测 $y - x_{bilinear}$ 而非直接预测 $y$
4. 使用MSE损失函数训练

### 第二阶段：扩散模型训练
1. 基于EDM（Elucidated Diffusion Model）架构
2. 以回归预测为条件输入
3. 学习残差的概率分布
4. 通过去噪过程生成细尺度细节

### 推理预测
1. 对新输入进行双线性插值
2. 运行回归模型得到基础预测
3. 运行扩散模型多次采样
4. 融合结果：$\hat{y} = \hat{y}_{diffusion} + \hat{y}_{regress}$

### 变量配置（4种组合）
| 组合 | 输入变量 | 输出变量 |
|-----|---------|---------|
| 1 | 10m风、2m温度、比湿、位势高度（多压层） | 同输入 |
| 2 | 添加雷达反射率、地形 | 添加雷达反射率 |
| 3 | 去除100/200hPa高层变量 | 去除高层输出 |
| 4 | 兼容SFF模型的变量子集 | 24个变量 |

## 方法特点

1. **两阶段架构**：回归模型提供条件均值，扩散模型学习细尺度校正
2. **残差学习**：加速收敛、提高精度
3. **概率预测**：多次采样产生不确定性估计
4. **物理一致性**：不同于GAN，扩散模型产生物理一致的结果

## 性能指标

| 变量 | CorrDiff MAE | CMA-MESO MAE |
|-----|-------------|-------------|
| 2m温度 | 0.8K | 1.2K |
| 10m风速 | 1.5m/s | 2.0m/s |
| 500hPa风 | 1.8m/s | 2.5m/s |
| 850hPa温度 | 0.9K | 1.4K |

## 应用场景

- 气象要素降尺度（25km→3km）
- CMAQ模式输出的统计降尺度
- 生成高分辨率天气预报
- 卫星数据与模式数据的融合

## 数据规格

### 输入
| 数据 | 格式 | 维度 | 单位 |
|-----|------|-----|------|
| ERA5再分析数据 | array | (192, 288, n_vars) | - |
| CMA-RRA目标数据 | array | (1600, 2400, n_vars) | - |

### 输出
| 数据 | 格式 | 单位 |
|-----|------|------|
| 高分辨率预测 | array | - |

## 方法指纹
MD5: corrective_diffusion_corrdiff_downscaling

## 随机性
- [x] 是（扩散模型带随机噪声采样）
