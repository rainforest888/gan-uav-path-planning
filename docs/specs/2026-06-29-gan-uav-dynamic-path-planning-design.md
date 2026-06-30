# 基于条件 WGAN-GP 潜空间连续演化的无人机动态路径规划 —— 设计文档

**日期**: 2026-06-29
**项目**: 西工智航-基于生成对抗网络的无人机智能路径规划方法研究
**作者**: 任德霖
**状态**: 已定稿

---

## 1. 概述

### 1.1 目标

构建一个基于生成对抗网络（GAN）的三维无人机动态路径规划系统。核心思路是利用 Conditional WGAN-GP 学习环境的潜空间表示，并通过**潜空间连续轨迹演化机制**处理动态障碍物——不丢弃旧路径重新生成，而是在潜空间中连续移动潜编码，使路径平滑变形。

### 1.2 应用场景

- **第一阶段（可行性验证）**: 简单通用 3D 场景（球/柱障碍物）
- **第二阶段（目标场景）**: 森林/山地野外环境——不规则障碍物，地形起伏
- **动态障碍物**: 环境中存在移动的障碍物（移动速度 ≤ 无人机最大飞行速度的 50%）

### 1.3 系统定位

| 我们不做 | 我们做 |
|---------|--------|
| 跟扩散模型比轨迹精度 | 用 GAN 单次前向 + 潜空间优化的极低延迟做动态重规划 |
| 跟 DRL 比动态避障成功率 | 输出完整全局路径，可解释、可约束、可验证 |
| 声称"首次将 GAN 用于路径规划" | 首次将 CWGAN-GP 潜空间范式引入 3D 无人机，并提出潜空间连续演化 |

---

## 2. 技术栈

| 项 | 选择 | 理由 |
|----|------|------|
| 仿真平台 | AirSim + Unreal Engine | Windows 兼容，UE 渲染森林场景逼真，可视化好 |
| 深度学习框架 | Python + PyTorch | GAN 生态最成熟，社区资源最丰富 |
| 路径表示 | 稀疏关键航点 (K×3) + Minimum Snap 插值 | 低维可控 (K≈8-12)，天然平滑，与深蓝学院课程衔接 |
| GAN 架构 | Conditional WGAN-GP | 训练稳定，Gradient Penalty 防止崩塌，路径规划文献验证有效 |
| 环境编码 | 3D 体素网格 (32³) → 3D Conv | 第一阶段使用；后续可替换为 PointNet 处理稀疏森林点云 |
| 动态处理 | 潜空间梯度优化 | 连续演化，保持前后路径一致性 |

---

## 3. 网络架构

### 3.1 环境编码器 E_env

```
输入:  3D 体素地图 V ∈ {0,1}^{32×32×32}（1=障碍物）
       起点坐标 s ∈ R³, 终点坐标 g ∈ R³（归一化到 [0,1]）

结构:
  V → 3D Conv(3→32, k=3, s=2) → BN → LeakyReLU
     → 3D Conv(32→64, k=3, s=2) → BN → LeakyReLU
     → 3D Conv(64→128, k=3, s=2) → BN → LeakyReLU
     → Global Average Pooling
     → 与 s,g 拼接 → FC(256+6→256) → LeakyReLU

输出:  条件向量 c ∈ R^{256}
```

### 3.2 生成器 G(z, c)

```
输入:  噪声 z ∈ R^{128} ~ N(0, I)
       条件 c ∈ R^{256}

结构:
  z + c → Concat → FC(384→512) → BN → LeakyReLU
        → FC(512→1024) → BN → LeakyReLU
        → FC(1024→2048) → BN → LeakyReLU
        → FC(2048→K×3)

其中 K 个输出通道对应 K 个关键航点的 (x, y, z) 坐标。
坐标通过 Tanh 激活限制在 [-1, 1]，再反归一化到世界坐标系。

输出:  K 个关键航点 ∈ R^{K×3}
       （后续经 Minimum Snap 插值生成 N 点光滑轨迹, N≈100）
```

### 3.3 判别器/Critic D(τ, c)

```
输入:  轨迹 τ ∈ R^{N×3}（N 个采样点）
       条件 c ∈ R^{256}

结构:
  τ → 1D Conv(3→64, k=5, s=2) → LeakyReLU
     → 1D Conv(64→128, k=5, s=2) → LeakyReLU
     → 1D Conv(128→256, k=3, s=2) → LeakyReLU
     → Global Average Pooling → 256-d 特征
     → 与 c 拼接 → FC(512→256) → LeakyReLU
     → FC(256→1)

输出:  标量分数（无 Sigmoid，WGAN-GP 的 critic）
```

### 3.4 路径编码器 E_path(τ, c)

```
输入:  轨迹 τ ∈ R^{N×3}
       条件 c ∈ R^{256}

结构:
  τ → 1D Conv(3→64, k=5, s=2) → LeakyReLU
     → 1D Conv(64→128, k=5, s=2) → LeakyReLU
     → 1D Conv(128→256, k=3, s=2) → LeakyReLU
     → Global Average Pooling
     → 与 c 拼接 → FC(512→256) → LeakyReLU
     → FC(256→128)

输出:  潜编码 ẑ ∈ R^{128}

约束:  G(E_path(τ, c), c) ≈ τ   （路径重建一致性）
```

> **注意**: 这是此架构与 Ando 2023 / Ocampo 2025 最关键的差异之一。Ando 只有 G 和 D，没有 E_path。E_path 的存在使得在线阶段可以将当前路径快速定位到潜空间中，然后做连续更新。

### 3.5 Minimum Snap 插值（训练与推理的不同处理）

```
训练阶段：
  G 输出 K 个关键航点 (含坐标和到达时间)
  → 使用可微三次样条 (Cubic Spline) 插值到 N 个轨迹点
  → 随后在此 N 点轨迹上计算碰撞损失 / 平滑度损失
  → 梯度可回传至 G

  ⚠ 训练阶段不用完整 Minimum Snap QP 求解，因为 QP 不可微。
  三次样条是纯矩阵乘法，天然可微，且输出足够光滑用于碰撞检测。

推理阶段：
  G 输出 K 个关键航点
  → 完整 Minimum Snap QP 求解（各航点间分配均匀时间）
  → 输出满足四旋翼动力学的最优光滑轨迹
  → 发送给飞控执行

这样设计的理由：
  - 训练需要梯度回传 → 可微三次样条做代理
  - 推理需要动力学可行性 → 完整 Minimum Snap
  - 两者轨迹形状高度相似，训练代理不会造成严重的分布偏移
```

---

## 4. 损失函数

### 4.1 WGAN-GP 对抗损失

```
L_adv_D = mean[D(G(z, c), c)] - mean[D(τ_real, c)]
L_adv_G = -mean[D(G(z, c), c)]

Gradient Penalty:
  x̂ = ε·τ_real + (1-ε)·G(z, c),  ε ~ U(0,1)
  GP = mean[(||∇_x̂ D(x̂, c)||₂ - 1)²]
  λ_gp = 10

L_D = L_adv_D + λ_gp · GP
```

### 4.2 路径约束损失（直接加在 G 上）

```
碰撞损失:
  注意：碰撞检测在 Minimum Snap 插值后的完整轨迹（N 点）上进行，
  而非仅在 K 个关键航点上。因为关键航点之间可能穿越障碍物。
  做法：G 输出 K 个关键航点 → Minimum Snap 插值 → N 点轨迹 → 体素采样
  L_collision = mean[ 体素查询(V, τ_point) ]  # 1=碰撞, 0=自由
  λ_col = 10.0

路径长度正则:
  L_length = mean[ Σᵢ||τ_i - τ_{i-1}||₂ ]
  λ_len = 0.5

平滑度正则（加速度最小化）:
  L_smooth = mean[ Σᵢ||τ_i - 2τ_{i-1} + τ_{i-2}||² ]
  λ_smooth = 1.0
```

### 4.3 路径重建损失（训练 E_path）

```
L_recon = ||G(E_path(τ_real, c), c) - τ_real||²
λ_recon = 1.0
```

### 4.4 潜空间凸性正则化（核心创新训练损失）

```
对每个 batch 随机采样两对 (z_a, τ_a) 和 (z_b, τ_b):
  α ~ U(0,1)
  z_interp = (1-α)·z_a + α·z_b
  τ_interp = G(z_interp, c_a)  # 使用第一个场景的条件

  计算 τ_interp 在 c_a 场景下的碰撞损失

L_convexity = mean[ 碰撞(τ_interp) ]
λ_convexity = 2.0
```

> 目的: 显式约束潜空间的几何性质——任意两点的线性插值应对应无碰撞路径。Ando 2023 发现这个性质隐式存在，但我们将其变为训练目标，使潜空间更适合动态演化。

### 4.5 总损失汇总

```
L_G = L_adv_G + λ_col·L_collision + λ_len·L_length + λ_smooth·L_smooth
L_D = L_adv_D + λ_gp·GP
L_E = L_recon + λ_convexity·L_convexity  （E_path 和 G 联合训练）
```

---

## 5. 训练流程

### 5.1 数据生成

1. 随机生成 3D 体素障碍物场景（位置、大小、密度随机，5000-10000 个场景）
2. 对每个场景随机生成起点-终点对
3. 用 A*（3D 网格搜索）生成 ground truth 路径，作为 τ_real
4. 数据格式: (体素地图 V, 起点 s, 终点 g, 真实路径 τ_real)

### 5.2 训练超参数

| 参数 | 值 |
|------|-----|
| Batch size | 64 |
| G 学习率 | 1e-4 |
| D 学习率 | 1e-4 |
| E_path 学习率 | 1e-4 |
| D 更新频率 | n_critic = 5 (每 5 次 D 更新 1 次 G/E 更新) |
| λ_gp | 10 |
| 优化器 | Adam (β₁=0, β₂=0.9) ← WGAN-GP 标准设置，β₁=0 保证稳定 |
| 训练轮数 | ~500 epochs（第一阶段） |
| 潜空间维度 | 128 |
| 关键航点数 K | 10 |
| 轨迹采样点数 N | 100 |

### 5.3 训练阶段划分

**阶段 A（数据驱动）**: 用 A* 生成的 ground truth 路径训练，目标是验证整个 pipeline 可运行。

**阶段 B（自监督增强）**: 减小 λ_recon 权重，增大碰撞/长度/平滑度的约束权重，尝试让 G 在无 A* 标签的情况下学习。

---

## 6. 在线推理（动态飞行）

### 6.1 初始化

```
t = 0:
  感知初始环境 → 构建体素地图 V₀
  z₀ ~ N(0, I) 或 z₀ = E_path(τ_init, c₀)  # 如果已有初始路径
  τ₀ = G(z₀, c₀) → Minimum Snap 插值 → 完整轨迹
```

### 6.2 重规划循环（频率: ~10Hz，即每 100ms）

```
1. 更新体素地图 V_t（障碍物位置变化 + 自身位置更新）
2. 构建新条件 c_t = E_env(V_t, s_new, g)
3. 潜空间优化（5 步梯度下降）:
   z_cur = z_{t-1}  （初始化 = 上一轮潜编码）
   for step in 1..5:
     τ_cur = G(z_cur, c_t)
     loss = 碰撞损失(τ_cur, V_t) + λ_continuity·||z_cur - z_{t-1}||²
     z_cur = z_cur - lr·∇loss
   z_t = z_cur
4. τ_t = G(z_t, c_t) → Minimum Snap → 发送给飞控
5. 执行轨迹的第一步，等待下一个周期
```

### 6.3 关键参数

| 参数 | 值 | 说明 |
|------|-----|------|
| λ_continuity | 0.5 | 潜编码连续性约束权重，越大前后路径越一致 |
| 在线优化学习率 | 0.01 | 梯度下降步长（远大于训练 lr，因为只做 5 步） |
| 在线优化步数 | 5 | 每轮梯度下降步数 |

### 6.4 设计说明

- **||z_cur - z_{t-1}||² 约束**: 保证潜编码不会跳太远，前后路径平滑过渡。这是"连续演化"的核心。
- **5 步梯度下降**: 约 5ms 延迟（相比扩散模型 100+ 步去噪的 100-500ms），保留实时性优势。
- **更新频率 10Hz**: 可在 AirSim 中调整为 20Hz（5ms 优化 + 2ms 前向 = 7ms < 50ms 周期）。

---

## 7. 实验设计

### 7.1 Baseline 对比

| 方法 | 描述 |
|------|------|
| A* 定期重规划 | 经典网格搜索，每 100ms 重新搜索 |
| cGAN 静态重生成（方案 A） | 每周期重新随机采样 z，生成新路径 |
| CWGAN-GP 静态重生成 | WGAN-GP 变体，同样独立重采样 |
| **我们的方法** | CWGAN-GP + 潜空间连续演化 + 凸性正则化 |

### 7.2 评估指标

| 指标 | 含义 |
|------|------|
| 成功率 | 到达目标且无碰撞的比例 |
| 路径长度 | 总飞行距离 |
| 重规划延迟 | 每次重规划耗时 (ms) |
| 轨迹平滑度 | 加速度 jerk 的均值 |
| 路径一致性 | 连续两帧路径之间的 Hausdorff 距离（越小越平滑） |
| 碰撞次数 | 飞行过程中的碰撞事件数 |

### 7.3 消融实验

| 消融项 | 目的 |
|--------|------|
| 去掉 E_path（随机初始化 z₀） | 验证编码器对在线初始化的作用 |
| 去掉潜空间连续性约束（λ_continuity=0） | 验证连续性约束对路径平滑度的贡献 |
| 去掉凸性正则化（λ_convexity=0） | 验证显式凸性约束的必要性 |
| 变 K 值（K=5, 10, 15, 20） | 关键航点数量对路径质量的影响 |
| 变 λ_continuity | 连续性约束强度的敏感性分析 |

### 7.4 场景设计

**静态场景**: 障碍物密度递增（稀疏 → 中等 → 密集），每种密度 100 个随机场景，评估路径质量和成功率。

**动态场景**: 1-5 个移动障碍物，速度 1-3 m/s，直/圆/蛇形轨迹。评估动态避障成功率和重规划延迟。

**森林场景**: AirSim 中导入树木模型构建野外场景，测试方法在目标环境的表现。

---

## 8. 与最相关文献的差异

|  | Ando 2023 | Ocampo 2025 | VAE-GAN 2025 | 我们的方法 |
|---|-----------|------------|-------------|----------|
| 平台 | UR5e 机械臂 | Baxter 机械臂 | 2D 地面 | **3D 四旋翼** |
| 环境输入 | 2D 深度图 | RGB-D 图像 | 2D 数据集 | **3D 体素/点云** |
| 生成模型 | cGAN | WGAN-GP | VAE+GAN | **CWGAN-GP** |
| 路径编码器 E | 无 | 无 | VAE编码器 | **E_path + 重建损失** |
| 动态处理 | 无 | 无 | 重生成 | **潜空间连续演化** |
| 凸性正则化 | 隐式 | 隐式 | 未讨论 | **显式训练正则化** |
| 动力学 | 无 | 无 | B-spline | **Min Snap 四旋翼动力学** |

---

## 9. 创新点总结

1. **领域迁移创新**: 首次将 Conditional WGAN-GP 潜空间路径规划范式从机械臂迁移到 3D 四旋翼无人机导航，处理更高维度的环境和不规则障碍物。

2. **机制创新——潜空间连续轨迹演化**: 提出在潜空间中做连续梯度优化而非丢弃-重生成，使动态路径更新天然平滑，解决了独立重生成中前后路径不一致的问题。

3. **方法创新——路径编码器 + 显式凸性正则化**: 引入路径编码器 E_path 实现路径→潜空间的快速映射，并用显式凸性正则化损失确保潜空间的几何性质适合连续演化。这两个组件在现有潜空间 GAN 规划文献中是首次出现。

---

## 10. 可行性评估

### 10.1 关键风险与缓解

| 风险 | 等级 | 缓解 |
|------|------|------|
| 3D GAN 训练不稳定 | 中 | WGAN-GP 机制已验证；先用 16³ 小体素验证，逐步增大；监控 critic loss |
| 潜空间凸性正则化不收敛 | 中高 | 先进后加——先验证基础 CWGAN-GP+E_path 可行，再加 L_convexity |
| 梯度优化延迟超预期 | 低 | 5 步优化 ≈ 一次额外前向+反向传播；可减少步数或减小网络 |
| AirSim 配置复杂 | 中 | 先用纯 Python 简单方块场景验证算法，再接入 AirSim |
| 扩散模型审稿质疑 | 中 | 设计 GAN vs Diffusion 推理延迟对比实验，明确讨论优劣 |
| 与 DRL 方法对比 | 中 | 引用 DRL 相关工作，强调 GAN 的完整路径可解释性优势 |

### 10.2 开发路线

```
Phase 1: 2D 简化验证（2-3周）
  ✓ 2D 平面 + 静态圆形障碍物
  ✓ 验证 CWGAN-GP 能学会生成无碰撞路径
  ✓ 验证 E_path 重建精度
  目的: 快速确认基础架构，debug 方便

Phase 2: 3D 静态场景（2-3周）
  ✓ 3D 体素 + 球/柱障碍物
  ✓ 对比 A* baseline
  目的: 确认 3D 扩展可行

Phase 3: 潜空间演化 + 动态障碍物（3-4周）
  ✓ 加入移动障碍物
  ✓ 验证潜空间连续更新 vs 重生成的差异
  ✓ 消融实验
  目的: 核心创新点验证

Phase 4: AirSim 集成 + 森林场景（3-4周）
  ✓ AirSim 环境搭建
  ✓ 森林/山地场景
  目的: 目标场景验证

Phase 5: 真机测试（视课程进度）
  ✓ 深蓝学院四旋翼对接
  ✓ 室内简单场景试飞
```

---

## 11. 参考文献

1. Ando, T., et al. "Learning-based collision-free planning on arbitrary optimization criteria in the latent space through cGANs." *Advanced Robotics*, 37(10):621–633, 2023.

2. Ocampo Jimenez, J. & Suleiman, W. "Improving Path Planning Performance through Multimodal Generative Models with Local Critics." arXiv:2306.09470, 2023.

3. Ocampo Jimenez, J. & Suleiman, W. "Enhancing Path Planning Performance through Image Representation Learning of High-Dimensional Configuration Spaces." arXiv:2501.06639, 2025.

4. Guan, L. & Li, B. "Obstacle-free robot path planning based on variational autoencoder and generative networks." *IJICT*, 26(7):17–31, 2025.

5. Lou, J., et al. "Real-Time On-the-Fly Motion Planning for Urban Air Mobility via Updating Tree Data of Sampling-Based Algorithms Using Neural Network Inference." *Aerospace*, 11(1):99, 2024.

6. Fan, X., et al. "Flying in Highly Dynamic Environments with End-to-end Learning Approach." arXiv:2503.14352, 2024. (IEEE RA-L 2025)

7. Xu, B., et al. "Flow-Aided Flight Through Dynamic Clutters From Point To Motion." arXiv:2511.16372, 2025.

8. Zhang, S. "A Physics-Informed Neural Network Approach for UAV Path Planning in Dynamic Environments." arXiv:2510.21874, 2025.

9. Das, A., et al. "DroneDiffusion: Robust Quadrotor Dynamics Learning with Diffusion Models." arXiv:2409.11292, 2024. (ICRA 2025)

10. Jiang, S., et al. "Perception-Aware-Based UAV Trajectory Planner via Generative Adversarial Self-Imitation Learning From Demonstrations." *IEEE IoT Journal*, 12(3), 2025.

11. Gulrajani, I., et al. "Improved Training of Wasserstein GANs." NeurIPS, 2017. (WGAN-GP 原始论文)

12. Mellinger, D. & Kumar, V. "Minimum snap trajectory generation and control for quadrotors." ICRA, 2011. (Min Snap 原始论文)
