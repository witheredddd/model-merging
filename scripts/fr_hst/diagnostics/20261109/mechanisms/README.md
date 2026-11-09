# DTD 训练轨迹上的机制实验

数据来自同一条经逐张量核验的 DTD AdamW 重放轨迹。包含步数 0–216 的 17 个参数状态、16 个相邻窗口和 8 个完整训练轮。每个状态使用原先两组各 32 张互不重叠的训练探测图。实验只读现有权重。跨种子参照是旧设置中已有的 3 个种子、42 个窗口，在两个图像组上测量得到 84 条记录。

## 实验结果

**事后低秩和时间外泛化。** 16 个实际参数增量的前四个主方向解释原始能量的 75.71%，归一化后为 74.53%。由前 8 个窗口拟合的四维子空间，在后 8 个窗口上的能量捕获率从 57.92% 逐段降至 4.69%。因此全轨迹低秩不足以确定后期更新的固定参数方向。

**块响应增强与梯度抵消。** 两组图的每个相邻窗口上，48 原子投影的联合响应能量都超过各块单独响应能量之和。比值中位数分别为 13.37 和 11.98。与此同时，投影 JVP 相对全参数 JVP 的误差中位数达到 82.28% 和 88.61%。强块间协同只描述这组候选分量，实际窗口更新还包含大量其他预测响应。独立的损失梯度分解给出相反的符号模式。交互项在每组 16 个窗口中有 12 个沿净梯度变化方向为负。平行分量中位数为 −2.66% 和 −4.28%，正交分量中位数为 21.72% 和 24.55%。旧三种子对照的 84 条记录平行分量均为正，说明交互符号依赖所用轨迹、参数范围和探测设置。

**同轮缩窗。** 每个短窗口的交互范数均低于它所属整轮的值。两组各 16 项比较的短／整轮范数比中位数为 0.542 和 0.565。除以两个主项范数之和之后，交互相对强度仍在 14/16 和 13/16 项中下降。这个趋势支持交互具有有限步长依赖性。当前窗口长 2–25 步，若要估计无穷小尺度下的阶数，需要更密的观测。

**概率轨迹与局部映射。** 两组概率 InPCA 的前四个方向解释正谱能量的 97.94% 和 97.72%。从预训练点到专家点，输入侧 Fisher Gram 的 Frobenius 余弦为 0.824 和 0.589，而输出侧受限 NTK 的余弦仅为 0.025 和 0.033。参数方向在功能度量下仍可保持部分相对强弱，同时它们在输出空间中的作用方式发生很大变化。所有状态对上，概率 Fisher–Rao 距离与 \(1-\cos(K_t,K_s)\) 的描述性 Pearson 相关为 0.928 和 0.941。局部 A 组拟合的前四方向在 B 组的解释率中位数为 68.65%，预训练点固定方向的中位数为 66.90%。这两项连同受限 NTK 应分别报告。

端点矩阵与先前独立测量的最大相对 Frobenius 误差为 \(5.70\times10^{-7}\)。每个状态的 Fisher 矩阵均通过半正定性和响应重构核对，梯度分解也通过数值闭合核对。

## 测量定义

**参数更新结构。** 用全视觉参数的实际窗口增量构造 \(G_{ij}=\langle\Delta\theta_i,\Delta\theta_j\rangle\)，分别保留原始尺度和单位范数版本。用前 8 个窗口拟合单位更新的 4 维参数子空间，在后 8 个窗口量化投影能量，避免把整条轨迹的事后低秩当成可迁移的方向。

**Fisher 块交互。** 在每个参考点使用同一组 48 个原子 \(U\)，每个原子是预训练点到第 8 轮 DTD 专家的一个矩阵更新，经独立单位范数化。对组内 \(n\) 张图，以 \(f(\theta)=\operatorname{vec}(2\sqrt{p_\theta}/\sqrt{n})\) 定义 \(R_t=Df_{\theta_t}U\)，计算 \(C_t=R_t^\top R_t\)。每个窗口用实际更新的投影系数 \(a=U^\top\Delta\theta\)。保留对角能量 \(\sum_i a_i^2 C_{ii}\)，并将 \(2a_i a_j C_{ij}\) 按符号分别求和。把全参数 JVP 与真实端点的 \(f(\theta_{t+1})-f(\theta_t)\) 比较，再把 48 原子投影的 JVP 与全参数 JVP 比较，分别量化线性化误差和候选空间截断误差。

**损失梯度交互。** 此处 \(J_t\) 指 logits 对全部视觉参数的 Jacobian，\(r_t=p_t-y\)。同一固定探测图和标签下，探测集交叉熵梯度为 \(g_t=J_t^\top r_t/n\)。四个端点交叉 VJP 给出 \(P=J_0^\top\Delta r/n\)、\(M=\Delta J^\top r_0/n\) 和 \(I=\Delta J^\top\Delta r/n\)。核验 \(\Delta g=P+M+I\)。交互的平行与正交分量分别按 \(\langle I,\Delta g\rangle/\|\Delta g\|^2\) 和 \(\|I_\perp\|/\|\Delta g\|\) 报告。时间细化把同一轮的两个实测子窗口分别与其整轮比较，保留每个窗口的步数和相对交互强度 \(\|I\|/(\|P\|+\|M\|)\)。

**概率轨迹与受限 NTK。** 概率距离使用两组图各自的 RMS Fisher–Rao 距离及平均 Bhattacharyya 距离。InPCA 对后者双中心化，解释率以正谱总量为分母，完整有符号谱保留在 JSON 中。受限 NTK 是 \(K_t=R_tR_t^\top\)，同组不同时间的 Frobenius 余弦用 \(\|R_t^\top R_s\|_F^2/(\|C_t\|_F\|C_s\|_F)\) 计算。Fisher 矩阵也用 A 组拟合前四和末四方向，再用 B 组复核其响应。每个窗口的原始结果均保留。

## 数据与复现

- `summary.json` 包含逐状态、逐窗口、逐图像组指标及同轮匹配比较。
- `mechanism_arrays.npz` 包含固定 48 方向的逐状态响应 `fisher_responses[17,2,1504,48]`、Fisher Gram `fisher_grams[17,2,48,48]`、参数增量 Gram、概率距离和受限 NTK，以及两种窗口尺度的全参数 JVP 和梯度项 Gram。梯度项轴依次为 \(P,M,I,\Delta g\)。
- `historical_interactions_verified.json` 是旧三种子诊断记录。由于训练设置和参数范围不同，单独作跨种子参照。

从带有既有模型、数据与权重的仓库根目录运行。

```bash
run=results/DTD_DYNAMIC_MECHANISMS_20261109
../afg_env/bin/python scripts/fr_hst/measure_dtd_dynamic_mechanisms.py \
  --trace results/DTD_DYNAMIC_TRACE_20261109/trace \
  --queue results/EXPERTS8_99_20261109 --output "$run"
OPENBLAS_NUM_THREADS=1 ../afg_env/bin/python scripts/fr_hst/summarize_dtd_dynamic_mechanisms.py \
  --measurements "$run" \
  --historical scripts/fr_hst/diagnostics/20261109/mechanisms/historical_interactions_verified.json \
  --reference scripts/fr_hst/diagnostics/20261109/new_dtd_expert_geometry.json \
  --output "$run/publish"
```

新轨迹只覆盖一个种子。各窗口共享训练路径，同轮的短窗口也共享端点，统计量用作描述性诊断。候选空间的低响应和受限 NTK 都只对所测方向与探测图成立。完整参数纤维的核维数和局部图的可拼接性需另外验证。
