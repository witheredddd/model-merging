# DTD 功能子空间、曲率与残差诊断

三项实验复用 `e328ebd` 的单种子 DTD AdamW 轨迹和两组各 32 张固定训练探测图。两个图像组在曲率实验中轮流拟合和复核。实验固定原模型权重，仅计算微分或评估虚拟参数更新。第四项非线性映射诊断单独留待检验。

## A. Fisher 谱演化

对 17 个状态的两组 $48\times48$ Fisher Gram 分别求谱，保留完整特征值和特征向量。前 $r\in\{2,4,8\}$ 维的能量占比使用特征值和除以迹。通过 $V_t^\top V_s$ 的奇异值求主角，稳定性用奇异值平方的均值表示。前八个方向用绝对内积最大权匹配，配对需结合谱隙解读。早期方向的保留率为 $\operatorname{tr}(V_0^\top C_tV_0)/\sum_{i=1}^r\lambda_i(C_t)$。时间匹配对照在相邻、不重叠的各 $r$ 个实际参数更新所张成的子空间之间测量相同指标，再比较对应结束状态的 Fisher 子空间。

相邻检查点的四维 Fisher 重叠中位数在两组上分别为 **0.958、0.980**。预训练点与专家点的重叠降至 **0.624、0.632**，最大主角达到 **66.2°、68.9°**。起点四维在终点仍保留当时最优四维响应的 **88.8%、82.7%**，说明响应强弱与几何角度给出互补信息。三组时间匹配的四维参数更新重叠为 **0.202、0.164、0.103**，相应 Fisher 重叠两组分别为 **0.950／0.937、0.851／0.868、0.597／0.662**。这个比较反映 48 个固定原子内的局部功能方向，相对于所测全参数增量跨度更稳定。原子取自同一训练轨迹的终点更新。

## B. 完整视觉参数 GGN 与 Krylov

在步 0、54、108、216，以交叉熵 logit 曲率 $C=\operatorname{diag}(p)-pp^\top$ 构造 $Gv=J^\top CJv/n$。这里的 $J$ 覆盖完整视觉参数。一次 logit JVP 和一次 VJP 实现一个 GGN 乘积。两组图交换拟合和复核角色。

主要的同预算对照均使用三维。五次微分调用对照中，Krylov 使用一次当前梯度和两次 GGN 乘积，梯度 PCA 使用五次梯度调用，同时记录单位范数与原始梯度的未中心化二阶矩。第 0 步缺少历史，使用当前图像的 bootstrap 梯度补足，单列为冷启动。另一套七次调用记录更长梯度历史、三次 GGN 随机值域素描，并为 Krylov 多测一次拟合侧曲率。调用数和墙钟时间分别记录。独立的 12 步 Lanczos 和六个 Rademacher 样本估计谱和参与率，三条主要 Ritz 方向也各自核验本征残差。这套谱参考使用额外预算。

可前瞻的三个参考点均用下一整轮真实参数更新作对照。将该更新正交投影到各候选子空间，记录在另一图像组上的真实 CE 收益、完整更新 CE 收益和参数能量保留。第 216 步只做局部测量。小步长对照把拟合侧投影下降方向缩放到统一范数，尺度为真实整轮更新范数的 1%、3%、10%、25%。末点的范数参照使用上一轮。二次模型在复核组计算 $-g^\top v-\tfrac12v^\top Gv$ 并与实际改善对照。真实更新只提供评估目标及统一范数标尺。

两个图像组和四个参考点上，当前梯度落在 GGN 前三 Ritz 方向的能量从 **9.4% 到 86.8%**。参与率估计从约 **7 到 56**。谱与梯度的对应关系随阶段和探测样本变化。参与率来自六次随机估计，原始样本、Ritz 值及其残差均保留。

严格五次调用的对照中，Krylov 的下一轮 CE 收益保留率中位数为 **2.08%**，归一化和原始梯度 PCA 分别为 **0.99%、1.04%**。Krylov 在 **4/6** 项比较中胜过两种 PCA，去掉第 0 步后为 **3/4**。三个方法的保留率整体偏低，比较只覆盖一条轨迹和三段前瞻窗口。步长取 1% 时三个方法均有 **5/8** 项降低 CE，取 25% 时均只有 **1/8** 项。二次模型的绝对 CE 误差中位数在 1% 时约为 **0.0027–0.0035**，在 25% 时升至 **0.139–0.202**。GGN 方差恒等式和双线性对称性的最大相对误差分别为 $2.67\times10^{-6}$ 和 $1.45\times10^{-6}$。

## C. 残差重放

用步 0–54 的前四个窗口拟合三维单位增量 PCA 载体。固定载体后，在后十二个窗口各自的真实起点，分别重放载体、残差、完整更新和载体加 25%、50%、75% 残差。记录探测组 CE 改善、从原预测到重放预测的 KL、准确率变化，以及完整收益减去载体和残差各自收益的交叉项。载体加完整残差的预测与记录端点按 $2\times10^{-5}$ 容差核对。

残差单独降低 CE 的窗口数分别为 **12/12、11/12**，而载体单独为 **12/12、4/12**。在完整更新改善 CE 的窗口中，残差单独保留收益的中位数为 **75.7%、87.7%**。CE 交叉项在两组均有 **10/12** 为正，中位数为 **0.0058、0.0072**。相邻残差余弦中位数为 **0.508**。用前六个残差拟合的三维子空间，在后六个残差上的能量捕获率从 **34.5%** 降至 **6.0%**。这支持残差在局部反复承担有效任务更新，同时提示残差方向也随时间变化。仅凭重放难以分辨动量、任务需求等来源。

## 文献对齐

- APML 用轨迹和 LTSA 学习参数流形。A 的结果主要说明局部功能子空间有时间连续性，比较 LTSA 的预测收益属于第四项诊断。[APML 原文](https://www.sciencedirect.com/science/article/pii/S0893608026011809)
- CoSO 从梯度 SVD 动态确定子空间。B 使用同类梯度统计作为方向生成基准，保留真正的留出探测结果。[CoSO 论文](https://arxiv.org/abs/2505.11816)
- Krylov Subspace Descent 从梯度与近似 Hessian 的反复作用构造子空间。这里的 GGN 只量化 logit 曲率拉回，真实 CE 重放覆盖有限步长以及模型非线性带来的差异。[Krylov Subspace Descent](https://proceedings.mlr.press/v22/vinyals12.html) · [局部损失几何](https://arxiv.org/abs/1910.05929)

## 数据和复现

`fisher_evolution.json`、`fisher_eigenpairs.npz` 保存 136 对时间状态和 17 组同时跨图像对照。`curvature.json` 保存四个参考点逐组方向、谱、预算及重放结果。`five_sweep_audit.json` 保存严格同预算的归一化和原始梯度 PCA、Krylov、小步长曲线及 GGN 方差与对称性核验。`residual_replay.json` 保存十二个留出窗口的所有重放条件、余弦与跨时间捕获率。此处的图像留出只针对诊断方向构造，两个探测组都来自原训练划分。

```bash
out=results/DTD_CARRIER_DIAGNOSTICS_20261109
OPENBLAS_NUM_THREADS=1 ../afg_env/bin/python scripts/fr_hst/analyze_dtd_fisher_evolution.py \
  --bank scripts/fr_hst/diagnostics/20261109/mechanisms/mechanism_arrays.npz --output "$out"
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 ../afg_env/bin/python \
  scripts/fr_hst/run_dtd_carrier_diagnostics.py \
  --trace results/DTD_DYNAMIC_TRACE_20261109/trace --queue results/EXPERTS8_99_20261109 --output "$out"
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 ../afg_env/bin/python \
  scripts/fr_hst/run_dtd_carrier_diagnostics.py \
  --trace results/DTD_DYNAMIC_TRACE_20261109/trace --queue results/EXPERTS8_99_20261109 \
  --output "$out" --audit-five-sweeps
```
