# DTD 结构优先写回 A/B

A 沿用原始 `adaptive_dynamics`。B 用联合坐标求解和结构优先写回，保留 A 的方向轴演化提案、出生判据、复杂度代价、冷却和组数上限。原规则提出的新组仅在本步采用结构候选时以零坐标提交，下一步才可用。完整 AdamW 候选和零步保持现有坐标、轴和二次系数。B 从 AdamW 接收当步原始候选，无高维回流状态。

两组使用同一个 DTD seed-0 训练/验证划分。训练 3384 张，留出验证 376 张。初始化、随机种子、预热 36 步和总计 216 步相同。历史 Test 1880 张在权重锁定后评测。历史约 82% 的专家使用完整训练集，其分数用于背景对照。评价 B 的性能差值采用本次同划分的 A。

## 局部选择

预热后使用训练批次前 32 张图，通过含跨组块的拉回 Fisher 联合求解旧组坐标。比较两个阻尼系数和四种步长，用实际 CE 选取每个几何方案的坐标步。原始算法提出一套有界轴与二次系数变化。B 在同一个衰减后锚点比较冻结和演化几何。混合候选的坐标步乘以 \(1-\beta\) 后重新精确解码，轴采用冻结或原始算法提出的有界状态。演化分量相对实际坐标位移的参数范数及 Fisher 响应范数均不超过 25%。结构与 AdamW 补偿的合计位移受当步原始 AdamW 范数约束。

在本批次下一组最多 32 张图上计算候选 CE。AdamW 补偿系数为 0、0.25、0.5、0.75 或 1。选择达到最佳可行 CE 下降 95% 的最小补偿系数，同一补偿系数下优先冻结几何。若都未下降而原始 AdamW 步在全训练批次的梯度方向上一阶下降，回到原 AdamW 步，否则选零步。全批次梯度用于 AdamW，坐标拟合、局部选择和验证各自使用分开的图像。

## 贡献归因

预热后逐步记录固定旧几何的精确坐标位移 \(d_z\)、同坐标下演化几何的额外位移 \(d_\Psi\) 和高维位移 \(d_H\)。A 的高维项含原始回流，B 的高维项是实际选择的 AdamW 补偿。衰减单列于三个分量之外。每隔原有记录间隔，在批次中未参与局部选择的图像上依次测量写入这三个分量的 CE 差值，并计算三项 Fisher 响应的 Gram。原始结果包含不足以单独取出监测图时的重叠标记，汇总排除此类记录。

逐步的 CE 分解依赖于写入次序，正值表示添加相应分量后 CE 下降。Fisher 响应能量归因使用 \(\sum_j G_{ij}/\sum_{ij}G_{ij}\)，含交互且可为负。终点使用两组各 32 张留出验证图，比较每个坐标双向精确扰动的 Fisher 预测和实际平方根预测概率位移。

## 复现

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
out=/dev/shm/dtd-update-allocation
for arm in adaptive_dynamics adaptive_allocation; do
  ./run-python -u scripts/fr_hst/train_fd_task.py --task DTD \
    --steps 216 --warmup 36 --max-components 8 --birth-penalty .0002 \
    --heldout --arm "$arm" --output "$out/$arm"
done
./run-python scripts/fr_hst/summarize_dtd_allocation.py \
  "$out/adaptive_dynamics/results.json" "$out/adaptive_allocation/results.json"
```
