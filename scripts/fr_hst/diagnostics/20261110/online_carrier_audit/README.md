# DTD 在线载体诊断

本轮沿用提交 3dea8f4 中的线性与非线性载体。已有服务器权重只包含两个第 8 轮终点，因此按原配置做确定性重放，在写入真实参数前额外记录模型、载体、弯曲系数、合成候选与创新残差。重放终点须逐张量匹配原视觉权重、载体和残差缓冲，弯曲系数也须一致。诊断使用两个各 32 张图的训练探测组，两组与原实验的三组探测图均不重合。这些图可能参与过原训练 minibatch。

## 结果

两组各 216 步的训练日志与原实验完全一致。终点视觉参数、载体、残差缓冲和弯曲系数均精确复现。

| 诊断 | 线性载体 | 非线性载体 |
|---|---:|---:|
| 原载体在新图像上的 Fisher 解释率中位数 | 18.0% | 19.2% |
| 原载体在新图像上降低 CE 的评分 | 11/14 | 10/14 |
| 当窗重新拟合时，拟合组 → 复核组 Fisher 解释率 | 35.0% → 16.5% | 25.1% → 15.6% |
| 冻结四步后的 Fisher 解释率中位数 | −0.18% | 9.75% |
| 同一后期位置的当期载体 Fisher 解释率中位数 | −2.46% | 2.46% |
| 冻结四步后保留的 CE 收益中位数 | 47.5% | 16.9% |
| 同一后期位置的当期载体 CE 收益保留率中位数 | 47.4% | 7.4% |

CE 收益保留率只对完整合成候选降低 CE 的评分取中位数。拟合组和复核组均来自训练划分。四步复用比较每组 6 个时间配对、12 次交换评分。少量 CE 收益能保留下来，功能匹配则随批次和候选变化。更新后的载体在这些配对上没有稳定优于冻结载体。

**残差吸收尚无足够证据。** 每个方法测量 21 个长间隔和 6 个四步配对。每个时间配对有两次拟合／复核交换。两种方法都没有配对通过两个图像组余弦均至少为 0.6 的复现判据。对每次交换取两个余弦的较小值后，线性和非线性的最大值分别为 **0.453 和 0.403**。未作复现筛选时，四步配对里当期载体相对旧载体对历史创新的解释率中位数分别增加 2.61 和 2.70 个百分点，但同时改善解释率和 CE 的评分只有 7/12 和 4/12。这些比较给出了局部变化，无法验证同一任务有效残差被持续吸收。

逐窗口、逐组原始数值见 results.json 与 short_absorption.json。它们保留复现余弦、早期创新本身的 CE、两个载体在同一个晚期点的解释率和有限步 CE，允许把覆盖变化和任务收益分开检查。

## 三项诊断

- **功能泛化**。在步骤 76、100、125、150、175、200、216 的写入前状态，用原载体实际给出的更新分量重放。两组都测 Fisher 响应解释率、只施加载体的有限步 CE 和完整合成候选的 CE。另把载体冻结，在 A 组按原坐标模型重新拟合系数并到 B 组评分，再交换两组。
- **时间复用**。对前六个锚点各取四步后的状态。在同一后期模型和同一真实合成候选上，用早期冻结载体和当期载体分别求系数，跨图像组比较功能解释和 CE。只重放候选更新，参数状态随后恢复。这是载体短期复用对照。
- **残差吸收**。对每对早、晚锚点，在同一个晚期参数点比较。先用早期冻结载体拟合晚期合成候选，以它剩下的功能响应和早期训练时实际产生的创新响应比较。早期创新归一化到晚期候选的参数范数，保证比较的目标方向和尺度相同。两组响应余弦均至少为 0.6 时标记为复现配对，另单列在复核组仍能降低 CE 的配对。每一对都保留。对同一早期创新，比较早期冻结载体与晚期载体在交叉复核图上的解释率和有限步 CE。还保留两个载体对晚期候选的评分。四步配对作为单独的短程补充，复用相同快照和同一个阈值。

功能响应使用概率嵌入 $2\sqrt p/\sqrt n$ 的 JVP。解释率为 $1-\|\delta_{\rm target}-\delta_{\rm carrier}\|^2/\|\delta_{\rm target}\|^2$。实际载体与合成候选均排除权重衰减，避免混淆载体的独立作用。两组和多个锚点共享一条训练轨迹，比较作描述性诊断。

## 复现

~~~bash
replay=results/DTD_CARRIER_REPLAY_20261110
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 ../afg_env/bin/python \
  scripts/fr_hst/train_dtd_online_carrier.py \
  --trace results/DTD_DYNAMIC_TRACE_20261109/trace \
  --baseline results/DTD_ROLLING_INNOVATION_20261109/seed42/adamw \
  --output "$replay" --record-carrier \
  --replay-reference results/DTD_ONLINE_CARRIER_20261109
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 ../afg_env/bin/python \
  scripts/fr_hst/diagnose_dtd_online_carrier.py \
  --replay "$replay" --output results/DTD_CARRIER_DIAGNOSTICS_20261110
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 ../afg_env/bin/python \
  scripts/fr_hst/diagnose_dtd_online_carrier.py \
  --replay "$replay" --output results/DTD_CARRIER_DIAGNOSTICS_20261110 --short-absorption
~~~

Git 保存逐组结果。逐步全视觉检查点和载体留在实验服务器。
