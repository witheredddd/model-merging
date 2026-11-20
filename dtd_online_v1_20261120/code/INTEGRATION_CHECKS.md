# A100 首步验收检查单

在任何多步运行前，仅运行 `--steps 1 --max-modes 1 --checkpoint-every 0`，检查以下硬门禁：

- `TrainingPositions.verify_normal_recipe()` 通过；输入 `trace` 的 step 0 参数形状与已冻结 checkpoint 一致。
- 参数映射恰好 48 个 Transformer 注意力/MLP 矩阵，共 84,934,656 个活动参数、86,192,640 总视觉参数。
- `FullFunctionalGeometry` 在每一步的当前参数上重新初始化；不会悄悄复用初始锚点的 Jacobian。
- A100 上真实梯度是 float64，形状与 flattened 视觉参数相同；无外层网络参数被更新。
- `birth.solver_converged` 是真实原始残差通过，不是仅满足递推 CG 残差。
- 如果出生了新模式，`birth.error_reduction` 超过阈值，p2 的真实 CE 也改善；否则本步写回必须为零或只由已有模式构成。
- `actual_outside_mode_span` 基本为零，且 `write_norm <= radius`。
- 本步 `steps.jsonl` 应同时含 `proposal_norm`、`write_norm`、`rejected_outside_write_norm`、出生判据与 JVP/VJP 审计。
- p3 在全部 step 完成前不得打开；若中途失败，不生成 `FINISHED.json`。

## 科学限制

先导使用冻结的 200 张 p0 图像训练，不能当 DTD 完整优化。更长训练应从 `src.datasets.registry.get_dataset('DTDVal', ...)` 获取全部训练样本并保留独立的 376 张验证集，且必须保证 p2/p3 及其他选择样本与训练流无重叠。不要在这套小 probe 上报告 Val >=80%。

## 算法替代路径

如果 A100 单步出现 CG 稳定性/速度问题，先记录残差与算子计数，不应改变训练目标为已有旧版的 `g^Tv+0.5||v||²+0.5β||Jv||²`。原方案反求的是**指定有标签 DTD 下降所诱导的预测响应**，不是为任意梯度求一个正则化代理更新。
