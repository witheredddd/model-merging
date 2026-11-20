# DTD 在线任务响应控制 V1 — 可执行先导

**状态：数学核心 + 在线更新闭环已实现，CPU 测试通过。A100 集成入口已编写，尚未在用户服务器实际执行。**

这个版本是对《任务功能几何与 EC 模型融合方案(5)》中**第 2 节软约束功能响应反求（公式 14–15）**、**第 3 节有保护任务时的条件补偿（公式 17–20）**和**第 4.2 节模式出生—在线写回**的最小可验证实现。它不是 11 月 20 日旧版的“四个小批次梯度 → H-Gram PCA → Schur 删除”实验。没有预定义类别正弦需求、梯度最大网络层、固定 rank，也没有在正常训练之后做事后审计。

## 研究对象、公式与实际代码

- 第 `t` 步在真实 DTD 训练批次求平均交叉熵梯度 `g_t`，无权重更新的 AdamW 提出 `u_t`。AdamW 只产生任务下降意图；**绝不直接执行 `optimizer.step()`**。
- 目标功能响应 `y_t=J_D(theta_t) P_active u_t`。它是 DTD **一个任务**的当前优化需求，不是自行制造的多种任务功能。
- **全活动空间响应反求（任务条件软约束）**：

  ```text
  a = argmin_v 0.5 ||J_D v-y_t||² + 0.5 λ||v||²
                   + 0.5 β||J_C v||²       (Cars 保护可选)
      subject to v = P_active v
  ```

  通过全空间矩阵自由 CG：`(λI + J_DᵀJ_D + βJ_CᵀJ_C)v = J_Dᵀy_t`。默认没有保护任务，此时 `β=0`，没有凭空创造 DTD 内部的“保护补偿”。这是 PDF 的加权最小二乘近似，不是硬约束 `J_Dv=y_t` 的精确实现。
- 已有控制字典 `B_t=[b_1,…]` 时，当前系数由

  ```text
  z_t = argmin_z ||J_D B_t z-y_t||² + μ||B_t z||²
                 + β||J_C B_t z||²
  ```

  的小矩阵闭式解产生。**只有 `B_t z_t` 可写回模型**，且必须通过独立训练侧选择集 p2 的真实 CE 守卫、物理更新半径和（若启用）Cars 保护守卫。
- 只有功能响应残差大、从残差生成的新全空间方向能减小响应误差，且在 p2 真实 CE 上相对旧结构有额外收益，才允许**在线出生**一个模式。默认每步至多出生一个；`max_modes` 是算力上限而非预设内在维数。
- 若启用 `--protect-task Cars`，每个模式保存**未经其他模式正交化修改的原始驱动—补偿图向量**。补偿由 `-(λI+βF_C)|_N^{-1}(βF_C)_{NR}d` 求得。实际控制计算使用与这些图向量张成相同空间的正交工作基。
- 每个训练位置重新计算 DTD Jacobian 对模式的作用；启用 Cars 时重新求保护补偿。谱 `J_DᵀJ_D` 相对 `J_CᵀJ_C+λI` 的有限维广义特征值记录为诊断量；正负 CE 方向仍由真实任务需求决定。

**研究边界：** 每个模式可以跨越所有 48 矩阵，但代码尚未声称发现唯一的物理参数支持或不可替代权重组。驱动—补偿图目前是针对**方向空间**的扩展，不是人为将原生张量当成功能组。真正的物理支持检验、八任务 EC Pareto、ODE 连续积分以及可达到 80% Val 的完整 DTD 训练均不在本先导交付范围；需在在线更新先导通过后接入真正完整数据加载器。

## 文件

| 文件 | 功能 |
|---|---|
| `core.py` | 线性算子 CG、DTD 响应反求、Cars 条件补偿图、小型控制拟合、选择性谱 |
| `controller.py` | AdamW 仅建议、在线残差出生、模式独立 CE 守卫、严格空间内写回、状态检查 |
| `run_a100.py` | 真实 CLIP/DTD 检查点及原 A100 `FullFunctionalGeometry` 适配；p0 训练、p2 选择、p3 最终检查 |
| `run_toy.py` | CPU 上真实 CE、真实平方根概率 JVP/VJP 的端到端演示 |
| `tests/` | 核心数学、保护补偿、写回、冻结对照、状态恢复、端到端 toy 测试 |

## 本地测试

```bash
cd task_functional_online_v1
python -m unittest discover -s tests -v
python run_toy.py --arm online --steps 6
python run_toy.py --arm frozen --steps 6
python run_toy.py --arm reference --steps 6
```

默认 toy 配置下，在线控制实际增加到 4 个模式，严格空间外写回误差在浮点精度范围；最终 toy 留出 CE 从 1.0986 降到 0.5104。冻结首个模式后的留出 CE 为 0.6406；受同半径、同 CE 守卫约束的参考更新为 0.4844。**这些仅是合成线性分类器程序验证，不能当作 DTD 科学结果。**

## A100 上的单位置最小执行

运行目录与现有 `dtd_schur_20261120` 并列；原先的 `dtd_schur_20261120/code/bootstrap.py` 在服务器上已能解析外部 `task_runtime`、`FullFunctionalGeometry`、`TrainingPositions`。如果这套 A100 私有源码不在原位置，必须先调整 `--runtime-code-dir` 或其 `bootstrap.py` 引用目录。

```bash
python run_a100.py \
  --runtime-code-dir /path/to/dtd_schur_20261120/code \
  --inventory /path/to/INVENTORY.json \
  --probes /path/to/probes.json \
  --trace /path/to/DTD_DYNAMIC_TRACE_20261109/trace \
  --output /new/empty/online_dtd_step0 \
  --arm online --anchor 0 --steps 2 --seed 42 \
  --max-modes 2 --radius 1e-4 --cg-maxiter 24
```

本例只用以前冻结的训练侧开发图像（p0 200 张计算梯度与 Jacobian，p2 106 张决定模式是否出生与写回，p3 94 张只在全部训练后评价）。**它是集成与真实写回的先导，不是完整 DTD 专家训练**。没有对官方 test 操作，也没有把 p3 参与在线选模。默认模型的训练批次只是 p0 中的小批次，正式训练阶段必须接入真正独立的完整训练集及严格不重叠的验证方案。

在不同的**全新输出目录**分别运行：

```bash
# 同一个种子和同一数据顺序，只允许第 0 步出生一个模式
python run_a100.py <相同四个路径参数> --output /new/empty/frozen_dtd_step0 \
  --arm frozen --anchor 0 --steps 2 --seed 42 --radius 1e-4

# 普通 AdamW 候选在同一个物理半径内、通过同一个选择 CE 守卫写回
python run_a100.py <相同四个路径参数> --output /new/empty/reference_dtd_step0 \
  --arm reference --anchor 0 --steps 2 --seed 42 --radius 1e-4
```

上面 `<相同四个路径参数>` 是占位文字，实际执行时展开为 `--runtime-code-dir`, `--inventory`, `--probes`, `--trace`；不可原样粘贴。`--radius` 代表**欧氏参数半径**，不是预测空间半径；其数值仅是先导保护预算，不是训练最优超参。

输出：`RUN_LOCK.json`、`steps.jsonl`、`STATUS.json`、`ENDPOINT.pt`、`FINISHED.json`。可使用 `--checkpoint-every 1` 启用额外的完整状态断点（大文件）；恢复需加 `--resume checkpoint_stepXXXX.pt` 并保持全部锁定配置一致。若发现 `FAILED.json`，读取错误，不要自动扩展 CG、批次或锚点。提供的默认算子迭代上限是计算保险；不收敛时不允许拿近似结果当作已验证方向。

## 关键验收条件

1. 对所有实际写回记录 `actual_outside_mode_span`，应接近数值零；不能通过审计或正向更新绕过这个门禁。
2. `birth.admitted` 必须同时满足当前 DTD 响应新颖性和 p2 独立 CE 收益；仅 CG 求解成功不够。
3. 固定字典 arm 与在线 arm 使用相同 DTD 数据顺序；在线写回轨迹必须和冻结控制轨迹有实质差异，才能叫作在线结构学习有效。
4. 真实训练阶段必须提高数据覆盖面。严格记录总体 JVP/VJP 时间、每步 CE、无写回步骤、字典维数及 Cars（若启用）的有限保护响应。
5. 未独立完成 full training 以前，任何 `rank=2/3/4` 都只是当前计算出的字典规模，绝不宣称模型内在功能秩。

## 技术限制与可信度

- 已运行 CPU `torch` 16 项测试，包含数值参考、真实平方根概率的 toy 在线写回。**没有 A100 运行结果**，服务器环境 `task_runtime` 源码并未完整入库，适配器仍需要一次真实单步集成验收。
- 在 DTD-only 时保护集为空，条件补偿图严格退化为零；不能将 DTD 自身 Fisher 充当“保护任务”然后宣称找到了跨任务补偿。
- 该版本没有完整物理参数支持发现算法。原 PDF 对“如何从全运动选择驱动支持”没有给出唯一算法；实现不应该假装已有唯一解。
- 在线用 p2 做有限 CE 守卫会对 p2 自适应选择；不能把 p2 当最终检验。p3 仅作先导最终结果，更不能声称官方 test。
