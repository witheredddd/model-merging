# DTD Schur Step 0 审计与修复建议（固定提交 04047a0）

审计范围：`dtd_schur_20261120/code/conditional_schur_core.py`、`schur_resumable.py`、`run_step0.py`、两组单元测试与 `results/step0_anchor0` 的 JSON/报告。**远端原始数据未修改。**

## 事实与缺陷

| 类型 | 具体位置 | 结论 | 修复 |
|---|---|---|---|
| 数学正确 | `conditional_schur_core.solve_restricted` | 输出空间 Woodbury 解与条件补偿符号正确；前提为正交投影、固定 H。 | 保留。 |
| 数值通过 | `NUMERICS.json` / `solves/*.json` | JVP/VJP 伴随误差 6.1e-15，Schur 恒等式相对误差约 1.5e-5；当前主要残差对应的目标上界约 1e-6。 | 不需要推翻现有 λ=1 的二次求解。 |
| 数值错误（条件性） | `schur_resumable.py` | `objective_gap_upper` 和停止规则用 `||r||²/2`；一般 λ 时应为 `||r||²/(2λ)`。 | 在派生实验目录使用 `patch_solver_lambda.py` 生成修正版；当前 λ=1 原结果不受影响。 |
| 断点风险 | `schur_resumable.py` + `run_step0.py` | 已有 checkpoint 仅按标签复用，未校验所有会改变线性系统的内容，如梯度、物理模式、Jacobians、λ/β。 | 生产程序应写 `solve_identity` 元数据（锁定锚点、探针、梯度/模式哈希、active mask、λ/β、J 定义），拒绝不匹配恢复；暂时不要在同一 output 下改参数并重跑。 |
| 实验选择偏差 | `run_step0.py`, `SELECTION.json` | rank2 仅击败 Fisher2/随机2，但 candidate2=6.354 仍弱于 mean_gradient1=11.580；没有 pooled regularized one-dimensional baseline，也无 mean+second direction 2D。 | 添加 `pooled_step`、`reduced_optimum` 的独立需求收益比例，不再仅凭删除值选 rank。 |
| 研究对象混淆 | `run_step0.py` | 删除 = 允许完整补集调整后的损失；不等于仅使用候选空间的优化收益（充分性）。 | 同时报告 `q_R / q_full` 和 `q_N - q_full`，不可互换。 |
| 数据不稳定 | `DEMANDS.json` | 200 张图像分为 59/47/47/47，梯度余弦接近 0；四个运动特征值近乎平坦。这可能是小批次梯度噪声，不能断言 DTD intrinsically high rank。 | 不增加预设功能；用 pooled CE 的正规化完整最优步与独立需求检验。 |
| 标定不一致 | `run_step0.py` | `kappa` 用完整 86M 梯度求，活动域却仅 48 矩阵；量纲与域不同。 | 审计脚本另算 active-domain kappa，仅作报告；改变 β 必须重新求 H/CG，不能沿用旧解。 |
| 诊断域不一致 | `run_step0.py` / `DEMANDS.json` | 四批次梯度的余弦用**86M 全域梯度**计算，而被研究运动限制在 48 矩阵（84.9M 坐标）。 | 需要同时报告 `cos(P_D g_i,P_D g_j)`；现有余弦不能直接解释活动域的梯度方向差异。 |
| 虚弱对照 | `run_step0.py` | `random_coordinates` 实为随机**稠密方向**，不是随机坐标支持，也不匹配任务梯度结构。 | 应准确重命名；以后对照包括同秩 H-mean+方向、随机旋转/同支持预算。 |
| 固定驱动自指 | `run_step0.py` | x 取 `P_R v_full`，精确二次模型下 `x+n(x)=v_full` 是必然恒等式，幅度线性也属模型代数性质。 | 标记求解器检验；额外用独立需求和有限步检查耦合泛化。 |
| 无效有限 CE | `evaluation.csv` | 直接应用 norm≈28 的更新，原标定 radius≈1.06e-4，所有 CE 大涨。 | 等半径方向对照与共同 τ 缩放分开，二者不得都称 Schur 删除。 |
| 实际记录错误 | `run_step0.py` | Anchor sentinel 行将 CE/Top1 写 0，而不是实际模型值。 | 审计脚本另生成真正的两个基准 probe 行。 |
| 验收缺口 | `run_step0.py` | `solve` 记录 `converged` 但不在所有后续用途执行 fail-closed；目前 8 次迭代的结果确实通过。 | 未来版本对 nonconverged 拒绝排序与指标结论。 |

## 更严格的当前结论

- 选出的候选二阶空间删除 gap `6.354/454.424 ≈1.40%`，**其补集仍可获得 98.6% 的当前二次目标收益**。
- 单位维数候选更弱：`candidate1=4.205`，`mean_gradient1=11.580`。该结果说明局部代理意义下候选模式不占优势，不是证明该低维结构全局无效。
- 四个预设独立批次里最大 H 特征值仅 32%，但是这是 m=4 个含噪运动的谱，不是模型内在功能秩。
- 补偿相对幅度 `1.768/8.193=21.6%`，是 H 下有条件交叉项的证据；`t=1` 恢复完整运动不是泛化证据。
- **没有实际的物理参数支持发现，也没有在线训练/EC 结果。**

## 本补丁范围

`step0_reaudit.py` 读取原始 A100 的三份 `.pt` 数据，不作任何新 CG。它进行以下独立审计：

1. 使用四个原最优运动按实际样本数合成正则化 pooled-CE 最优运动，并用原始驻点残差验证线性一致性。
2. 计算仅使用候选空间、平均梯度方向、正则化 pooled-CE 方向和加入第二方向的**小矩阵最优收益**，独立于 Schur 删除结果。
3. 使用现有 λ/β 保持原二次目标；另计算 `kappa_active` 以提示标定问题，但不改变旧实验解。
4. 小步长真实前向：分别执行**同物理半径**方向比较和**共同缩放系数**的 full/restricted/joint 诊断；记录真实 baseline CE/Top1。
5. 输出 `reaudit_step0/{sufficiency.json,finite_small_steps.csv,RUN_AUDIT.json}`；不覆盖旧代码或文件。

本地 `reaudit_math.py` 5 项合成数值测试通过，`step0_reaudit.py` 经 `py_compile`；**未在 A100 执行**，不能声称物理实验已验证。模型数据在服务器、二进制向量未入库。

## 严格实验边界

- p2 参与算法选择；p3 不准用于选择后调整算法，只用于预先固定所有比较的最终结果。
- 原子个数、方向秩不是通过 `q_N - q_full` 单调性筛出的科学发现。
- 因为扩展远端有源代码哈希冻结，建议把新增脚本复制到运行目录，而不是更改旧的已冻结计算文件后原地重启。
- 若需要修改 λ/β、更新构造需求或重新建空间，必须生成新的运行目录和新的 CG，不得静默加载旧 `.npz`。
