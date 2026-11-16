# 动态功能补偿与功能组 EC 融合：代码设计书 v1.1

**版本日期：**2026-11-16  
**状态：**设计冻结候选；**未修改仓库、未提交代码、未启动训练或 EC。**  
**本次目标：**在不改变既有 r2 条件补偿学习器算法和门槛的前提下，为其已有联合模式增加功能组描述、确定性参数解码、原生多目标 EC 接口，以及位置更新后的离散局部几何图。

## 0. 核心实施契约（代码审查的第一依据）

**唯一方法链：**

```
原任务功能几何 (p0, J_D/J_C, K_D/K_C, h_D/h_C)
  -> 原近似纤维约束优化、teacher 只用于候选
  -> 原 driver + 联合 compensator / Schur 求解
  -> 原 Mode 出生、重整、控制空间内真实写回
  -> 一个 Mode 对应一个可追踪的功能联动组及控制振幅
  -> 冻结局部图 B_k，EC 优化 z（仅真实归一化 Top-1）
  -> 按显式选定的 Pareto 分支更新锚点
  -> 在新位置调用原活动刷新及重整，建立 B_{k+1}
  -> 继续 EC，跨图保存真实 Pareto 档案
```

**不可改动约束：**

1. **不改原有 `fiber_core.py`、`fast_online.py`、`run_a100.py`、`resume_state.py` 的算法语义。** 优先复用已有函数；若为了调用边界必须抽取外层流程，先做同输入、同随机状态回归，确保结果等价。原文件保持不变，扩展实现放在独立新目录。
2. **禁止新增任何研究性通行门槛。** 不引入“Fisher 达到 99% 才继续”、“低维前两轴占比够了才进入 EC”、“若组数量低于 N 则运行下一阶段”等条件。新监控只写日志，不参与出生、写回、EC 个体选择或图迁移。
3. **禁止硬编码新低维维数。** EC 染色体长度 `n_var = len(chart.modes)`，与 `rank(B)` 分开记录；不能固定 2、4、8、10 等。即使存在历史对照中的 `global2d`，也仅是对照臂，与本方法无关。
4. **既有 r2 策略的门槛既不删也不改。** `capacity_target`、`novelty_tol`、计算上限、真实有限步保护约束等均保持原函数的判定；它们属于继承行为，不得包装成数学上自然发现的维数或组数。
5. **304 维不是输出低维维数。** 它是现有专家重组字典的输入坐标数。`86,192,640` 个视觉参数仍被完整重建及前向评价，但原 r2 的 Jacobian、驱动和补偿搜索限定于其现有专家坐标空间；不宣称本版已实现任意 86M 方向列生成。
6. **教师解绝对不得作为自由高维更新直接写回。** 原 r2 的真实接受更新必须由已学习模式承担；EC 候选也只能由冻结模式列解码，且不同候选从同一图锚点独立构造。
7. **原数据角色不变。** 本次 DTD/Cars：p0 用于几何和先导搜索；已使用的 p4/p5 只是历史探索性诊断；p6 和官方 Test 不自动打开。不得把本次先导称为正式 Test 或生产官方 Val EC。
8. **历史轨迹含义不改。** 原学习器历史只能写入通过原 `finite_audit` 等规则接受的 `control_learning` 更新。EC 移动锚点为 `ec_chart_move`；不得将其伪造为原学习器已接受步或直接塞入旧历史门控。

## 1. 已核对的来源与无法核对的依赖

### 1.1 GitHub 当前可核对的源文件

仓库 `witheredddd/model-merging`，本次只读核对的 `main`：`97d43ee63999b8103e55a45ca25b88fd75ee26c5`。

| 源文件（仓库相对路径） | 经核对的核心行为 | 新实现使用方式 |
|---|---|---|
| `scripts/adaptive_fiber_20261116_r2/fiber_core.py` | `Geometry`、`Scenario`、`Mode`、`optimize`、`mode_from_driver`、`IncrementalController.learn/recondition/accept_trajectory` | 直接复用；不改源码中的筛选、求解和出生规则 |
| `scripts/adaptive_fiber_20261116_r2/fast_online.py` | `shortlist`、`refresh`、`exact_basis`、`drift_error`；先刷新活动支持、重整、出生，再在模式空间内有限步接受 | 复用已有刷新策略；保留旧 step 与 `actual_responses` 含义 |
| `scripts/adaptive_fiber_20261116_r2/run_a100.py` | `setup`、`current_geometry`、`all_metrics`、`finite_audit`、完整权重 `s.point(a)` | 唯一沿用真实模型评价/重建路径；不要运行旧总入口作为 EC |
| `scripts/adaptive_fiber_20261116_r2/resume_state.py` | r1/r2 断点恢复，原模式和有限功能响应历史恢复 | 先恢复再建立图；不覆盖旧 checkpoint |
| `scripts/conditional_compensation_20261116/cache.py` | 初始 p0 几何与固定参考预算；`Space` 来自 A100 外部 `compromise_geometry_20261115` | 冻结初始字典、参考预算和 probe 来源 |
| `scripts/adaptive_fiber_20261116_r2/R2_EXECUTION.md` | r2 冻结的在线策略、历史延续和数据角色 | 作为算法行为基线 |

源码链接：

- https://github.com/witheredddd/model-merging/blob/97d43ee63999b8103e55a45ca25b88fd75ee26c5/scripts/adaptive_fiber_20261116_r2/fiber_core.py
- https://github.com/witheredddd/model-merging/blob/97d43ee63999b8103e55a45ca25b88fd75ee26c5/scripts/adaptive_fiber_20261116_r2/fast_online.py
- https://github.com/witheredddd/model-merging/blob/97d43ee63999b8103e55a45ca25b88fd75ee26c5/scripts/adaptive_fiber_20261116_r2/run_a100.py

### 1.2 尚需从原 A100 环境核验、不能假装已本地检查的对象

`Space` 与参数展开由 `/home/shangxueyuan/merging/tsv/scripts/compromise_geometry_20261115/space.py` 和对应 `space.pt` 提供；现有 GitHub r2 文件引用了这些路径，但它们的完整实现及 private `space.pt` **未在本轮容器中直接可用**。原 NSGA-II 包装、`numerics.py`、A100 生产 EC 主文件也需要在实施前从原环境逐项对照。附件 v1 给出的哈希作为核验线索，**不能代替当前环境重新计算哈希**。

这些外部依赖缺失时，可以完成纯接口与单元测试，但不能声称已完成真实 A100 复现。

## 2. 数据与数学约定：一个语义只允许有一个来源

### 2.1 参数空间

- 完整视觉参数向量：`theta ∈ R^P`，当前报告 `P = 86,192,640`。
- 原专家重组字典：`Q ∈ R^(P × d_E)`，原运行 `d_E = 304`；初始按 152 个张量组织，每张量两个专家来源坐标。**这是原输入字典，不是我们规定“每组二维”。**
- 候选坐标：`a ∈ R^d_E`，完整参数 `theta(a) = theta_ref + Q a`。实现只能通过原 `Space.point(a)`/`Space.expand(v)` 写回，不重复写另一个容易顺序错位的 Q。
- 原 `Mode.vector`：归一化的 **a 坐标方向**，长度为 `d_E`。一个 Mode 有一个组级振幅。
- 组语义：`Mode.driver_key`、`Mode.supports`、`Mode.source_task`、`Mode.map_matrix`、`Mode.birth_step`。实际涉及张量的贡献额外记录，但不按新阈值重新裁组。

### 2.2 功能几何与原约束

原 `Geometry` 给出：

\[
\Phi_k(\theta)=\mathrm{vec}\!\left(\tfrac{2}{\sqrt{N_k}}\sqrt{p_{\theta,k}}\right),\quad
J_k=D_a\Phi_k(\theta(a)),\quad K_k=J_k^\top J_k,
\]

以及真实标签交叉熵对 `a` 的梯度 `h_k`。原 `optimize` 保持：

\[
\min_v\ h_T^\top v+\tfrac12 v^\top K_Tv
\quad \text{s.t. }\|v\|\le R,\ \|J_Pv\|\le\ell,\ h_P^\top v\le0.
\]

注意原 `fast_online` 允许活动列刷新、非活动列带年龄缓存；304D teacher 因此是混合新旧列的**近似候选信号**，并非该位置精确304D能力上界。不能用监控模块改变此近似策略。

### 2.3 原联合条件补偿

对于驱动坐标集合 `M` 与已有支持 `N`，原函数求：

\[
A=K_{P,NN}+\lambda I,\qquad C=K_{P,NM},\qquad A\Gamma=-C,
\]

\[
\widetilde b=E_M x_M+E_N\Gamma x_M,\quad
\beta=\|\widetilde b\|_2,\quad b=\widetilde b/\beta.
\]

原 `mode_from_driver` 采用联合线性求解 `np.linalg.solve`；阻尼为 `damping_scale * max(||K_NN||_2, 1e-30)`，其中默认 `damping_scale=1e-8`。保持完全一致。完整参数模式由原 `Space.expand(b)` 表示。

原 `recondition` **固定旧 Mode 的 driver 和支持，只用新位置的保护任务 K 重算 `Gamma` 和归一化 `b`**。不能称它已经在动态改变组的参数成员。

### 2.4 控制模式、组与有效秩

定义原模式矩阵：

\[
B_{\mathrm{mode}}=[b_1,\ldots,b_m],\quad
r_{\mathrm{eff}}=\operatorname{rank}(B_{\mathrm{mode}}).
\]

- `m` = `len(policy.modes)`：EC 的**可解释组振幅变量个数**。
- `r_eff` = `mode_basis` 计算的数值秩：诊断有效自由度。
- 原局部优化内部使用 `U = ortho(B_mode)`，不等于 `B_mode` 的各组语义列。不能拿正交化列 U 直接冒充功能组。
- EC 使用 `B_mode @ z`；不把 `U @ xi` 与 `B_mode @ z` 混淆。若必须内部换坐标，需显式证明并记录线性变换关系。
- 一个张量可参与多个模式；允许组间重叠，不把它们强制划成互斥 partition。

## 3. 新增模块：只加适配层，不另造算法

建议新目录 `scripts/functional_group_ec/`；**原 r2 文件和旧 checkpoint 不覆盖**。

| 文件 | 唯一职责 | 禁止事项 |
|---|---|---|
| `legacy_adapter.py` | 通过原 `setup / restore / refresh / exact_basis / policy.recondition / policy.learn` 建立运行时；读取原 budgets 与数据协议 | 不改控制规则，不重写 Schur、birth、finite audit |
| `chart_state.py` | 冻结锚点、复制 Mode 向量与支持、模式矩阵、测量位置/缓存年龄、来源哈希 | 不做聚类、剪枝、模式出生 |
| `decode_fitness.py` | `decode(chart,z)` 和原 `Top1 → fitness`；每个个体从同一锚点重新构建参数 | 不在模式外更新，不加 CE / Fisher EC 淘汰约束 |
| `ec_adapter.py` | 使用原 NSGA-II ask/evaluate/tell；基因维数按当前图读取；保存每个候选的真实参数坐标与目标 | 不重写 NSGA-II、不默认二维或 `z∈[0,1]^m` |
| `chart_loop.py` | 组织“冻结图内 EC → 明确的 Pareto 分支 → 新位置几何刷新 → 下一图” | 不把 EC 移动写入旧学习器有限步接受历史 |
| `state_io.py` | 只在新输出目录原子保存状态、个体、种群、RNG、来源和预算进度 | 不覆盖旧状态，不用仅恢复模型替代完整恢复 |
| `monitor.py` | 追加日志；接口永远不返回 `accept/reject/birth/next_stage` | 禁止根据监控值改控制流 |

### 3.1 核心状态（接口草图，不是已实现代码）

```python
@dataclass(frozen=True)
class FrozenChart:
    chart_id: str
    source_checkpoint_sha256: str
    anchor_a: np.ndarray                # (d_E,)
    mode_matrix: np.ndarray            # (d_E, n_modes), original Mode.vector columns
    modes_snapshot: tuple               # driver_key, supports, source_task, Gamma, birth_step
    geometry_point_a: np.ndarray        # actual point where J/K were measured
    column_ages: np.ndarray            # stale column provenance
    parent_candidate_id: str | None
    probe_role: str                     # 'p0_exploratory' unless explicit override
    protocol_id: str

@dataclass(frozen=True)
class EvaluationRecord:
    chart_id: str
    candidate_id: str
    z: np.ndarray
    a: np.ndarray
    parameter_digest: str
    normalized_top1: tuple
    ec_objectives: tuple
    ce_kl_monitor: dict                # never influences EC filtering

@dataclass(frozen=True)
class RunProtocol:
    original_fitness_source: str
    control_domain_source: str          # required; cannot fabricate from old alpha bounds
    ec_budget_source: str               # required and recorded
    branch_policy_source: str | None   # None => only the first chart
    population_transition_source: str | None
```

**所有 ndarray/Mode 需要深拷贝或冻结快照**，否则原 `policy.recondition` 就地修改 `Mode.vector` 会无意改变已归档的旧图，导致旧 EC 候选不再可复现。

### 3.2 唯一解码器

```python
def decode(chart: FrozenChart, z: np.ndarray, space):
    # 仅为设计级伪代码：完全依赖原 Space.point 的权重顺序和 dtype。
    assert z.shape == (chart.mode_matrix.shape[1],)  # 接口形状检查，不是研究门槛
    a = chart.anchor_a + chart.mode_matrix @ z
    point = space.point(torch.as_tensor(a, device='cuda', dtype=torch.float64))
    return a, point
```

**不得把解码解释为“连续 ODE 已经积分”。** 它是单张局部图上的仿射表示；跨图刷新后才得到位置依赖的分段控制结构。

### 3.3 评价器

- 用原 `all_metrics(c, point, probe=0)` 计算每个真实模型的 DTD、Cars Top-1；与原 EC 的**负归一化 Top-1 目标**对接。
- 归一化常数从已锁定的原 EC 协议读取并记录文件哈希；附件草稿给出 DTD `0.8209219858156028`、Cars `0.8648648648648649`，但真正编码前必须核对原 A100 目标来源，不私自写入新硬编码。
- 允许额外记录 CE、KL、功能位移、Fisher 响应、参数范数；不得将这些监控量插入 NSGA-II 的可行性规则或改变 Pareto 目标。
- 所有个体从相同图锚点解码，**不能**在上一位个体权重上累加。
- 用真实坐标与模型哈希去重/缓存评价时，必须与**选定的原 EC 实现**的缓存语义一致；不能额外引入改变评价预算的隐式去重。

## 4. 实际运行的状态机：阶段来自运行流程，不来自新门槛

### 4.1 只读恢复与建立首张局部图

1. 明确指定 r1/r2 的已有、**已接受** checkpoint；调用原 `restore(...)`，恢复 `a`、modes、旧 Γ、原实际有限响应历史和参考预算。
2. 用原 `Space.point(a)` 逐元素核对完整视觉参数与 checkpoint 中 `full_visual_flat`。核对失败是数据完整性错误，不能继续以错配状态运行；这不是功能性研究门槛。
3. 分清 checkpoint 中 `a` 对应写回后的点，而 Mode 可能尚位于更新前的几何。建立首张 EC 图之前，**在同一 `a` 位置执行原活动支持刷新与 `policy.recondition`**；需要保留原 refresh 的列年龄语义。
4. 如协议明确要求延续原学习器，则调用原 `learn`/原有限步写回流程；不能因“进入 EC 阶段”而重新设计出生机制。学习器的 `step`、`max_new` 必须来自原步骤历史，不由 EC 图号代替。
5. 冻结当前 Mode 列、元数据及几何测量点，创建 `FrozenChart`。

### 4.2 图内 EC（无新增功能阈值）

\[
a_k(z)=a_k+B_k z,\qquad
\theta_k(z)=\texttt{Space.point}(a_k(z)).
\]

- `n_var = B_k.shape[1]`；`n_obj = len(task_order)`。
- 由**已确认原协议**提供 EC 优化器、种群规模、随机种子、目标、总评价预算和 `z` 的搜索域。
- `z` 是归一化 Mode 的振幅，**不能沿用旧全局专家权重 `alpha∈[0,1]`**。若尚无来源，这个字段保持未定义，不自动猜一个范围。
- NSGA-II 按原 ask/evaluate/tell 流程迭代，只有真实负归一化 Top-1 作为目标。
- 保存所有真实参数点和 fitness，不将保护 CE 监控量当作 EC 的新增约束。
- 停止来自明确评价预算/原 EC 终止条件，**不来自监控阈值**。

### 4.3 从图 k 到图 k+1（需明确的分支协议，不自动替你选）

1. 保留图 k 的全部非支配**真实**候选及对应原 fitness。
2. 通过显式 `BranchPolicy` 选出需继续探索的 Pareto 候选；**没有“最佳唯一点”**，不能偷换成最大平均准确率或“过阈值才移动”。如果该协议未给定，运行合法地停在单图 EC，不造下一个锚点。
3. 新锚点取所选 EC 候选的**真实 a**，事件记作 `ec_chart_move`，不是原 r2 已接受的 `control_learning` 步。
4. 在新锚点调用原位置的活动几何刷新；依据保留的 driver/support 调用原 `recondition`。如果还要出生新 Mode，则调用已锁定的原 `policy.learn` 路径并如实记录这是一种扩展到 EC 锚点的策略应用；不得宣称复现旧五步轨迹。
5. 将新 Mode 及新 Γ 冻结成下一张图；新图的 `n_var` 自然取新模式数量，图内不临时改变染色体维数。
6. 旧候选如果需要迁移到新图，可解最小二乘 `min_z ||a_{k+1}+B_{k+1}z-a_old||^2`。保存重表示误差；**投影点是新候选，必须重新做真实评价，不得沿用原 fitness**。
7. 新种群迁移、分支选择和图预算必须有运行配置来源；它们是新 EC 调度协议，不能以“忠于原 r2”名义暗自创造。

### 4.4 ODE 的位置：只监控其离散一致性

原方案有固定支持、固定保护任务条件下的关系：

\[
A(\theta)\Gamma(\theta)+C(\theta)=0,
\quad
\frac{d\Gamma}{ds}=-A^{-1}\left(\frac{dC}{ds}+\frac{dA}{ds}\Gamma\right).
\]

本版**不对每个 EC 个体做 ODE 积分**，只在局部图锚点更新时重新计算 Γ。可以记录 `||AΓ+C||`、新旧 Γ 差异、Mode 方向旋转、有限差分一致性，但这些量不得控制图移动。组支持改变时不跨事件硬算同一个导数；基底重整和模式归一化改变也要单独记录。

## 5. 必须原封保留的旧规则（不是新增门槛）

以下是 GitHub r2 `fiber_core`、`fast_online`、`run_a100` 中核对的**既有代码行为**。真实执行以选定 checkpoint 所属版本和原配置为准；不得拿新默认值覆盖恢复状态。

| 原字段/代码 | 当前源码行为 | 新设计处理 |
|---|---|---|
| `capacity_target=0.90` | 学习器在已有容量比例达到该值时停止继续出生 | 保持，不称学得内在维数 |
| `novelty_tol=1e-3` | 新模式的功能新颖性和历史响应门控 | 保持；监控不新增第二门 |
| `numeric_gain_tol=1e-11` | 生成功能模式时比较收益 | 保持 |
| `max_modes=16`, `max_compensators=8` | 旧模式容量与补偿支持循环上限 | 保持；显式标记为原工程上限 |
| `driver_limit=2`, `compensator_limit=4` | 原候选粗筛的数目 | 保持，**不是**最终组数/维数 |
| `max_new=1` 首步，后续最多 2 | 原 `fast_online.run` 的出生次数调度 | 保持原 step 语义 |
| `R=.003*norm(s.aD-s.aC)` | 原参数搜索尺度 | 保持，实际完整参数范数另记录 |
| `tau in {.01,.1,.3,1.}` | 原保护任务响应预算网格 | 保持，不能解释为 EC 组系数域 |
| `reliability=0.2` | 活动 JVP/有限响应线性化误差的原数值策略 | 保持，不加另一可靠性门 |
| `finite_audit` 原 CE/响应/半径检查 | 原学习器的接受条件 | 只在原学习器实际写回中调用并沿用原返回判定；EC 不另造筛选 |
| 原回溯、停滞、异常校准上限 | 原计算调度 | 保持；实验预算停止不报告为自然秩收敛 |

**必须明确的边界：**用户要求“不加门槛”，本设计严格执行。若未来也要删除原旧阈值，那将是单独的算法变更，不能在本轮以“忠实落地”名义偷偷进行。

## 6. 监控内容：写日志但不驱动算法

`monitor.record(event)` 返回 `None`；算法代码不得从 monitor 读取阈值或决策。

| 类型 | 仅记录的量 |
|---|---|
| 模式 / 组 | `driver_key`, `supports`, `source_task`, `birth_step`, 模式范数、每张量实际贡献、支持重叠 |
| 模式容量 | `n_modes`, 原 `rank(B)`, 全奇异值谱、内部 U/外部 B 关系 |
| 条件补偿 | 实际阻尼 λ、`||AΓ+C||`、纯响应泄漏、补偿成本、规范化 β |
| 几何时效 | `geometry_point`, `column_ages`, 当前活动精确 JVP 次数、teacher 缓存年龄 |
| 动态 | 同一模式在相邻图间 Γ 变化、支持变化事件、方向角、归一化变化 |
| 任务性能 | 原 EC fitness、两任务 Top-1、CE、KL、有限预测响应、联合响应余弦 |
| 真正写回 | `control_learning` 的实际步长、模式系数、已接受轨迹；EC 事件另记 |
| EC | 每次 z、真实 a/hash、原始 Top-1、负归一化 Top-1、非支配归档、图 ID |
| 计算成本 | JVP/VJP、任务前向调用、缓存命中/陈旧列、线性求解、CUDA 时间、墙钟时间 |

新增监控**不得引入**“观察值好看才进入 EC”的阶段门槛。

## 7. 恢复、留痕与无副作用测试

### 7.1 文件输出（仅新目录）

```
experiment_manifest.json       固定源码 commit、依赖哈希、数据角色、原运行配置
charts/<id>/chart_state.pt      冻结锚点、Mode 快照、B、Γ、几何测量点和列年龄
charts/<id>/learning_events.jsonl
charts/<id>/ec_state.pt         种群、原目标、优化器状态、RNG、评价预算位置
charts/<id>/evaluations.jsonl  原模型参数与真正评价结果
charts/<id>/transition.json    EC 分支选择来源、父个体和真实锚点
archive.jsonl                 不同图上的真实 Pareto 个体，绝不只存重投影点
trajectory.npz                按控制学习/EC 图移动分开记录
REPORT.md                     已完成事实、未完成项、成本与来源
```

**恢复条件：**相同 chart、同一 Mode 快照、相同种群、RNG 和预算位置。只恢复模型却重新初始化 EC，不叫精确恢复。

### 7.2 实现验证（代码一致性，不是科学实验门槛）

1. **旧行为回归：**给相同 r2 保存状态、几何和随机状态，扩展前后 `learn/recondition` 的支持、事件、Γ、Mode 向量、接受结果完全一致（按原数值精度比较）。
2. **解码确定性：**同一 chart/z、不同评价顺序得到相同参数；组支持重叠时向量自然叠加。
3. **坐标一致性：**`B_mode @ z` 与内部 `U @ xi` 代表同一坐标更新；新模块不对 `Mode.vector` 二次归一化。
4. **维数来自状态：**使用合成 Mode 数量不同的状态测试 `n_var=len(modes)`；测试数据规模不是算法固定维数。
5. **原目标复用：**同一模型点的 EC `F_k` 与原负归一化 Top-1 完全一致；改变监控 CE/KL 不改变原 fitness。
6. **历史隔离：**`ec_chart_move` 不会增加 `policy.actual_responses` 或 `policy.trajectories`；只有原接受更新会写入。
7. **图迁移：**旧候选坐标重表示后必须有新候选 ID 和重新评价；原候选保持原哈希和 fitness。
8. **保存恢复：**在确定的评价序号断点恢复后，后续个体顺序、RNG、预算、目标保持一致。
9. **监控无副作用：**监控开/关，全部原算法决策和 EC ask/tell 输出相同。

这些验证失败时修代码和状态兼容性；**不改变算法策略去“通过某个科学门槛”**。不新增 seed、大规模对照或训练任务。

## 8. 不得私自编造的 EC 运行协议

目前附件与已核对的源文件**尚不能唯一确定**下面四个执行参数。设计必须给出类型和来源，不能凭直觉补数值。

| 必须配置 | 为什么当前不能自作主张 | 处理方式 |
|---|---|---|
| **z 控制域 / 尺度** | z 作用于单位化的 Mode，而非原全局专家系数；旧 `alpha∈[0,1]` 不能照搬 | `ControlDomain` 必填，记录来源；解码器本身不对 z 新增过滤 |
| **EC 种群与评价预算** | 原先导和 A100 生产 EC 使用不同的终止协议 | `RunProtocol` 指向指定原配置，不拼装互不相干的旧数值 |
| **Pareto 分支移动规则** | 一个非支配集合没有唯一最佳点；旧 anchor 25/50/75% 选点不是当前跨图迁移规则 | `BranchPolicy` 明确来源；未指定时只完成单图 EC，不自动选最优平均准确率 |
| **种群跨图迁移** | `B_k`、`B_{k+1}` 一般不同，旧 z 不等于新 z | 显式 `PopulationTransition`，若重表示必须重新真实评价 |

这些是**执行规范缺项**，不是“新门槛”。任何人都不能借“自动补齐配置”给新方法添加假设。

## 9. 动工边界与交付顺序

- **本轮仅交付设计书**，不修改仓库，不运行 GPU，不创建 EC 实验结果。
- 随后的编码顺序：先 `legacy_adapter` 和恢复/冻结图，再 `decode_fitness`，然后 `ec_adapter` 的单图接口，最后 `chart_loop`；并行加入纯日志监控。
- 首先执行原行为回归与无副作用测试；这属于基本软件正确性验证，不是“指标达到某比例才允许下一阶段”的实验门槛。
- 只有原 EC 来源与必要运行协议可访问后，才可能执行真实 EC；缺项不能静默用“2 维”、`z∈[-1,1]`、额外 99% Fisher 截断等代替。
- 动态部分第一版仅实现离散局部图，连续 ODE / FPCA / 任意全参数 VJP 扩维均不在本次编码范围内。若需加入，必须单独以算法修改立项，不能冒充原实现。

## 10. 唯一可信的本轮完成定义

**“设计书完成”**：目标、旧算法复用边界、公式、模式/组语义、EC 目标、无新增门槛约束、接口、状态机、监控、恢复和未知运行配置全部有文字可审查。  
**“实现完成”**（未来）：在原环境下可用既有 checkpoint 从一个已学习的模式控制空间执行真实多目标 EC，保存真实候选与目标；随后按明示分支策略建立新图，原补偿重整影响新的组级搜索；旧算法行为对照一致。  
**“研究效果成立”**（未来，且不被软件测试替代）：实际多任务 Pareto 性能优于选定基线，且任何关于动态补偿增益的论断有相应的真实验证。
