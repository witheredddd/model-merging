> Corrected A100 r1 implementation: see REVIEW_AND_RUN.md. Original package narrative below is retained for provenance; its commands and unverified claims are superseded.

# 2026-11-16：任务纤维诱导的在线自适应维数与多张量补偿

这是一套**可执行的下一阶段算法原型**，不是又一份只读取日志的事后分析。核心是：实际参数写回只使用在线学得的控制模式，而非直接写入 304D 局部最优方向。

## 已知的实验依据

原始 `conditional_compensation_20261116` 的 304D Jacobian 缓存已完成；`reachable_geometry_20261116` 显示现有条件 2D 在最严格保护预算下仅达到完整 304D **预测** Cars GN 收益的 22.04%–23.82%。预算扫描所得的单目标最优响应前两轴约覆盖 95%–98%，但此数字**不是通用内在维数**。

## 本轮要真正执行的算法

1. 在 p0 当前参数点读取 DTD/Cars 的 Jacobian、GN 和 CE 梯度，建立一组 **Cars→保护 DTD**、**DTD→保护 Cars** 的不同预算情形。
2. 初始控制字典为空。完整 304D 受限优化**仅作为局部容量教师/出生信号**；从其当前未解释的功能改善中选择一个原始张量作为驱动，并在其专家更新坐标中确定**一个方向**。不继承旧固定 2D 的秩规定。
3. 根据保护任务 Fisher 耦合，逐步引入补偿张量：对给定驱动方向，在当前全部补偿支持上**联合重求**条件最小二乘映射。新增张量必须使保护任务响应能量降低、目标任务可行改善提高。每个新模式始终只有一个独立振幅；支持可跨多个张量。
4. 当已学模式集合仍不能覆盖不同目标/预算下有价值的局部功能改善时，检验其相对于现有响应和已执行**轨迹响应**的独立性，允许新控制模式出生；否则复用已有模式。`max_modes`、`max_compensators` 只是计算安全上限，并**不是模型被规定为固定维数**。
5. 实际更新使用 `B_t z_t`，其中 `B_t` 只包含已学模式。在当前点用真实 p0 CE 和平方根预测分布检查目标改善、保护任务不恶化和有限位移预算；必要时沿同一模式缩步。**没有无约束全 304D 参数写回**。
6. 每次成功写回后，计算已用参数支持的新 JVP/Jacobian 列，重新求条件补偿映射，保存实际功能轨迹；其在线功能新颖性直接影响下一轮模式出生。可配置周期性全 304D 重线性化。不同 anchor 分开学习，最终比较独立控制维数、支持规模、两任务 CE/Top1 及容量覆盖率。

### 最主要的对照

`run_comparators_a100.py` 从同一个 p0 Pareto 锚点独立启动四个真实迭代轨迹：

- `static_pair2d`：原来 p0 选择的条件双张量，映射固定且仅两个控制坐标。
- `dynamic_pair2d`：支持与两维驱动固定，每步随保护任务重算条件映射。
- `global2d`：固定全局专家二维空间。
- `full304`：完整空间受限局部优化，仅作为容量对照，需单列其高自由度和成本。

全部使用相同 p0 更新协议、半径和保护预算。注意每种方法沿自己的真实轨迹行进，最终比较是**同起点、同优化协议**，不把不同轨迹上的单步增益直接当成配对观测。

### 如何立即在既有 A100 服务器启动

服务器代码路径沿用先前实验（见脚本顶部）。先将整个目录放在：

`/home/shangxueyuan/merging/tsv/scripts/adaptive_fiber_20261116/`

之后执行：

```bash
cd /home/shangxueyuan/merging/tsv/scripts/adaptive_fiber_20261116
bash launch_a100.sh
```

对应的程序可直接手动启动（更改 `--anchors`/`--steps` 即可）：

```bash
/home/shangxueyuan/merging/afg_env/bin/python run_a100.py \
  --output /media/shangxueyuan/data1/adaptive_fiber_20261116/main \
  --anchors 0,1,2 --steps 4 --births-per-step 2 \
  --max-modes 16 --max-compensators 8 --heldout
```

在没有权限启动 A100 的外部环境中，可使用 `pytest -q` 检查纯算法和有限步验证的模拟测试。**模拟测试通过不等于真实 CLIP 训练完成。**

### 输出与重要的区分

- `experiment_lock.json`：配置、代码哈希和数据角色。
- `anchor*_live.json`：每次接受/拒绝的更新、补偿支持、控制秩、出生事件、容量曲线、轨迹和实际有限响应。
- `anchor*_completed.json`/`FINAL.json`：迭代完成的结构与评价。
- 单独的 comparator 目录里保存 `BASELINES_FINAL.json`。
- 读写只在**新目录**进行，原有 conditional/reachability 实验不被覆盖。
- p0 用于在线控制与候选选择；p4/p5 **只能作为已经使用过的训练侧探索性验证**，绝不能声称是全新盲测。p6 与官方 Test 不打开。
- 当前受限 304D 的 DTD Jacobian 满列秩，因此算法研究的是**有预算的近似纤维保持**，不是声称发掘到了非零的精确 DTD 纤维切向量。
- 活跃参数列在成功更新后刷新；其余列若未设置全刷新频率会逐步陈旧，任何预测-实际偏差应当如实记录。建议首次结构实验后在可用计算预算内设置 `--full-refresh-every 2` 对照。

### 如何判定维数不是人为指定的

报告每个锚点随真实更新的秩曲线 `0→1→...→d`，并在**不同任务方向、不同保护预算**下对比同点 full304 的容量；只在累计有益功能收益与跨任务验证支持时出生新方向。重点不是某次训练选择了 d=2，而是**d=2 不足时确实会出生 d=3，新增的从属支持又不增加 d**。本方法只是算法原型，最终是否稳定优于静态 2D 与 304D 仍须由实测结果判定。
