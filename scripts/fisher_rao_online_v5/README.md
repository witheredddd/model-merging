# 在线功能几何 v5：代码实现与验收边界

按用户提供的 v5 实施规格新建，旧 `9b4ff39`、`062e87f` 代码和已停止实验保持原样。初版 `9b01696` 交付时只做 CPU 验证；随后用户明确要求运行完整 DTD。首次 A100 启动在冷候选超高矩阵 SVD 处遇到 cuSOLVER 错误，尚未进入坐标训练。该失败目录保留，当前改用逐列两次重正交化构造相同候选跨度，再从新目录运行。

`dictionary.py` 不对 D×n 参数矩阵调用 SVD，只用 FP64 点积和向量操作；小型功能几何矩阵仍使用 QR/SVD。25 项 CPU 测试通过，包括禁止调用高维 SVD、跨度保持和重复方向处理。A100 完整训练状态与实际结果需以随后运行记录为准。

随后审查确认了旧持续性判据缺少跨窗口结构一致性；该轮长跑已停止、检查点保留，未形成完整训练报告。最新修正及“4”仅为预算值的说明见 [REVIEW_RESPONSE.md](REVIEW_RESPONSE.md)。最新配置以 `runner.py --describe` 为准。

最新修正版通过 42 项 CPU 测试：跨窗口残差在同一观测切空间比较，区分同类子空间复现和异类新颖性；记录剩余可达秩，出生必须增加实际功能联合秩；组数及候选数不再默认固定为四。初始分组支持 `responses/single/random`，获选几何还要经过独立的新训练批次确认。[A100 受控 DTD 记录](evidence/20261112/controlled_clip/REPORT.md)已完成，亦暴露坐标冗余导致反复刷新，随后改为约化子空间求解；该次轨迹不能作为求解器修复后的长程验收。长跑保持停止。

整体数学设计成立于局部可微、数值可达等条件下。固定任务预测映射并不固定当前 Jacobian；功能图由真实模型复合产生，不预设全局光滑纤维丛。动态低秩表示不能仅靠表示终点证明低控制自由度。

## 参数状态与坐标步

唯一实际模型表示是：

```text
theta = theta0 + S @ sum(group.Z @ group.a)
U_group = current_J_S @ group.Z
```

`state.py` 不保存可自由优化的历史高维锚点或额外残差；全部 online 更新经 `S,Z,a`。S 对全部活动参数稠密作用；当前资产适配器沿用 48 个视觉矩阵的活动范围，冻结头及其他参数，并非声称训练 CLIP 的每个偏置/归一化参数。S 的列正交性校验按字典版本缓存，张量被意外原地改动时会重新验证。

`initialization.py` 从 theta0 上的可配置训练批次提取梯度与真实 JVP，只观察候选、不做 AdamW 预热。第一步模型严格等于原始基座。实际初始字典秩由这些候选支持，不凑满 d；95% 响应能量只初始化容量。响应暂定分组、单组与随机分组保持相同联合秩；默认不强制组数，不解释为组数量真值。

`geometry.py` 在 G 所代表的字典子空间 Q 内解小系统 `H = (AQ)^T(AQ) + (lambda+epsilon) I`，再最小范数提升回组坐标。重叠组的纯坐标核不参与条件数判断；epsilon 正则字典系数增量。真正病态时回退该子空间内的欧氏方向并记录原因；单批拒绝不终止训练。记录新增量的近核能量，不将历史总位移强制重新投成水平向量。信赖半径由当前功能敏感度得到，与任何预热位移无关。

真实预测与当前局部一阶模型的误差影响步长接受/收缩。监测子集保留全探测集的 FR 嵌入归一化；局部误差含绝对尺度，避免零位移相对误差病态。坐标 CE 使用可配置的小噪声容许量，而非固定 `1e-10` 严格下降；明显失控仍会拒绝。

## 在线窗口、字典和组结构

`trainer.py` 默认每 8 个批次重新计算实际当前模型的 `q` 和 `A=J S`。这只更新小型局部参考 `reference_x`，**不会改变 theta0 或把当前高维模型折入自由锚点**。局部近似恶化也会触发刷新。

`evidence.py` 从当前模型取得独立候选 `v` 并计算真实 `J_theta v`，分别记录其可达投影、字典外残差、独立训练 CE 作用。责任包含覆盖、相对其他组的残差新颖性和独立候选作用，并设“未归属”分量；不强制所有响应归组，不以当前坐标幅度分配责任。坐标伴随与真实 JVP 不混为一类信号。

`transforms.py` 的字典替换保护总位移 `S x`，优先替换剩余任务证据较低的方向，组贡献只是软优先。转移 Z 后重算当前 A/提升关系，禁止复用旧字典 Jacobian。维数增长受持久结构/异类新颖性、剩余容量、维数预算和容量代价约束，默认可以进入等成本竞争；`--fixed-dictionary-rank` 是关闭扩维的对照。

组方向通过真实响应的加权 SVD、可达提升与有界混合提出；Procrustes 对齐、联合坐标重拟合后，明确区分纯重表达和真实参数移动。支持暂定出生、同秩合并、淘汰。它们是可被否定的候选，不等于已发现语义模块。

`lookahead.py` 比较相同批次、相同独立训练检查集及相同计费图像成本的可配置正整数步未来更新。允许候选起始 CE 在安全范围内上升；以后续任务收益、容量代价和软连续性共同决定。覆盖项用于构造提议，不能靠响应重建收益掩盖未来 CE 退化。新旧分支及失败候选的计算都计费；提交的是已付费分支状态，没有免费训练步。

几何事务失败会整体回滚 S/Z/a 和局部几何，成本不退款，下一批次继续。旧单列字典无法同时保持非零模型并替换为独立方向时，正常拒绝；不会用隐藏残差绕过限制。几何变化记录主角度、组数量/坐标数、实际参数联合秩、实际功能联合秩及前瞻 CE。

## 数据、计费与对照

`clip_runtime.py` 仅复用 v4 冻结的基座、头和数据加载模块。它不读取 warmup/candidate/expert 缓存。固定无标签 `X_fr` 从原训练池抽取，并从优化训练样本中排除；Vcal 和独立 Xprime 不进入在线方向学习。原 DTD 镜像官方折号仍未独立验证。

`runner.py` 提供手动 online 入口与独立 `adamw_baseline.py` 对照，均从同一 theta0 开始。所有在线梯度、真实 JVP、监测、候选 CE 和两条前瞻分支都计费；逻辑操作数与底层实际 runtime 计数另报，不把图像数相同称为 FLOPs 相同。

`reporting.py` 在固定图像成本网格观察**已提交的主模型**。尚未被选择的 lookahead 分支不会冒充即时模型；同一网格上的 Vcal 最优模型另报并保存。初始化探测已经花掉预算但尚无任务写回时，网格报告的仍是原始基座。最终和 Vcal 选定模型另在完整独立 Xprime 上评估，不使用官方 test。

支持固定字典、冻结 S/Z、单组、随机同总秩早期划分、无组响应旋转等控制。`control_audit.py` 提供冻结未来窗口、同总秩合并和删除后同容量替换；`fingerprints.py` 使用独立实际 JVP、参数归一组方向、随机划分参照与跨种子匹配。它们不能独自证明组身份。

## CPU 验证与手动入口

```bash
# 不加载 CLIP，不使用 GPU
python -m unittest discover -s scripts/fisher_rao_online_v5 -v
python scripts/fisher_rao_online_v5/cpu_smoke.py --output /path/to/cpu-smoke.json
python scripts/fisher_rao_online_v5/runner.py --describe
```

下面是长程手动入口；受控运行见上方记录，当前长跑保持停止：

```bash
python scripts/fisher_rao_online_v5/runner.py --method online \
  --root /home/shangxueyuan/merging/tsv --output /path/to/new-v5-output \
  --seed 11 --steps 64 --probe-size 128 --audit

# 固定同一图像预算的独立 AdamW 对照；应比较实际成本和统一成本网格
python scripts/fisher_rao_online_v5/runner.py --method adamw \
  --root /home/shangxueyuan/merging/tsv --output /path/to/new-adamw-output \
  --seed 11 --until-budget --budget-images 1000000
```

`--steps` 是开发/smoke 上限，同步数不代表同计算量；正式比较应对双方用相同 `--until-budget --budget-images`、相同数据与报告网格，并检查实际未用完的预算。输出目录必须为空，不恢复已停止的 v4 作业。离线审计成本与在线成本分开，不反向修改训练。

## 尚未获得的证据与实现限制

- CPU 测试和两窗口真实 CE 合成问题已验证关键不变量、有限差分、等成本前瞻、刷新后下一次实际写回及失败回滚。它们不是 DTD/CLIP 结果。
- **尚未运行 A100 两窗口 CLIP smoke、两个训练种子的短程 DTD 或四种子正式比较。** 当前没有 DTD v5 性能表，Val≥80% 是待检验目标。
- 规则中的容量代价、连续性权重、持久窗口、重叠阈值和噪声容许量属于显式启发式，不能保证辨识固有组数、全局收敛或迁移。初始响应划分只是暂定组织，单组与随机对照保持同一联合秩；后续出生/合并仍需独立证据。
- 字典增长的等成本比较给较小分支加入真实外部候选 JVP 检查；小矩阵系统维数仍可能不同，因此不是严格等 FLOPs。全参数候选梯度及密集 S 的真实显存/速度仍需 CLIP profiling。
- 当前支持局部误差控制和在线重线性化；没有开启任意自由 B、多步高维 CG、纤维优化或 EC。不把预留机制计作已完成实验。
- 长程冻结复用、跨种子功能身份和容量匹配的整套正式统计尚未验收。不能因 S 可随模型移动，就宣称整个训练只需一个固定低维空间。

[DTD and SUN397 medium-run protocol, complete resume and historical checkpoints](MEDIUM_RUN.md): 46 CPU tests passed; this is not a claim that the real medium runs have completed.
