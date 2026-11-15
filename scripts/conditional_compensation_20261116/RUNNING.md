# 条件功能补偿：固定锚点的第二阶段先导

GitHub快照时间：2026-11-15T18:26:24.567734+00:00。第一锚点304/304已完成；当前第二锚点59/304，三锚点总进度363/912。尚无有限补偿性能结果，未生成最终REPORT.md。服务Result=success仅表示当前未记录退出失败，不代表实验完成。代码与本次服务器源码SHA256一致；原始模型、完整Jacobian缓存与图像继续保存在服务器。

本目录是运行源码与状态证据快照。执行还依赖服务器既有compromise_geometry_20261115的space/context和其CLIP运行环境，不能把本目录单独视为完整离线复现包。

2026-11-16用户授权：不依赖未收敛轨迹、不做聚类前置，直接从Jacobian/Fisher交叉块构造条件补偿，继续DTD保护/Cars改善实验。与旧w延拓解族分开，不宣称Schur补偿仍是旧流形切线。

## 当前作业

- A100 shangxueyuan@10.159.23.24
- Python /home/shangxueyuan/merging/afg_env/bin/python
- 源码 /home/shangxueyuan/merging/tsv/scripts/conditional_compensation_20261116
- 结果 /media/shangxueyuan/data1/conditional_compensation_20261116
- 缓存服务 conditional-cache-20261116.service
- 配对枚举/有限评价 conditional-pilot-20261116.service
- 成对审计 conditional-audit-20261116.service，主任务完成后只读后处理

02:06已确认缓存第一个锚点265/304列。缓存PID1782783、主任务PID1784275均未改变；补充审计等待主候选冻结，不使用GPU。三个锚点依p0线性Global EC原始Pareto fitness的弧长四分位选取，不用p4/p5选锚点。全新计算各锚点的J_D/J_C及真实标签CE方向导数；旧均值点Fisher不直接移用。

## 本轮追加审计：不改主任务

按用户“补一下，不要打扰跑动”的要求，新增独立supplement.py与conditional-extra-audit-20261116.service。没有修改或重启当前缓存和主程序，没有更换目标、主候选、预算或数据划分。额外CLIP前向/JVP只在主任务完成后开始；等待期间只登记协议，主候选冻结后读取小矩阵构造附加对照。

- extra_preregistration.json已登记随机种子202611160154及匹配规则。每个选中配对/预算/锚点另抽一个随机有向张量对，优先精确匹配两端张量元素数与控制秩；不存在替代对时，使用确定的最近20对候选池，并明确标记支持规模不匹配。随机配对使用同一求解器、控制维数、位移与源任务预算。
- damping_nullspace_audit.json对照最小范数数值伪逆（相对秩容差1e-10）、当前映射与相对块谱尺度1e-10/1e-8/1e-6/1e-4的阻尼映射；保存块特征值、秩/条件数、源零空间中的Cars Fisher迹和CE梯度、固定驱动方向的源响应/成本，以及映射变化。
- 保留主任务原比例的驱动/补偿分量消融，另外在完整同等预算下分别重新优化两端。附加有限评价包含随机对照、伪逆与扰动阻尼，以及单端重新优化对照；不覆盖主结果。
- cross_anchor_map_audit.json对同一个有向支持使用固定的典型驱动方向，比较三个锚点的T差异、功能响应与有限效果，不把跨锚点移用默认视为满足新锚点源预算。
- extra_validation_audit.json保存rho_cancel、单独/联合DTD响应能量、有限响应的抵消与叠加误差；分别报告Q域内构造、G外参数比例，以及Cars响应相对J_C G的独立范数比例eta_C。
- extra_controls_lock.json在主候选出现后冻结完整附加向量；实际是否早于主任务首个p4/p5模型文件，按时间检查如实记录，不能事后声称已预注册。当前已登记的是种子和规则，完整候选尚在等待主任务选择。
- p0用于发现；p4为复用稳定性检查、p5为复用训练侧留出，两者均不是新的盲测。逐项配对CE区间不作多重检验后的发现认证。p6_confirmation_bundle.json仅保存未来确认所需的候选哈希与比较清单；不采样、不打开p6，正式确认前仍需固定效应界限与多重比较协议。

附加服务独立记录extra_audit_status.json、extra_audit_failure.json及extra_audit_complete.json。失败只影响附加审计，不触碰主作业。附加GPU工作排在主结果之后，预计额外数十分钟，实际时长取决于主任务选中的不同支持数量。

## 参数域与目标

复用经重建审计的304维分块正交基，全部152张量均保留原支持。每个张量最多2个坐标，Q和M=I保持一致，不将全局旋转后的任意坐标误当张量。没有使用旧失败轨迹、FPCA/SSC、有符号图或L1稀疏选择。

候选由min h_C^T v+.5 v^T K_C v构造，同时约束DTD预测响应范数、真实DTD一阶CE不增加（h_D^T v<=0）、参数位移预算。K_C是CE的预测Fisher/Gauss–Newton正半定近似，非精确CE Hessian。真实评价仍记录负归一化top1 Pareto，CE只是可微构造信号，没有用平均准确率替换fitness。

## 条件映射与公平对照

每个有向m→n使用T=−(K_D_nn+lambda I)^−1 K_D_nm，x_n=T*x_m。lambda只在2×2块病态时用于数值稳定；保存T、阻尼和Schur恒等式。真实泄漏使用[I;T]^T K_D [I;T]，不把含lambda*T^T*T的正则Schur值当纯功能泄漏。

对[I;T]做参数欧氏QR只改变驱动坐标表示，联合控制仍最多2维，与单张量2维对照匹配；不是事后删除功能分量。原始相对联动比例保留。

- 全部152单张量最优局部控制；每个预算都取最佳单张量比较。
- 全部11,476无向对的22,952有向映射，每个映射扫描4个源预算；每个锚点共91,808个小凸问题，不对每对都跑CLIP。
- p0局部预测中，相对最佳单张量增益最高的两个不同无向对/源预算进入有限评价。这是前向名额，不是预设组数或最终控制维数；无正增益或无抵消也照实报告，不强行准入。
- 同一对支持上的4维联合优化只作上界；完整304维也是能力上界，不能把更高自由度的收益称为补偿证据。
- 线性全局专家span中的2维局部CE控制使用相同保护和运动预算，作为更强的同维对照，不冒称又跑了一次历史Global EC。
- 对选中方向，另以它的实际控制成本重新比较全部单张量，并记录驱动/补偿各自单独写入的效果。

## 预算曲线与有限验证

主要R=.003*||Delta_D−Delta_C||；检查1/3、1、10/3倍半径。源范数预算=R*tau*||J_D(-h_C/||h_C||)||，tau=.01/.1/.3/1。平方后才是v^T K_D v的能量预算。这些是扫描的控制预算，不是判断组存在的研究阈值。

每个源预算和半径下，各对照统一约束。选中支持的模式随半径在同一局部几何下重新求解；有限评价实际写入完整参数并核验CE/Accuracy/KL。只做一次局部构造和多尺度有限验证，不把它冒称已完成非线性重线性化迭代。

真正的源抵消同时报告−2<J_D v_m,J_D v_n>、单独源能量与联合源能量；独立数据上直接重新计算。实际有限输出响应额外计算非线性联合能量降低及叠加误差，区分局部交叉项与高阶交互。条件方向相对线性G的新增参数及Cars功能响应独立性另记。

所有系数、支持和比较点在p4/p5预测前冻结。p0选模型；p4诊断、p5最终训练侧留出。它们已被上一阶段使用过，本轮不称全新盲测，官方Test不开。原始归一化常数保持不变。补充脚本对联合方向与匹配单张量、4维支持上界、全局2维及全304上界做同图像配对CE区间；区间为逐项估计，不冒称多重检验后的发现认证。

FP32写入需要的整体径向缩放在看p4/p5之前冻结，驱动/补偿消融继承同一缩放，保持两部分相加关系。不单独剪系数。材料化位移实际范数始终记录。

## 数值验证与计算

两项复合测试通过：条件映射抵消、欧氏预算/一阶CE约束、Schur阻尼恒等式。当前实际缓存块的200次小优化全部通过，Schur恒等式误差约5.2e-17。源码不依赖旧未收敛曲线。

三个锚点各需304×2次JVP，缓存预计共约75分钟；全部成对小问题预计数分钟，有限模型验证和成对响应审计另需数十分钟。整轮约2–3小时。缓存和主任务分开，每列缓存保留，不会因求解器修改重算已有列。必要修改仅重启等待依赖的主程序，未中断缓存。

```bash
cat /media/shangxueyuan/data1/conditional_compensation_20261116/status.json
systemctl --user show conditional-cache-20261116.service conditional-pilot-20261116.service conditional-audit-20261116.service -p Id -p ActiveState -p Result
```

主要输出anchor_selection_lock.json、experiment_lock.json、anchor*_program_audit.json、anchor*_selected_pair_lock.json、all_candidates_frozen.json、finite_results.json、independent_cancellation_audit.json、paired_compensation_effects.json及REPORT.md。完整每对得分、全部候选304维坐标与logits在private/。

只有看完双张量/全空间能力差距后才判断是否需要多张量扩张；当前没有自动学习聚类网络、预设稀疏组或进入EC-z控制。
