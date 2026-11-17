# model-merging — functional geometry / TFPML

## Protective mechanism verification

[Conditional graphs, selection spectra and fixed-response comparisons](scripts/protective_mechanism_20261118). The shared residual bank is a numerical baseline, not discovered protective groups. Arbitrary parameter supports and reoptimized removals are supported; automatic support discovery and EC integration remain unfinished.

## Current research: adaptive conditional functional spaces

[Adaptive entry, streaming algebra, tests and method](scripts/functional_space_diagnostic_20261117). The current entry is `adaptive.py`. It retains existing shared coordinates and CG states, evaluates candidate extensions by their actual improvement of the same functional WLS, and schedules tasks, positions, independent probes, refinement and regional diagnostics without a unified numerical admission gate. New-position demands are regenerated at that position, not copied from an old tangent space. Parameter algebra uses disk-backed blocks and incremental Gram updates, with no mode-count cap. Frozen, accumulated and evolving spaces remain distinct. Runtime cost, uncertainty and provisional structure are recorded; source tests do not establish successful low-dimensional coupling.

## Earlier observation prototype: conditional functional-space identification

[Space-identification source and tests](scripts/functional_space_20261117). Existing ordinary-training parameter positions supply the carrier. Task functional demands and protection constraints generate the motions and drive/compensation modes to be studied. No new CE optimization, accuracy acceptance gate, synthetic training path, or trajectory-PCA substitute is used. Counterfactual mode-controlled models are observed without changing the input carrier. Earlier controlled-update experiments remain archived separately.

## Earlier experimental controlled-update implementation: r3

[Protected compensation graphs, selectivity modes and discrete real-model control](scripts/functional_control_r3_20261117). This revision separates exact-response and joint weighted-fit objectives, constructs full-complement compensation actions after full-space motion, uses graph-specific spectral modes for real parameter updates, and rebuilds geometry at committed new anchors. Numerical uncertainty affects actuation; candidate histories are retained. Earlier r1/r2 code remains historical. Only source, tests and method notes are published; model resources and experiment artifacts remain external.

## A100 control source update (2026-11-17)

Runtime source: [r1](scripts/eight_task_online_repair_r1_20261117), [r2](scripts/eight_task_online_repair_r2_20261117). [Diagnostic source and tests](scripts/coordination_audit_20261117). This update publishes code and usage notes only; experiment artifacts remain outside this update.

当前发布 **`functional-geometry-v2.4.0-sun397-metric`**。文件校验见 [SHA256SUMS](SHA256SUMS)。

[SUN397 参数度量改版](functional_geometry_sun397_v2_4)。归档时已完成 4/20 轮。最低验证 CE 对应第 3 轮，准确率 66.05%，CE 1.1958。本轮训练进行中，测试结果待完成后生成。

DTD 连续几何修正版已完成：Val 82.45%、Test 78.78%；80% 最低门槛通过。本轮仅验证 DTD 任务学习，不验证多任务功能结构或 EC。

[修正版代码](functional_geometry_dtd_v2_2)、[结果与曲线](functional_geometry_dtd_v2_2/results/REPORT.md)、[修正边界](functional_geometry_dtd_v2_2/CORRECTION_NOTE.md)。

上一版（已失败归档）：[八任务在线功能几何代码](functional_geometry_v2)、[结果报告](functional_geometry_v2/evidence/run_report.md)、[完整原始证据](functional_geometry_v2/evidence/final_evidence.zip)。**八任务各 2,000 次更新已完成，但 DTD Val 59.31%、Test 57.82%，未达到 80% 验证底线；EC pilot 也未成功。** 旧 TFPML 迭代及其历史结果保持独立。

本仓库保存任务驱动傅里叶参数更新曲面的实现、DTD 原始实验结果与诊断证据。
**v3任务表达阶段通过：锁定第7轮映射的Val为82.45%、Test为81.91%。这不证明局部几何或冻结后低维控制已经成立。**

| 迭代 | 内容 | 验证状态 |
|---|---|---|
| `tfpml-0.1.0-dtd20` | [原始训练代码](tfpml/iterations/v1)、[原始结果](tfpml/results) | 20轮完成，冻结后Val 73.14%、Test 70.85% |
| `tfpml-0.2.0-audit.20261106` | [当前代码](tfpml)、[诊断报告](tfpml/DIAGNOSIS.md)、[原始诊断](tfpml/audit_results) | 完整元梯度有限差分和真实CLIP更新通过；不能宣称已提升准确率 |
| `tfpml-0.3.0-expression` | [逐权重傅里叶映射](tfpml/dense_fourier.py)、[任务训练入口](tfpml/train_expression.py)、[原始结果](tfpml/results_v3) | 持续更新完整映射；达到验证门槛后锁定第7轮模型，Test 81.91% |

先阅读：[版本变更](CHANGELOG.md)、[迭代与哈希元数据](ITERATIONS.json)、[运行说明](RUNNING.md)。

v3：[完整结果说明](tfpml/RESULT_V3.md)、[方法与资源配置](tfpml/V3_EXPRESSION.md)、
[最终原始指标](tfpml/results_v3/expression_final_result.json)、
[测试前选择依据](tfpml/results_v3/expression_selection_before_test.json)、
[独立重建复核](tfpml/results_v3/reconstruction_verification.json)。
训练映射约27.18亿参数、控制变量4个、峰值分配显存48.94GiB。
移除跨4096输出块共享的129维输出字典；16组频率仍是明确的有限函数表示选择。
上限20轮，达到验证性能要求后锁定7轮/371步的模型；并非跑完20轮。
v3按验证性能门槛选定，v1按冻结阶段最小CE选定，不能隐去选择策略差异。

## 证据索引

- [最终性能](tfpml/results/result.json)、[逐轮曲线](tfpml/results/history.json)、[逐步日志](tfpml/results/steps.jsonl)
- [冻结几何诊断](tfpml/results/geometry.json)、[原配置](tfpml/results/config.json)
- [数据划分](tfpml/results/split.json)、[模型选择锁定](tfpml/results/selection_lock.json)
- [梯度链路、切空间对齐、真实下降测试](tfpml/audit_results/diagnosis.json)
- [完整元梯度有限差分](tfpml/audit_results/full_meta_finite_difference.json)
- [真实CLIP完整元梯度](tfpml/audit_results/full_meta_smoke.json)
- [标准内外批次16的集成检查](tfpml/audit_results/full_meta_integration.json)
- [环境包版本](tfpml/audit_results/environment_packages.json)

没有上传数据集、模型权重、虚拟环境或凭据。未上传模型的SHA256列在ITERATIONS.json。
Git提交哈希标识仓库快照；SHA256SUMS标识文件字节，两者用途不同。

## 已知限制

以下保留旧 TFPML 迭代的历史限制；本轮 DTD 修正版的范围与限制见其独立报告。

旧分类头缺失，目前使用按原函数和模板重建的冻结零样本头，尚未与旧头逐张量核对。
本机缺少完整DTD专家权重，无法完成专家表示能力检查。
原始v1早期映射checkpoint未保存，不能补造第5轮冻结实验。
满秩Jacobian只代表局部可移动性，不证明各方向有任务收益或非线性优势。

支持模块来自现有TSV项目快照，其许可保留于[THIRD_PARTY_TSV_LICENSE](THIRD_PARTY_TSV_LICENSE)。
