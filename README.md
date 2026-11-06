# model-merging — TFPML / DTD

当前发布：**`tfpml-0.2.0-audit.20261106`**。文件校验：[SHA256SUMS](SHA256SUMS)。

本仓库保存任务驱动傅里叶参数更新曲面的实现、DTD 原始实验结果与诊断证据。
**v1 的最终测试准确率为70.85%，未达到80%目标；v2是通过实现检查的修正版，尚未完成新的20轮性能验证。**

| 迭代 | 内容 | 验证状态 |
|---|---|---|
| `tfpml-0.1.0-dtd20` | [原始训练代码](tfpml/iterations/v1)、[原始结果](tfpml/results) | 20轮完成，冻结后Val 73.14%、Test 70.85% |
| `tfpml-0.2.0-audit.20261106` | [当前代码](tfpml)、[诊断报告](tfpml/DIAGNOSIS.md)、[原始诊断](tfpml/audit_results) | 完整元梯度有限差分和真实CLIP更新通过；不能宣称已提升准确率 |

先阅读：[版本变更](CHANGELOG.md)、[迭代与哈希元数据](ITERATIONS.json)、[运行说明](RUNNING.md)。

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

旧分类头缺失，目前使用按原函数和模板重建的冻结零样本头，尚未与旧头逐张量核对。
本机缺少完整DTD专家权重，无法完成专家表示能力检查。
原始v1早期映射checkpoint未保存，不能补造第5轮冻结实验。
满秩Jacobian只代表局部可移动性，不证明各方向有任务收益或非线性优势。

支持模块来自现有TSV项目快照，其许可保留于[THIRD_PARTY_TSV_LICENSE](THIRD_PARTY_TSV_LICENSE)。
