# 当前八任务运行记录

这是一份启动与验证记录，不是最终性能报告。

- 服务器：`shangxueyuan@10.159.23.24`，A100 80GB。
- 算法版本：`functional_geometry_ec_v2.1`，来源为本轮提供的 no-expert v2 压缩包。
- 任务：Cars、DTD、EuroSAT、GTSRB、MNIST、RESISC45、SVHN、SUN397，均已实际执行梯度训练。
- 每任务计划 2,000 次更新，任务均衡交替；初始有效批量 64、微批量 32。
- 冷启动完成：8×4 条全参数梯度，初始秩 12，实际已按独立数据的 CE 检查扩张至 14；新增时八个专家的参数变化误差均为 0。
- 当前保存的第 150 次更新验证结果：DTD 48.67%，尚不合格，不能作为最终成功结果。
- 实际训练峰值显存约 23.8 GB，普通任务步骤约 0.5 秒；完整运行仍需数小时。

## 自动继续运行

服务器用户服务 `functional-geometry-v2.service` 正在托管监督程序。本机关闭、SSH 断开不影响它在服务器继续运行。每小时保存一次完整诊断，异常退出自动从检查点恢复；OOM 时逐级减小微批量并保持有效批量 64，所有改动留痕。服务管理器会恢复监督程序本身的异常退出。

结果目录：`/home/shangxueyuan/merging/tsv/results/functional_geometry_v2_no_expert`。

主要监控文件：`monitor/latest.json`、`monitor/latest.md`、`monitor/diagnosis_*.json`、`monitor/events.jsonl`。

训练结束后，程序自动继续执行三个共同参考点的 Fisher/共同块检验、探索性 EC pilot、锁定后的测试评估，生成 `result.json`、`geometry.json`、`history.json`、`config.json`、`run_report.md`、性能曲线与 `final_evidence.zip`。未满足 DTD 80% 门槛或共同结构检验时，报告明确标记失败，不伪装为成功。

## 已完成的检查

1. 原始预训练 checkpoint SHA256 核验通过：`3a381398c0af297b467cee275ec67bcb3efe488acb969fee4205cac13a91258a`。
2. 当前训练脚本 SHA256：`e5cd64df4e8741fbe8a439d55cf7c138588dd23f0c09342525c15f528b663faf`。代码、依赖和分类头的完整哈希见服务器 `artifact_manifest.json`。
3. 基底正交误差约 `5.31e-8`。DTD/MNIST 真实模型上，预测 Fisher 的 KL 二阶近似与实际扰动一致；有符号 CE 导数通过有限差分核验。
4. 扫描 159,295 张图像，发现 SUN397 一张可恢复截断图；保留原始字节和样本划分，开启进程内兼容解码并记录路径、哈希。
5. EuroSAT 使用源镜像的明确三划分，跨划分图像字节哈希交集为 0。
6. 修正原压缩包候选排序使用留出误差的问题，保留原文件；现在训练子集负责拟合排序，独立子集负责验证。
7. 初始共同块完成真实预测 Fisher、独立留出、第三审计、bootstrap 与旋转空假设检查，目前没有通过全部检验的稳定共同块。
8. 已在隔离 `/tmp` 合成夹具中检查后处理、EC、选模锁定、测试缓存恢复、曲线和证据包生成。合成测试输出不混入真实训练指标。

后续应以服务器实时日志为准，本文件和 `run_snapshot` 是启动阶段快照。
