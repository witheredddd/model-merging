# Functional Geometry EC v2.1 — 八任务无专家在线几何学习

本轮已经完成八任务各 2,000 次更新及锁定后的评估。**研究性能目标未达到：DTD Val 59.31%、Test 57.82%，低于 80% 验证底线。** 结构化 EC pilot 的 DTD Test 为 46.54%；不能宣称融合成功或近乎无损。

这是新的共享梯度子空间研究路线，独立于仓库旧 TFPML/Fourier 迭代。初始秩 12，最终秩 48；不读取已有专家 checkpoint。原始训练代码保持与服务器记录的文件哈希一致，没有为发布改写训练算法。

- [实现与可执行算法](IMPLEMENTATION.md)、[方法原文](functional_geometry_ec_v2_no_expert.md)、[配置](config.json)
- [结果报告](evidence/run_report.md)、[最终指标](evidence/result.json)、[几何检验](evidence/geometry.json)
- [完整原始证据 ZIP](evidence/final_evidence.zip)：包含 history.json、逐步日志、Fisher/JVP 小矩阵及逐图像估计、数据划分、每小时诊断、选模锁定与测试结果。
- [真实 CLIP 数值检查](evidence/implementation_checks.json)、[隔离合成集成测试](evidence/pipeline_integration_check.json)
- [运行环境](evidence/environment.json)、[服务器原始哈希清单](evidence/artifact_manifest.json)、[最终完成标记](evidence/final_complete.json)
- [训练开始阶段的历史记录](STARTUP_RECORD.md)不是当前运行状态；本轮已完成，服务正常结束。

功能矩阵仅使用每任务每组 8 张图像。theta0 和均值融合参考点的共同块通过当前局部检验，中间参考点未通过；不能解释为跨状态或全局隔离成立。末期 DTD 的梯度投影能量比例均值约 2.58%，容量上限触发后仍存在大量不足记录。这是诊断线索，不是对失败原因的最终证明。

## 运行入口与依赖

代码保留本次执行使用的服务器布局常量 `/home/shangxueyuan/merging/tsv`，以及 A100/CUDA 和已有 CLIP 资产路径。它不是已经移植为任意路径的安装包。

将本目录部署至 `tsv/scripts/functional_geometry_v2`，将仓库的 `src`、`scripts/dtd_grassmann4.py`、`scripts/tfpml/runtime.py`、`scripts/fetch_matched_eight.py` 放到相应位置。先按 `evidence/environment.json` 准备依赖和原始预训练 checkpoint，再执行：

```bash
cd /home/shangxueyuan/merging/tsv/scripts/functional_geometry_v2
python prepare.py
python -c "from engine import Engine,cold; cold(Engine())"
python supervise.py
```

`functional-geometry-v2.service` 提供用户级 systemd 托管，每小时诊断不暂停训练。不要在已有结果目录直接重新执行冷启动；新实验应配置独立 OUT 路径，并同步监督程序及 service 路径。

`prepare.py` 复用并替换旧下载辅助函数中的 download 实现，支持本次 Python 环境；不要把旧 `fetch_matched_eight.py` 直接作为本轮主入口。

`reconstruct.GeometryBundle` 提供 theta0+Bz 专家及融合模型接口。数据集、模型权重、密钥和环境未上传；原预训练与分类头的哈希在原始清单中。复现数值时应同时核对数据划分与精度设置。

## 代码验证

```bash
python functional_commutant_core.py --self-test
python -m py_compile engine.py train.py supervise.py analyze.py finalize.py report.py reconstruct.py
```

`verify.py` 检查真实 CLIP 的 CE 导数与 Fisher/KL 近似；`test_pipeline.py` 使用隔离的 /tmp 合成夹具检查后处理和恢复，合成数字不混入真实指标。`functional_commutant_core.original.py` 保留压缩包原实现，当前版本修正了用留出误差排序候选的问题。
