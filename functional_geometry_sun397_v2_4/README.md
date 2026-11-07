# SUN397 v2.4.0 参数度量实验

沿用 SUN397 的 20 维共享初始化、种子 42、数据划分和 20 轮预算。坐标与水平基底步共用学习率，采用阻尼 0.05、有限参数位移上限 0.10 和秩一极分解回缩。步长检查使用有限参数位移和一阶 CE 下降量。日志抽查完整位移与更新后同批 CE。多状态轨迹目标留待单独实验。

归档时已完成 4/20 轮。最低验证 CE 对应第 3 轮，准确率 66.05%，CE 1.1958。本轮训练进行中，测试结果待完成后生成。

先前两组 20 维角度上限实验的最终指标及本轮归档时的配置、历史和源码哈希见 [server_snapshot.json](server_snapshot.json)。旧角度上限 0.05 和 0.02 组的锁定测试准确率分别是 74.21% 和 75.07%。

在现有云端部署中，将本目录和相邻的 `functional_geometry_sun397_v2_3` 源码部署到 `/root/autodl-tmp/tsv/scripts/`。训练复用项目根目录的 `src`、`scripts/dtd_grassmann4.py` 和 `scripts/tfpml/runtime.py`。依赖已准备的 SUN397 数据、冻结分类头、预训练权重及旧角度实验的共享初始基底和配置。输入身份及 48 个参数矩阵布局会在启动时核验。

```bash
/root/miniconda3/bin/python -B /root/autodl-tmp/tsv/scripts/functional_geometry_sun397_v2_4/metric_step.py
/root/miniconda3/bin/python -B -u /root/autodl-tmp/tsv/scripts/functional_geometry_sun397_v2_4/supervise_metric.py
```

输出位于 `/root/autodl-tmp/tsv/results/SUN397_rank20_parameter_metric`。最佳基底持久化到项目数据盘，最近一轮恢复状态放在 `/dev/shm`，实例重启后可回到持久化最佳检查点。最终按最低验证 CE 锁定一组权重后测试。
