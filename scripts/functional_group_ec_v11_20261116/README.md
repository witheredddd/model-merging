# 动态功能组 EC：v1.1 实际运行版本与结果

本目录包含 A100 实际执行的完整新增源码、用户冻结的运行协议、设计书，以及本轮两张图的 5,000 次真实评价。执行代码以 `results/experiment_manifest.json` 中的 SHA256 为准；旧 r2 算法文件没有修改。

## 最终结果

| 项目 | 第一图 | 第二图 |
| --- | --- | --- |
| NSGA-II | 50 个体，50 代，seed=1 | 50 个体，50 代，seed=1 |
| 真实评价次数 | 2,500 | 2,500 |
| 实际模式数 | 9 | 11 |
| 全部候选 DTD Top-1 | 97% | 97% |
| 全部候选 Cars Top-1 | 97% | 97% |

起点同样为 DTD 97%、Cars 97%。5,000 个实际模型哈希不同，但只有一个不同的双目标 fitness。因此本轮没有取得 Top-1 增益，也没有得到可区分的准确率 Pareto 权衡；模式增长或 CE 小幅变化不改变这个结论。结果仅来自 p0，不能称为独立验证或官方 Test 成绩。

正式 EC 计时约 10,154 秒；此前完整断点恢复及活动几何准备单独计时，没有算作零成本。

## 实现与协议

- 参数运动仍在原 304D 专家重组字典中，完整视觉参数重建为 86,192,640 维。
- 联合控制域为 `||Bz|| <= R` 且 `||QBz|| <= R`，`R` 继承原 r2；确定性径向映射处理控制域。有限包围盒由实际 Gram 推导，不使用任意组振幅区间。
- EC 基因来自实际模式；仅按原数值秩规则处理冗余坐标。无固定二维设置。
- 第一图按真实目标去重后的 Pareto 前沿 25% 弧长位置选点。该图目标全部相同，故按明确的并列规则选取首个真实候选。
- 换图后重新计算原活动几何、补偿映射，并调用原出生机制，得到 11 个模式。第二图重新初始化种群，不迁移个体。
- 目标保持各任务负归一化 Top-1。没有新增 CE、Fisher、新颖性或性能筛选条件；原学习器已有规则保留。实际评价使用 p0。

## 文件

- [执行设计书](CODE_DESIGN_EXECUTION.md)、[原 v1.1 参考稿](CODE_DESIGN_v1.1.md)、[实施说明](IMPLEMENTATION.md)。
- [冻结协议](RUN_PROTOCOL.json)：本轮所有显式调度选择及来源。
- [最终报告](results/REPORT.md)、[数值汇总](results/RESULTS_SUMMARY.json)。
- [实际运行清单](results/experiment_manifest.json)：实际代码、原依赖、源断点与目标常数来源哈希。
- `results/evaluations.jsonl.gz`：按图顺序保存全部 5,000 条真实评价，保留原记录全部字段，包括原始基因、实际模式振幅、304D 模型坐标、完整模型哈希、目标、CE/KL 和成本。
- `results/charts/000/`、`001/`：两图完整模式快照的 JSON 表示、控制域、功能组、真实种群结果及换图记录。
- [结果索引](results/RESULTS_INDEX.json)：导出文件 SHA256，以及留在 A100 的模型权重、完整优化器断点和未压缩档案路径/哈希。

大权重与优化器 pickle 仍保存在 A100；仓库保留可读坐标和结果，不重复上传数百 MB 模型文件。

## 在原 A100 环境运行

此代码依赖既有 r2、`compromise_geometry_20261115`、CLIP 模型、原专家与固定 probe。private `space.pt`、数据和模型不在 Git 中，路径和哈希见实际运行清单；不声称裸仓库可以脱离这些资源直接复现。

```bash
cd /home/shangxueyuan/merging/tsv/scripts/functional_group_ec_v11_20261116
export PYTHONPATH=/home/shangxueyuan/merging/compromise_geometry_deps
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
PY=/home/shangxueyuan/merging/afg_env/bin/python

$PY -m unittest -v test_implementation

# 每次使用新的输出目录，禁止覆盖本轮结果。
$PY chart_loop.py prepare \
  --checkpoint /media/shangxueyuan/data1/adaptive_fiber_20261116_r2/archived_other_anchors/anchor0/adaptive/checkpoints/step004.pt \
  --output /media/shangxueyuan/data1/functional_group_ec_new_run/prepared

$PY chart_loop.py run \
  --prepared /media/shangxueyuan/data1/functional_group_ec_new_run/prepared \
  --protocol RUN_PROTOCOL.json \
  --output /media/shangxueyuan/data1/functional_group_ec_new_run/ec
```

恢复 EC 使用相同协议、输出目录及 `--resume`，会恢复种群、随机数状态、当前代中已评价个体和预算位置。只恢复权重而重新初始化 EC 不是精确恢复。

本轮 8 项软件检查已在 A100 通过，覆盖原学习行为、解码、联合控制域、原数值秩、弧长选择、监控无副作用及原生 NSGA-II 逐个体断点恢复。它们只验证实现，不是研究效果证明。
