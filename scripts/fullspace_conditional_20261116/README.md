# Full-space conditional realization — numerical validation branch

本目录是独立完整参数数学核心，旧 `0f2d922` 及阶段A结果保留。本次只做四项数值验收，不继续旧模式审计，不训练、学习结构或运行EC。

主线冻结为：功能需求族 → 完整空间条件实现 → 联动结构 → 选择性控制 → 在线演化 → EC。**当前只实现前两项接口并进行数值验证，不声称整条方法已完成。**

## 实现

- `geometry.py`、`fullspace_runtime.py`：全部共同视觉参数上的JVP/VJP、CE梯度和模型点重建；没有Space/Q、304D字典或支持筛选。
- `precision.py`：同一数学模型的FP64数值参照，不改网络结构或参数可行域。
- `conditional.py`：完整KKT、显式M=I下的精确输出Schur、显式选择的加权拟合；分别记录原始与对偶残差。无输出ridge或结构截断。
- `demands.py`：CE、当前点完整运动JVP、当前点专家Log；锚点及输出坐标一致性、来源可达性和实际数值可行性分开。原幅度保留，归一化视图不参与求解。
- `validate_a100.py`：随机完整方向、CE方向和专家差方向的多尺度中心差分、JVP/VJP伴随及参数舍入误差。
- `test_*.py`：概率嵌入直接自动微分、精确/加权方程、KKT/Schur同解、CE一阶关系、Log球面半径和样本平均度量、不可达目标及坐标错配检查。

## 实际数值结果

见 `results/NUMERICAL_REPORT.md` 和完整FP32/FP64误差曲线。已检查6个任务/方向组合的FP64最佳差分误差约1e-9—4.3e-7，伴随误差约1e-14或更低。原FP32 DTD三个完整KKT候选均未在80步内收敛，不能称条件最优解；剩余旧精度扫描已停止并保留结果。

FP64完整Schur80步仍未收敛，当前运行同问题的输出对角预条件修复，**完整求解收敛尚未最终验收**。不以导数精度改善代替条件最优性。

## 验证

在已有A100运行环境：

```bash
OPENBLAS_NUM_THREADS=1 /home/shangxueyuan/merging/afg_env/bin/python \
  -m unittest -v test_precision test_conditional test_demands
```

三类方向检查：

```bash
python validate_a100.py --checkpoint <完整模型断点> --probes <原固定probe锁> \
  --output <新输出目录> --precision float64 --directions random,ce,expert \
  --epsilons 0.1,0.01,0.001,0.0001,0.00001,0.000001,0.0000001,0.00000001
```

复用原A100的完整模型加载器、数据与专家文件。`results/fp32/executed_source/`保留导数验收实际执行的数值源码版本；后续输出坐标指纹属于元数据正确性修订，未改变该次数值算子。

新接口的一般M须由调用方显式指定并具有SPD契约；I仅为此前用户明确选择的数值基线。需求族聚合、最小支持容差、模式数及最终维数没有自动默认。详情见`DESIGN.md`、`REVISION_REVIEW.md`和`ACCEPTANCE.md`。

