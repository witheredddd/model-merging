# 核心数值数据快照（2026-11-19）

该数据包按用户要求发布，供独立复核、重算补偿关系和尝试新研究思路。它是运行中实验的一次快照，不代表实验完成或自动保护性分组已成功。

## 从这里读起

| 文件/目录 | 内容 |
| --- | --- |
| `snapshot.json.gz` | 捕获时的参数位置、需求目录、空间来源、各位置状态、作业元数据、运行版本和原17个位置的身份哈希 |
| `evaluations.csv` | 完整有限候选、受限最优与独立probe比较，包含目标误差和目标差上下界 |
| `solver_history.csv` | 每个问题的迭代数、真实正规残差、参数/功能误差界及目标值变化 |
| `functional_updates.csv` | 每次候选的功能收益、保护能量变化、目标保持及源可信度 |
| `joint_mode_curves.csv` | WLS折中与严格固定响应曲线分列；不自动选择维数 |
| `timing.csv` | 分项实际耗时；不能把全部标签相加当作独立总墙钟 |
| `run/adaptive_evaluations/` | 所有已导出比较的完整JSON：活动/冻结/累积、学习/验证、逐任务能量及系数 |
| `run/solves/` | 各轮数值报告；每个问题最近已完成参考的八任务功能响应NPZ |
| `run/demands/` | 数值目标响应y，包括不同位置、目标任务、来源和独立probe |
| `run/regional_coupling/` | 已计算的区域任务响应和参数范数平方；可重新构造保护Fisher交叉矩阵 |
| `mechanism/` | Schur图T/L/S、选择性模式、固定驱动补偿、重新优化消融及跨目标需求对照；保留观察版本 |
| `MANIFEST.json` | 捕获时间、状态哈希、文件校验、原始文件哈希及NPZ数组形状/类型 |
| `SHA256SUMS` | 本数据包全部发布文件的校验（不包含清单本身） |

## 数值与作用域

- 参数维数86,192,640。参数位置来自既有DTD普通训练配方重放的17个保存位置，末端与原专家逐张量一致；不是新训练写回，也不代表认证每个原始中间步。
- 所有误差为比例，乘100才是百分比。任务响应为逐样本平方根概率嵌入 `2 sqrt(p)/sqrt(N)` 的数值切向量，NPZ保留原始数组精度。
- 完整参考通常是有限CG候选，不能直接称为严格最优；用对应正规残差和误差界解释目标差。
- WLS目标同时包含目标拟合、保护扰动和参数正则。WLS降低不保证保护能量降低。
- 固定响应曲线与WLS折中不是同一种评价；不可达条目不能按零保护能量解释。
- 区域数据只包含已经计算的运动方向。每个区域的一根运动不代表区域全部自由度，不是自动发现的组成员。
- 图族合并与原区域空间相同的终点值是等价性校验。跨目标需求测试在同一物理区域基上进行；不同位置各自有结果，不等于完成了冻结保护结构的跨位置实测。
- probe=1是与学习probe分离的数值开发探针，不是官方测试集，也不保证专家训练未见这些图像。验证参考未用于扩展学习空间。
- 活动坐标、历史累计坐标、冻结坐标不是参数组数量；新位置的活动空间可以仍处于重生成初期。

## 读取示例

```python
import gzip, json
from pathlib import Path
import numpy as np

root = Path('data/functional_space_20261119')
with gzip.open(root/'snapshot.json.gz', 'rt', encoding='utf-8') as f:
    snapshot = json.load(f)

region = next((root/'run/regional_coupling').glob('*/*.npz'))
with np.load(region, allow_pickle=False) as z:
    print({key: (z[key].shape, z[key].dtype) for key in z.files})

# 对同位置、同源运动、同probe的区域向量：
# C_ij^P = sum_k w_k * dot(response_k_i, response_k_j)
# 原生区域互不相交，参数Gram对角元为parameter_norm_squared。
# 不要把不同位置或不同源控制的区域文件拼成同一物理基。
```

## 发布边界与可复核性

不包含图像、图像ID列表、模型权重、原始参数轨迹、完整参数方向或大Krylov状态。服务器绝对路径替换为`$RUN`、`$MECHANISM`、`$RESOURCE`。这些文件仍留在A100；快照中保留来源哈希及相对引用，因此有些引用文件有意不随包发布。

JSON使用gzip，数组使用无损NPZ压缩。`MANIFEST.json`中的原文件哈希与发布文件哈希分开，重新压缩和路径清理会改变文件字节，但不会降精度。运行中尚未落盘的数据不在本快照内；使用捕获时间和文件清单确定覆盖范围。这里不包含源码目录之外新设计方案的验证结论。
