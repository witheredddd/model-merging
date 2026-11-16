# v1.1 实施记录

本次最新用户指令授权实现和运行，替代附件第9节旧的“仅交付设计书”状态。

原 r2 文件保持不变。新增接口使用原 Mode 列，EC 变量数取实际模式数量；没有新增 CE/Fisher 个体过滤，没有固定二维替代。原学习器的数值策略、保护预算和近似缓存语义保持原样。

## 实际模块

- `legacy_adapter.py`：原 A100 恢复、活动刷新、原 recondition/learn；从原 fast_lock 读取预算网格及可靠性设置。
- `chart_state.py`：深快照、功能组与联合模式，不二次归一化或截断。
- `decode_fitness.py`：完整模型重建，原标签 Top-1 与原负归一化目标。
- `ec_adapter.py`：原生 pymoo NSGA-II，可在一代内逐个体精确保存/恢复。
- `chart_loop.py`：真实单图 EC、明确分支协议驱动的后续新图、原补偿重整及可配置原模式出生。
- `state_io.py`、`monitor.py`：原子状态和无决策日志。
- `protocol.py`：运行协议必须显式给出来源，不猜测 z 范围、预算、分支或迁移方式。

## 运行

```bash
PY=/home/shangxueyuan/merging/afg_env/bin/python
$PY chart_loop.py prepare --checkpoint <原已接受断点> --output <新准备目录>
$PY chart_loop.py run --prepared <准备目录> --protocol <明确的运行配置.json> --output <新EC结果目录>
```

`prepare` 真正恢复 A100 完整模型并按原函数刷新活动几何和补偿映射，不产生新参数运动；不需要未定义的 z 控制域。它不等于 EC 已开始。

用户随后明确给定运行协议，已经保存为 `RUN_PROTOCOL.json`：原 r2 半径诱导的联合运动控制域，两张图，每图原生 NSGA-II 50 个体、50 代、seed=1；第一图真实 Pareto 前沿按原规则取25%归一化目标弧长点，第二图刷新原几何及补偿并允许原出生逻辑，不迁移种群。

`control_domain.py` 由实际 B 和原 Space 计算 G、H，以两种参数范数的确定性径向映射解码。包围盒来自 Gram 逆的坐标极值，绝不使用任意 [-1,1]。满列秩时基因直接对应原模式振幅；只有数值秩亏时按原 `ortho` 规则保留有效坐标，并记录至全部原模式振幅的映射。

本轮只用 p0，不打开 p4/p5/p6 或官方 Test。不增加研究对照。

输入字典仍为原 304D 专家重组空间；不是任意 86M 方向搜索。没有训练新专家，没有扩展 seed，没有增加研究对照臂。

## 输出语义

`PREPARED.json` 只表示真实恢复和当前活动几何已就绪；`charts/*/FINAL.json` 才表示真实 EC 图完成；最终 `FINAL.json` 保存跨图实际 Pareto 点。任何软件测试、服务 active 或准备完成都不能当作融合结果。
