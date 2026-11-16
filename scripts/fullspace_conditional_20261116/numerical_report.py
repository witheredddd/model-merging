"""Summarize measured numerical checks without selecting modes or changing solvers."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main(root):
    data={name:json.loads((root/name/'RESULTS.json').read_text()) for name in ['fp32','fp64']}
    lines=['# 完整空间数学核心：数值验收报告','',
           '研究架构冻结；本次只检查坐标、导数、条件方程和需求一致性。没有结构学习、模式筛选或EC。', '',
           '## 导数结果','',
           '同一完整模型参数点，两个任务、三类完整方向；每种精度均保留8个中心差分步长。下表的最小误差用于描述误差曲线，不是只挑一个步长替代完整记录。', '',
           '|任务|方向|FP32最小相对差分误差|FP64最小相对差分误差|FP64对应h|FP64伴随相对误差|',
           '|---|---|---|---|---|---|']
    fig,axes=plt.subplots(2,3,figsize=(13,7),constrained_layout=True)
    for i,task in enumerate(['DTD','Cars']):
        for j,direction in enumerate(['random','ce','expert']):
            rows={name:next(r for r in d['records'] if r['task']==task and r['direction']==direction)
                  for name,d in data.items()}
            best={name:min(r['finite_differences'],key=lambda a:a['JVP_relative_error']) for name,r in rows.items()}
            lines.append(f"|{task}|{direction}|{best['fp32']['JVP_relative_error']:.5g}|{best['fp64']['JVP_relative_error']:.5g}|{best['fp64']['epsilon']:g}|{rows['fp64']['adjoint_relative_error']:.5g}|")
            for name,r in rows.items():
                axes[i,j].loglog([f['epsilon'] for f in r['finite_differences']],
                                [f['JVP_relative_error'] for f in r['finite_differences']],'o-',label=name)
            axes[i,j].set(title=f'{task}: {direction}',xlabel='Central difference h',ylabel='Relative JVP error')
            axes[i,j].invert_xaxis();axes[i,j].grid(alpha=.2);axes[i,j].legend()
    fig.savefig(root/'derivative_curves.png',dpi=160);plt.close(fig)
    lines += ['', 'FP64保持同一模型数学表达、参数点、任务头与样本；本模型无需替换LayerNorm模块。FP64不是重新训练的专家。FP32与FP64的差别是数值计算精度，不是参数支持或结构维数。', '',
              '这些结果支持已检查方向上的解析导数一致性。FP32的误差随方向和尺度而异，DTD CE方向存在强局部曲率，较小步长又受FP32舍入影响。不能仅凭伴随一致性证明任意有限步长的线性化都准确。', '',
              '## 条件求解状态','',
              '原FP32三组DTD候选均达到80步上限，未收敛。完整参数点、候选与真实有限评价保留在A100；停止剩余扫描以先完成数值验收，不把候选当条件最优解。', '',
              '|gamma|目标约束相对残差|对偶驻点相对残差|参数范数|数值状态|',
              '|---|---|---|---|---|']
    candidates=json.loads((root/'kkt_fp32_candidates.json').read_text())
    for r in candidates['rows']:
        a=r['solver']
        lines.append(f"|{r['gamma']:g}|{a['response_relative_residual']:.6g}|{a['stationarity_relative_residual']:.6g}|{a['parameter_norm']:.6g}|info={a['solver_info']}|")
    lines += ['', '完整KKT与精确输出Schur已在同一可显式求解的小问题上交叉验证；同时验证了块标量缩放不改变参数解、整体成本缩放只改变乘子、CE行空间方向导数保持。Schur只消去数值未知量，最终方向仍有全部86192640个参数，没有目标ridge或结构截断。', '',
              'A100上的FP64 Schur在80步内仍未收敛：目标残差0.00522494、驻点残差0.00911130。已转为输出对角预条件数值修复，保持同一参数问题和80步预算。不能把小问题或导数测试代替完整问题收敛验收。', '',
              '## 四项边界','',
              '- 完整参数域：JVP/VJP共同使用全部视觉参数的相同展开次序，随机完整方向、CE方向和专家差方向都在此域内。没有Q字典、mask、张量候选或top-k。',
              '- 需求对象：原始尺度保留，单位需求视图仅用于分析；任务、锚点、probe、样本锁、类别头、嵌入定义及精度均参与坐标身份检查。专家Log按逐样本2/sqrt(N)半径计算，重复样本测试验证平均度量。',
              '- 可达性：按构造可达和未知来源标记，与实际目标残差分开；不可达例子即使求解器停止也不会标记为数值可行。加权拟合不会被称为精确匹配。',
              '- M/W：本轮实际数值基线使用显式I，具有SPD证书；测试W为已知PSD。一般矩阵自由外部M/W仍须提供相应数学契约，不能用有限次随机探测冒充全局SPD/PSD证明。', '',
              '结论：已解决原导数差分问题的精度参照，条件求解器的完整问题收敛仍未全部验收。阶段A关闭，结构学习、在线受控训练与EC均未开启。','']
    (root/'NUMERICAL_REPORT.md').write_text('\n'.join(lines),encoding='utf8')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);main(p.parse_args().root)

