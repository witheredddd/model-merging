"""Read-only diagnostics of the actually trained basis and performance."""
import hashlib,json,math,time,zipfile
from pathlib import Path
import numpy as np
import torch
ROOT=Path('/home/shangxueyuan/merging/tsv');OUT=ROOT/'results/functional_geometry_dtd_v2_2'

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()

def main():
    torch.set_num_threads(8)
    result=json.loads((OUT/'result.json').read_text());history=json.loads((OUT/'history.json').read_text())
    first=torch.load(OUT/'initial_basis.pt',map_location='cpu',weights_only=True)['B']
    selected=torch.load(OUT/'best.pt',map_location='cpu',weights_only=True);last=selected['B']
    cross=torch.zeros(first.shape[1],last.shape[1],dtype=torch.float64);G=cross.clone();maxdiff=0.
    for a,b in zip(first.split(262144),last.split(262144)):
        cross+=a.double().T@b.double();G+=b.double().T@b.double();maxdiff=max(maxdiff,float((a-b).abs().max()))
    angles=torch.linalg.svdvals(cross).clamp(0,1).acos()*180/math.pi
    steps=[];bad=[]
    for i,line in enumerate((OUT/'steps.jsonl').read_text().splitlines(),1):
        try:steps.append(json.loads(line))
        except ValueError:bad.append(i)
    byepoch={}
    for s in steps:
        key=(s['epoch'],s['step']);byepoch[key]=s
    unique=list(byepoch.values())
    geom=dict(initial_B_sha256=sha(OUT/'initial_basis.pt'),selected_checkpoint_sha256=sha(OUT/'best.pt'),
        initial_to_selected_principal_angles_degrees=angles.tolist(),selected_orthogonality_max_error=float((G-torch.eye(len(G))).abs().max()),
        basis_entry_max_change=maxdiff,nonzero_basis_updates_in_unique_step_log=sum(bool(s['basis_updated']) for s in unique),
        formal_basis_updates=result['basis_updates'],formal_steps=result['formal_steps'],malformed_interrupted_log_lines=bad,
        no_columns_appended=first.shape==last.shape,first_step_zero_z_behavior=not unique[0]['basis_updated'],
        training_B_elements=first.numel(),coordinate_variables=first.shape[1],basis_trainable_dimension=first.shape[1]*(len(first)-first.shape[1]),
        task_learning_only=True,frozen_coordinate_control_verified=False,multitask_functional_structure_verified=False)
    (OUT/'geometry_learning_evidence.json').write_text(json.dumps(geom,indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    xx=[h['epoch'] for h in history];ax[0].plot(xx,[100*h['validation']['accuracy'] for h in history],'.-',label='Validation')
    ax[0].axhline(80,color='red',ls='--',label='80% minimum');ax[0].axhline(81.7,color='green',ls=':',label='81.7% primary target')
    ax[0].set(xlabel='Epoch',ylabel='Accuracy (%)');ax[0].legend()
    ax[1].plot(xx,[h['validation']['ce'] for h in history],label='Validation CE')
    ax[1].plot(xx[1:],[h['training_ce'] for h in history[1:]],label='Training CE');ax[1].set(xlabel='Epoch',ylabel='CE');ax[1].legend()
    fig.savefig(OUT/'curves.png',dpi=170);plt.close(fig)
    test=result.get('test');test_text='未执行（验证门槛失败）' if test is None else f"{100*test['accuracy']:.2f}% / CE {test['ce']:.6f}"
    text=f'''# DTD 连续 Grassmann 学习修正版 v2.2.0

最佳验证 CE 模型：第 {result['best_epoch']} 轮，Val Accuracy **{100*result['validation']['accuracy']:.2f}%**，Val CE {result['validation']['ce']:.6f}。

锁定后测试：{test_text}。

80% 最低门槛：{result['minimum_80_gate_passed']}；81.7% 主要目标：{result['primary_81_7_gate_passed']}。

正式更新 {result['formal_steps']} 次；实际基底更新 {result['basis_updates']} 次；维数固定 {result['rank']}，没有追加方向。零坐标首步的基底梯度为零，此后每批必须持续更新已有方向。

初始与选定基底的主夹角（度）：{', '.join(f'{x:.2f}' for x in angles.tolist())}。

选定模型基底正交误差：{geom['selected_orthogonality_max_error']:.3g}。训练基底显式存储 {geom['training_B_elements']:,} 个 FP32 元素，不能把训练自由度说成只有 12 个坐标。

此轮仅验证 DTD 任务损失能持续学习基底和坐标。不声称已学出多任务共享/隔离结构，不声称冻结后低维控制有效，不执行 EC。原始基模型及非目标参数保持不变。测试集已在历史研究中暴露，不能称为全新盲测。

相对 v2.1，本轮同时改变为单任务、轨迹梯度初始化、坐标 SGD/余弦调度和每步基底学习；未做控制变量消融，不能把全部性能差异归因于其中单一因素。

原始配置、每步基底移动、数学与真实模型有限差分检查、完整 B/z 检查点在对应文件中。采样索引启动问题在首个有效更新前已修正，旧错误日志保留。
'''
    (OUT/'REPORT.md').write_text(text)
    paths=[p for p in OUT.rglob('*') if p.is_file() and p.suffix in ['.json','.jsonl','.md','.png','.log']]
    with zipfile.ZipFile(OUT/'evidence.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in paths:z.write(p,p.relative_to(OUT))
    print(text)

if __name__=='__main__':main()
