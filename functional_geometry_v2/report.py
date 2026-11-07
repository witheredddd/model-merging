"""Generate reviewable raw evidence and curves after final locked evaluation."""
import hashlib,json,zipfile
from pathlib import Path
import numpy as np
from engine import OUT,TASKS

def build(result,history):
    lines=['# 八任务在线功能几何 v2.1 结果','',
        f"最终秩：{result['rank']}；DTD 80% 验证门槛：{'通过' if result['DTD_validation_80_percent_gate_passed'] else '未通过，不能宣称成功'}。",'',
        '| 任务 | 专家验证准确率 | 专家测试准确率 | 融合测试准确率 |','|---|---:|---:|---:|']
    for k,task in enumerate(TASKS):
        val=result['expert_validation'][k]['accuracy'];tests=result['tests'][task]
        lines.append(f"| {task} | {100*val:.2f}% | {100*tests['expert']['accuracy']:.2f}% | {100*tests['merge']['accuracy']:.2f}% |")
    lines+=['','共同结构：'+('通过全部预设检验。' if result['common_structure_validated'] else '未检测到通过全部检验的稳定共同块，融合使用全局系数基线。'),'',
        'EC 仅为小规模探索性 pilot。功能矩阵每任务每分组仅 8 张图像；不能将局部响应小解释为全局互不干扰。',
        '所有选模在测试前锁定；未额外训练全量微调对照；当前结果不能与历史不同划分的数字直接混比。']
    (OUT/'run_report.md').write_text('\n'.join(lines)+'\n')
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(2,4,figsize=(15,7),constrained_layout=True)
        for k,(task,ax) in enumerate(zip(TASKS,axes.flat)):
            rows={}
            for row in history['validation']:
                if row['task']==task:rows[0 if row.get('initial') else row['cycle']+1]=row['current']
            xx=sorted(rows);ax.plot(xx,[100*rows[x]['accuracy'] for x in xx],marker='.',label='Validation accuracy')
            if task=='DTD':ax.axhline(80,color='red',ls='--',label='80% gate')
            ax.set(title=task,xlabel='Updates per task',ylabel='Accuracy (%)');ax.grid(alpha=.2)
        fig.savefig(OUT/'validation_curves.png',dpi=160);plt.close(fig)
        fig,axes=plt.subplots(2,4,figsize=(15,7),constrained_layout=True)
        for task,ax in zip(TASKS,axes.flat):
            rows={r['cycle']+1:r['ce'] for r in history['training'] if r['task']==task};xx=sorted(rows);yy=np.array([rows[x] for x in xx])
            if len(yy)>=20:ax.plot(xx[19:],np.convolve(yy,np.ones(20)/20,mode='valid'),label='Training CE (20 updates)')
            val={0 if r.get('initial') else r['cycle']+1:r['current']['ce'] for r in history['validation'] if r['task']==task}
            vx=sorted(val);ax.plot(vx,[val[x] for x in vx],label='Validation CE');ax.set(title=task,xlabel='Updates per task',ylabel='CE');ax.legend(fontsize=7);ax.grid(alpha=.2)
        fig.savefig(OUT/'loss_curves.png',dpi=160);plt.close(fig)
    except ImportError:(OUT/'plot_unavailable.txt').write_text('matplotlib not installed; complete raw history is available in history.json and JSONL logs.\n')
    path=OUT/'final_evidence.zip'
    with zipfile.ZipFile(path.with_suffix('.tmp'),'w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(OUT.rglob('*')):
            if p.is_file() and p.suffix in ['.json','.jsonl','.npz','.md','.png','.log','.txt']:
                z.write(p,p.relative_to(OUT))
    path.with_suffix('.tmp').replace(path)
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    (OUT/'final_evidence.sha256').write_text(h.hexdigest()+'  final_evidence.zip\n')
