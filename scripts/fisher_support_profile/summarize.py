"""Summarize sampled response support profiles without claiming optimal support."""
import hashlib,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    here=Path(__file__).resolve().parent;out=here/'evidence/20261114'
    report=json.loads((out/'report.json').read_text());cfg=report['config']
    assert cfg['source_sha256']==hashlib.sha256((here/'run.py').read_bytes()).hexdigest()
    if 'prior_source_sha256' in cfg:
        assert cfg['prior_source_sha256']==hashlib.sha256((here/'run_initial.py').read_bytes()).hexdigest()
    assert cfg['no_fixed_support_budget'] and cfg['no_coefficient_refit']
    profiles=report['profiles'];r=cfg['input_dimension']
    assert len(profiles)==2*(r+1)
    for profile in profiles:
        assert profile['executed_source_sha256'] in [cfg['source_sha256'],cfg.get('prior_source_sha256')]
        rows=profile['measurements'];assert rows[-1]['k']==cfg['D']
        assert all(p['error']<1e-12 for p in rows[-1]['probes'])
    fig,axes=plt.subplots(1,2,figsize=(12,4.3),constrained_layout=True)
    for p in profiles:
        if p['entity']=='whole_subspace':continue
        ax=axes[0 if p['ranking']=='parameter' else 1]
        ax.semilogx([100*x['fraction'] for x in p['concentration_curve']],[x['mass'] for x in p['concentration_curve']],label=p['entity'])
    for ax,title in zip(axes,['Per-direction parameter mass','Per-direction absolute rho mass (not response retained)']):
        ax.set(xlabel='Original coordinate support (%)',ylabel='Cumulative mass',title=title,ylim=(0,1.02));ax.grid(alpha=.2)
    axes[0].legend(fontsize=7,ncol=2);fig.savefig(out/'concentration.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.3),constrained_layout=True)
    for p in profiles:
        if p['entity']=='whole_subspace':continue
        rows=p['measurements'];ax=axes[0 if p['ranking']=='parameter' else 1]
        ax.semilogx([100*x['fraction'] for x in rows],[x['probes'][1]['error'] for x in rows],marker='.',label=p['entity'])
    for ax,title in zip(axes,['Parameter-ranked masks: held local response','Absolute-rho-ranked masks: held local response']):
        ax.set(xlabel='Original coordinate support (%)',ylabel='Relative response error',title=title);ax.grid(alpha=.2)
        ax.axhline(.1,color='grey',linestyle='--')
    axes[0].legend(fontsize=7,ncol=2);fig.savefig(out/'mode_responses.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(12,7),constrained_layout=True)
    for p in profiles:
        if p['entity']=='whole_subspace':continue
        row=0 if p['ranking']=='parameter' else 1
        for col,field in enumerate(['cosine','amplitude']):
            axes[row,col].semilogx([100*x['fraction'] for x in p['measurements']],
                [x['probes'][1][field] for x in p['measurements']],'.-',label=p['entity'])
    for row,ranking in enumerate(['parameter','absolute rho']):
        for col,field in enumerate(['cosine','amplitude']):
            axes[row,col].set(title=ranking+' ranked: held '+field,xlabel='Original coordinate support (%)')
            axes[row,col].grid(alpha=.2)
    axes[0,0].legend(fontsize=6,ncol=2);fig.savefig(out/'mode_geometry.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(14,4),constrained_layout=True)
    for p in profiles:
        if p['entity']!='whole_subspace':continue
        for probe in (0,1):
            rows=p['measurements'];label=p['ranking']+f' probe{probe}'
            for ax,field in zip(axes,['error','cosine','amplitude']):
                ax.semilogx([100*x['fraction'] for x in rows],[x['probes'][probe][field] for x in rows],'.-',label=label)
    for ax,title in zip(axes,['Whole-space Frobenius error','Whole-space response alignment','Whole-space response amplitude']):
        ax.set(xlabel='Original coordinate support (%)',title=title);ax.grid(alpha=.2)
    axes[0].legend(fontsize=7);fig.savefig(out/'whole_response.png',dpi=180);plt.close(fig)
    summary=[]
    for p in profiles:
        summary.append(dict(entity=p['entity'],ranking=p['ranking'],mass_support_fraction=p['mass_support_fraction'],
            functional_thresholds=p['functional_thresholds'],negative_rho_sum=p['negative_rho_sum'],positive_rho_sum=p['positive_rho_sum']))
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    lines=['# Fisher Subspace Support Profile','',
        '2026-11-14，A100 完成。分析现有输入空间，维数由文件读取；没有重新选择十维、没有固定1%预算、没有控制器拟合或连续训练。',
        '',f"活动坐标 D={cfg['D']:,}，输入空间维数 r={r}，旧 v5 第128步；两批独立开发图各128张。",
        '逐方向名称按历史平均受限 Fisher 的特征值降序排列。留一历史状态算子后的单方向绝对余弦最小值范围为 '
        +f"{min(cfg['minimum_mode_LOO_cosine']):.4f}–{max(cfg['minimum_mode_LOO_cosine']):.4f}。这只是该扰动下的稳健性检查，近谱值方向仍需谨慎解释；整体指标对正交基底旋转不变。",'',
        '## 参数质量与绝对 Fisher 分配的支持比例','',
        '完整排序累计曲线直接反求所需坐标数，以下百分比是该排名下的精确质量阈值。绝对 rho 质量不是功能响应保留率。','',
        '| 对象 | 排名 | 50%质量所需 | 80%质量所需 | 90%质量所需 | 95%质量所需 | 99%质量所需 |','|---|---|---:|---:|---:|---:|---:|']
    for p in profiles:
        lines.append('| '+p['entity']+' | '+p['ranking']+' | '+' | '.join(f"{100*p['mass_support_fraction'][q]:.3f}%" for q in ['0.5','0.8','0.9','0.95','0.99'])+' |')
    lines+=['','正负分配项另列如下；负值不删除，正负分配不能直接解释成任务促进或冲突。','',
        '| 对象 | 正 rho 总量 | 负 rho 总量 |','|---|---:|---:|']
    for p in profiles:
        if p['ranking']=='parameter':lines.append(f"| {p['entity']} | {p['positive_rho_sum']:.6g} | {p['negative_rho_sum']:.6g} |")
    lines+=['','![质量累计曲线](concentration.png)','','## 原方向 mask 的直接局部响应','',
        '掩码后不重归一化，不优化系数，直接计算 J M_G b 或 J M_G B。rho 排名只使用第一批，第二批使用完全相同的 mask。',
        'E 是相对响应残差，A 是余弦，N 是幅值比。所谓90%响应标准在这里明确指 E≤0.10，不是能量保留90%。',
        '以下是已测嵌套排名路径上的最小达标点；支持点包含预设的宽范围网格及每条质量曲线反求出的质量阈值。未假设响应误差单调，也未声称找到全局最小支持；数据分辨率之外的最小值尚未确定。','',
        '| 对象 | 排名 | 独立 E≤0.20 | 独立 E≤0.10 | 独立 E≤0.05 | 独立 E≤0.01 |','|---|---|---:|---:|---:|---:|']
    for p in profiles:
        t=p['functional_thresholds'][1]
        lines.append('| '+p['entity']+' | '+p['ranking']+' | '+' | '.join(f"{100*t[q]['first_observed_support_fraction']:.3f}%" if t[q]['first_observed_support_fraction'] is not None else '未达标' for q in ['0.8','0.9','0.95','0.99'])+' |')
    lines+=['','![逐方向响应曲线](mode_responses.png)','','![逐方向夹角与幅值](mode_geometry.png)','','![整体响应误差、方向与幅值](whole_response.png)','',
        '## 边界与溯源','',
        '这些是固定锚点的 JVP 一阶响应，不是有限参数步的真实 CE 或准确率验收。参数质量集中、绝对 rho 集中、功能响应保持必须分开。整体 Frobenius 误差按响应能量加权，不等于所有方向或任意线性组合的最坏相对误差；原始记录另存共同 mask 下的逐列误差。曲线只能说明所测排名路径上的承载分布，不能证明参数子集唯一或不可替代，也不能称为跨任务冲突模式。',
        '输入仍为旧轨迹产生的空间，不是全参数 Fisher 自动发现结果。无明显间隙或对扰动不稳的单方向不能推广为天然功能原子；本次不强行构造新功能分组。',
        '源码哈希与执行 config 一致。每个支持点单独原子保存 JSON，完整排序索引保存在服务器 private/，不公开模型权重或排序坐标。中断后按已完成支持点继续；本轮未做中断重放验收。',
        'assets.json 是冻结适配器的初始化记录，其中初始 X 等标签不是本次观测集；实际观测样本以 config.probes 为准，两批均来自未参与方向选取的独立开发样本。',
        '输出目录 `/media/shangxueyuan/data1/DTD_fisher_support_profile_20261114`。本实验取代此前固定1%控制作为当前研究对象，旧结果保留。']
    if 'previous_output' in cfg:
        lines+=['','恢复说明：初版在全部20条逐方向曲线完成后，于整体分数的向量化表达式处发生 CUDA invalid configuration argument。保留 run_initial.py 和原目录；新版本逐列等价累加分数，读取旧曲线并仅补算整体曲线。每条 profile 记录其实际执行版本 SHA。最终目录为 `/media/shangxueyuan/data1/DTD_fisher_support_profile_20261114_r2`。']
    wp=next(p for p in profiles if p['entity']=='whole_subspace' and p['ranking']=='parameter')
    wr=next(p for p in profiles if p['entity']=='whole_subspace' and p['ranking']=='absolute_rho')
    values=[p['mass_support_fraction']['0.95'] for p in profiles if p['entity']!='whole_subspace' and p['ranking']=='parameter']
    rho95=next(x for x in wr['measurements'] if x['k']==wr['mass_support_k']['0.95'])
    lines+=['','## 本次结果的解释','',
        f"- 单方向95%参数质量所需支持比例为{100*min(values):.2f}%–{100*max(values):.2f}%，没有从这些曲线强行划出 localized/distributed 两类。整体共同 mask 的95%参数质量需要{100*wp['mass_support_fraction']['0.95']:.2f}%。",
        f"- 整体95%绝对 rho 质量需要{100*wr['mass_support_fraction']['0.95']:.2f}%坐标，但该点独立响应残差为{100*rho95['probes'][1]['error']:.2f}%。归因累计量不能替代响应验收。",
        '- 参数排名的相对响应误差与归因排名必须在第二批图像上比较；排名样本上的改善不自动泛化。严格阈值下，绝对 rho 排名没有表现出一致的独立优势。',
        '- 整体参数质量是共同 mask 的结果，逐方向各自支持集合则可以不同；不能将方向维数当作原坐标支持数，也不能将两种支持实验混成固定预算控制。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps([s for s in summary if s['entity']=='whole_subspace']))


if __name__=='__main__':main()
