"""Audit raw Cars Fisher matrices and report diagonalization versus validation."""
import hashlib,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    here=Path(__file__).resolve().parent;out=here/'evidence/20261114'
    r=json.loads((out/'report.json').read_text());c=r['config'];s=r['spectrum'];g=r['geometry']
    assert hashlib.sha256((here/'run.py').read_bytes()).hexdigest()==c['source_sha256']
    data=np.load(out/'matrices.npz');Fs=data['F'];V=data['eigenbasis'];G=data['diagonalized'];O=data['common_rotation'];labels=data['block_labels']
    assert Fs.shape==(3,2,g['candidate_dimension'],g['candidate_dimension'])
    np.testing.assert_allclose(V.T@Fs[2,0]@V,G[2,0],atol=1e-7,rtol=1e-7)
    np.testing.assert_allclose(V.T@V,np.eye(len(V)),atol=1e-10)
    pair_ratios=[]
    blocks=[np.where(labels==x)[0] for x in np.unique(labels)]
    for state in range(3):
        vals=[]
        for probe in range(2):
            A=O.T@Fs[state,probe]@O;ratios=[]
            for i,b in enumerate(blocks):
                for a in blocks[:i]:
                    ratios.append(float(np.linalg.norm(A[np.ix_(a,b)])/max(np.sqrt(max(0.,np.trace(A[np.ix_(a,a)])*np.trace(A[np.ix_(b,b)]))),1e-30)))
            vals.append(max(ratios,default=0.))
        pair_ratios.append(vals)
    held_leak=[r['common_blocks']['leakage'][0][1],*r['common_blocks']['leakage'][1]]
    held_pairs=[pair_ratios[1][0],*[p[1] for p in pair_ratios]]
    passed=len(blocks)>1 and max(held_leak)<=.05 and max(held_pairs)<=.1
    T=data['tangent_coordinates'];A0=T.T@Fs[2,0]@T;A1=T.T@Fs[2,1]@T
    te,TV=np.linalg.eigh(A0);te=te[::-1];TV=TV[:,::-1];tw=np.maximum(te,0);tw/=tw.sum()
    transformed_tangent=TV.T@A1@TV
    tangent_audit=dict(eigenvalues=te.tolist(),effective_rank_relative1e_6=int((te>te[0]*1e-6).sum()),
        participation_rank=float(1/(tw@tw)),trace_energy_dimensions={str(q):int(np.searchsorted(np.cumsum(tw),q))+1 for q in [.8,.9,.95,.99]},
        other_probe_offdiagonal_ratio=float(np.linalg.norm(transformed_tangent-np.diag(np.diag(transformed_tangent)))/np.linalg.norm(transformed_tangent)))
    _,H=np.linalg.eigh(Fs[2,1]);H=H[:,::-1];overlap={}
    for q,k in s['trace_energy_dimensions'].items():
        overlap[q]=dict(k=k,principal_cosines=np.linalg.svd(V[:,:k].T@H[:,:k],compute_uv=False).tolist())
    finite=r['finite_interventions'];local=[x for x in finite if x['radius']==.001 and x['predicted_KL']>1e-9]
    ratios=[x['KL']/x['predicted_KL'] for x in local]
    audit=dict(strict_common_block_pass=bool(passed),normalized_pair_coupling=pair_ratios,tangent_Fisher=tangent_audit,
        global_leakage_limit=.05,pair_coupling_limit=.1,expert_subspace_overlap_by_primary_trace_energy=overlap,
        finite_KL_above_floor_count=len(local),finite_KL_actual_over_predicted_quantiles=np.quantile(ratios,[0,.1,.5,.9,1]).tolist() if ratios else None,
        caution='Single-matrix diagonalization is algebraic; independent probe off-diagonal and overlap assess reproducibility. Common variants are not a trajectory.')
    (out/'validation_audit.json').write_text(json.dumps(audit,indent=2))
    eig=np.array(s['eigenvalues']);fig,axes=plt.subplots(1,3,figsize=(13,3.8),constrained_layout=True)
    axes[0].semilogy(np.arange(1,len(eig)+1),np.maximum(eig,1e-30),'o-');axes[0].set(xlabel='Index (all retained candidates)',ylabel='Eigenvalue',title='Cars expert restricted Fisher spectrum')
    maxv=np.log10(np.maximum(np.abs(Fs[2,0]),1e-12)).max()
    for ax,A,title in [(axes[1],G[2,0],'Fit probe in fitted eigenbasis'),(axes[2],G[2,1],'Other probe in same eigenbasis')]:
        im=ax.imshow(np.log10(np.maximum(np.abs(A),1e-12)),vmin=maxv-8,vmax=maxv,cmap='viridis');ax.set(title=title);fig.colorbar(im,ax=ax,label='log10 |entry|')
    fig.savefig(out/'spectrum_diagonalization.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    for probe in range(2):
        rows=[x for x in local if x['probe']==probe]
        axes[0].scatter([x['predicted_KL'] for x in rows],[x['KL'] for x in rows],s=12,label=f'probe{probe}')
    axes[0].set(xscale='log',yscale='log',xlabel='Predicted KL',ylabel='Actual finite KL',title='All modes, radius0.001, both signs');axes[0].legend()
    for q,value in overlap.items():axes[1].plot(range(1,value['k']+1),value['principal_cosines'],'o-',label=f'energy{q}, k{value["k"]}')
    axes[1].set(xlabel='Principal index',ylabel='Cosine',title='Independent-probe eigenspace stability',ylim=(0,1.02));axes[1].legend(fontsize=7)
    fig.savefig(out/'finite_stability.png',dpi=180);plt.close(fig)
    lines=['# Cars Fisher 子空间谱与对角化','',
        '2026-11-14，A100完成。复用 Cars adaptive_dynamics 已训练专家和保存的学习结构，不重新训练，不使用旧 DTD v5 十维，不做 Support Profile、leverage、掩码或 EC。','',
        '## 可直接复用的模型与结构','',
        f"Cars 专家：`EXPERTS8_FD_REVIEWED_20261110/Cars/adaptive_dynamics/epoch08.pt`。历史87.08%，本次锁定 Test 复现 **{100*r['test_reproduction']['accuracy']:.4f}%**（{r['test_reproduction']['correct']}/{r['test_reproduction']['n']}），CE {r['test_reproduction']['CE']:.6f}。Test只核验锁定权重，不参与方向/阈值选择。",
        f"保存{c['learned_components']}个分量、{c['coordinate_count']}个坐标，每组3条非线性载体轴。原训练上限为{c['original_component_cap']}组，已碰上限；这些数目不是自然 Fisher 维数。实际坐标切空间数值秩为{g['actual_coordinate_tangent_rank']}。",
        f"完整解码相对专家终点位移误差为{c['decoder_relative_endpoint_error']:.6f}，所以专家精度不能当作载体独立承担训练的证明。DTD约82.07%的同框架结构可用；81.86%是不同的adaptive_allocation版本，仅找到归档指标，未找到其检查点，不能混同。",'',
        '## Fisher 协议','',
        f"全视觉端 D={c['D']:,}，不是旧48矩阵范围。候选包括保存的全部载体轴、实际专家位移和保存的 frozen-decoder 位移；FP64两遍重正交化，相对1e-5/绝对1e-12阈值，保留 **{g['candidate_dimension']}维**，未设维数上限。该维数是候选范围的数值秩，不是完整 Cars Fisher 的天然维数。",
        '固定预测嵌入 q=2sqrt(p)/sqrt(N)，两批互不重叠训练侧图各128张。专家曾在全部8144张上训练；两批只是本次探测划分分离，不是模型从未见过的数据。逐列真实 JVP 计算 R=JQ，再构造 A=R^T R。',
        '三锚点是预训练基座、已保存 frozen-decoder 模型、完整专家；它们不是历史训练轨迹。专家主结果用当前专家锚点，另外两者仅用于变化检查。共同分块拟合基座/专家的第一批，decoder锚点和第二批作为留出检查；不能称为时间共同不变结构或跨任务验证。','',
        '## 实际谱与对角化','',
        f"数值有效秩（最大特征值的1e-6）为{s['effective_rank_relative1e_6']}；参与率秩{s['participation_rank']:.4f}；熵秩{s['entropy_rank']:.4f}。解释80/90/95/99%受限 Fisher迹需要的维数分别为 "+', '.join(str(s['trace_energy_dimensions'][q]) for q in ['0.8','0.9','0.95','0.99'])+'。',
        f"拟合矩阵在自身特征基中的非对角 Frobenius 比为{s['expert_fit_offdiagonal_ratio']:.3g}，这是代数性质。另一批图在同一基中为 **{s['expert_other_probe_offdiagonal_ratio']:.4f}**；后者才提供重现性检查。独立子空间余弦见 validation_audit.json，不能把近重根的单向量旋转直接判成子空间失败。",'',
        f"单独限制到真正的已学坐标切空间（{T.shape[1]}维），其Fisher数值有效秩为{tangent_audit['effective_rank_relative1e_6']}，参与率秩{tangent_audit['participation_rank']:.4f}；解释80/90/95/99%迹所需维数为 "+', '.join(str(tangent_audit['trace_energy_dimensions'][q]) for q in ['0.8','0.9','0.95','0.99'])+f"。该切空间拟合特征基在另一批图的非对角比为{tangent_audit['other_probe_offdiagonal_ratio']:.4f}。它与25维候选矩阵必须区分。",'',
        '![谱与对角化](spectrum_diagonalization.png)','',
        f"多模型锚点共同块候选大小为{r['common_blocks']['sizes']}。按原DTD的严格标准：全局块外泄漏≤5%、归一化块对耦合≤10%，留出验收结果为 **{passed}**。本审计补上块对判据；原始运行器blocks.json的held_support只包含全局泄漏，不作为最终验收。",'',
        '## 真实有限扰动','',
        '对全部拟合特征方向施加±0.001、±0.003的单位参数范数扰动，直接测两批真实预测 KL 和 CE变化。比较 epsilon² v^T Fv /2。没有优化方向或坐标，没有将 Fisher 能量解释成任务冲突。',
        'radius0.001且预测KL>1e-9的实际/预测比值分位数（min/10%/median/90%/max）为 '+str(audit['finite_KL_actual_over_predicted_quantiles'])+'。弱响应模式的数值精度与有限步长非线性需分别看，完整逐模式记录保留。','',
        '![有限扰动与稳定性](finite_stability.png)','',
        '## 溯源与边界','',
        '公开原始/对角化/共同旋转矩阵NPZ、CSV、谱、全部有限扰动和源哈希。Q与完整模型权重留在A100 private/。每个响应缓存及有限模式记录可复用；本轮未做硬中断恢复验收。',
        '输出 `/media/shangxueyuan/data1/Cars_fisher_diagonalization_20261114`。源码和分块求解器哈希见config；归一化、参数次序和两批ID均明确记录。',
        '本轮交付 Cars 的受限 Fisher 子空间对角化。没有进行 DTD–Cars 共同锚点交叉测量，也没有认定真实负迁移；后续跨任务实验需另行在同一锚点重算两任务的 Fisher。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');print(json.dumps(dict(geometry=g,spectrum=s,test=r['test_reproduction'],strict_blocks=passed)))


if __name__=='__main__':main()
