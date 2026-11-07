"""Evaluate exactly the two pre-registered, validation-locked SUN397 models."""
import hashlib,json,math,time,zipfile
from pathlib import Path
import numpy as np
import torch
from PIL import ImageFile
import train as t

def main():
    ImageFile.LOAD_TRUNCATED_IMAGES=True;torch.set_num_threads(8)
    torch.backends.mha.set_fastpath_enabled(False);torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    pair=t.PAIR;locked=json.loads((pair/'both_selections_locked.json').read_text());names=['rank20_angle005','rank20_angle002']
    a=locked['configs'][names[0]]['protocol'];b=locked['configs'][names[1]]['protocol']
    assert {k for k in a if a[k]!=b[k]}=={'experiment_id','angle_cap'}
    enc,model,found,base=t.runtime.build_model(t.u,t.u.ClassificationHead.load(str(t.ROOT/'models/checkpoints/ViT-B-16/head_SUN397Val.pt')))
    for p in model.parameters():p.requires_grad_(False)
    model.eval();_,_,vl,_=t.data(enc,42)
    testdata=t.u.get_dataset('SUN397',enc.val_preprocess,str(t.ROOT/'datasets'),num_workers=0);testloader=t.u.loader(testdata.test_dataset)
    rows=[]
    for name in names:
        out=pair/name;config=locked['configs'][name];ck=torch.load(out/'best.pt',map_location='cpu',weights_only=False)
        t.check_state(ck,config['identity'],len(base),20)
        B=ck['B'].cuda();z=ck['z'].cuda();epoch=ck['epoch'];del ck
        t.u.materialize(found,base,B,z);measured=t.u.evaluate(model,vl)
        assert abs(measured['ce']-locked['selections'][name]['validation']['ce'])<2e-6
        model_hash=t.u.sha(out/'best.pt');cached=out/'locked_test.json'
        if cached.exists():
            test=json.loads(cached.read_text());assert test['checkpoint_sha256']==model_hash and test['identity']==config['identity']
        else:
            test=dict(identity=config['identity'],checkpoint_sha256=model_hash,metrics=t.u.evaluate(model,testloader),time=time.time())
            t.write(cached,test)
        history=json.loads((out/'history.json').read_text());tail=history[-5:]
        row=dict(experiment=name,rank=20,angle_cap=config['protocol']['angle_cap'],selected_epoch=epoch,validation=measured,test=test['metrics'],
            checkpoint_sha256=model_hash,last5_validation_ce_std=float(np.std([h['validation']['ce'] for h in tail])),
            last5_validation_accuracy_std=float(np.std([h['validation']['accuracy'] for h in tail])),
            mean_angle_cap_fraction=float(np.mean([h['angle_cap_fraction'] for h in history[1:]])),
            formal_steps=history[-1]['step'],basis_updates=history[-1]['basis_updates'],peak_gpu_bytes=max(h.get('peak_gpu_bytes',0) for h in history))
        rows.append(row);del B,z
    result=dict(task='SUN397',experiments=rows,only_changed_learning_parameter='maximum geodesic rotation angle',same_initialization=True,
        test_evaluated_only_after_both_selections_locked=True,selection='minimum validation CE independently in each pre-registered arm',
        functional_control_or_EC_verified=False,time=time.time())
    t.write(pair/'result.json',result)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for name in names:
        hist=json.loads((pair/name/'history.json').read_text());xx=[h['epoch'] for h in hist]
        axes[0].plot(xx,[100*h['validation']['accuracy'] for h in hist],label=name)
        axes[1].plot(xx,[h['validation']['ce'] for h in hist],label=name)
    for ax in axes:ax.set_xlabel('Epoch');ax.legend(fontsize=8);ax.grid(alpha=.2)
    axes[0].set_ylabel('Validation accuracy (%)');axes[1].set_ylabel('Validation CE');fig.savefig(pair/'curves.png',dpi=170);plt.close(fig)
    text=['# SUN397 rank20 rotation-cap pair','', '| Arm | Selected epoch | Val accuracy | Val CE | Test accuracy | Test CE |','|---|---:|---:|---:|---:|---:|']
    for r in rows:text.append(f"| angle {r['angle_cap']} | {r['selected_epoch']} | {100*r['validation']['accuracy']:.2f}% | {r['validation']['ce']:.6f} | {100*r['test']['accuracy']:.2f}% | {r['test']['ce']:.6f} |")
    text+=['','Both runs use the same shared, identity-checked rank20 initialization, seed, split, training budget and minimum-validation-CE selection. No DTD accuracy gate is applied to SUN397.',
        'The two configurations were registered before either training run. Both validation selections were locked before either test evaluation; test scores do not select new hyperparameters.',
        'This pair studies the rotation cap within SUN397 rank20. It is not a DTD rank12-to-rank20 capacity comparison, a proof of functional control, or an EC experiment.',
        'Storage: complete best checkpoints are retained. Obsolete resume snapshots are removed only after a valid full-state best checkpoint is available; this permits recovery from the best if an interrupted resume write occurs.']
    (pair/'REPORT.md').write_text('\n'.join(text)+'\n')
    with zipfile.ZipFile(pair/'evidence.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in pair.rglob('*'):
            if p.is_file() and p.suffix in ['.json','.jsonl','.md','.png','.log']:z.write(p,p.relative_to(pair))
    t.write(pair/'complete.json',dict(time=time.time(),result='result.json',report='REPORT.md'))
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
