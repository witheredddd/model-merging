"""Matched clean-label DTD continuation from the CE Surfaces warmup anchor."""

import argparse
import fcntl
import gc
import hashlib
import time
from pathlib import Path

import torch
from torch.nn import functional as F

import dtd_trajectory_projection as projection
from run_dtd_carrier_diagnostics import Operator
from task_data import test_protocol, test_set
from train_dtd_rolling_innovation import LEARNING_RATE, ShadowAdamW
from train_dtd_surfaces import COMPARISON_STEPS, SEED, WARMUP
from train_eight_expert_queue import BASE, BATCH_SIZE, HEADS, ROOT, build_model, cosine_lr, existing, loader, read, save, splits


FINAL_STEP = 270
ARMS = ('adamw','pca4','ltsa4','pca8','ltsa8')
EARLY_ARMS = ('early_ltsa','early_task')
EARLY_WARMUP = 12
STEP_FRACTIONS = (1.,.5,.25,.125)
SUFFIXES = ('attn.in_proj_weight','attn.out_proj.weight','mlp.c_fc.weight','mlp.c_proj.weight')


def digest(x):
    return hashlib.sha256(x.contiguous().numpy().data).hexdigest()


def cpu_weights(visual):
    return {key:value.detach().cpu() for key,value in visual.state_dict().items()}


def block_snapshot(params):
    return torch.cat([p.detach().flatten().cpu() for p in params])


def gradient(model, op, images, labels, chosen=None):
    with torch.autocast('cuda',dtype=torch.bfloat16):
        loss = F.cross_entropy(model(images).float(),labels)
    loss.backward()
    if chosen is not None:
        for i,p in enumerate(op.params):
            if i not in chosen:
                p.grad = None
    torch.nn.utils.clip_grad_norm_(op.params,1.,error_if_nonfinite=True)
    return float(loss.detach())


def schedule_for(optimizer):
    initial = cosine_lr(optimizer,LEARNING_RATE,20,COMPARISON_STEPS)
    def schedule(step):
        if step<=COMPARISON_STEPS:
            initial(step-1)
        else:
            optimizer.param_groups[0]['lr'] = .10*LEARNING_RATE
        return float(optimizer.param_groups[0]['lr'])
    return schedule


def minibatches(data, count, start, end):
    for epoch in range((start-1)//count,(end-1)//count+1):
        data.epoch = epoch
        for i,(x,y,ids) in enumerate(loader(data,4,BATCH_SIZE,SEED+epoch,True)):
            step = epoch*count+i+1
            if step<start:
                continue
            if step>end:
                return
            yield step,x.cuda(non_blocking=True),y.cuda(non_blocking=True),ids


@torch.no_grad()
def ce(model, x, y):
    training = model.training
    model.eval()
    try:
        return float(F.cross_entropy(model(x).float(),y))
    finally:
        model.train(training)


def warmup(model, op, params, data, count, expected, output, val_loader, early=False):
    shadow = ShadowAdamW(op)
    schedule = schedule_for(shadow.optimizer)
    snapshots = [block_snapshot(params)]
    early_anchor = None
    model.train()
    for step,x,y,_ in minibatches(data,count,1,WARMUP):
        shadow.clear()
        schedule(step)
        gradient(model,op,x,y)
        shadow.propose()
        with torch.no_grad():
            for p,w in zip(op.params,shadow.shadows):
                p.copy_(w)
        if (early and step<=EARLY_WARMUP) or (not early and step%projection.SNAPSHOT_EVERY==0):
            snapshots.append(block_snapshot(params))
        if early and step==EARLY_WARMUP:
            optimizer = shadow.optimizer.state_dict()
            optimizer = dict(optimizer,state={i:{k:v.detach().cpu().clone() if torch.is_tensor(v) else v
                                                 for k,v in slot.items()} for i,slot in optimizer['state'].items()},
                             param_groups=[g.copy() for g in optimizer['param_groups']])
            early_anchor = (cpu_weights(op.visual),optimizer,snapshots.copy(),
                dict(step=step,anchor_sha256=digest(block_snapshot(op.params)),snapshot_steps=list(range(step+1)),
                     optimizer='shared replayed AdamW state at step 12',validation=existing.evaluate(model,val_loader)))
            model.train()
    anchor = block_snapshot(op.params)
    if not torch.equal(anchor,expected):
        raise RuntimeError(f'Warmup did not exactly match the saved anchor, max difference {(anchor-expected).abs().max()}')
    optimizer = shadow.optimizer.state_dict()
    for state in optimizer['state'].values():
        for key,value in state.items():
            if torch.is_tensor(value):
                state[key] = value.cpu()
    if {int(s['step']) for s in optimizer['state'].values()}!={WARMUP}:
        raise RuntimeError('Unexpected AdamW step counters after replay')
    evidence = dict(step=WARMUP,exact_anchor_match=True,anchor_sha256=digest(anchor),
                    optimizer='deterministically replayed original 36 AdamW steps',
                    snapshot_steps=list(range(0,WARMUP+1,projection.SNAPSHOT_EVERY)),
                    validation=existing.evaluate(model,val_loader))
    existing.write(output/'warmup.json',evidence)
    print(f'WARMUP_EXACT {evidence["validation"]}',flush=True)
    if early:
        existing.write(output/'early_anchor.json',early_anchor[-1])
        return early_anchor
    return cpu_weights(op.visual),optimizer,snapshots,evidence


@torch.no_grad()
def search_step(model, op, selected, params, deltas, radius, x, y):
    size = float(torch.stack([d.square().sum() for d in deltas]).sum().sqrt())
    if size==0 or radius==0:
        return 0.,0.
    model.eval()
    score = float(F.cross_entropy(model(x).float(),y))
    scale = .99999*radius/size
    chosen,best,best_norm = 0.,None,0.
    names = [op.names[i] for i in sorted(selected)]
    for fraction in STEP_FRACTIONS:
        weights = {key:p+(fraction*scale)*d for key,p,d in zip(names,params,deltas)}
        norm = float(torch.stack([(weights[k]-p).square().sum() for k,p in zip(names,params)]).sum().sqrt())
        if norm>radius:
            continue
        logits = torch.func.functional_call(model,weights,(x,)).float()
        candidate = float(F.cross_entropy(logits,y))
        if candidate<score:
            score,chosen,best,best_norm = candidate,fraction,weights,norm
    model.train()
    if best is None:
        return 0.,0.
    for key,p in zip(names,params):
        p.copy_(best[key])
    return best_norm,chosen


def run_arm(name, model, op, params, selected, blocks, data, count, warm, val_loader, folder):
    folder.mkdir(exist_ok=True)
    if (folder/'train.json').exists():
        summary = read(folder/'train.json')
        if existing.sha(folder/'best.pt')!=summary['checkpoint_sha256']:
            raise ValueError('Cached selected weights differ from the recorded checkpoint')
        return summary
    weights,initial_optimizer,initial_snapshots,anchor = warm
    op.visual.load_state_dict(weights)
    is_full = name=='adamw'
    early = name in EARLY_ARMS
    rank = None if is_full else (8 if early else int(name[-1]))
    method = 'ltsa' if early else name[:-1]
    warmup_step = anchor['step']
    snapshots = list(initial_snapshots)
    snapshots_at = list(anchor['snapshot_steps'])
    projected = None
    moments = [(initial_optimizer['state'][i]['exp_avg'].flatten(),
                initial_optimizer['state'][i]['exp_avg_sq'].flatten()) for i in sorted(selected)]
    shadow = ShadowAdamW(op) if is_full or early else None
    group = initial_optimizer['param_groups'][0]
    if shadow:
        shadow.optimizer.load_state_dict(initial_optimizer)
        scheduler = schedule_for(shadow.optimizer)
    else:
        # Use the same schedule helper without allocating ambient optimizer states.
        class ScheduleState:
            param_groups = [{'lr':LEARNING_RATE}]
        scheduler = schedule_for(ScheduleState())
    records,epochs,refreshes = [],[],[]
    best = dict(step=warmup_step,validation=anchor['validation'])
    save(folder/'best.pt',dict(visual=weights,step=warmup_step))
    epoch_loss,epoch_n = 0.,0
    task_gradient = None
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for step,x,y,ids in minibatches(data,count,warmup_step+1,FINAL_STEP):
        model.train()
        tick = time.perf_counter()
        lr = scheduler(step)
        if not is_full and (step-warmup_step-1)%projection.REFRESH==0:
            versions = [p._version for p in op.params]
            stamp = time.perf_counter()
            projected,blocks_info = projection.refresh(snapshots,blocks,rank,method,moments,projected,task_gradient)
            task_gradient = None
            if versions!=[p._version for p in op.params]:
                raise RuntimeError('Refreshing the geometry changed weights')
            refreshes.append(dict(step=step,fit_steps=snapshots_at.copy(),
                total_rank=sum(b['rank'] for b in blocks_info),blocks=blocks_info,
                seconds=time.perf_counter()-stamp,weight_preserved=True))
        if shadow:
            shadow.clear()
        else:
            model.zero_grad(set_to_none=True)
        value = gradient(model,op,x,y,selected)
        epoch_loss += value*len(y)
        epoch_n += len(y)
        if name=='early_task':
            g = torch.cat([p.grad.flatten().cpu() for p in params])
            if task_gradient is None:
                task_gradient = g
            else:
                task_gradient.add_(g)
        probe_x,probe_y = (x[32:64],y[32:64]) if early else (x[:32],y[:32])
        before = ce(model,probe_x,probe_y)
        extra = {}
        if is_full:
            shadow.propose()
            with torch.no_grad():
                norm = float(torch.stack([(shadow.shadows[i]-op.params[i]).square().sum()
                                          for i in sorted(selected)]).sum().sqrt())
                for i in selected:
                    op.params[i].copy_(shadow.shadows[i])
            cosine = 1.
        else:
            if early:
                shadow.propose()
                with torch.no_grad():
                    radius = float(torch.stack([(shadow.shadows[i]-op.params[i]).square().sum()
                                                for i in sorted(selected)]).sum().sqrt())
                deltas,cosine = projection.direction(params,projected,step,lr,group['betas'],group['eps'],group['weight_decay'])
                norm,fraction = search_step(model,op,selected,params,deltas,radius,x[:32],y[:32])
                extra = dict(reference_norm=radius,selected_fraction=fraction)
                del deltas
            else:
                norm,cosine = projection.apply(params,projected,step,lr,group['betas'],group['eps'],group['weight_decay'])
            if step%projection.SNAPSHOT_EVERY==0:
                snapshots = [*snapshots,block_snapshot(params)][-projection.SNAPSHOTS:]
                snapshots_at = [*snapshots_at,step][-projection.SNAPSHOTS:]
        after = ce(model,probe_x,probe_y)
        records.append(dict(step=step,lr=lr,train_ce=value,update_norm=norm,gradient_projection_cosine=cosine,
                            train_probe_ce_gain=before-after,seconds=time.perf_counter()-tick,
                            batch_ids_sha256=digest(ids),**extra))
        if step%count==0 or step in (COMPARISON_STEPS,FINAL_STEP):
            val = existing.evaluate(model,val_loader)
            epochs.append(dict(step=step,train_ce=epoch_loss/epoch_n,train_n=epoch_n,validation=val))
            epoch_loss,epoch_n = 0.,0
            if val['ce']<best['validation']['ce']:
                best = dict(step=step,validation=val)
                save(folder/'best.pt',dict(visual=op.visual.state_dict(),step=step))
            existing.write(folder/'progress.json',dict(epochs=epochs,best=best))
            print(f'{name} {step}/{FINAL_STEP} val_acc={val["accuracy"]:.6f} val_ce={val["ce"]:.6f}',flush=True)
    return finish_arm(folder,name,rank,best,epochs,records,refreshes,started,projected)


def finish_arm(folder,name,rank,best,epochs,records,refreshes,started,projected):
    row = dict(method='APML-inspired reproduction' if name.startswith('ltsa') else name,rank_per_block=rank,
               epochs=epochs,best=best,steps=records,refreshes=refreshes,
               seconds=time.perf_counter()-started,peak_cuda_allocated=torch.cuda.max_memory_allocated(),
               final_basis_bytes=sum(u.numel()*u.element_size() for u,_,_ in projected or []),
               checkpoint_sha256=existing.sha(folder/'best.pt'))
    existing.write(folder/'train.json',row)
    return row


def run(args):
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    existing.ROOT = ROOT
    existing.seed_all(SEED)
    source = torch.load(args.surface_best,map_location='cpu',weights_only=True,mmap=True)
    provenance = source['provenance']
    if (provenance['task'],provenance['seed'],provenance['warmup'],provenance['split'])!=('DTD',SEED,WARMUP,'train_holdout_seed_0'):
        raise ValueError('Surface reference has incompatible provenance')
    head_path = HEADS/'DTD.pt'
    if (existing.sha(BASE),existing.sha(head_path))!=(provenance['base_sha256'],provenance['head_sha256']):
        raise ValueError('Pretraining or head changed')
    head = existing.ClassificationHead.load(str(head_path)).cuda()
    encoder,model,_,flat = build_model(existing,head)
    del flat
    visual = encoder.model.visual
    model.requires_grad_(False)
    visual.requires_grad_(True)
    op = Operator(model,visual)
    selected = {i for i,key in enumerate(op.keys) if key.endswith(SUFFIXES)}
    params = [op.params[i] for i in sorted(selected)]
    if len(params)!=48 or any(not p.is_contiguous() or p.ndim!=2 for p in params):
        raise ValueError('Expected the 48 contiguous visual transformer matrices')
    blocks,end = [],0
    for p in params:
        blocks.append((end,end+p.numel()))
        end += p.numel()
    training,validation = splits('DTD',encoder.train_preprocess,encoder.val_preprocess)
    if (len(training),len(validation))!=(3384,376):
        raise ValueError('Unexpected DTD split')
    data = existing.EpochData(training,SEED)
    count = (len(training)+BATCH_SIZE-1)//BATCH_SIZE
    val_loader = loader(validation,2,64,SEED)
    warm = warmup(model,op,params,data,count,source['surface']['warmup_anchor'],args.output,val_loader,args.early)
    del source
    results = {}
    for name in args.arm or (EARLY_ARMS if args.early else ARMS):
        results[name] = run_arm(name,model,op,params,selected,blocks,data,count,warm,val_loader,args.output/name)
        gc.collect()
        torch.cuda.empty_cache()
    reference = [r['batch_ids_sha256'] for r in next(iter(results.values()))['steps']]
    if any([s['batch_ids_sha256'] for s in r['steps']]!=reference for r in results.values()):
        raise RuntimeError('Continuation data order changed across arms')
    if args.arm:
        return
    # Lock every validation selection before the historical test pass.
    existing.write(args.output/'locked.json',{a:dict(best=r['best'],sha256=r['checkpoint_sha256']) for a,r in results.items()})
    test = test_set('DTD',encoder.val_preprocess)
    protocol = test_protocol('DTD',test)
    batches = loader(test,4,64,SEED)
    for name,row in results.items():
        state = torch.load(args.output/name/'best.pt',weights_only=True,map_location='cpu',mmap=True)
        visual.load_state_dict(state['visual'])
        row['test'] = dict(protocol=protocol,**existing.evaluate(model,batches))
        print(f'{name} LOCKED_TEST {row["test"]["accuracy"]:.6f}',flush=True)
    existing.write(args.output/'results.json',dict(warmup=warm[-1],parameters=[op.keys[i] for i in sorted(selected)],
        parameter_count=end,seed=SEED,train_n=len(training),validation_n=len(validation),final_step=FINAL_STEP,
        early=args.early,step_fractions=STEP_FRACTIONS if args.early else None,
        geometry=dict(neighbors=projection.NEIGHBORS,refresh=projection.REFRESH,
                      snapshots=projection.SNAPSHOTS,stride=projection.SNAPSHOT_EVERY),arms=results))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--surface-best',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--arm',choices=ARMS,action='append')
    parser.add_argument('--early',action='store_true',help='Run only the two early LTSA diagnostics')
    args = parser.parse_args()
    if args.early and args.arm:
        parser.error('--early always runs both matched diagnostic arms')
    args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run(args)


if __name__=='__main__':
    main()
