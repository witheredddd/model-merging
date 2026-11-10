"""Train persistent DTD surfaces until the holdout confirms overfitting."""

import argparse
import errno
import fcntl
import hashlib
from pathlib import Path
import shutil

import numpy as np
import torch
from torch.nn import functional as F

import dtd_surface_learning as surface
from run_dtd_carrier_diagnostics import Operator
from task_data import test_protocol, test_set
from train_dtd_online_carrier import initialize
from train_dtd_rolling_innovation import LEARNING_RATE, ShadowAdamW
from train_eight_expert_queue import BASE, BATCH_SIZE, HEADS, ROOT, build_model, cosine_lr, existing, loader, read, save, splits
from train_fd_task import SEED, WARMUP, audit_coordinates


COMPARISON_STEPS = 216
VALIDATION_MARGIN = .003
TRAIN_MARGIN = .005
OVERFIT_PATIENCE = 5
LR_PATIENCE = 4


def cpu_state(state):
    if state is None:
        return None
    return dict(state,frames=state['frames'].cpu(),z=state['z'].tolist(),q=state['q'].tolist(),
                offset_coefficients=[v.tolist() for v in state['offset_coefficients']])


def overfit(epochs, best):
    if best is None or len(epochs)<8 or len(epochs)-best['epoch']<OVERFIT_PATIENCE:
        return False
    recent = epochs[-OVERFIT_PATIENCE:]
    high_validation = all(r['validation']['ce']>best['validation']['ce']+VALIDATION_MARGIN for r in recent)
    lower_training = np.mean([r['train_ce'] for r in recent[-3:]])<best['train_ce']-TRAIN_MARGIN
    return bool(high_validation and lower_training)


def train(out, scratch, max_components, early=None, expected_anchor=None):
    warmup = 12 if early else WARMUP
    final_step = 270 if early else None
    existing.seed_all(SEED)
    existing.ROOT = ROOT
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    head_path = HEADS/'DTD.pt'
    head_hash = existing.sha(head_path)
    if read(head_path.with_suffix('.json'))['sha256'] != head_hash:
        raise ValueError('DTD classification head hash mismatch')
    head = existing.ClassificationHead.load(str(head_path)).cuda()
    encoder,model,_,flat = build_model(existing,head)
    del flat
    visual = encoder.model.visual
    model.requires_grad_(False)
    visual.requires_grad_(True)
    op = Operator(model,visual)
    shadow = ShadowAdamW(op)
    training,validation = splits('DTD',encoder.train_preprocess,encoder.val_preprocess)
    data = existing.EpochData(training,SEED)
    batches = (len(data)+BATCH_SIZE-1)//BATCH_SIZE
    provenance = dict(task='DTD',algorithm='persistent_ce_surfaces_v1',seed=SEED,warmup=warmup,
                      comparison_steps=COMPARISON_STEPS,train_n=len(training),validation_n=len(validation),
                      split='train_holdout_seed_0',base_sha256=existing.sha(BASE),head_sha256=head_hash,
                      max_components=max_components,geometry_window=surface.WINDOW,fit_batches=surface.FIT_BATCHES,
                      confirmation_batches=surface.CONFIRM_BATCHES,replay_images=surface.REPLAY_IMAGES,
                      inner_rounds=surface.COORDINATE_ROUNDS,surface_rounds=surface.SURFACE_ROUNDS,
                      axis_cap=surface.AXIS_CAP,bend_cap=surface.BEND_CAP,birth_margin=surface.BIRTH_MARGIN,
                      overfit_patience=OVERFIT_PATIENCE,validation_margin=VALIDATION_MARGIN,train_margin=TRAIN_MARGIN)
    if early:
        provenance.update(early_arm=early,final_step=final_step,first_refresh=25,expected_anchor=expected_anchor)
    if (out/'results.json').exists():
        if read(out/'results.json')['provenance'] != provenance:
            raise ValueError('Completed run has different provenance')
        return
    val_loader = loader(validation,8,BATCH_SIZE,SEED)
    schedule = cosine_lr(shadow.optimizer,LEARNING_RATE,20,COMPARISON_STEPS)
    resume,best_path = scratch/'resume.pt',scratch/'best.pt'
    ends = {round(warmup*i/5) for i in range(1,6)}
    history,replay,records,epochs = [],[],[],[]
    window = torch.zeros(sum(op.sizes),device='cuda')
    state,best = None,None
    continuation_rate,stale = .10*LEARNING_RATE,0
    gradient_sum = None
    if resume.exists():
        core = np._core if hasattr(np,'_core') else np.core
        with torch.serialization.safe_globals([core.multiarray.scalar,np.dtype,type(np.dtype('float64'))]):
            saved = torch.load(resume,weights_only=True,map_location='cpu',mmap=True)
        if saved['provenance'] != provenance:
            raise ValueError('Resume provenance mismatch')
        visual.load_state_dict(saved['visual'])
        shadow.optimizer.load_state_dict(saved['optimizer'])
        history = [d.cuda() for d in saved['history']]
        window = saved['warmup_window'].cuda()
        replay,records,epochs = saved['replay'],saved['records'],saved['epochs']
        best,continuation_rate,stale = saved['best'],saved['continuation_rate'],saved['stale']
        state = saved['surface']
        if early=='b' and saved.get('gradient_sum') is not None:
            gradient_sum = saved['gradient_sum'].clone()
        if state is not None:
            state.update(frames=state['frames'].cuda(),z=np.array(state['z']),q=np.array(state['q']),
                         cumulative=[d.clone() for d in state['cumulative']])
        del saved
    while len(records)<final_step if early else not overfit(epochs,best):
        epoch = len(epochs)+1
        data.epoch = epoch-1
        model.train()
        training_ce = []
        for i,(images,labels,ids) in enumerate(loader(data,8,BATCH_SIZE,SEED+data.epoch,True)):
            step = (epoch-1)*batches+i+1
            n = surface.REPLAY_IMAGES
            replay_item = images[:n].clone(),labels[:n].clone(),ids[:n].clone()
            if step<=COMPARISON_STEPS:
                schedule(step-1)
            else:
                shadow.optimizer.param_groups[0]['lr'] = continuation_rate
            shadow.clear()
            images,labels = images.cuda(non_blocking=True),labels.cuda(non_blocking=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                objective = F.cross_entropy(model(images).float(),labels)
            objective.backward()
            torch.nn.utils.clip_grad_norm_(op.params,1.,error_if_nonfinite=True)
            decay,raw = shadow.propose()
            value = float(objective.detach())
            training_ce.append(value*len(labels))
            row = dict(step=step,training_ce=value,lr=float(shadow.optimizer.param_groups[0]['lr']))
            if early:
                row['batch_ids_sha256'] = hashlib.sha256(ids.numpy().data).hexdigest()
            if step<=warmup:
                shadow.apply(shadow.shadows,decay,window)
                if step in ends:
                    history.append(window)
                    window = torch.zeros_like(window)
            else:
                if state is None:
                    anchor = torch.cat([p.detach().flatten().cpu() for p in op.params])
                    state = dict(frames=initialize(history).unsqueeze(0),z=np.zeros((1,2)),q=np.array([[.08,0.,.08]]),
                                 scale=16*float(raw.norm()),warmup_anchor=anchor,cumulative=[torch.zeros_like(anchor)])
                    state['offset_coefficients'] = surface.offset_coefficients(state)
                    history.clear()
                    if early:
                        anchor_sha = hashlib.sha256(anchor.numpy().data).hexdigest()
                        if expected_anchor and expected_anchor!=anchor_sha:
                            raise ValueError('Step-12 anchor differs from the verified replay')
                        initial = hashlib.sha256(state['frames'].cpu().numpy().data).hexdigest()
                        score = existing.evaluate(model,val_loader)
                        existing.write(out/'anchor.json',dict(step=warmup,sha256=anchor_sha,initial_surface_sha256=initial,
                                                             initial_scale=state['scale'],validation=score))
                        model.train()
                correction = None
                if early=='b':
                    current_gradient = torch.cat([p.grad.detach().flatten().cpu() for p in op.params])
                    if step>13 and (step-13)%surface.WINDOW==0 and gradient_sum is not None:
                        from functools import partial as bind
                        from dtd_surface_residual import correct
                        correction = bind(correct,gradient=gradient_sum,next_gradient=current_gradient)
                        gradient_sum = None
                    if gradient_sum is None:
                        gradient_sum = current_gradient.clone()
                    else:
                        gradient_sum.add_(current_gradient)
                del decay,raw
                shadow.clear()
                recent = [r['surface'] for r in records[-surface.WINDOW:] if 'surface' in r]
                row['surface'] = surface.step(op,shadow,images,labels,state,replay,recent,
                                              step-warmup,max_components,13 if early else 1,correction)
                if early=='b':
                    del current_gradient,correction
                if row['surface']['window'] is not None:
                    win = row['surface']['window']
                    print(f'Window {step}: {win["policy"]} groups={row["surface"]["components"]}',flush=True)
            replay = [*replay,replay_item][-surface.REPLAY:]
            records.append(row)
            if step%10==0:
                print(f'DTD surface step {step} train_ce={value:.5f} M={0 if state is None else len(state["z"])}',flush=True)
        score = existing.evaluate(model,val_loader)
        if state is not None:
            canonical = surface.canonical_total(state)
            actual = torch.cat([p.detach().flatten().cpu() for p in op.params])
            if not torch.equal(actual,canonical.float()):
                raise RuntimeError('The weights diverged from the canonical group decomposition')
            del actual,canonical
        current = dict(epoch=epoch,step=len(records),train_ce=float(sum(training_ce)/len(training)),validation=score,
                       groups=0 if state is None else len(state['z']),lr=float(shadow.optimizer.param_groups[0]['lr']))
        epochs.append(current)
        if state is not None and (best is None or score['ce']<best['validation']['ce']):
            stale = 0 if best is None or score['ce']<best['validation']['ce']-VALIDATION_MARGIN else stale+1
            best = current
            save(best_path,dict(visual=visual.state_dict(),surface=cpu_state(state),best=best,provenance=provenance))
            save(out/'best_visual.pt',dict(visual=visual.state_dict(),best=best,provenance=provenance))
        elif len(records)>=COMPARISON_STEPS:
            stale += 1
        if not early and len(records)>=COMPARISON_STEPS and stale>=LR_PATIENCE:
            continuation_rate = max(.005*LEARNING_RATE,.5*continuation_rate)
            stale = 0
        resume_state = dict(visual=visual.state_dict(),optimizer=shadow.optimizer.state_dict(),surface=cpu_state(state),
                         history=history,warmup_window=window,replay=replay,records=records,epochs=epochs,
                         best=best,continuation_rate=continuation_rate,stale=stale,provenance=provenance)
        if early=='b':
            resume_state['gradient_sum'] = gradient_sum
        save(resume,resume_state)
        del resume_state
        existing.write(out/'progress.json',dict(provenance=provenance,epochs=epochs,best=best,
                                               next_lr=continuation_rate,stop=len(records)>=final_step if early else overfit(epochs,best)))
        print(f'Epoch {epoch} step {len(records)} train_ce={current["train_ce"]:.5f} '
              f'val_ce={score["ce"]:.5f} val_acc={score["accuracy"]:.5f}',flush=True)

    # The validation-selected checkpoint is locked before loading the historical Test.
    del state
    locked = torch.load(best_path,weights_only=True,map_location='cpu',mmap=True)
    if locked['best'] != best:
        raise ValueError('Best checkpoint and validation selection disagree')
    visual.load_state_dict(locked['visual'])
    final_state = locked['surface']
    final = out/'epoch_best.pt'
    partial = final.with_suffix('.pt.tmp')
    checkpoint_hash = existing.sha(best_path)
    try:
        shutil.copyfile(best_path,partial)
        if existing.sha(partial)!=checkpoint_hash:
            raise ValueError('Persistent checkpoint copy failed verification')
        partial.replace(final)
    except OSError as exc:
        if exc.errno!=errno.ENOSPC:
            raise
        partial.unlink(missing_ok=True)
        final = best_path
        print('Data disk full. The locked full checkpoint remains in scratch.',flush=True)
    final_state.update(frames=final_state['frames'].cuda(),z=np.array(final_state['z']),q=np.array(final_state['q']))
    test = test_set('DTD',encoder.val_preprocess)
    protocol = test_protocol('DTD',test)
    score = existing.evaluate(model,loader(test,8,BATCH_SIZE,SEED))
    coordinate_audit = audit_coordinates(op,validation,final_state)
    existing.write(out/'results.json',dict(provenance=provenance,epochs=epochs,records=records,best=best,
        stop_reason='fixed_comparison_budget' if early else 'sustained_validation_ce_increase_with_lower_training_ce',checkpoint_sha256=checkpoint_hash,
        checkpoint_path=str(final),
        test=dict(protocol=protocol,**score),learned=dict(groups=len(final_state['z']),coordinates=2*len(final_state['z']),
            z=final_state['z'].tolist(),q=final_state['q'].tolist(),cumulative_group_norms=[float(v.norm()) for v in final_state['cumulative']],
            coordinate_audit=coordinate_audit)))
    print(f'DTD locked test {score["accuracy"]:.6f}',flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--scratch',type=Path,required=True)
    parser.add_argument('--max-components',type=int,default=6)
    parser.add_argument('--early',choices=('a','b'))
    parser.add_argument('--expected-anchor')
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    args.scratch.mkdir(parents=True,exist_ok=True)
    with (args.output/'train.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        train(args.output,args.scratch,args.max_components,args.early,args.expected_anchor)


if __name__=='__main__':
    main()
