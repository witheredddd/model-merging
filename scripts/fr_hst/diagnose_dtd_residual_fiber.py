"""Read-only residual diagnostics during an unchanged DTD dynamics replay."""

import argparse
from collections import Counter
from pathlib import Path
import platform
import shutil

import numpy as np
import torch
from torch.nn import functional as F

import train_fd_task as trainer
from task_data import test_set
from train_dtd_online_carrier import response
from train_dtd_rolling_innovation import probe_mode
from train_eight_expert_queue import ROOT, existing, loader, read


FIXED_STEPS = (37, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 215, 216)
DOSES = (0., .25, .5, 1.)
CE_TOL = .002
FUNCTION_RELATIVE_FLOOR = .01
SVD_CUTOFF = .01
LOW_NOVELTY, HIGH_NOVELTY = .35, .65
PROBE_SEED = 149603


def interval(values, seed):
    x = np.asarray(values, dtype=np.float64)
    choices = np.random.default_rng(seed).integers(len(x), size=(2048, len(x)))
    return dict(mean=float(x.mean()), ci95=np.quantile(x[choices].mean(axis=1), [.025,.975]).tolist())


def energy(vector):
    return np.square(vector).sum(axis=1)


def cosine(a, b):
    return float(np.vdot(a,b) / max(np.linalg.norm(a)*np.linalg.norm(b),1e-30))


def fit_svd(matrix, target):
    matrix = matrix.reshape(-1,matrix.shape[-1])
    u, s, v = np.linalg.svd(matrix,full_matrices=False)
    keep = s > max(SVD_CUTOFF*s[0],1e-12)
    coefficient = v[keep].T @ ((u[:,keep].T @ target.ravel()) / s[keep])
    return coefficient, int(keep.sum()), s.tolist()


def cross_projection(bases, responses):
    fits = [fit_svd(a,b) for a,b in zip(bases,responses)]
    fractions = []
    within = []
    for i,(basis,target) in enumerate(zip(bases,responses)):
        denominator = max(energy(target).mean(),1e-30)
        within.append(float(energy(target-basis @ fits[i][0]).mean()/denominator))
        fractions.append(float(energy(target-basis @ fits[1-i][0]).mean()/denominator))
    return dict(cross_novelty=fractions,within_novelty=within,ranks=[a[1] for a in fits],
                singular_values=[a[2] for a in fits]), [fits[1][0],fits[0][0]]


def tangent_update(frames, z, q, coefficients, like):
    weights = (trainer.tangent(z,q) @ coefficients).reshape(-1,3)
    vector = torch.zeros_like(like)
    for frame, weight in zip(frames,weights):
        axis = frame.to(like.device)
        vector.add_(trainer.tensor(weight,axis) @ axis)
    return vector


def image_response(op, x, v):
    return response(op,x,v).reshape(len(x),-1) * np.sqrt(len(x))


def response_basis(op, groups, frames, z, q, like):
    per_group = [[] for _ in groups]
    derivative = trainer.tangent(z,q)
    for m,frame in enumerate(frames):
        axes = frame.to(like.device)
        weights = trainer.tensor(derivative[3*m:3*m+3,2*m:2*m+2].T,axes)
        for weight in weights:
            direction = weight @ axes
            for bucket,(x,_) in zip(per_group,groups):
                bucket.append(image_response(op,x,direction))
    return [np.stack(rows,axis=-1) for rows in per_group]


def classify(groups, projection, dose, novel_gain):
    weak = all(g['function_above_floor']['ci95'][1] < 0 for g in groups)
    strong = all(g['function_above_floor']['ci95'][0] > 0 for g in groups)
    bounds = [d['pooled']['ci95'] for d in dose[1:]]
    equivalent = all(-CE_TOL < lo <= hi < CE_TOL for lo,hi in bounds)
    endpoint = dose[-1]
    benefit = endpoint['pooled']['ci95'][0] > CE_TOL and all(x > 0 for x in endpoint['group_means'])
    harmful = endpoint['pooled']['ci95'][1] < -CE_TOL and all(x < 0 for x in endpoint['group_means'])
    if weak and equivalent:
        return 'V'
    if strong and (harmful or equivalent):
        return 'H3'
    if strong and benefit:
        novelty = projection['cross_novelty']
        if max(novelty) <= LOW_NOVELTY:
            return 'H1'
        if min(novelty) >= HIGH_NOVELTY and novel_gain['pooled']['ci95'][0] > CE_TOL and min(novel_gain['group_means']) > 0:
            return 'H2'
    return 'undetermined'


def combined(groups, seed):
    return dict(pooled=interval(np.concatenate(groups),seed),group_means=[float(x.mean()) for x in groups],
                per_image=[x.tolist() for x in groups])


class Observer:
    def __init__(self, output, reference):
        self.output = output
        self.reference = reference
        self.groups = None
        self.indices = None

    def sample(self, step, state, cap):
        trial = len(state['frames']) < cap and step-state['last_birth'] >= 16 and step%8 in (0,1)
        return step in FIXED_STEPS or trial or step == state['last_birth']+1

    def probe_groups(self, op):
        if self.groups is None:
            dataset = test_set('DTD',op.model.image_encoder.val_preprocess)
            ids = torch.randperm(len(dataset),generator=torch.Generator().manual_seed(PROBE_SEED))[:64].tolist()
            self.indices = [ids[:32],ids[32:]]
            self.groups = []
            for group in self.indices:
                samples = [dataset[i] for i in group]
                self.groups.append((torch.stack([x for x,_ in samples]).cuda(),
                                    torch.tensor([y for _,y in samples],device='cuda')))
        return self.groups

    def record(self, op, step, old, state, proposal, c, r, details, delay):
        print(f'Fiber diagnostic step {step}, M={len(old["z"])} → {len(state["z"])}',flush=True)
        groups = self.probe_groups(op)
        seed = PROBE_SEED+step
        generator = torch.Generator(device=r.device).manual_seed(seed)
        random = torch.randn(r.shape,device=r.device,generator=generator)
        random.mul_(r.norm()/random.norm().clamp_min(1e-20))
        optimal = old['optimality'].to(r.device)
        optimal_norm = float(optimal.norm())
        optimal.mul_(-r.norm()/optimal.norm().clamp_min(1e-20))
        geometric = old['geometry_update'].to(r.device)
        birth_direction = old.get('birth_direction')
        if birth_direction is not None:
            birth_direction = birth_direction.to(r.device)*r.norm()
        geometry = []
        yr = []
        ce_by_dose = [[] for _ in DOSES]
        ce_novel = []
        ce_evolution = []
        with probe_mode(op.model):
            bases = response_basis(op,groups,old['frames'],old['z'],old['q'],r)
            for i,(x,y) in enumerate(groups):
                fc, fr, fn, fo = [image_response(op,x,v) for v in (c,r,random,optimal)]
                fv = fc+fr
                er,ev,en = map(energy,(fr,fv,fn))
                yr.append(fr)
                row = dict(proposal_fisher=float(ev.mean()),structured_fisher=float(energy(fc).mean()),
                           residual_fisher=float(er.mean()),random_fisher=float(en.mean()),
                           cross_term=float(2*(fc*fr).sum(axis=1).mean()),
                           residual_to_proposal=float(er.mean()/max(ev.mean(),1e-30)),
                           function_above_floor=interval(er-np.maximum(en,FUNCTION_RELATIVE_FLOOR*ev),seed+i),
                           optimality_fisher_cosine=cosine(fr,fo),
                           geometry_evolution_fisher=float(energy(image_response(op,x,geometric)).mean()),
                           image_residual_fisher=er.tolist(),image_proposal_fisher=ev.tolist(),image_random_fisher=en.tolist())
                if birth_direction is not None:
                    fb = image_response(op,x,birth_direction)
                    row['birth_fisher'] = float(energy(fb).mean())
                    row['birth_residual_fisher_cosine'] = cosine(fr,fb)
                geometry.append(row)
            projection, coefficients = cross_projection(bases,yr)
            after_birth = None
            if details['birth'] and details['birth']['added']:
                new_bases = response_basis(op,groups,state['frames'],state['z'],state['q'],r)
                after_birth = dict(all_components=cross_projection(new_bases,yr)[0],
                                   evolved_without_newborn=cross_projection([a[:,:,:-2] for a in new_bases],yr)[0])
            for i,(x,y) in enumerate(groups):
                def ce(delta):
                    return F.cross_entropy(op.evaluate(x,delta).double(),y,reduction='none').cpu().numpy()
                baseline = ce(c)
                geometry[i]['carrier_ce_gain'] = interval(ce(None)-baseline,seed+i)
                for bucket,alpha in zip(ce_by_dose,DOSES):
                    bucket.append(baseline-ce(c+alpha*r) if alpha else np.zeros_like(baseline))
                covered = tangent_update(old['frames'],old['z'],old['q'],coefficients[i],r)
                geometry[i]['covered_parameter_norm'] = float(covered.norm())
                ce_novel.append(ce(c+covered) - (baseline-ce_by_dose[-1][i]))
                ce_evolution.append(ce(c-geometric)-baseline)
        dose = [dict(alpha=a,**combined(rows,seed)) for a,rows in zip(DOSES,ce_by_dose)]
        novel_gain = combined(ce_novel,seed)
        row = dict(step=step,old_components=len(old['z']),new_components=len(state['z']),
                   probe_indices=self.indices,geometry=geometry,projection=projection,after_birth=after_birth,
                   residual_class=classify(geometry,projection,dose,novel_gain),ce_doses=dose,
                   ce_novel_gain=novel_gain,ce_geometry_evolution=combined(ce_evolution,seed),
                   birth=details['birth'],optimality_norm=optimal_norm,
                   optimality_residual_euclidean_cosine=float((r@optimal)/(r.norm()*optimal.norm()).clamp_min(1e-20)),
                   proposal_parameter_norm=float(proposal.norm()),carrier_parameter_norm=float(c.norm()),
                   residual_parameter_norm=float(r.norm()),geometry_parameter_norm=float(geometric.norm()),
                   closure_relative=float((proposal-c-r).norm()/proposal.norm().clamp_min(1e-20)),
                   innovation_written_fraction=1-delay,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
        existing.write(self.output/'windows'/f'{step:03d}.json',row)
        print(f'Step {step} residual class {row["residual_class"]}; peak allocated {row["peak_allocated_gib"]:.2f} GiB',flush=True)

    def finish(self, op, state, records, provenance, resume):
        dataset = test_set('DTD',op.model.image_encoder.val_preprocess)
        score = existing.evaluate(op.model,loader(dataset,4,64,42))
        source = read(self.reference/'historical_test.json')
        windows = [read(p) for p in sorted((self.output/'windows').glob('*.json'))]
        count = Counter(w['residual_class'] for w in windows)
        energy_totals = {k:sum(np.mean([g['residual_fisher'] for g in w['geometry']])
                              for w in windows if w['residual_class']==k) for k in count}
        result = dict(provenance=provenance,source_checkpoint=source,source_births=self.source_births,
                      runtime=dict(python=platform.python_version(),torch=torch.__version__,cuda=torch.version.cuda,
                                   gpu=torch.cuda.get_device_name(0)),
                      diagnostics=dict(fixed_steps=FIXED_STEPS,probe_seed=PROBE_SEED,ce_tolerance=CE_TOL,
                                       function_relative_floor=FUNCTION_RELATIVE_FLOOR,svd_cutoff=SVD_CUTOFF,
                                       novelty_thresholds=[LOW_NOVELTY,HIGH_NOVELTY],image_groups='historical_test'),
                      final=dict(test=score,z=state['z'].tolist(),q=state['q'].tolist(),scale=state['scale'],
                                 components=len(state['z']),coordinates=int(state['z'].size),
                                 peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30),
                      class_counts=dict(count),class_fisher_energy=energy_totals,windows=windows,records=records)
        existing.write(self.output/'results.json',result)
        resume.unlink(missing_ok=True)
        return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--scratch',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'windows').mkdir(exist_ok=True)
    args.scratch.mkdir(parents=True,exist_ok=True)
    source = read(args.reference/'complete.json')
    provenance = source['provenance']
    assert provenance['task']=='DTD' and provenance['arm']=='adaptive_dynamics'
    assert provenance['batch_size']==trainer.BATCH_SIZE and provenance['seed']==trainer.SEED
    assert provenance['lr_base']==trainer.LEARNING_RATE and provenance['steps']==216
    assert shutil.disk_usage(args.scratch).free >= 12*2**30
    trainer.TASK,trainer.STEPS = 'DTD',provenance['steps']
    existing.ROOT = ROOT
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    observer = Observer(args.output,args.reference)
    observer.source_births = [r['step'] for r in source['records'] if (r.get('birth') or {}).get('added')]
    import fcntl
    with (args.output/'replay.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (args.output/'results.json').exists():
            raise FileExistsError(args.output/'results.json')
        trainer.train('adaptive_dynamics',args.output,provenance['max_components'],provenance['warmup'],
                      provenance['birth_penalty'],observer=observer,scratch=args.scratch)


if __name__ == '__main__':
    main()
