"""Replay DTD with an exact update ledger and restricted local fiber probes."""

import argparse
import fcntl
from pathlib import Path
import platform

import numpy as np
import torch
from torch.nn import functional as F

import train_fd_task as trainer
from diagnose_dtd_residual_fiber import interval
from task_data import test_set
from train_dtd_online_carrier import response
from train_dtd_rolling_innovation import probe_mode
from train_eight_expert_queue import ROOT, existing, loader, read


ANCHORS = (48, 81, 105, 137, 168, 216)
PROBE_SEED = 351179
PROBE_N = 48
SPECTRAL_CUTOFF = .01
GRAM_CUTOFF = 1e-6


def flat_cpu(op):
    return torch.cat([p.detach().flatten().cpu() for p in op.params])


def whitening(gram):
    value, vector = np.linalg.eigh(gram)
    keep = value > max(GRAM_CUTOFF*value[-1],1e-12)
    return vector[:,keep] / np.sqrt(value[keep]), value


def spectrum(features, white):
    matrix = features @ white
    values, vectors = np.linalg.eigh(matrix.T @ matrix)
    return values, vectors


def features(op, x, direction):
    return np.concatenate([response(op,part,direction)*np.sqrt(len(part)/len(x)) for part in x.split(24)])


def embedding(logits):
    return 2 * logits.double().softmax(-1).sqrt()


def readings(before, after, labels, seed):
    p, q = [z.double().log_softmax(-1) for z in (before,after)]
    ce = F.nll_loss(p,labels,reduction='none') - F.nll_loss(q,labels,reduction='none')
    kl = (p.exp()*(p-q)).sum(-1)
    fr = (embedding(before)-embedding(after)).square().sum(-1)
    return dict(ce_gain=interval(ce.cpu().numpy(),seed),kl=interval(kl.cpu().numpy(),seed),
                fisher_rao_chord_squared=interval(fr.cpu().numpy(),seed))


class Dictionary:
    def __init__(self, rows):
        self.norms = [float(r.norm()) for r in rows]
        self.rows = torch.stack([r/max(n,1e-20) for r,n in zip(rows,self.norms)])
        self.gram = np.zeros((len(rows),len(rows)))
        for block in self.rows.split(524288,dim=1):
            block = block.double()
            self.gram += (block @ block.T).numpy()

    def vector(self, coefficients, device):
        return (torch.as_tensor(coefficients,dtype=self.rows.dtype) @ self.rows).to(device)

    def norm(self, coefficients):
        return float(np.sqrt(max(0.,coefficients @ self.gram @ coefficients)))


def finite(op, groups, baseline, dictionary, coefficient, seed):
    delta = dictionary.vector(coefficient,groups[0][0].device)
    return dict(parameter_norm=float(delta.norm()),groups=[readings(z,op.evaluate(x,delta),y,seed+i)
                  for i,((x,y),z) in enumerate(zip(groups,baseline))])


def choose_horizontal_dose(op, groups, baseline, dictionary, fit, direction, covered, radius, seed):
    x,y = groups[fit]
    before = F.cross_entropy(baseline[fit].double(),y)
    fit_gains = [dict(fraction=0.,ce_gain=0.)]
    for fraction in (.03125,.0625,.125,.25,.5,1.):
        delta = dictionary.vector(fraction*radius*direction,x.device)
        loss = F.cross_entropy(op.evaluate(x,delta).double(),y)
        fit_gains.append(dict(fraction=fraction,ce_gain=float(before-loss)))
    chosen = max(fit_gains,key=lambda row:row['ce_gain'])['fraction']
    full = finite(op,groups,baseline,dictionary,chosen*radius*direction,seed)
    approximation = finite(op,groups,baseline,dictionary,chosen*radius*covered,seed)
    x,y = groups[1-fit]
    def losses(coefficient):
        delta = dictionary.vector(chosen*radius*coefficient,x.device)
        return F.cross_entropy(op.evaluate(x,delta).double(),y,reduction='none')
    return dict(fit_gains=fit_gains,selected_fraction=chosen,horizontal=full,covered=approximation,
                heldout_covered_minus_horizontal_ce=interval((losses(covered)-losses(direction)).cpu().numpy(),seed))


def continue_path(op, groups, baseline, dictionary, low, horizontal, jacobian, radius, seed):
    x = groups[0][0]
    reference = embedding(baseline[0])
    matrix = jacobian @ horizontal
    position = np.zeros_like(low)
    rows = []
    for k in range(1,4):
        distance = k*radius/2
        position = position + radius/2*low
        prediction = position.copy()
        def error(coef):
            logits = op.evaluate(x,dictionary.vector(coef,x.device))
            return ((embedding(logits)-reference)/np.sqrt(len(x))).flatten().cpu().numpy()
        for _ in range(2):
            residual = error(position)
            correction = horizontal @ np.linalg.lstsq(matrix,-residual,rcond=.01)[0]
            accepted = False
            for fraction in (1.,.5,.25):
                attempt = position + fraction*correction
                if dictionary.norm(attempt-distance*low) > .5*distance:
                    continue
                if np.linalg.norm(error(attempt)) < np.linalg.norm(residual):
                    position = attempt
                    accepted = True
                    break
            if not accepted:
                break
        rows.append(dict(vertical_progress=distance,correction_norm=dictionary.norm(position-distance*low),
                         predicted=finite(op,groups,baseline,dictionary,prediction,seed),
                         corrected=finite(op,groups,baseline,dictionary,position,seed),
                         straight_low=finite(op,groups,baseline,dictionary,distance*low,seed)))
    return rows


class Audit:
    def __init__(self, output, reference):
        self.output, self.reference = output, reference
        self.decay, self.history, self.closure = 1., [], []
        self.anchor = self.carrier = self.residual = self.telescope = None
        self.groups = None

    def snapshot(self):
        return {k:getattr(self,k) for k in ('decay','history','closure','anchor','carrier','residual','telescope')}

    def restore(self, saved, step):
        self.__dict__.update(saved)
        for path in (self.output/'anchors').glob('*.json'):
            if int(path.stem)>step:
                path.unlink()

    def start(self, op):
        self.anchor = flat_cpu(op)
        self.carrier = torch.zeros_like(self.anchor,dtype=torch.float64)
        self.residual = torch.zeros_like(self.carrier)
        self.telescope = torch.zeros_like(self.carrier)

    def advance(self, op, step, decay_point, a, structured, residual, delay):
        c,r = structured.detach().cpu(),residual.detach().cpu()
        self.decay *= float(a)
        self.carrier.mul_(float(a)).add_(c)
        self.residual.mul_(float(a)).add_(r,alpha=1-delay)
        self.telescope.add_(c)
        self.history = [*self.history[-1:],torch.cat([(p.detach()-q).flatten().cpu()
                                                     for p,q in zip(op.params,decay_point)])]
        if step in ANCHORS:
            delta = flat_cpu(op)-self.anchor
            error = self.anchor.double().mul_(self.decay).add_(self.carrier).add_(self.residual).sub_(flat_cpu(op))
            self.closure.append(dict(step=step,l2=float(error.norm()),max_abs=float(error.abs().max()),
                                     relative_to_displacement=float(error.norm()/delta.norm())))

    def sample(self, step, state, cap):
        return step in ANCHORS

    def probes(self, op):
        if self.groups is None:
            dataset = test_set('DTD',op.model.image_encoder.val_preprocess)
            ids = torch.randperm(len(dataset),generator=torch.Generator().manual_seed(PROBE_SEED))[:2*PROBE_N].tolist()
            self.indices = [ids[:PROBE_N],ids[PROBE_N:]]
            self.groups = []
            for group in self.indices:
                rows = [dataset[i] for i in group]
                self.groups.append((torch.stack([x for x,_ in rows]).cuda(),
                                    torch.tensor([y for _,y in rows],device='cuda')))
        return self.groups

    def record(self, op, step, old, state, proposal, c, r, details, delay):
        print(f'Restricted fiber anchor {step}',flush=True)
        groups = self.probes(op)
        z,q,frames = old['z'],old['q'],old['frames']
        tangent = trainer.tangent(z,q)
        axes = []
        for i,frame in enumerate(frames):
            weights = state['scale']*tangent[3*i:3*i+3,2*i:2*i+2].T
            axes.extend((torch.as_tensor(weights,dtype=frame.dtype) @ frame).unbind())
        nu = len(axes)
        labels = [f'U{i+1}' for i in range(nu)]+['proposal','negative_optimality']
        rows = axes+[proposal.detach().cpu(),-old['optimality']]
        for i,previous in enumerate(reversed(self.history),1):
            labels.append(f'executed_minus_decay_t-{i}')
            rows.append(previous)
        if 'birth_direction' in old:
            labels.append('birth_candidate')
            rows.append(old['birth_direction'])
        dictionary = Dictionary(rows)
        del rows,axes
        white,p_values = whitening(dictionary.gram)
        white_u,_ = whitening(dictionary.gram[:nu,:nu])
        seed = PROBE_SEED+step
        radius = max(float(proposal.norm()),.002)
        with probe_mode(op.model):
            baseline = [op.evaluate(x) for x,_ in groups]
            jacobians = []
            for x,_ in groups:
                jacobians.append(np.stack([features(op,x,d.to(x.device)) for d in dictionary.rows],axis=1))
            group_stats, trials = [], []
            for i,j in enumerate(jacobians):
                j_other = jacobians[1-i]
                lam,basis = spectrum(j,white)
                u_lam,_ = spectrum(j[:,:nu],white_u)
                u_scale = np.asarray(dictionary.norms[:nu])
                _,y = groups[i]
                p_y = baseline[i].double().softmax(-1).gather(1,y[:,None]).flatten().cpu().numpy()
                g = -(j.reshape(len(y),-1,len(labels))[np.arange(len(y)),y.cpu().numpy()]
                      / p_y[:,None].clip(1e-24)**.5).sum(axis=0)/np.sqrt(len(y))
                low,high = white @ basis[:,0],white @ basis[:,-1]
                h = white @ basis[:,lam > SPECTRAL_CUTOFF*lam[-1]]
                heldout_scale = spectrum(j_other,white)[0][-1]
                group_stats.append(dict(fisher=(j.T@j).tolist(),eigenvalues=lam.tolist(),
                                        normalized_eigenvectors=(white@basis).tolist(),carrier_eigenvalues=u_lam.tolist(),
                                        carrier_gram=((j[:,:nu]*u_scale).T@(j[:,:nu]*u_scale)).tolist(),
                                        low_holdout_energy=float(np.linalg.norm(j_other@low)**2),
                                        low_holdout_relative=float(np.linalg.norm(j_other@low)**2/heldout_scale),
                                        high_holdout_energy=float(np.linalg.norm(j_other@high)**2),
                                        carrier_identifiable=int((u_lam>SPECTRAL_CUTOFF*lam[-1]).sum())))
                # Sign and regularized direction use only the fit group.
                active = lam > SPECTRAL_CUTOFF*lam[-1]
                horizontal_descent = -h @ ((h.T@g)/(lam[active]+.01*lam[-1]))
                direction = horizontal_descent / max(dictionary.norm(horizontal_descent),1e-20)
                coverage = np.linalg.lstsq(j[:,:nu],j@direction,rcond=.01)[0]
                covered = np.pad(coverage,(0,len(labels)-nu))
                trial = dict(fit_group=i,heldout_group=1-i,low_coefficient=low.tolist(),
                             high_coefficient=high.tolist(),horizontal_coefficient=direction.tolist(),
                             covered_coefficient=covered.tolist(),
                             uncovered_fit=float(np.linalg.norm(j@(direction-covered))**2/max(np.linalg.norm(j@direction)**2,1e-30)),
                             uncovered_holdout=float(np.linalg.norm(j_other@(direction-covered))**2/max(np.linalg.norm(j_other@direction)**2,1e-30)),
                             horizontal=finite(op,groups,baseline,dictionary,radius*direction,seed),
                             covered=finite(op,groups,baseline,dictionary,radius*covered,seed))
                random = np.random.default_rng(seed+i).standard_normal(white.shape[1])
                random = white @ (random/np.linalg.norm(random))
                trial['straight'] = [dict(radius=d*radius,low=finite(op,groups,baseline,dictionary,d*radius*low,seed),
                                     high=finite(op,groups,baseline,dictionary,d*radius*high,seed),
                                     random=finite(op,groups,baseline,dictionary,d*radius*random,seed)) for d in (.5,1.,1.5)]
                trial['path'] = continue_path(op,groups if i==0 else groups[::-1],
                                              baseline if i==0 else baseline[::-1],dictionary,low,h,j,radius,seed)
                if step in (168,216):
                    trial['fit_selected_dose'] = choose_horizontal_dose(op,groups,baseline,dictionary,i,direction,covered,radius,seed)
                trials.append(trial)
        existing.write(self.output/'anchors'/f'{step:03d}.json',dict(step=step,components=len(z),birth=details['birth'],
                       indices=self.indices,labels=labels,parameter_norms=dictionary.norms,parameter_gram=dictionary.gram.tolist(),
                       parameter_eigenvalues=p_values.tolist(),probe_geometry=group_stats,radius=radius,trials=trials,
                       peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30))
        print(f'Anchor {step} saved; peak allocated {torch.cuda.max_memory_allocated()/2**30:.2f} GiB',flush=True)

    def finish(self, op, state, records, provenance, resume):
        actual = flat_cpu(op)
        a = self.anchor.double()*self.decay
        c,r = self.carrier,self.residual
        expert = a+c+r
        telescoped = trainer.decode(state['frames'],state['z'],state['q']).cpu()*state['scale']
        diff = self.telescope-telescoped
        reductions = dict(decay_product=self.decay,closure_l2=float((expert-actual).norm()),
                          closure_max_abs=float((expert-actual).abs().max()),
                          closure_relative_displacement=float((expert-actual).norm()/(actual-self.anchor).norm()),
                          telescope_l2=float(diff.norm()),telescope_relative=float(diff.norm()/telescoped.norm()),
                          carrier_norm=float(c.norm()),residual_norm=float(r.norm()),
                          carrier_residual_cosine=float((c@r)/(c.norm()*r.norm())))
        targets = dict(A=a,AC=a+c,AR=a+r,ACR=expert)
        offsets = {name:(value-actual).float().cuda() for name,value in targets.items()}
        dataset = test_set('DTD',op.model.image_encoder.val_preprocess)
        models = {k:dict(ce=0.,accuracy=0.,kl_from_actual=0.,max_logit_difference=0.) for k in ('actual',*targets)}
        with probe_mode(op.model):
            for x,y in loader(dataset,4,64,42):
                x,y = x.cuda(),y.cuda()
                before = op.evaluate(x)
                p = before.double().log_softmax(-1)
                for name,row in models.items():
                    logits = before if name=='actual' else op.evaluate(x,offsets[name])
                    q = logits.double().log_softmax(-1)
                    row['ce'] += float(F.nll_loss(q,y,reduction='sum'))
                    row['accuracy'] += int((logits.argmax(-1)==y).sum())
                    row['kl_from_actual'] += float((p.exp()*(p-q)).sum())
                    row['max_logit_difference'] = max(row['max_logit_difference'],float((logits-before).abs().max()))
        for row in models.values():
            for key in ('ce','accuracy','kl_from_actual'):
                row[key] /= len(dataset)
            row['n'] = len(dataset)
        result = dict(provenance=provenance,runtime=dict(torch=torch.__version__,python=platform.python_version(),
                      gpu=torch.cuda.get_device_name()),source_test=read(self.reference/'historical_test.json'),
                      anchors=[read(p) for p in sorted((self.output/'anchors').glob('*.json'))],records=records,
                      ledger=dict(**reductions,steps=self.closure,models=models,
                                  ce_interaction=models['ACR']['ce']-models['AC']['ce']-models['AR']['ce']+models['A']['ce']),
                      final=dict(z=state['z'].tolist(),q=state['q'].tolist(),scale=state['scale']))
        existing.write(self.output/'results.json',result)
        resume.unlink(missing_ok=True)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('output','scratch','reference'):
        parser.add_argument('--'+name,type=Path,required=True)
    args = parser.parse_args()
    for path in (args.output/'anchors',args.scratch):
        path.mkdir(parents=True,exist_ok=True)
    source = read(args.reference/'complete.json')['provenance']
    if source['task']!='DTD' or source['arm']!='adaptive_dynamics' or source['steps']!=216:
        raise ValueError('Expected the reviewed 216-step DTD expert')
    trainer.TASK,trainer.STEPS = 'DTD',source['steps']
    existing.ROOT = ROOT
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    audit = Audit(args.output,args.reference)
    with (args.output/'replay.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (args.output/'results.json').exists():
            raise FileExistsError(args.output/'results.json')
        trainer.train('adaptive_dynamics',args.output,source['max_components'],source['warmup'],
                      source['birth_penalty'],observer=audit,scratch=args.scratch,ledger=audit)


if __name__ == '__main__':
    main()
