"""Audit the DTD learned groups at the full expert and at actual births."""

import argparse
import fcntl
from itertools import islice
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

import train_fd_task as trainer
from diagnose_dtd_fiber_paths import Dictionary, embedding, features, whitening
from diagnose_dtd_residual_fiber import interval
from run_dtd_carrier_diagnostics import Operator
from train_dtd_rolling_innovation import probe_mode
from train_eight_expert_queue import HEADS, ROOT, build_model, existing, loader, read, splits


SEED = 76219
FIT_N = 32
BIRTH_BUDGET = 40
RANDOM_BIRTHS = 3


def loss(logits, y):
    return F.cross_entropy(logits.double(), y, reduction='none').cpu().numpy()


def logits(op, x, delta=None):
    return torch.cat([op.evaluate(part,delta) for part in x.split(64)])


def metrics(old, new, y, seed):
    p,q = [x.double().log_softmax(-1) for x in (old,new)]
    kl = (p.exp()*(p-q)).sum(-1).cpu().numpy()
    return dict(ce=float(loss(new,y).mean()),accuracy=float((new.argmax(-1)==y).double().mean()),
                ce_gain=interval(loss(old,y)-loss(new,y),seed),kl=interval(kl,seed),
                fr_squared=float((embedding(old)-embedding(new)).square().sum(-1).mean()))


class Data:
    def __init__(self, op):
        transform = op.model.image_encoder.val_preprocess
        train,held = splits('DTD',transform,transform)
        self.dataset = train.dataset
        order = np.random.default_rng(SEED).permutation(train.indices)
        self.fit = order.tolist()
        self.val = list(held.indices)

    def groups(self, step=None):
        exclude = set()
        if step is not None:
            epoch,batch = divmod(step-1,30)
            order = loader(range(len(self.dataset)),0,128,trainer.SEED+epoch,True)
            exclude = set(next(islice(iter(order),batch,None)).tolist())
        indices = [[i for i in self.fit if i not in exclude][:FIT_N],
                   [i for i in self.val if i not in exclude]]
        groups = []
        for ids in indices:
            items = [self.dataset[i] for i in ids]
            groups.append((torch.stack([x for x,_ in items]).cuda(),
                           torch.tensor([y for _,y in items],device='cuda')))
        return groups,indices,sorted(exclude)


class Bank:
    def __init__(self, op, frames, x):
        self.dictionary = Dictionary(frames)
        self.device = x.device
        self.size = len(frames)
        self.j = np.stack([features(op,x,row.to(self.device)) for row in self.dictionary.rows],axis=1)

    def vector(self, coef):
        return self.dictionary.vector(coef,self.device)

    def norm(self, coef):
        return self.dictionary.norm(coef)


class Chart:
    def __init__(self, bank, tangent, decode):
        self.bank = bank
        self.tangent = tangent
        self.decode = decode
        self.white,_ = whitening(tangent.T @ bank.dictionary.gram @ tangent)

    def update(self, point, radius):
        point = point / max(1.,np.linalg.norm(point))
        dz = radius*self.white @ point
        coef = self.decode(dz)
        if self.bank.norm(coef) > radius:
            lo,hi = 0.,1.
            for _ in range(20):
                mid = (lo+hi)/2
                if self.bank.norm(self.decode(mid*dz)) <= radius:
                    lo = mid
                else:
                    hi = mid
            dz = lo*dz
            coef = self.decode(dz)
        return coef,dz


def nonlinear(bank, z, q, scale, groups):
    columns = [i for m in groups for i in (3*m,3*m+1,3*m+2)]
    norm = np.array(bank.dictionary.norms)[columns]
    z,q = np.asarray(z)[groups],np.asarray(q)[groups]
    tangent = np.zeros((bank.size,2*len(groups)))
    tangent[columns] = scale*trainer.tangent(z,q)*norm[:,None]

    def decode(dz):
        coef = np.zeros(bank.size)
        coef[columns] = scale*norm*np.concatenate([trainer.decoder(v+delta,b)[0]-trainer.decoder(v,b)[0]
                                                   for v,b,delta in zip(z,q,dz.reshape(-1,2))])
        return coef

    return Chart(bank,tangent,decode)


def linear(bank, tangent):
    return Chart(bank,tangent,lambda z:tangent@z)


def ce_derivative(j, old, y):
    p_y = old.double().softmax(-1).gather(1,y[:,None]).flatten().cpu().numpy()
    rows = j.reshape(len(y),-1,j.shape[-1])[np.arange(len(y)),y.cpu().numpy()]
    return -(rows/p_y[:,None].clip(1e-24)**.5).sum(axis=0)/np.sqrt(len(y))


def optimize(op, group, chart, radius, baseline, budget, origin=None, warm=None):
    x,y = group
    bank = chart.bank
    shift = np.zeros(bank.size) if origin is None else origin
    j = bank.j @ chart.tangent @ chart.white
    if origin is None:
        g = (chart.tangent@chart.white).T @ ce_derivative(bank.j,baseline,y)
        start = float(loss(baseline,y).mean())
        def objective(pred):
            return float(loss(pred,y).mean())
    else:
        target = embedding(baseline)
        shifted = logits(op,x,bank.vector(shift))
        gap = ((embedding(shifted)-target)/np.sqrt(len(x))).flatten().cpu().numpy()
        g = j.T @ gap
        start = float(gap@gap)
        def objective(pred):
            return float((embedding(pred)-target).square().sum(-1).mean())
    gram = j.T@j
    n = len(g)
    largest = max(float(np.linalg.eigvalsh(gram)[-1]),1e-12)
    point = np.zeros(n)
    best,coef,coordinates = start,np.zeros(bank.size),np.zeros(chart.tangent.shape[1])
    trace = []

    def try_point(p):
        nonlocal point,best,coef,coordinates
        p = p/max(1.,np.linalg.norm(p))
        attempt,dz = chart.update(p,radius)
        score = objective(logits(op,x,bank.vector(shift+attempt)))
        if score < best:
            point,best,coef,coordinates = p,score,attempt,dz
        trace.append(best)

    if warm is not None:
        try_point(np.linalg.lstsq(chart.white,warm/radius,rcond=None)[0])
    for damp in (.01,.1,1.):
        direction = -np.linalg.solve(gram+damp*largest*np.eye(n),g)/radius
        direction /= max(1.,np.linalg.norm(direction))
        for fraction in (1.,.25):
            try_point(fraction*direction)
    width,previous = .25,best
    for k in range(budget-len(trace)):
        i,sign = (k//2)%n,(-1,1)[k%2]
        proposal = point.copy()
        proposal[i] += sign*width
        try_point(proposal)
        if (k+1)%(2*n)==0:
            if previous-best < 1e-7:
                width *= .5
            previous = best
    return coef,dict(initial=start,best=best,trace=trace,norm=bank.norm(coef),dimension=n,budget=budget,
                     coordinates=coordinates.tolist())


def group_rows(frames):
    return [axis for frame in frames for axis in frame]


def random_axes(old_tangent, bank, seed):
    rows = (torch.as_tensor(old_tangent.T,dtype=torch.float32) @ bank.dictionary.rows)
    gram = (rows@rows.T).double()
    generator = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(2):
        v = torch.randn(rows.shape[1],generator=generator)
        v -= torch.linalg.solve(gram,(rows@v).double()).float() @ rows
        for axis in out:
            v -= (v@axis)*axis
        out.append(v/v.norm())
    return out


class CompletedBirthAudit(Exception):
    pass


class Births:
    def __init__(self, output):
        self.output = output
        self.data = None

    def sample(self, step,state,cap):
        return len(state['frames']) < cap and step-state['last_birth']>=16 and step%8==1

    def record(self, op, step, old, state, proposal, c, r, details, delay):
        if not (details['birth'] or {}).get('added'):
            return
        if self.data is None:
            self.data = Data(op)
        groups,ids,excluded = self.data.groups(step)
        fit,val = groups
        radius = float(c.norm())
        seed = SEED+step
        n = len(old['z'])
        z = np.vstack((old['z'],np.zeros((1,2))))
        q = np.vstack((old['q'],np.zeros((1,3))))
        old_rows = group_rows(old['frames'])
        true_rows = group_rows(state['frames'][-1:].detach().cpu())
        print(f'Birth {step}: fitting old, real and {RANDOM_BIRTHS} random charts',flush=True)
        with probe_mode(op.model):
            baseline = [logits(op,x) for x,_ in groups]
            bank = Bank(op,old_rows+true_rows,fit[0])
            charts = [nonlinear(bank,z,q,state['scale'],list(range(n))),
                      nonlinear(bank,z,q,state['scale'],list(range(n+1)))]
            _,pilot = optimize(op,fit,charts[0],radius,baseline[0],BIRTH_BUDGET)
            result = {}
            val_losses = {}
            def evaluate(name,chart):
                warm = np.pad(pilot['coordinates'],(0,chart.tangent.shape[1]-2*n))
                coef,fit_stats = optimize(op,fit,chart,radius,baseline[0],BIRTH_BUDGET,warm=warm)
                pred = logits(op,val[0],chart.bank.vector(coef))
                val_losses[name] = loss(pred,val[1])
                result[name] = dict(fit=fit_stats,validation=metrics(baseline[1],pred,val[1],seed))
                print(f'Birth {step} {name} fit gain={fit_stats["initial"]-fit_stats["best"]:.6f}',flush=True)
            evaluate('old',charts[0])
            evaluate('real',charts[1])
            for i in range(RANDOM_BIRTHS):
                pair = random_axes(charts[0].tangent,bank,seed+i)
                # A zero quadratic term matches the new chart at its birth coordinates.
                alternate = Bank(op,old_rows+pair+[pair[0]],fit[0])
                evaluate(f'random_{i+1}',nonlinear(alternate,z,q,state['scale'],list(range(n+1))))
                del alternate,pair
        contrast = {name:interval(values-val_losses['real'],seed) for name,values in val_losses.items() if name!='real'}
        existing.write(self.output/'births'/f'{step:03d}.json',dict(step=step,old_groups=n,radius=radius,
                       ids=ids,excluded_training_ids=excluded,birth=details['birth'],shared_old_fit=pilot,
                       arms=result,real_gain_over=contrast))
        print(f'Birth {step} saved',flush=True)
        if len(list((self.output/'births').glob('*.json')))==4:
            raise CompletedBirthAudit()

    def finish(self,op,state,records,provenance,resume):
        return None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=('endpoint','births'),required=True)
    for name in ('output','scratch','reference'):
        p.add_argument('--'+name,type=Path,required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    args.scratch.mkdir(parents=True,exist_ok=True)
    source = read(args.reference/'complete.json')
    provenance = source['provenance']
    if provenance['task']!='DTD' or provenance['arm']!='adaptive_dynamics' or provenance['steps']!=216:
        raise ValueError('Expected the reviewed DTD full expert')
    trainer.TASK,trainer.STEPS = 'DTD',216
    existing.ROOT = ROOT
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    with (args.output/'audit.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.stage=='endpoint':
            from audit_dtd_group_endpoint import run
            run(args.output,args.reference,source)
            return
        (args.output/'births').mkdir(exist_ok=True)
        try:
            trainer.train('adaptive_dynamics',args.output,provenance['max_components'],provenance['warmup'],
                          provenance['birth_penalty'],observer=Births(args.output),scratch=args.scratch)
        except CompletedBirthAudit:
            pass
        births = [read(p) for p in sorted((args.output/'births').glob('*.json'))]
        if len(births)!=4:
            raise RuntimeError(f'Expected four observed births, found {len(births)}')
        existing.write(args.output/'births.json',dict(provenance=provenance,births=births))
        (args.scratch/'resume.pt').unlink(missing_ok=True)


if __name__=='__main__':
    main()
