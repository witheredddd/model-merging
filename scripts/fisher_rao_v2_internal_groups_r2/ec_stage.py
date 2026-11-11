"""Deferred group-level EC. Never called before eight independently validated tasks."""
import numpy as np
import torch

TASKS={'DTD','SUN397','Cars','EuroSAT','GTSRB','MNIST','RESISC45','SVHN'}


def validate_tasks(tasks):
    if set(tasks)!=TASKS:raise ValueError('EC requires all eight independently learned tasks')
    if not all(t['internal_structure_passed'] for t in tasks.values()):raise ValueError('Single-task necessity/replication gates not passed')
    if len({t['base_sha256'] for t in tasks.values()})!=1:raise ValueError('Incompatible common base')
    if len({t['active_hash'] for t in tasks.values()})!=1:raise ValueError('Incompatible active parameter coordinates')


def atoms_from_tasks(tasks):
    validate_tasks(tasks);atoms=[];names=[]
    for name,task in tasks.items():
        # Warm offsets are explicit task-level genes and are included once.
        atoms.append(task['warm_offset']);names.append((name,'warm_offset'))
        for group,delta in enumerate(task['group_parameter_updates']):
            atoms.append(delta);names.append((name,group))
    return atoms,names


def combine_parameters(theta0,atoms,weights):
    assert len(atoms)==len(weights)
    theta=theta0.clone()
    for delta,weight in zip(atoms,weights):theta.add_(delta.to(theta.device),alpha=float(weight))
    return theta


def pareto_indices(values):
    values=np.asarray(values)
    return [i for i,x in enumerate(values) if not any(np.all(y<=x) and np.any(y<x) for j,y in enumerate(values) if i!=j)]


def evolutionary_front(tasks,theta0,evaluate_development,population=24,generations=30,seed=119):
    atoms,names=atoms_from_tasks(tasks);rng=np.random.default_rng(seed);dim=len(atoms)
    genes=rng.uniform(-.1,1.,size=(population,dim));genes[0]=0
    archive=[];losses=[]
    for generation in range(generations):
        values=np.array([evaluate_development(combine_parameters(theta0,atoms,w)) for w in genes])
        archive.extend(genes.copy());losses.extend(values.copy())
        selected=pareto_indices(values)
        parents=genes[selected]
        if len(parents)<2:parents=genes[np.argsort(values.mean(1))[:2]]
        children=[]
        for _ in range(population):
            a,b=parents[rng.integers(len(parents),size=2)];mix=rng.random(dim)
            child=mix*a+(1-mix)*b+rng.normal(0,.08,size=dim)
            children.append(np.clip(child,-1,2))
        genes=np.array(children);genes[:min(len(parents),population)]=parents[:population]
    indices=pareto_indices(losses)
    return dict(gene_names=names,weights=np.array(archive)[indices],development_losses=np.array(losses)[indices],
                objectives='task-specific development CE',shared_specific_conflict='evaluate gradients and actual merged task losses',test_used=False)


def joint_responses_after_validation(tasks,runtimes,theta0,atoms):
    validate_tasks(tasks)
    # All task/group atoms are now observed at the SAME base with compatible heads.
    return torch.stack([torch.cat([rt.jvp(theta0,delta.to(theta0.device))/len(runtimes)**.5 for rt in runtimes]) for delta in atoms],1)
