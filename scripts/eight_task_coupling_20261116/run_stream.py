"""Eight joint one-versus-seven problems and a growing full-motion controller.

Iteration chunks and the wall clock limit computation, never structural rank.
Unconverged candidates are retained separately and never called optimal teachers.
"""
import argparse,json,sys,time,os,hashlib
from pathlib import Path
import numpy as np
import torch

CORE=Path(__file__).resolve().parent.parent/'fullspace_conditional_20261116'
if not CORE.exists():CORE=Path(__file__).resolve().parent.parent/'fullspace_conditional'
sys.path.insert(0,str(CORE))
from geometry import EuclideanMetric,FullFunctionalGeometry
from conditional import ConditionalMotion
from precondition import output_diagonal
from task_runtime import load_eight_geometry
from controls import FunctionalControls
from prepare import write,sha
from run import spectral_scale


class NumericalTimeBudget(Exception):pass


def plain(x):
    if isinstance(x,dict):return {k:plain(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [plain(v) for v in x]
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    return x


def save_array(path,**arrays):
    temp=Path(str(path)+'.tmp')
    with temp.open('wb') as f:np.savez(f,**arrays)
    temp.replace(path)


def demand_stream(g):
    # Scheduling small output systems first changes neither the eight tasks nor
    # their simultaneous seven-task protection objective.
    order=sorted(g.tasks,key=lambda t:(g.output_dimensions[t],g.tasks.index(t)))
    for task in order:yield dict(task=task,source='CE',name=task+'_CE')
    for task in order:yield dict(task=task,source='expert_motion_JVP',name=task+'_expert_motion')
    classes={t:sorted({int(y) for _,labels in g.c.batches[t,g.probe] for y in labels.cpu()}) for t in order}
    for i in range(max(map(len,classes.values()))):
        for task in order:
            if i<len(classes[task]):
                label=classes[task][i]
                yield dict(task=task,source='class_CE',label=label,name=f'{task}_class{label}')


def make_demand(g,spec):
    task=spec['task']
    if spec['source']=='CE':
        y,gradient,u,info=g.ce_unit_target(task)
        return y,dict(info,origin='Current supervised CE unit direction; reference not executed')
    if spec['source']=='class_CE':
        gradient=g.class_ce_gradient(task,spec['label']);m=float(gradient.norm())
        if m==0:return None,dict(defined=False,reason='Exactly zero class CE gradient')
        return g.jvp(task,-gradient/m),dict(defined=True,origin='Class CE unit-gradient demand',label=spec['label'])
    expert=g.c.experts[g.tasks.index(task)].to(device=g.device,dtype=g.point.dtype)
    motion=expert-g.point;m=float(motion.norm())
    if m==0:return None,dict(defined=False,reason='Expert equals current visual model')
    return g.jvp(task,motion/m),dict(defined=True,origin='Current-anchor JVP of full expert displacement',
        expert_displacement_norm=m,not_a_parameter_search_dictionary=True,
        not_the_nonlinear_expert_Log=True)


def bank_checkpoint(bank,out):
    write(out/'CONTROL_STATE.json',plain(dict(columns=bank.columns,events=bank.events,
        parameter_gram=bank.parameter_gram,parameter_gram_eigenvalues=np.linalg.eigvalsh(bank.parameter_gram),
        machine_resolved_parameter_rank=int(np.linalg.matrix_rank(bank.parameter_gram)) if bank.columns else 0,
        preset_dimension=None,preset_supports=None,natural_dimension_convergence_claim=False)))


def frozen_validation(bank,out,rtol,gammas,scales):
    if not bank.columns:
        write(out/'HELDOUT.json',dict(performed=False,reason='No numerically accepted full-space teacher yet'))
        return
    g=FullFunctionalGeometry(bank.g.c,bank.g.point,1,precision='float64')
    frozen=bank.reexpress_at(g);rows=[]
    for task in g.tasks:
        y,_,_,info=g.ce_unit_target(task)
        if y is None:
            rows.append(dict(task=task,defined=False,info=info));continue
        weights={t:1. for t in g.tasks if t!=task}
        for gamma in gammas:
            rho=gamma*sum(scales[t] for t in weights)
            result=frozen.validate_unseen(task,y,weights,rho,parameter_metric_gram=frozen.parameter_gram,
                numerical_tolerance=rtol)
            rows.append(dict(task=task,gamma=gamma,result=plain(result)))
    write(out/'HELDOUT.json',dict(performed=True,rows=rows,
        data='Disjoint locked training-side probe1; not an expert-training-unseen or official Test claim',
        demand='Probe1 CE target never used to add controls',structure_updated=False,
        learned_columns=len(bank.columns),new_demand_fullspace_optimality_comparison=False))


def main(args):
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    if (out/'LOCK.json').exists():raise ValueError('Use a new run directory; solver states are preserved per demand')
    validation=json.loads((args.operator_validation/'COMPLETE.json').read_text())
    if not validation['all_checked_directions_meet_numeric_accuracy']:
        raise ValueError('Eight-task operators did not meet requested numerical accuracy')
    started=time.monotonic();deadline=started+args.wall_seconds
    runtime=out/'runtime';runtime.mkdir(exist_ok=True)
    import shutil
    shutil.copy2(args.operator_validation/'runtime/probes.json',runtime/'probes.json')
    g,source=load_eight_geometry(runtime,args.inventory,args.checkpoint)
    hashes={str(p):sha(p) for folder in (Path(__file__).parent,CORE) for p in folder.glob('*.py')}
    write(out/'LOCK.json',dict(source=source,gammas=args.gammas,solver_rtol=args.rtol,
        solver_chunk_iterations=args.chunk_iterations,wall_seconds=args.wall_seconds,
        power_iterations=args.power_iterations,precondition_seconds=args.precondition_seconds,
        parameter_metric='I, explicit baseline',protection='Joint equal-weight sum of seven mean Fishers',
        normalization='rho = gamma * sum of seven scalar power estimates; not a certified operator norm',
        task_count=8,structural_rank_limit=None,top_k=None,energy_cutoff=None,
        mode_birth_gate=None,EC=False,model_writeback=False,code_hashes=hashes,
        learning='Every numerically accepted nonzero full conditional motion retained; no energy truncation',
        learned_controller_is_used_before_next_teacher=True))
    scales={};histories={}
    for task in g.tasks:
        write(out/'status.json',dict(stage='eight_task_scalar_scales',task=task))
        scales[task],histories[task]=spectral_scale(g,task,args.power_iterations)
    write(out/'SCALES.json',dict(scales=scales,power_history=histories))
    diagonals,daudit=output_diagonal(g,g.tasks,args.precondition_seconds,1)
    save_array(out/'PRECONDITIONER.npz',**diagonals);write(out/'PRECONDITIONER.json',daudit)
    bank=FunctionalControls(g,out/'controls');rows=[];pending=[];stopped=False
    iterator=iter(demand_stream(g))
    # The first CE pass covers all eight one-versus-seven objectives before any
    # unfinished solve receives additional iteration chunks.
    first_specs=[next(iterator) for _ in g.tasks]
    for gamma in args.gammas:
        for spec in first_specs:pending.append(dict(spec=spec,gamma=gamma,iterations=0))
    def process(case):
        spec=case['spec'];task=spec['task'];gamma=case['gamma']
        label=f"{spec['name']}_gamma{gamma:g}";folder=out/'demands'/label;folder.mkdir(parents=True,exist_ok=True)
        response_file=folder/'response.npy'
        if response_file.exists():y=g.numpy_to_tensor(np.load(response_file))
        else:
            y,info=make_demand(g,spec)
            write(folder/'DEMAND.json',dict(spec=spec,info=info,anchor=source['checkpoint_sha256'],
                output_coordinates=g.output_coordinate_ids[task],reachable_by_construction=y is not None))
            if y is None:return True
            np.save(response_file,g.tensor_to_numpy(y))
        if float(y.norm())==0:
            write(folder/'ZERO_RESPONSE.json',dict(target_functional_response_is_zero=True,
                  no_nonzero_control_manufactured=True))
            return True
        weights={t:1. for t in g.tasks if t!=task}
        rho=gamma*sum(scales[t] for t in weights)
        if rho<=0:raise ValueError('Scalar scale estimate vanished; it does not certify seven zero Fishers')
        before=bank.solve_demand(task,y,weights,rho,parameter_metric_gram=bank.parameter_gram,
            numerical_tolerance=args.rtol)
        write(folder/f'CONTROLLER_BEFORE_{case["iterations"]:06d}.json',plain(before))
        pd=np.concatenate([1+diagonals[t]/rho for t in weights])
        td=diagonals[task]/rho;td=np.where(td>0,td,1.)
        problem=ConditionalMotion(g,task,weights,EuclideanMetric(),rho)
        state_file=folder/'OUTPUT_ITERATE.npz';state=None
        if state_file.exists():
            with np.load(state_file) as saved:state={k:saved[k] for k in saved.files}
            case['iterations']=int(state['iterations'])
        count=[case['iterations']]
        def callback(x):
            count[0]+=1
            save_array(state_file,**problem.schur_krylov_state)
            write(out/'status.json',dict(stage='joint_seven_protection_solve',target=task,gamma=gamma,
                iterations=count[0],teacher_controls=len(bank.columns),remaining_cases=len(pending),
                elapsed_seconds=time.monotonic()-started))
            if time.monotonic()>=deadline:raise NumericalTimeBudget()
        inner_rtol=case.get('inner_rtol',args.rtol)
        v,audit=problem.solve_exact_response_schur_euclidean(y,relative_tolerance=inner_rtol,
            max_iterations=args.chunk_iterations,protected_diagonal_scale=pd,target_diagonal_scale=td,
            callback=callback,krylov_state=state,checkpointable=True)
        case['iterations']=count[0]
        save_array(state_file,**problem.schur_krylov_state)
        accepted=bool(audit['response_relative_residual'] is not None and
            audit['response_relative_residual']<=args.rtol and
            audit['stationarity_relative_residual'] is not None and
            audit['stationarity_relative_residual']<=args.rtol)
        write(folder/'LATEST_NUMERICAL_AUDIT.json',dict(audit,iterations=count[0],
            teacher_usable_at_requested_numerical_accuracy=accepted,
            numerical_test_is_not_structure_dimension_selection=True))
        torch.save(dict(full_motion=v.detach().cpu(),accepted_teacher=accepted,source=spec,
                        gamma=gamma,iterations=count[0]),folder/'LATEST_MOTION.pt')
        if not accepted:
            if audit['solver_info']==0:
                worst=max(audit['response_relative_residual'],audit['stationarity_relative_residual'])
                case['inner_rtol']=inner_rtol*args.rtol/worst
                if case['inner_rtol']<np.finfo(float).eps:
                    raise ArithmeticError('Original KKT accuracy not attained before working precision; no structural conclusion')
            elif audit['solver_info']<0:
                raise ArithmeticError('CG nonpositive numerical curvature; retain state, no teacher admitted')
            return False
        # This full P-dimensional motion is learned only after both original KKT
        # residuals satisfy the explicitly requested numerical solve accuracy.
        bank.add_full_motion(v,dict(demand=spec,gamma=gamma,numerical_audit=audit))
        full_norm=bank.columns[-1]['full_motion_norm']
        functional_energies={t:float(np.linalg.norm(bank.responses[t][-1])**2*full_norm**2) for t in g.tasks}
        after=bank.solve_demand(task,y,weights,rho,parameter_metric_gram=bank.parameter_gram,
            numerical_tolerance=args.rtol)
        write(folder/'CONTROLLER_AFTER.json',plain(after));bank_checkpoint(bank,out)
        rows.append(dict(demand=spec,gamma=gamma,iterations=count[0],numerical_audit=audit,
            controls_before=plain(before),controls_after=plain(after),task_response_energies=functional_energies))
        write(out/'LEARNING.json',plain(dict(rows=rows,raw_controls=len(bank.columns),
            intrinsic_low_dimension_proven=False,discrete_parameter_blocks_discovered=False)))
        return True
    try:
        while time.monotonic()<deadline:
            if not pending:
                try:spec=next(iterator)
                except StopIteration:break
                pending=[dict(spec=spec,gamma=gamma,iterations=0) for gamma in args.gammas]
            case=pending.pop(0)
            if not process(case):pending.append(case)
            # Do not introduce later demand families until the initial eight
            # problems have either converged or exhausted the actual time budget.
            write(out/'PENDING.json',plain(dict(cases=pending)))
    except NumericalTimeBudget:
        stopped=True
        pending.insert(0,case)
    bank_checkpoint(bank,out)
    write(out/'PENDING.json',plain(dict(cases=pending)))
    write(out/'status.json',dict(stage='frozen_unseen_demand_validation',learned_columns=len(bank.columns)))
    frozen_validation(bank,out,args.rtol,args.gammas,scales)
    write(out/'COMPLETE.json',dict(execution_ended=True,accepted_teachers=len(rows),
        learned_columns=len(bank.columns),unfinished_cases=len(pending),elapsed_seconds=time.monotonic()-started,
        wall_budget_exhausted=stopped or time.monotonic()>=deadline,
        natural_dimension_convergence_claim=False,eight_task_low_dimension_claim=False,
        operator_counts=g.operator.audit()))
    write(out/'status.json',dict(stage='ended',learned_columns=len(bank.columns),unfinished_cases=len(pending)))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--inventory',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--operator-validation',type=Path,required=True)
    p.add_argument('--gammas',type=lambda s:[float(x) for x in s.split(',')],required=True)
    p.add_argument('--rtol',type=float,required=True)
    p.add_argument('--chunk-iterations',type=int,required=True)
    p.add_argument('--power-iterations',type=int,required=True)
    p.add_argument('--precondition-seconds',type=float,required=True)
    p.add_argument('--wall-seconds',type=float,required=True)
    args=p.parse_args()
    try:main(args)
    except Exception:
        import traceback
        write(args.output/'FAILURE.json',dict(traceback=traceback.format_exc(),scientific_failure=False))
        raise
