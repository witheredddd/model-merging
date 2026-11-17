import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
import numpy as np
import torch
from bootstrap import R3
from test_research import Geometry
from io_state import Transactions
from space_worker import ObservedOutputWLS
from runner import Runner
from diagnostics import compare,error_budget,condition_grid


def small_runner(folder):
    r=Runner.__new__(Runner);r.out=Path(folder);r.args=SimpleNamespace(quantum=4,relative_error_budget=1e-6,gamma=.1)
    r.store=Transactions(r.out);r.g=Geometry();r.g.probe=0;r.stop=False
    r.state=dict(position=0,bank=[],observations=[],solvers={},numerical_floor_events=[],run_state='RUNNING')
    r.positions=SimpleNamespace(rows=[dict(step=0)])
    r.scales=dict.fromkeys(r.g.tasks,1.)
    r.diagonals=None
    return r


class DiagnosticTests(unittest.TestCase):
    def test_quantum_repeats_until_fresh_error_budget_and_resumes_without_restart(self):
        with tempfile.TemporaryDirectory() as f:
            r=small_runner(f);weights=dict.fromkeys(r.g.tasks,1.);d={t:np.ones(3) for t in weights}
            session=ObservedOutputWLS(r.g,weights,.7,d)
            report=None
            for i in range(12):
                report=r.refine(session,'test',d)
                if i==1:
                    self.assertIsNone(report);self.assertEqual(r.store.cg(r.state,'test')['iterations'],4)
                    # Restore exactly the committed solver state, not its direction restarted.
                    r.state=r.store.load();session=ObservedOutputWLS(r.g,weights,.7,d)
                if report:break
            self.assertIsNotNone(report);self.assertGreater(report['iterations'],4)
            H=.7*np.eye(13)+sum(r.g.J[t].T@r.g.J[t] for t in weights)
            exact=np.linalg.solve(H,sum(r.g.J[t].T@d[t] for t in weights))
            np.testing.assert_allclose(r.load_motion(report),exact,atol=1e-6)
            self.assertTrue(report['budget']['ready'])

    def test_three_errors_are_separate_and_finite_reference_gap_interval_is_valid(self):
        Y={'T':np.eye(2),'P':np.zeros((2,2))};G=np.eye(2);y=np.array([1.,2.]);rho=100.
        optimum=y/101.;full={'T':optimum,'P':np.zeros(2)}
        cert=dict(objective_gap_upper_bound=0.,parameter_error_bound=0.,joint_energy_error_bound=0.)
        report=compare(Y,G,'T',y,{'P':1.},rho,full,float(optimum@optimum),cert,gain=.01)
        self.assertGreater(report['e_full'],.99)
        self.assertLess(report['e_repr'],1e-12)
        self.assertAlmostEqual(report['e_demand'],report['e_full'])
        self.assertGreater(report['e_applied'],report['e_demand'])
        self.assertAlmostEqual(report['objective_excess_vs_full_optimum_interval'][1],0.)

    def test_full_parameter_residuals_grow_shared_space_without_tensor_or_count_cap(self):
        with tempfile.TemporaryDirectory() as f:
            r=small_runner(f);rng=np.random.default_rng(6)
            # Deliberately exceed the old initial one-column capacity; all tasks
            # access the same bank and all vector coordinates may participate.
            for i in range(6):
                condition=dict(id=f'c{i}',target=str(i%8),family='normal_response')
                vector=torch.tensor(rng.normal(size=13),dtype=torch.float64)
                file=r.out/f'v{i}.pt';torch.save(dict(motion=vector),file)
                ref=dict(motion_file=file.name,certificate=dict(parameter_error_bound=1e-12))
                r.add_reference(ref,condition)
            self.assertEqual(len(r.state['bank']),6)
            vectors,_,G=r.bank_data();np.testing.assert_allclose(G,np.eye(6),atol=1e-12)
            self.assertTrue(all(e['kind']=='full_conditional_residual' for e in r.state['bank']))
            self.assertTrue(all('driver_start' not in e for e in r.state['bank']))
            self.assertTrue(all(torch.count_nonzero(v)==13 for v in vectors))

    def test_full_family_is_processed_before_position_advances(self):
        r=Runner.__new__(Runner);r.state=dict(stage='STUDY_SELECT',position=0,condition_index=16,conditions=condition_grid([str(i) for i in range(8)]))
        r.save=lambda *a,**k:None
        r.summarize_joint_coordinates=lambda:None
        r.step()
        self.assertEqual(r.state['position'],0);self.assertEqual(r.state['stage'],'VALIDATE_POSITION')
        self.assertEqual(len(r.state['conditions']),16)

    def test_budget_does_not_admit_four_steps_without_fresh_bound(self):
        c=dict(parameter_error_bound=.2,joint_energy_error_bound=.1)
        self.assertFalse(error_budget(c,1.,1.,.001)['ready'])

    def test_demand_catalog_expands_with_functional_classes_not_a_fixed_rank(self):
        small=condition_grid(['T','P'],dict(T=[0,1],P=[0]))
        large=condition_grid(['T','P'],dict(T=list(range(20)),P=[0]))
        self.assertGreater(len(large),len(small))
        self.assertTrue(any(c.get('class_label')==19 for c in large))

    def test_relocalization_rebuilds_all_residual_recipes_not_only_the_last(self):
        with tempfile.TemporaryDirectory() as f:
            r=small_runner(f);r.args.relative_error_budget=1e-7
            demand_dir=r.out/'demands';demand_dir.mkdir()
            conditions=[];demands=[]
            for i in range(2):
                c=dict(id=f'c{i}',target=str(i),family='normal_response');y=np.array([1.+i,.3,-.2])
                weights,rho=r.settings(c['target'])
                H=rho*np.eye(13)+sum(r.g.J[t].T@r.g.J[t] for t in r.g.tasks)
                v=torch.tensor(np.linalg.solve(H,r.g.J[c['target']].T@y))
                file=r.out/f'source{i}.pt';torch.save(dict(motion=v),file)
                np.savez(demand_dir/f'p0_{c["id"]}.npz',y=y)
                r.add_reference(dict(motion_file=file.name,certificate=dict(parameter_error_bound=1e-12)),c)
                conditions.append(c);demands.append(y)
            self.assertEqual(len(r.state['bank']),2)
            # A genuinely different new Jacobian, with the source demands frozen.
            for t in r.g.tasks:r.g.J[t]=r.g.J[t]+.07*np.eye(3,13)
            r.state.update(transport_bank=[],transport_index=0)
            for _ in range(40):
                if r.transport():break
            self.assertEqual(r.state['transport_index'],2)
            rebuilt=r.state['transport_bank'];self.assertEqual(len(rebuilt),2)
            expected={}
            for entry,y in zip(r.state['bank'],demands):
                weights,rho=r.settings(entry['condition']['target'])
                H=rho*np.eye(13)+sum(r.g.J[t].T@r.g.J[t] for t in r.g.tasks)
                v=np.linalg.solve(H,r.g.J[entry['condition']['target']].T@y)
                for pid,z in zip(entry['parent_ids'],entry['projection_coefficients']):v-=z*expected[pid]
                expected[entry['id']]=v/entry['residual_norm']
            for entry in rebuilt:np.testing.assert_allclose(r.load_motion(entry),expected[entry['id']],atol=1e-6)

    def test_heldout_response_cache_cannot_reuse_learning_response(self):
        with tempfile.TemporaryDirectory() as f:
            r=small_runner(f);file=r.out/'basis.pt';torch.save(dict(motion=torch.ones(13,dtype=torch.float64)),file)
            r.state['bank']=[dict(id='shared',motion_file=file.name)]
            original=r.g.jvp
            r.g.jvp=lambda task,v:original(task,v)*(1+r.g.probe)
            _,learn,_=r.bank_data();r.g.probe=1;_,heldout,_=r.bank_data()
            for t in r.g.tasks:np.testing.assert_allclose(heldout[t],2*learn[t])

    def test_validation_reference_never_expands_the_learned_space(self):
        with tempfile.TemporaryDirectory() as f:
            r=small_runner(f);(r.out/'demands').mkdir();r.state['validation_index']=0
            r.demand=lambda *a,**k:np.ones(3)
            r.full_reference=lambda *a,**k:dict(valid=True)
            observed=[];r.evaluate=lambda *a,**k:observed.append(r.g.probe)
            r.add_reference=lambda *a,**k:(_ for _ in ()).throw(AssertionError('Held-out reference must not enter bank'))
            r.validate_position()
            self.assertEqual(observed,[1]);self.assertEqual(r.state['bank'],[])
            self.assertEqual(r.g.probe,0);self.assertEqual(r.state['validation_index'],1)

    def test_exact_expert_endpoint_has_zero_demand_without_dividing_by_zero(self):
        with tempfile.TemporaryDirectory() as f:
            r=small_runner(f);r.g.c=SimpleNamespace(experts=[r.g.point.clone() for _ in r.g.tasks])
            y=r.demand(dict(id='expert_endpoint',target='0',family='expert_response'))
            np.testing.assert_array_equal(y,np.zeros(3));self.assertTrue(np.isfinite(y).all())


if __name__=='__main__':unittest.main()
