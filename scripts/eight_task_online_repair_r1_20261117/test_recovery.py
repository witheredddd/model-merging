import json,os,subprocess,sys,tempfile,unittest
from pathlib import Path
import numpy as np
from scipy.sparse.linalg import LinearOperator
from bootstrap import CORE
from checkpoint import Store
from scheduler import build_catalog,take,finish
from iterative import continuing_cg


def fixture(folder,total,kill_after_commit=False):
    store=Store(folder)
    if (Path(folder)/'CURRENT.json').exists():state=store.load(restore_random_state=True)
    else:
        state=dict(generation=0,phase='RUNNING',scheduler=build_catalog(['T','P'],{'T':[0,1],'P':[0]},[.1]),completed=0)
        store.save(state)
    rng=np.random.default_rng(19);Q=rng.normal(size=(17,17));M=Q.T@Q+.3*np.eye(17)
    A=LinearOperator(M.shape,matvec=lambda x:M@x,dtype=np.float64);b=rng.normal(size=17)
    while state['completed']<total:
        dispatch=take(state['scheduler']);case_id=dispatch['case_id'];case=state['scheduler']['cases'][case_id]
        dispatch.setdefault('start_iteration',case['iterations']);store.save(state)
        cg=store.load_cg(case)
        progressed=(int(cg['iterations'])-dispatch['start_iteration']) if cg is not None else 0
        if progressed<1:
            def callback(x,current):
                case['iterations']=int(current['iterations']);store.save(state,current,case_id)
                if kill_after_commit:os._exit(97)
            _,_,cg=continuing_cg(A,b,rtol=1e-14,max_iterations=1,state=cg,callback=callback)
        state['completed']+=1;finish(state['scheduler'],resolved=False);store.save(state,cg,case_id)


class TestProcessRecovery(unittest.TestCase):
    def call(self,folder,total,kill=False):
        args=[sys.executable,__file__,'fixture',str(folder),str(total)]
        if kill:args.append('kill')
        return subprocess.run(args,capture_output=True,text=True)

    def test_fresh_process_after_abrupt_exit_matches_continuous_run(self):
        with tempfile.TemporaryDirectory() as temp:
            left=Path(temp)/'continuous';right=Path(temp)/'restarted'
            self.assertEqual(self.call(left,11).returncode,0)
            killed=self.call(right,11,True);self.assertEqual(killed.returncode,97,killed.stderr)
            resumed=self.call(right,11);self.assertEqual(resumed.returncode,0,resumed.stderr)
            a=Store(left).load();b=Store(right).load()
            self.assertEqual(a['completed'],b['completed'])
            for key in ('cursor','inflight','dispatches','strata'):
                self.assertEqual(a['scheduler'][key],b['scheduler'][key])
            for case_id,case in a['scheduler']['cases'].items():
                other=b['scheduler']['cases'][case_id]
                self.assertEqual(case['iterations'],other['iterations'])
                cg=Store(left).load_cg(case);other_cg=Store(right).load_cg(other)
                if cg is not None:
                    for key in cg:np.testing.assert_array_equal(cg[key],other_cg[key])

    def test_unresolved_ce_never_blocks_expert_and_class_dispatch(self):
        state=build_catalog(['T','P'],{'T':[0,1,2],'P':[0,1]},[.1,1.])
        visited=[]
        for _ in range(12):
            dispatch=take(state);case=state['cases'][dispatch['case_id']]
            visited.append((case['spec']['task'],case['spec']['source']))
            finish(state,resolved=False)
        self.assertEqual(set(visited),{(t,s) for t in ('T','P') for s in ('CE','expert_motion_JVP','class_CE')})


if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='fixture':fixture(sys.argv[2],int(sys.argv[3]),len(sys.argv)>4)
    else:unittest.main()
