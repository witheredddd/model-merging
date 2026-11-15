import numpy as np
from fiber_core import Geometry,Scenario,ConstrainedSolver,mode_from_driver,IncrementalController,ortho,optimize,functional_novelty


def test_compensation_and_fiber_residual():
    # Task D observations permit an exact nonzero fiber in this synthetic toy.
    JD=np.array([[1.,1.,0.],[0.,0.,1.]])
    JC=np.array([[1.,-1.,0.],[0.,0.,1.]])
    G=Geometry(JD,JC,np.array([.5,.5,0.]),np.array([-1.,1.,0.]))
    d=np.array([1.,0.,0.])
    v,T=mode_from_driver(G,d,[0],[1],"DTD",damping_scale=1e-12)
    assert np.linalg.norm(JD@v)<1e-10
    assert np.linalg.norm(JC@v)>1
    assert T.shape==(1,1)


def test_grouping_and_not_fixed_rank():
    JD=np.array([[1.,1.,0.,0.],[0.,0.,1.,1.]])
    JC=np.array([[1.,-1.,0.,0.],[0.,0.,1.,-1.]])
    G=Geometry(JD,JC,np.array([0.,0.,0.,0.]),np.array([-1.,.2,-1.,.2]))
    learner=IncrementalController({'a':[0],'b':[1],'c':[2],'d':[3]},ConstrainedSolver(),
           capacity_target=.9,max_modes=4,max_compensators=3)
    S=[Scenario('Cars',.05,1.)]
    ev=learner.learn(G,S,max_new=2)
    assert len(ev)>=1
    assert learner.basis(G.dim).shape[1]>=1
    assert learner.modes[0].supports
    assert all(e['effective_rank']<=4 for e in ev)
    x,a=optimize(G,learner.basis(G.dim),S[0],ConstrainedSolver())
    assert a['passed'] and np.linalg.norm(x)>0


def test_functional_novelty():
    rng=np.random.default_rng(3);JD=rng.normal(size=(8,4));JC=rng.normal(size=(9,4))
    G=Geometry(JD,JC,rng.normal(size=4),rng.normal(size=4))
    b=np.array([1.,0.,0.,0.]);v=np.array([2.,0.,0.,0.])
    assert functional_novelty(G,v,b[:,None])<1e-10
    assert functional_novelty(G,np.array([0.,1.,0.,0.]),b[:,None])>.01


def test_dynamic_reconditioning_and_history():
    JD=np.array([[1.,1.,0.,0.],[0.,0.,1.,1.]])
    JC=np.array([[1.,-1.,0.,0.],[0.,0.,1.,-1.]])
    G=Geometry(JD,JC,np.zeros(4),np.array([-1.,.2,-1.,.2]))
    learner=IncrementalController({'a':[0],'b':[1],'c':[2],'d':[3]},ConstrainedSolver(),
           capacity_target=.9,max_modes=4,max_compensators=3)
    s=Scenario('Cars',.05,1.)
    learner.learn(G,[s],max_new=2)
    vec=learner.modes[0].vector.copy()
    # A functional trajectory is used to gate subsequent mode births.
    tr=learner.accept_trajectory(G,vec*.01,s,-.001,0.)
    assert tr['active_controls']>=1
    newer=Geometry(JD*np.array([[1.1,1.0,1.0,1.0],[1.,1.,1.,1.]]),JC,np.zeros(4),np.array([-1.,.2,-1.,.2]))
    learner.recondition(newer)
    assert learner.events[-1]['event']=='recondition'
    assert learner.trajectories[0]['trajectory_novelty']>0
