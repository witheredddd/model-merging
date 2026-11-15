import numpy as np
from fiber_core import Geometry,IncrementalController,ConstrainedSolver,Scenario,Mode

def test_frozen_budget_reference_survives_geometry_scale_changes():
    g=Geometry(np.eye(3),2*np.eye(3),np.ones(3),np.ones(3))
    g.budget_references={'Cars':7.,'DTD':11.}
    assert g.source_reference('Cars')==7. and g.source_reference('DTD')==11.

def test_birth_receives_unexplained_residual_not_full_teacher():
    g=Geometry(np.eye(3),np.eye(3),np.zeros(3),-np.ones(3))
    c=IncrementalController({'a':[0],'b':[1],'c':[2]},ConstrainedSolver())
    c.modes=[Mode('a',np.array([1.,0.,0.]),vector=np.array([1.,0.,0.]))]
    candidates=c._driver_candidates(g,np.array([1.,1.,0.]))
    residual=candidates[0][2]
    assert abs(residual[0])<1e-8 and abs(residual[1]-1)<1e-8

def test_actual_history_is_preserved_instead_of_linear_prediction():
    g=Geometry(np.eye(2),np.eye(2),np.zeros(2),-np.ones(2))
    c=IncrementalController({'a':[0],'b':[1]},ConstrainedSolver())
    finite=np.array([1.,2.,3.,4.]);v=np.array([.01,0.])
    tr=c.accept_trajectory(g,v,Scenario('Cars',1.,1.),-.1,0.,actual_response=finite)
    assert np.array_equal(c.actual_responses[0],finite)
    assert tr['actual_finite_response_norm']==np.linalg.norm(finite)
