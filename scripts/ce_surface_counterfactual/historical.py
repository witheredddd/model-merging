"""Metadata-only historical attribution; does not reconstruct missing states."""
import argparse,json,collections,math
from pathlib import Path
import torch,numpy as np

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();saved=torch.load(a.checkpoint,map_location='meta',weights_only=False)
    rows=[dict(step=r['step'],**r['surface']) for r in saved['records'] if 36<r['step']<=216 and 'surface' in r]
    accepted=[r for r in rows if r['coordinate_norm']>0];rejected=[r for r in rows if r['coordinate_norm']==0]
    reason=collections.Counter('fit_no_gain_or_zero_iterations' if r['fit_ce_gain']<=0 or r['coordinate_iterations']==0
        else 'positive_fit_but_unwritten_gate_or_fp32_unseparated' for r in rejected)
    last=rows[-1];tilts=[]
    for z,q in zip(last['z'],last['q']):
        b=[2*q[0]*z[0]+q[1]*z[1],q[1]*z[0]+2*q[2]*z[1]]
        tilts.append(dict(a3_coefficients=b,within_group_orthonormal_axes_assumed=True,
            conditional_tangent_gram_condition=1.+sum(v*v for v in b),
            conditional_tangent_cosine=b[0]*b[1]/math.sqrt((1+b[0]**2)*(1+b[1]**2))))
    result=dict(archive_step=len(saved['records']),provenance=saved['provenance'],historical_step_range=[37,216],
        recorded_steps=len(rows),accepted=len(accepted),rejected=len(rejected),recorded_rejection_categories=dict(reason),
        mean_coordinate_CE_gain=sum(r['coordinate_ce_gain'] for r in rows)/len(rows),
        mean_adamw_reference_CE_gain=sum(r['reference_ce_gain'] for r in rows)/len(rows),
        multiple_coordinate_rounds=sum(r['coordinate_iterations']>1 for r in rows),
        milestone216_tangent_coefficients=tilts,
        limitations=['No original step216 weights or axes are reconstructed','Historical cross-batch vs FP32 gate flips cannot be separated from these records','Step432 counterfactuals do not diagnose particular historical rejections'],
        records=rows)
    a.output.write_text(json.dumps(result,indent=2,default=lambda v:v.item() if isinstance(v,np.generic) else str(v)))
    print(json.dumps({k:v for k,v in result.items() if k!='records'},default=lambda v:v.item() if isinstance(v,np.generic) else str(v)))

if __name__=='__main__':main()
