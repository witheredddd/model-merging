"""Read existing region observations and build explicit restricted graphs."""
import argparse,json,time
from pathlib import Path
import numpy as np
from bootstrap import R3
from graphs import mechanism_report,exact_control
from io_state import Transactions,atomic_json,digest

def observe(source,output):
    state=Transactions(source).load();records=[]
    for key,ps in state.get('adaptive',{}).get('position_states',{}).items():
        item=ps.get('coupling') or ps.get('completed_coupling')
        if not item:continue
        destination=output/f'position{key}.json'
        if destination.exists():
            old=json.loads(destination.read_text())
            if old.get('source_observation_index')==item['region_index']:
                records.append(dict(position=key,regions=old['observed_region_count'],graphs=len(old['graphs'])));continue
        folder=source/item['output_folder'];files=sorted(f for f in folder.glob('[0-9]*.npz') if int(f.stem)<item['region_index'])
        if len(files)<2:continue
        rows=[];norms=[];ids=[]
        for file in files:
            with np.load(file) as z:
                n=float(z['parameter_norm_squared'])
                if n==0:continue
                rows.append({t:z[t].copy() for t in z.files if t!='parameter_norm_squared'});norms.append(n);ids.append(int(file.stem))
        if len(rows)<2:continue
        target=item['condition']['target'];Y={t:np.column_stack([r[t] for r in rows]) for t in rows[0]}
        weights={t:1. for t in Y if t!=target}
        # Original source config paths are frozen in the recorded dependency map.
        scale_file=next(Path(p) for p in state['source_hashes'] if p.endswith('/SCALES.json'))
        assert digest(scale_file)==state['source_hashes'][str(scale_file)]
        scales=json.loads(scale_file.read_text())['scales'];rho=state['gamma']*sum(scales[t] for t in weights)
        G=np.diag(norms)
        # Compare at an identical, guaranteed reachable response in THIS observed
        # span. This is not a claim that the original requested y is reachable.
        y=Y[target]@np.ones(len(rows))
        reports=[mechanism_report(Y,G,target,y,weights,rho,[i],include_ablations=i==0) for i in range(len(rows))]
        for i,report in enumerate(reports):
            report.update(graph_id=f'position{key}_{target}_driver_region{ids[i]}',revision=item['region_index'])
        family=np.column_stack([r['graph']['joint_modes'] for r in reports])
        record=dict(position=int(key),target=target,region_ids=ids,observed_region_count=len(rows),
            source_observation_index=item['region_index'],
            joint_graph_family_fixed_target=exact_control(Y,G,target,y,weights,rho,family),
            source_control_file=item['motion_file'],source_control_sha256=digest(source/item['motion_file']),
            parameter_mode_recipe='Partition source control by region_ids; multiply region vectors by graph joint_modes coefficients',
            complete_parameter_partition=False,graphs=reports,
            scope='One observed direction per available region; single-coordinate probes, not discovered groups',
            arbitrary_multicoordinate_graph_API_available=True,
            source_files={str(f):digest(f) for f in files},
            group_discovery_complete=False,original_request_fulfilment_not_inferred=True)
        atomic_json(output/f'position{key}_observation{item["region_index"]}.json',record)
        atomic_json(output/f'position{key}.json',record);records.append(dict(position=key,regions=len(rows),graphs=len(reports)))
    atomic_json(output/'status.json',dict(observations=records,main_learning_state_untouched=True))
    return records

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--watch',action='store_true');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    while True:
        print(json.dumps(observe(a.source,a.output)),flush=True)
        if not a.watch:break
        time.sleep(30)
