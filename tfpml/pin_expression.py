"""Pin an atomically completed checkpoint BEFORE test access or job stop."""
import json,os
from pathlib import Path
import torch

out=Path('/home/shangxueyuan/merging/tsv/results/TFPML_v3_expression_k16_d3')
source=out/'resume.pt';target=out/'validation_target_candidate.pt'
# link() atomically pins either completed inode if resume.pt is being replaced;
# it cannot reference resume.pt.tmp or affect the running trainer's writes.
os.link(source,target)
ck=torch.load(target,map_location='cpu',weights_only=False)
assert ck['validation']['accuracy']>=ck['config']['target_val']
record=dict(policy='validation performance gate; selected before any test access',
    checkpoint=target.name,epoch=ck['epoch'],generator_updates=ck['step'],validation=ck['validation'],
    target_val=ck['config']['target_val'],original_maximum_epoch_budget=ck['config']['epochs'],
    test_accessed=False,not_minimum_validation_CE_selection=True)
(out/'expression_selection_before_test.json').write_text(json.dumps(record,indent=2))
print(json.dumps(record),flush=True)
