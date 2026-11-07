"""Validate the explicit EuroSAT mirror splits, without using test labels."""
import collections,concurrent.futures,hashlib,json
from pathlib import Path
import pyarrow.parquet as pq
from engine import ROOT,OUT,write

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()

if __name__=='__main__':
    base=ROOT/'datasets/EuroSAT_splits';splits={};hashsets={}
    for name in ['train','val','test']:
        paths=sorted((base/name).glob('*/*.jpg'))
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:hs=list(pool.map(digest,paths))
        hashsets[name]=set(hs);splits[name]=dict(images=len(paths),class_counts=dict(collections.Counter(p.parent.name for p in paths)),
            path_list_sha256=hashlib.sha256('\n'.join(str(p.relative_to(base)) for p in paths).encode()).hexdigest(),
            content_list_sha256=hashlib.sha256('\n'.join(hs).encode()).hexdigest())
    overlaps={f'{a}:{b}':len(hashsets[a]&hashsets[b]) for a,b in [('train','val'),('train','test'),('val','test')]}
    files=list((ROOT/'datasets/_downloads/blanchon__EuroSAT_RGB/data').glob('*.parquet'))
    parquet={p.name:dict(rows=pq.ParquetFile(p).metadata.num_rows,sha256=digest(p)) for p in files}
    report=dict(dataset='EuroSAT',source='blanchon/EuroSAT_RGB',splits=splits,cross_split_identical_image_bytes=overlaps,source_parquet=parquet,
        correction='Use explicit mirror train/validation/test splits. The legacy helper writes a generic seed-0 provenance sentence even when explicit validation already exists; that sentence does not describe this run.')
    write(OUT/'assets/EuroSAT_split_audit.json',report);print(json.dumps(report,indent=2))
    assert not any(overlaps.values()),'Identical image bytes cross EuroSAT splits'
