"""Original DTD held-out test assets. Never used by optimizer/selection."""
from pathlib import Path
import hashlib,json,requests,yaml
import pyarrow.parquet as pq

root=Path('/home/shangxueyuan/merging/tsv/datasets');host='https://hf-mirror.com';repo='tanganke/dtd'
def get(url):
    r=requests.get(host+url,timeout=(20,120));r.raise_for_status();return r
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
meta=yaml.safe_load(get('/datasets/'+repo+'/raw/main/README.md').text.split('---')[1])['dataset_info']
names=next(f for f in meta['features'] if f['name']=='label')['dtype']['class_label']['names']
if isinstance(names,dict):names=[names[str(i)] for i in range(len(names))]
files=get('/api/datasets/'+repo+'/tree/main?recursive=true&limit=1000').json()
count=0;records=[]
for item in files:
    if not item['path'].endswith('.parquet') or not Path(item['path']).name.startswith('test-'):continue
    path=root/'_downloads/tanganke__dtd'/item['path'];path.parent.mkdir(parents=True,exist_ok=True)
    expected=item['lfs']['oid']
    if not path.exists() or sha(path)!=expected:
        r=requests.get(host+'/datasets/'+repo+'/resolve/main/'+item['path'],stream=True,timeout=(20,120));r.raise_for_status()
        tmp=path.with_suffix('.part')
        with tmp.open('wb') as f:
            for b in r.iter_content(1024*1024):f.write(b)
        assert sha(tmp)==expected;tmp.replace(path)
    for batch in pq.ParquetFile(path).iter_batches(batch_size=128):
        for row in batch.to_pylist():
            dst=root/'dtd/val'/names[int(row['label'])]/f'test_{count:06d}.jpg'
            dst.parent.mkdir(parents=True,exist_ok=True)
            if not dst.exists():dst.write_bytes(row['image']['bytes'])
            count+=1
    records.append(dict(path=item['path'],sha256=expected))
assert count==1880,count
(root/'DTD_test_provenance.json').write_text(json.dumps(dict(n=count,files=records)))
print('TEST_ASSETS_READY',count,flush=True)
