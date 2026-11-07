"""Find decoder failures without changing original dataset bytes or sample splits."""
import argparse,concurrent.futures,hashlib,json,time
from pathlib import Path
from PIL import Image,ImageFile
from engine import OUT,ROOT,write

def check(path):
    try:
        with Image.open(path) as image:image.load()
        return None
    except Exception as e:return dict(path=str(path),error=str(e),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),bytes=path.stat().st_size)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--all',action='store_true');args=ap.parse_args()
    base=ROOT/'datasets' if args.all else ROOT/'datasets/EuroSAT_splits'
    paths=[p for p in base.rglob('*') if p.suffix.lower() in ['.jpg','.jpeg','.png','.ppm','.bmp','.tif','.tiff']]
    ImageFile.LOAD_TRUNCATED_IMAGES=False
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:bad=[r for r in pool.map(check,paths) if r is not None]
    ImageFile.LOAD_TRUNCATED_IMAGES=True
    for row in bad:
        try:
            with Image.open(row['path']) as im:im.load();row.update(recoverable=True,size=list(im.size),mode=im.mode)
        except Exception as e:row.update(recoverable=False,recovery_error=str(e))
    write(OUT/'assets/image_decoder_audit.json',dict(scanned=len(paths),dataset='all image datasets' if args.all else 'EuroSAT',decoder_failures=bad,
        original_bytes_unchanged=True,policy='Pillow truncated-JPEG decoding enabled per training process; no sample removed',time=time.time()))
    print(json.dumps(dict(scanned=len(paths),failures=bad),indent=2))
    assert all(r['recoverable'] for r in bad)
