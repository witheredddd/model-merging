"""Disk-backed parameter algebra; O(block + r*r), never a D-by-r matrix."""
from pathlib import Path
import math
import mmap
import sqlite3
import numpy as np
import torch
from bootstrap import R3
from control_math import quadratic_solve
from io_state import atomic_json,digest


class DiskAlgebra:
    def __init__(self,root,block_elements=262144):
        self.root=Path(root);self.block=int(block_elements)
        if self.block<1:raise ValueError('Positive I/O block required')
        self.stats=dict(dot_products=0,read_elements=0,max_working_elements=0)
        folder=self.root/'stream_gram';folder.mkdir(exist_ok=True,parents=True)
        self.pairs=sqlite3.connect(folder/'parameter_pairs.sqlite')
        self.pairs.execute('CREATE TABLE IF NOT EXISTS products (a TEXT,b TEXT,value REAL,error REAL,PRIMARY KEY(a,b))')
        self.seeded={}

    def identity(self,e):return e['id']+':'+e.get('motion_sha256',e['motion_file'])

    def remember_pair(self,a,b,value,error):
        ka,kb=sorted((self.identity(a),self.identity(b)))
        with self.pairs:self.pairs.execute('INSERT OR REPLACE INTO products VALUES (?,?,?,?)',(ka,kb,value,error))

    def open(self,entry):
        path=self.root/entry['motion_file']
        if path.suffix=='.npy':return np.load(path,mmap_mode='r')
        return torch.load(path,map_location='cpu',mmap=True,weights_only=True)['motion'].numpy()

    def release(self,a):
        mm=getattr(a,'_mmap',None)
        if mm is not None and hasattr(mm,'madvise'):
            try:mm.madvise(mmap.MADV_DONTNEED)
            except (OSError,ValueError):pass

    def dot(self,a,b):
        if len(a)!=len(b):raise ValueError('Parameter dimensions differ')
        total=np.longdouble(0);absolute=np.longdouble(0);error=np.longdouble(0)
        eps=np.finfo(np.longdouble).eps
        for start in range(0,len(a),self.block):
            stop=min(len(a),start+self.block);n=stop-start
            av=np.asarray(a[start:stop],dtype=np.longdouble);bv=np.asarray(b[start:stop],dtype=np.longdouble)
            p=av*bv;s=np.sum(p,dtype=np.longdouble);ab=np.sum(np.abs(p),dtype=np.longdouble)
            gamma=n*eps/(1-n*eps);error+=(gamma+eps)*ab;total+=s;absolute+=ab
            self.stats['read_elements']+=2*n
            self.stats['max_working_elements']=max(self.stats['max_working_elements'],4*n)
        self.stats['dot_products']+=1
        error+=eps*(1+math.ceil(len(a)/self.block))*absolute+np.finfo(float).eps*abs(total)
        return float(total),float(error)

    def gram(self,entries,label):
        folder=self.root/'stream_gram';folder.mkdir(exist_ok=True)
        cache=folder/(label+'.npz');identities=[self.identity(e) for e in entries]
        r=len(entries);G=np.zeros((r,r));E=np.zeros_like(G);prefix=0
        if cache.exists():
            with np.load(cache) as z:
                old=list(z['identities'])
                if old==identities[:len(old)]:
                    prefix=len(old);G[:prefix,:prefix]=z['G'];E[:prefix,:prefix]=z['E']
        seeded=self.seeded.get(label,[])
        seed_start=len(seeded) if seeded==identities[:len(seeded)] else 0
        with self.pairs:
            for j in range(seed_start,prefix):
                for i in range(j+1):
                    ka,kb=sorted((identities[i],identities[j]))
                    self.pairs.execute('INSERT OR IGNORE INTO products VALUES (?,?,?,?)',(ka,kb,G[i,j],E[i,j]))
        for j in range(prefix,r):
            b=self.open(entries[j])
            for i in range(j+1):
                ka,kb=sorted((identities[i],identities[j]))
                cached=self.pairs.execute('SELECT value,error FROM products WHERE a=? AND b=?',(ka,kb)).fetchone()
                if cached is not None:value,error=cached
                else:
                    a=b if i==j else self.open(entries[i]);value,error=self.dot(a,b)
                    self.pairs.execute('INSERT OR REPLACE INTO products VALUES (?,?,?,?)',(ka,kb,value,error))
                    self.release(a)
                G[i,j]=G[j,i]=value;E[i,j]=E[j,i]=error
            self.release(b)
        self.pairs.commit();self.seeded[label]=list(identities)
        temp=cache.with_suffix('.tmp')
        with temp.open('wb') as f:np.savez(f,G=G,E=E,identities=np.asarray(identities,dtype=str))
        temp.replace(cache)
        return G,E

    def write_combination(self,entries,coefficients,path,reference=None,reference_scale=0.):
        path=self.root/path;path.parent.mkdir(parents=True,exist_ok=True)
        template=reference if reference is not None else self.open(entries[0])
        temp=Path(str(path)+'.tmp');out=np.lib.format.open_memmap(temp,mode='w+',dtype=np.float64,shape=(len(template),))
        for start in range(0,len(template),self.block):
            stop=min(len(template),start+self.block)
            value=np.array(template[start:stop],dtype=float)*reference_scale if reference is not None else np.zeros(stop-start)
            for e,c in zip(entries,coefficients):
                a=self.open(e);value+=float(c)*a[start:stop];self.release(a)
            out[start:stop]=value
        out.flush();del out;temp.replace(path)
        return np.load(path,mmap_mode='r')

    def innovation(self,entries,reference,path,label):
        G,E=self.gram(entries,label);r=len(entries)
        rhs=np.array([self.dot(self.open(e),reference)[0] for e in entries])
        z=quadratic_solve(G,rhs) if r else np.empty(0)
        residual=self.write_combination(entries,-z,path,reference,1.)
        # A second streamed orthogonalization pass removes cancellation error.
        rhs2=np.array([self.dot(self.open(e),residual)[0] for e in entries])
        z2=quadratic_solve(G,rhs2) if r else np.empty(0)
        if r:
            residual=self.write_combination(entries,-(z+z2),path,reference,1.);z+=z2
        squared,error=self.dot(residual,residual);norm=math.sqrt(max(0.,squared))
        original=math.sqrt(max(0.,self.dot(reference,reference)[0]))
        # Bound actual streamed arithmetic; no eps*D rank threshold.
        gram_error=float(np.linalg.norm(E,2)) if r else 0.
        arithmetic=math.sqrt(max(0.,error))+np.finfo(float).eps*(original+sum(abs(c)*math.sqrt(max(0.,G[i,i])) for i,c in enumerate(z)))*(r+4)
        if norm>arithmetic:
            out=np.lib.format.open_memmap(self.root/path,mode='r+')
            for start in range(0,len(out),self.block):out[start:start+self.block]/=norm
            out.flush();del out
        return dict(norm=norm,coefficients=z,numerically_nonzero=norm>arithmetic,
            arithmetic_error_bound=arithmetic,coordinate_gram_error=gram_error,
            motion_file=str(path),motion_sha256=digest(self.root/path))

    def convert(self,entry):
        if Path(entry['motion_file']).suffix=='.npy':return entry
        a=self.open(entry);path=Path('stream_vectors')/(entry['id']+'.npy')
        self.write_combination([],[],path,reference=a,reference_scale=1.)
        return dict(entry,motion_file=str(path),motion_sha256=digest(self.root/path))
