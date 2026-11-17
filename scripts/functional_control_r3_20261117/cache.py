"""Derived GPU cache limited by bytes; eviction never deletes a learned mode."""
from collections import OrderedDict
from pathlib import Path
import torch


class MotionCache:
    def __init__(self,device,byte_budget):
        self.device=device;self.budget=int(byte_budget);self.cache=OrderedDict();self.used=0
        self.loads=0;self.hits=0

    def get(self,path):
        key=str(Path(path).resolve())
        if key in self.cache:
            self.hits+=1;v=self.cache.pop(key);self.cache[key]=v;return v
        self.loads+=1
        v=torch.load(key,map_location='cpu',mmap=True,weights_only=True)['motion'].to(self.device,dtype=torch.float64)
        size=v.numel()*v.element_size()
        while self.cache and self.used+size>self.budget:
            _,old=self.cache.popitem(last=False);self.used-=old.numel()*old.element_size()
        if size<=self.budget:self.cache[key]=v;self.used+=size
        return v

    def clear(self):self.cache.clear();self.used=0
