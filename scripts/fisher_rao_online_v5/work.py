"""Count every online input image, including branches, JVP and monitoring.

JVP/backward operation counts are separate; image equality is not a FLOPs claim.
"""
from dataclasses import dataclass,field


class BudgetEnd(RuntimeError):
    pass


@dataclass
class Ledger:
    limit: int = 1000000
    images: dict = field(default_factory=dict)
    operations: dict = field(default_factory=dict)
    on_progress: object = None

    @property
    def used(self):
        return sum(self.images.values())

    def charge(self,kind,n,backward=False,jvp=False):
        if n<0 or self.used+n>self.limit:
            raise BudgetEnd('Online compute budget exhausted before '+kind)
        if self.on_progress is not None:self.on_progress(self.used,self.used+n)
        self.images[kind]=self.images.get(kind,0)+int(n)
        self.operations['forward']=self.operations.get('forward',0)+1
        if backward:self.operations['backward']=self.operations.get('backward',0)+1
        if jvp:self.operations['jvp']=self.operations.get('jvp',0)+1

    def merge_branch(self,branch):
        if self.used+branch.used>self.limit:raise BudgetEnd('Unfunded branch merge')
        if self.on_progress is not None:self.on_progress(self.used,self.used+branch.used)
        for kind,n in branch.images.items():self.images[kind]=self.images.get(kind,0)+n
        for kind,n in branch.operations.items():self.operations[kind]=self.operations.get(kind,0)+n


class MeasuredOracle:
    def __init__(self,oracle,ledger,prefix=''):
        self.raw=oracle;self.ledger=ledger;self.prefix=prefix
        self.theta0=oracle.theta0;self.n=oracle.n

    def sample(self,seed,step,exclude=()):
        return self.raw.sample(seed,step,exclude)

    def gradient(self,state,sample):
        self.ledger.charge(self.prefix+'task_gradient',len(sample['y']),backward=True)
        return self.raw.gradient(state.parameters(),sample)

    def loss(self,state,sample):
        self.ledger.charge(self.prefix+'task_loss',len(sample['y']))
        return self.raw.loss(state.parameters(),sample)

    def phi(self,state,positions=None):
        self.ledger.charge(self.prefix+'FR',self.n if positions is None else len(positions))
        return self.raw.phi(state.parameters(),positions)

    def jvp(self,state,v):
        self.ledger.charge(self.prefix+'JVP',self.n,jvp=True)
        return self.raw.jvp(state.parameters(),v)

    def raw_loss_at(self,theta,sample):
        self.ledger.charge(self.prefix+'read_only_direction_probe',len(sample['y']))
        return self.raw.loss(theta,sample)
