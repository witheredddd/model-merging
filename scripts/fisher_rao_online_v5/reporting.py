"""Exact cost-grid views of the committed model, separate from Vcal selection.

Unselected lookahead states are never reported as the main model. Pending
boundary evaluations wait until the current atomic operation commits or rejects.
"""


class CostReporter:
    def __init__(self,costs,get_state,evaluate,save_best):
        self.costs=sorted(set(costs))
        if not self.costs or self.costs[0]!=0:raise ValueError('Cost grid must include zero')
        self.get_state=get_state;self.evaluate=evaluate;self.save_best=save_best
        self.rows=[];self.pending=None;self.next=0;self.best=None

    def _evaluate(self,cost):
        state=self.get_state();metrics=self.evaluate(state)
        row=dict(exact_online_image_cost=cost,instant_validation=metrics)
        if self.best is None or metrics['CE']<self.best['CE']:
            self.best=dict(**metrics,model_online_cost=cost)
            self.save_best(state,cost)
        row['Vcal_selected']=dict(self.best)
        self.rows.append(row)

    def flush(self):
        if self.pending is not None:
            self._evaluate(self.pending);self.pending=None

    def bootstrap_base(self,used):
        # Initialization probes never move the base model. Report that base at
        # any checkpoints already consumed before attaching the observer.
        while self.next<len(self.costs) and self.costs[self.next]<=used:
            self._evaluate(self.costs[self.next]);self.next+=1

    def before_progress(self,before,after):
        if after<before:raise ValueError('Compute ledger moved backwards')
        self.flush()
        while self.next<len(self.costs) and self.costs[self.next]<=after:
            cost=self.costs[self.next];self.next+=1
            if cost==after:self.pending=cost
            else:self._evaluate(cost)
