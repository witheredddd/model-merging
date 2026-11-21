"""Does a loss-scale change move the balance structure, or only translate it?

lambda*_j(a) = |g_C| / (a|g_D| + |g_C|), so the log-odds
    t_j = log(lambda*/(1-lambda*)) = log(|g_C|/|g_D|) - log a
is translated by exactly -log a when the DTD loss is scaled by a. If that holds,
the edge fractions the pilot reports are an artifact of binning lambda with fixed
edges: the shape of the t distribution is unchanged.
"""
import json
from pathlib import Path

import numpy as np

D = Path('/media/shangxueyuan/data1/pareto_balance_pilot/physical')
gd = np.load(D / 'g_d_train.npy', mmap_mode='r')
gc = np.load(D / 'g_c_train.npy', mmap_mode='r')
n = len(gd)
chunk = 4_000_000
bins = 200
edges = np.linspace(-10, 10, bins + 1)
centres = (edges[:-1] + edges[1:]) / 2

scales = (1.0, 0.5, 2.0)
hist = {s: np.zeros(bins) for s in scales}
shift_seen = {s: [] for s in scales}
opp_total = 0
for lo in range(0, n, chunk):
    hi = min(n, lo + chunk)
    d = np.asarray(gd[lo:hi], dtype=np.float64)
    c = np.asarray(gc[lo:hi], dtype=np.float64)
    opp = (d * c < 0) & np.isfinite(d) & np.isfinite(c) & (d != 0) & (c != 0)
    opp_total += int(opp.sum())
    if not opp.any():
        continue
    t1 = np.log(np.abs(c[opp])) - np.log(np.abs(d[opp]))
    for s in scales:
        t = t1 - np.log(s)
        hist[s] += np.histogram(np.clip(t, -10, 10), bins=edges)[0]
        if s != 1.0 and lo == 0:
            shift_seen[s].append(float(np.median(t - t1)))

print(f'coordinates {n}, opposite-sign {opp_total} ({100*opp_total/n:.2f}%)')
print('median(t(a) - t(1)) for a=0.5,2.0:', shift_seen[0.5], shift_seen[2.0])
print('predicted -log a:', -np.log(0.5), -np.log(2.0))

# centre each histogram at its own weighted mean and compare shapes
shapes = {}
for s in scales:
    w = hist[s] / max(hist[s].sum(), 1e-30)
    mean = float((centres * w).sum())
    shapes[s] = (w, mean)
    print(f'  scale {s}: weighted mean t = {mean:+.6f}')

base_w, base_m = shapes[1.0]
for s in (0.5, 2.0):
    w, m = shapes[s]
    # translate the scaled histogram back by the predicted shift
    shifted = np.interp(centres - (base_m - m), centres, w, left=0., right=0.)
    l1 = float(np.abs(base_w - shifted).sum())
    raw_l1 = float(np.abs(base_w - w).sum())
    print(f'  scale {s}: mean shift {m - base_m:+.6f} (predicted {-np.log(s):+.6f}); '
          f'L1 before recentring {raw_l1:.4f}, after {l1:.4f}')

out = D.parent / 'SCALE_SHAPE_CHECK.json'
out.write_text(json.dumps(dict(
    n=n, opposite=opp_total, scales=list(scales),
    weighted_mean_t={str(s): shapes[s][1] for s in scales},
    predicted_shift={str(s): float(-np.log(s)) for s in scales},
    note='t = log(lambda/(1-lambda)); a DTD loss scaling by a translates t by -log a, '
         'so fixed lambda edges must change their occupancy without any structural change',
), indent=2))
print('wrote', out)
