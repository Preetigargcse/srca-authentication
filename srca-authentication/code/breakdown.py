"""Per-operation and per-source breakdown at the deployed configuration (Table 6 and Section 5.5)."""
import os, json, sys, numpy as np
from paths import RESULTS
from srca_core import relevance_map
from data_attacks import load_pool, sample_attack
from analysis import load, F, GRID

pool = load_pool({'natural': 40, 'biomedical': 60}, seed=0)
rm = np.random.default_rng(1234 + 10)                      # same seeds as precompute.py (targeted)
seeds = rm.integers(0, 2**31, len(pool)); donors = rm.integers(0, len(pool), len(pool))
kinds = []
for i, P in enumerate(pool):
    w = relevance_map(P['img'])
    _, k = sample_attack(w, np.random.default_rng(seeds[i]), donor=pool[donors[i]]['img'], targeted=1.0)
    kinds.append(k)
R, FF, dom, src = load('tgt')
g = GRID.index((0.15, 0.2, 1, 0)); gp = GRID.index((0.25, 0.2, 1, 0))
out = {}
for a in ['copy_move', 'splice', 'inpaint_blur', 'lesion_inject', 'lesion_remove']:
    idx = [i for i, k in enumerate(kinds) if len(k) == 1 and k[0] == a]
    X = R[idx, g]; tp, fp, fn = X[:, F['tp']].sum(), X[:, F['fp']].sum(), X[:, F['fn']].sum()
    out[a] = dict(n=len(idx), miss=float(X[:, F['miss']].mean()), miss_plugin=float(R[idx, gp, F['miss']].mean()),
                  f1=float(2*tp/(2*tp+fp+fn)), ff_miss=float(FF[idx, F['miss']].mean()))
ps = {}
for s in np.unique(src):
    m = src == s
    ps[str(s)] = dict(miss=float(R[m, g, F['miss']].mean()), wpsnr=float(10*np.log10(255**2/R[m, g, F['D']].mean())),
                      prot=float(R[m, g, F['n_prot']].mean()/900), dom=str(dom[m][0]))
json.dump(dict(attack=out, source=ps), open(os.path.join(RESULTS, 'results_breakdown.json'), 'w'), indent=1)
print(json.dumps(out, indent=1))
