import os
from paths import LOSS, RESULTS, FIGS, DATA
import json, sys, numpy as np
from collections import Counter
from analysis import *

CFG = dict(alpha_m=0.15, alpha_d=0.10, beta=0.10, delta=0.10, dmax=0.10)
out = {}
R, FF, dom, src = load('tgt')
N = R.shape[0]
print('N', N, 'grid', len(GRID))

# E1 main
res, ch = run_splits(R, FF, CFG, T=300, seed=1)
out['main'] = summarize(res, CFG)
out['main_chosen'] = {k: Counter([str(c) for c in v]).most_common(5) for k, v in ch.items()}
# test-risk samples for figure
out['main_samples'] = {k: [[x['miss'], x['diag_viol'], x['infeasible']] if x else None for x in v] for k, v in res.items()}

# E2 alpha sweep
out['alpha_sweep'] = {}
for a in [0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30]:
    cfg = dict(CFG, alpha_m=a)
    r, _ = run_splits(R, FF, cfg, T=100, seed=2)
    s = summarize(r, cfg)
    out['alpha_sweep'][a] = {k: s[k] for k in ['SRCA', 'Plug-in', 'SRCA-miss-only']}

# E2b diagnostic-budget sweep
out['dmax_sweep'] = {}
for dm in [0.03, 0.05, 0.10, 0.20]:
    cfg = dict(CFG, dmax=dm)
    r, ch2 = run_splits(R, FF, cfg, T=100, seed=5)
    s = summarize(r, cfg)
    out['dmax_sweep'][dm] = {'SRCA': s['SRCA'], 'chosen': Counter([str(c) for c in ch2['SRCA']]).most_common(3)}

# E3 calibration-size sweep: test on a fixed half, calibrate on subsets of the other half
rng = np.random.default_rng(3)
out['ncal_sweep'] = {}
for nc in [60, 100, 150, 250, 380]:
    acc = []
    for t in range(100):
        perm = rng.permutation(N); calpool, test = perm[:380], perm[380:]
        cal = calpool[:nc]
        g = select(R[cal], CFG, 'joint')
        acc.append(None if g is None else evaluate(R[test], g, dmax=CFG['dmax']))
    out['ncal_sweep'][nc] = summarize({'SRCA': acc}, CFG)['SRCA']

# E4 shift experiments
nat, bio = np.flatnonzero(dom == 'natural'), np.flatnonzero(dom == 'biomedical')
out['shift'] = {}
for name, cp, tp in [('nat->bio', nat, bio), ('bio->nat', bio, nat)]:
    r, _ = run_splits(R, FF, CFG, T=100, seed=4, idx_pool=cp, test_pool=tp, n_cal=min(len(cp), 240))
    out['shift'][name] = summarize(r, CFG)
# in-domain reference for bio (split bio in halves)
for name, pool_ in [('bio->bio', bio), ('nat->nat', nat)]:
    acc = []
    for t in range(100):
        p = rng.permutation(pool_); h = len(p) // 2
        g = select(R[p[:h]], CFG, 'joint')
        acc.append(None if g is None else evaluate(R[p[h:]], g, dmax=CFG['dmax']))
    out['shift'][name] = {'SRCA': summarize({'SRCA': acc}, CFG)['SRCA']}
# threat-model shift: calibrate under targeted attacks, deploy under uniform placement (and the reverse)
RU, FFU, _, _ = load('uni')
for name, Rc, Rt in [('targeted->uniform', R, RU), ('uniform->uniform', RU, RU), ('uniform->targeted', RU, R)]:
    acc = []
    for t in range(100):
        perm = rng.permutation(N); cal, test = perm[:380], perm[380:]
        g = select(Rc[cal], CFG, 'joint')
        acc.append(None if g is None else evaluate(Rt[test], g, dmax=CFG['dmax']))
    out['shift'][name] = {'SRCA': summarize({'SRCA': acc}, CFG)['SRCA']}
mu, _, _ = losses(RU, CFG['dmax'])
out['uniform_best_miss'] = float(mu.mean(0).min())
out['uniform_best_cfg'] = str(GRID[int(mu.mean(0).argmin())])
# landscape for figure: mean risks per grid point (all data)
miss, diag, infe = losses(R, CFG['dmax'])
out['landscape'] = dict(grid=[list(g) for g in GRID], miss=miss.mean(0).tolist(), diag=diag.mean(0).tolist(),
                        infe=infe.mean(0).tolist(), D=R[..., F['D']].mean(0).tolist())
out['domain_counts'] = {'natural': int(len(nat)), 'biomedical': int(len(bio))}
json.dump(out, open(os.path.join(RESULTS, 'results.json'), 'w'), indent=1, default=str)
print(json.dumps(out['main'], indent=1, default=str)[:6000])
