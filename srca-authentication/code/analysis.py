"""Calibration (Learn-then-Test with Bonferroni over the configuration grid) and evaluation over
repeated random calibration/test splits. Produces results/*.json and tables used in the paper."""
import os
from paths import LOSS, RESULTS, FIGS, DATA
import json, numpy as np
from scipy.stats import binom
from precompute import GRID, FIELDS

F = {k: i for i, k in enumerate(FIELDS)}
GRID = [tuple(g) for g in GRID]

def load(prefix):
    a, b = np.load(os.path.join(LOSS, f'{prefix}_0.npz')), np.load(os.path.join(LOSS, f'{prefix}_1.npz'))
    R = np.concatenate([a['R'], b['R']]); FF = np.concatenate([a['FF'], b['FF']])
    ids = np.concatenate([a['ids'], b['ids']]); o = np.argsort(ids)
    dom = np.concatenate([a['domain'], b['domain']]); src = np.concatenate([a['source'], b['source']])
    return R[o], FF[o], dom[o], src[o]

# ------------------------------------------------------------------ p-values
def p_hb(losses, alpha):
    """Hoeffding-Bentkus p-value for H0: E[loss] > alpha, losses in [0,1]."""
    n = losses.shape[0]; r = losses.mean(0)
    ph = np.exp(-2 * n * np.clip(alpha - r, 0, None) ** 2)
    pb = np.e * binom.cdf(np.ceil(n * r), n, alpha)
    return np.minimum(np.minimum(ph, pb), 1.0)

def p_binom(ind, alpha):
    """Exact binomial p-value for H0: P(event) > alpha."""
    n = ind.shape[0]; k = ind.sum(0)
    return binom.cdf(k, n, alpha)

def losses(R, dmax):
    miss = R[..., F['miss']]
    diag = (R[..., F['D']] > dmax).astype(float)
    infe = 1.0 - R[..., F['feasible']]
    return miss, diag, infe

def select(R, cfg, mode='joint', rng=None):
    a_m, a_d, beta, delta, dmax = cfg['alpha_m'], cfg['alpha_d'], cfg['beta'], cfg['delta'], cfg['dmax']
    miss, diag, infe = losses(R, dmax)
    G = len(GRID)
    meanD = R[..., F['D']].mean(0)
    if mode == 'joint':
        p = np.maximum.reduce([p_hb(miss, a_m), p_binom(diag, a_d), p_binom(infe, beta)])
        ok = p <= delta / G
    elif mode == 'miss_only':
        ok = p_hb(miss, a_m) <= delta / G
    elif mode == 'plugin':
        ok = (miss.mean(0) <= a_m) & (diag.mean(0) <= a_d) & (infe.mean(0) <= beta)
    else:
        raise ValueError(mode)
    if not ok.any():
        return None
    cand = np.flatnonzero(ok)
    return int(cand[np.argmin(meanD[cand] + 1e-6 * R[..., F['n_prot']].mean(0)[cand])])

def evaluate(Rt, g=None, FFt=None, dmax=None):
    X = FFt if g is None else Rt[:, g]
    tp, fp, fn = X[:, F['tp']].sum(), X[:, F['fp']].sum(), X[:, F['fn']].sum()
    D = X[:, F['D']]
    out = dict(
        miss=float(X[:, F['miss']].mean()),
        diag_viol=float((D > dmax).mean()),
        infeasible=float(1 - X[:, F['feasible']].mean()),
        wpsnr=float(10 * np.log10(255 ** 2 / max(D.mean(), 1e-12))),
        psnr=float(np.mean(10 * np.log10(255 ** 2 / np.maximum(X[:, F['mse']], 1e-12)))),
        prot_frac=float(X[:, F['n_prot']].mean() / 900),
        payload_kbit=float(X[:, F['payload']].mean() / 1000) if g is not None else 900 * 40 / 1000,
        far=float(fp / max(X[:, F['prot_untamp']].sum(), 1)),
        f1=float(2 * tp / max(2 * tp + fp + fn, 1)),
        rec_psnr=(float(np.nanmean(X[:, F['rec_psnr']])) if (g is not None and GRID[g][3] > 0 and np.isfinite(X[:, F['rec_psnr']]).any()) else float('nan')),
        header_fail=float(1 - X[:, F['header_ok']].mean()),
    )
    return out

def run_splits(R, FF, cfg, T=300, n_cal=None, seed=0, fixed=None, idx_pool=None, test_pool=None):
    rng = np.random.default_rng(seed)
    N = R.shape[0]
    res = {k: [] for k in ['SRCA', 'SRCA-miss-only', 'Plug-in', 'Fixed-H1', 'Fixed-H2', 'Full-fragile']}
    chosen = {k: [] for k in res}
    fixed = fixed or {'Fixed-H1': (0.5, 0.3, 1, 24), 'Fixed-H2': (0.3, 0.3, 2, 0)}
    for t in range(T):
        if idx_pool is None:
            perm = rng.permutation(N); nc = n_cal or N // 2
            cal, test = perm[:nc], perm[nc:]
        else:  # shift experiment: calibrate on one pool, test on another
            cal = rng.choice(idx_pool, size=min(n_cal or len(idx_pool), len(idx_pool)), replace=False)
            test = test_pool
        for name, mode in [('SRCA', 'joint'), ('SRCA-miss-only', 'miss_only'), ('Plug-in', 'plugin')]:
            g = select(R[cal], cfg, mode)
            if g is None:
                res[name].append(None); chosen[name].append(None); continue
            res[name].append(evaluate(R[test], g, dmax=cfg['dmax'])); chosen[name].append(GRID[g])
        for name, lam in fixed.items():
            g = GRID.index(lam)
            res[name].append(evaluate(R[test], g, dmax=cfg['dmax'])); chosen[name].append(lam)
        res['Full-fragile'].append(evaluate(None, None, FF[test], dmax=cfg['dmax'])); chosen['Full-fragile'].append('all')
    return res, chosen

def summarize(res, cfg):
    S = {}
    for k, v in res.items():
        vv = [x for x in v if x is not None]
        if not vv:
            S[k] = dict(abstain=1.0); continue
        keys = vv[0].keys()
        S[k] = {m: (float(np.nanmean([x[m] for x in vv])), float(np.nanstd([x[m] for x in vv]))) for m in keys}
        S[k]['viol_miss'] = float(np.mean([x['miss'] > cfg['alpha_m'] for x in vv]))
        S[k]['viol_diag'] = float(np.mean([x['diag_viol'] > cfg['alpha_d'] for x in vv]))
        S[k]['viol_feas'] = float(np.mean([x['infeasible'] > cfg['beta'] for x in vv]))
        S[k]['viol_any'] = float(np.mean([(x['miss'] > cfg['alpha_m']) or (x['diag_viol'] > cfg['alpha_d']) or
                                          (x['infeasible'] > cfg['beta']) for x in vv]))
        S[k]['abstain'] = 1 - len(vv) / len(v)
    return S
