"""Run the full embed -> attack -> blind verify pipeline for every (image, configuration)
and store per-instance losses and metrics. Usage: python precompute.py <targeted> <out.npz> <shard> <nshards>"""
import os, sys, time, numpy as np
from srca_core import *
from data_attacks import load_pool, sample_attack

TAUS = [float(x) for x in np.round(np.arange(0.05, 0.801, 0.05), 2)]
OMEGAS = [0.10, 0.20, 0.30, 0.40]
BITS = [1, 2, 3]
KRS = [0, 24]
GRID = [(t, o, b, r) for t in TAUS for o in OMEGAS for b in BITS for r in KRS]

def full_fragile(img, cache, key=KEY):
    """Baseline FF: every interior block carries a 16-bit MAC of its own 7 MSBs (in its own LSBs)
    and 24 recovery bits of a keyed partner block (in its own LSBs) -> 40 of 64 LSBs used."""
    n = img.shape[0] // BS
    bm = border_mask(n).ravel(); idx = np.flatnonzero(~bm)
    blocks = to_blocks(img).copy()
    msb = blocks & 0xFE
    tags = mac_bits(msb, idx, key)
    rec = recovery_bits(blocks)
    partner = idx[keyed_perm(len(idx), b"ffpart", key)]
    pos = keyed_perm(BS*BS, b"ffpos", key)[:PB]
    pay = np.concatenate([tags, rec[partner]], 1)
    sub = blocks[idx][:, pos]
    blocks[np.repeat(idx[:, None], PB, 1), np.repeat(pos[None, :], len(idx), 0)] = (sub & 0xFE) | pay
    cache['ff_pos'] = pos; cache['ff_idx'] = idx
    return from_blocks(blocks)

def ff_verify(rx, cache, key=KEY):
    n = rx.shape[0] // BS
    blocks = to_blocks(rx); idx = cache['ff_idx']; pos = cache['ff_pos']
    tags = mac_bits(blocks & 0xFE, idx, key)
    ext = blocks[idx][:, pos[:KA]] & 1
    flags = np.zeros(n*n, bool); flags[idx] = np.any(ext != tags, 1)
    return flags

def metrics(img, wm, rx, w, flags, S, header_ok, recovered=None):
    n = img.shape[0] // BS
    bm = border_mask(n).ravel(); wf = w.ravel()
    B0, B1, B2 = to_blocks(img).astype(float), to_blocks(wm).astype(float), to_blocks(rx)
    T = np.any(B2 != to_blocks(wm), 1) & ~bm
    se = ((B1 - B0) ** 2).mean(1)
    D = float((wf * se)[~bm].sum() / max(wf[~bm].sum(), 1e-9))
    mse = float(((B1 - B0) ** 2).mean())
    wT = wf[T].sum()
    if not header_ok: lm = 1.0 if wT > 0 else 0.0
    else: lm = float((wf * (T & ~flags)).sum() / wT) if wT > 0 else 0.0
    fp = int((flags & ~T).sum()); tp = int((flags & T).sum()); fn = int((T & ~flags).sum())
    prot_untamp = int((S & ~T).sum())
    rec_psnr = np.nan
    if recovered is not None:
        k = flags & T & S
        if k.any():
            Rr = to_blocks(recovered).astype(float)
            e = ((Rr[k] - B0[k]) ** 2).mean()
            rec_psnr = 10*np.log10(255**2 / max(e, 1e-9))
    return [lm, D, mse, tp, fp, fn, prot_untamp, int(T.sum()), float(wT), rec_psnr]

FIELDS = ['miss', 'D', 'mse', 'tp', 'fp', 'fn', 'prot_untamp', 'n_tamp', 'w_tamp', 'rec_psnr',
          'feasible', 'n_prot', 'payload', 'capacity', 'header_ok']

if __name__ == '__main__':
    targeted = float(sys.argv[1]); out = sys.argv[2]; shard, nsh = int(sys.argv[3]), int(sys.argv[4])
    pool = load_pool({'natural': 40, 'biomedical': 60}, seed=0)
    rng_master = np.random.default_rng(1234 + int(targeted * 10))
    seeds = rng_master.integers(0, 2**31, len(pool))
    donors = rng_master.integers(0, len(pool), len(pool))
    ids = [i for i in range(len(pool)) if i % nsh == shard]
    if os.environ.get('SRCA_LIMIT'):          # quick smoke test: first k images of the shard
        ids = ids[:int(os.environ['SRCA_LIMIT'])]
    R = np.full((len(ids), len(GRID), len(FIELDS)), np.nan)
    FF = np.full((len(ids), len(FIELDS)), np.nan)
    t0 = time.time()
    for a, i in enumerate(ids):
        img = pool[i]['img']; w = relevance_map(img)
        rng = np.random.default_rng(seeds[i])
        f, kinds = sample_attack(w, rng, donor=pool[donors[i]]['img'], targeted=targeted)
        cache = {}
        for g, lam in enumerate(GRID):
            wm, meta = embed(img, w, lam, cache=cache)
            rx = f(wm)
            res = verify(rx, ref_blocks=to_blocks(wm), ref_tags=cache['tags'])
            m = metrics(img, wm, rx, w, res['flags'], meta['S'], res['header_ok'], res['recovered'])
            R[a, g] = m + [meta['feasible'], meta['n_prot'], meta['payload_bits'], meta['capacity_bits'], res['header_ok']]
        wm = full_fragile(img, cache); rx = f(wm)
        fl = ff_verify(rx, cache)
        n = img.shape[0] // BS
        FF[a] = metrics(img, wm, rx, w, fl, ~border_mask(n).ravel(), True) + [1, int((~border_mask(n)).sum()), 0, 0, 1]
        if a % 25 == 0: print(shard, a, len(ids), round(time.time() - t0, 1), flush=True)
    np.savez_compressed(out, R=R, FF=FF, ids=np.array(ids), grid=np.array(GRID),
                        domain=np.array([pool[i]['domain'] for i in ids]), source=np.array([pool[i]['source'] for i in ids]))
    print('done', out, time.time() - t0)
