"""
SRCA: Risk-Calibrated Selective Image Authentication through Joint Payload
Feasibility and Diagnostic-Loss Control -- reference implementation (numpy).

Per image (8-bit grayscale, N x N, 8x8 blocks), configuration lam = (tau, omega, b):
  S = protected blocks (w >= tau, interior)  -> left bit-exact, authenticated + recovery bits
  H = host blocks (w < omega, interior, not S) -> b LSB planes carry payload slots
  R = border ring                              -> header (lam + class map) x3 + MAC, 1 LSB
  remaining interior blocks: untouched and unprotected.
Payload per protected block: KA-bit keyed MAC (BLAKE2b) of the full 8-bit block + index,
and KR recovery bits (four 4x4 sub-block means at 6 bits). One protected block per host slot;
a host with b planes offers floor(64 b / (KA+KR)) slots. If |S| exceeds the slot budget the
sender keeps the most relevant blocks (graceful degradation) and records the payload as
infeasible. The receiver is blind: everything it needs is in the header and the key.
"""
import hashlib, numpy as np
from scipy import ndimage as ndi

BS, KA, KR = 8, 16, 24
PB = KA + KR
HDR_REP = 4
KEY = b"srca-demo-key-2026"

def border_mask(n):
    m = np.zeros((n, n), bool); m[0, :] = m[-1, :] = m[:, 0] = m[:, -1] = True
    return m

def relevance_map(img):
    """Block-level diagnostic-relevance proxy in [0,1] from the 5 MSBs (invariant to <=3-LSB
    embedding): smoothed gradient energy + local contrast, gated by a foreground mask."""
    x = (img >> 3).astype(np.float64) * 8.0
    xs = ndi.gaussian_filter(x, 1.5)
    gm = np.hypot(ndi.sobel(xs, 0), ndi.sobel(xs, 1))
    lstd = np.sqrt(np.maximum(ndi.uniform_filter(x**2, 9) - ndi.uniform_filter(x, 9)**2, 0))
    fg = (xs > np.percentile(xs, 10) + 4).astype(float)
    n = img.shape[0] // BS
    bmean = lambda a: a.reshape(n, BS, n, BS).mean(axis=(1, 3))
    r = bmean(gm) / (np.percentile(gm, 99) + 1e-6) + 0.5 * bmean(lstd) / (np.percentile(lstd, 99) + 1e-6)
    r = r * (0.3 + 0.7 * bmean(fg))
    r = ndi.gaussian_filter(r, 0.8)
    r = np.clip(r / (np.percentile(r, 99) + 1e-9), 0, 1)
    r[border_mask(n)] = 0.0
    return r

def to_blocks(img):
    n = img.shape[0] // BS
    return img.reshape(n, BS, n, BS).transpose(0, 2, 1, 3).reshape(n*n, BS*BS)

def from_blocks(flat):
    nn = flat.shape[0]; n = int(round(np.sqrt(nn)))
    return flat.reshape(n, n, BS, BS).transpose(0, 2, 1, 3).reshape(n*BS, n*BS)

def mac_bits(blocks, idx, key=KEY):
    out = np.zeros((len(idx), KA), np.uint8)
    for t, j in enumerate(idx):
        h = hashlib.blake2b(blocks[j].tobytes() + int(j).to_bytes(4, 'little'), key=key, digest_size=4).digest()
        v = int.from_bytes(h, 'little')
        out[t] = (v >> np.arange(KA)) & 1
    return out

def recovery_bits(blocks):
    b = blocks.reshape(-1, 2, 4, 2, 4).astype(float).mean(axis=(2, 4)).reshape(-1, 4)
    q = np.clip(np.round(b / 255 * 63), 0, 63).astype(np.uint8)
    return ((q[:, :, None] >> np.arange(6)) & 1).reshape(-1, 24).astype(np.uint8)

def recovered_block(bits24):
    q = (bits24.reshape(4, 6).astype(int) * (1 << np.arange(6))).sum(1) / 63 * 255
    blk = np.empty((BS, BS)); blk[:4, :4], blk[:4, 4:], blk[4:, :4], blk[4:, 4:] = q
    return blk

def keyed_perm(m, salt, key=KEY):
    seed = int.from_bytes(hashlib.blake2b(key + salt, digest_size=8).digest(), 'little')
    return np.random.default_rng(seed).permutation(m)

def slots_per_host(b, kr=KR):
    return (BS * BS * b) // (KA + kr)

def _assignment(s_idx, h_idx, b, kr=KR, key=KEY):
    """Receiver-reproducible slot assignment: keyed order of protected and host blocks."""
    s_ord = np.sort(s_idx)[keyed_perm(len(s_idx), b"prot", key)] if len(s_idx) else s_idx
    h_ord = np.sort(h_idx)[keyed_perm(len(h_idx), b"host", key)] if len(h_idx) else h_idx
    pb = KA + kr
    spb = slots_per_host(b, kr)
    host = np.repeat(h_ord, spb)[:len(s_ord)]
    slot = np.tile(np.arange(spb), max(len(h_ord), 1))[:len(s_ord)]
    pos = keyed_perm(BS*BS*b, b"pos" + bytes([b]), key)
    bitpos = pos[(slot[:, None] * pb + np.arange(pb)[None, :])] if len(s_ord) else np.zeros((0, pb), int)
    return s_ord, host, bitpos % (BS*BS), bitpos // (BS*BS)

def _header_bits(tau, omega, b, kr, cls_interior):
    lam = np.concatenate([(int(round(tau*100)) >> np.arange(8)) & 1, (int(round(omega*100)) >> np.arange(8)) & 1,
                          (b >> np.arange(2)) & 1, [1 if kr > 0 else 0]]).astype(np.uint8)
    cm = np.stack([cls_interior & 1, (cls_interior >> 1) & 1], 1).ravel().astype(np.uint8)
    return np.concatenate([lam, cm])

def _ring_segments(bm, key=KEY):
    """Border pixels split into HDR_REP spatially contiguous segments along the ring."""
    n = bm.shape[0]
    ring = [(0, c) for c in range(n)] + [(r, n-1) for r in range(1, n)] + \
           [(n-1, c) for c in range(n-2, -1, -1)] + [(r, 0) for r in range(n-2, 0, -1)]
    ring_idx = np.array([r*n + c for r, c in ring])
    segs = np.array_split(ring_idx, HDR_REP)
    out = []
    for s, seg in enumerate(segs):
        px = (seg[:, None] * (BS*BS) + np.arange(BS*BS)[None, :]).ravel()   # flat (block*64+pix)
        out.append(px[keyed_perm(len(px), b"hdr" + bytes([s]), key)])
    return out

def _write_header(blocks, bm, hdr, key=KEY):
    flat = blocks.ravel()
    fits = True
    for seg in _ring_segments(bm.reshape(int(np.sqrt(bm.size)), -1), key):
        if len(hdr) > len(seg): fits = False
        p = seg[:len(hdr)]
        flat[p] = (flat[p] & 0xFE) | hdr[:len(p)]
    return flat.reshape(blocks.shape), fits

def _read_header(blocks, bm, L, key=KEY):
    flat = blocks.ravel()
    copies = []
    for seg in _ring_segments(bm.reshape(int(np.sqrt(bm.size)), -1), key):
        copies.append((flat[seg[:L]] & 1).astype(np.uint8))
    def mac_ok(h):
        body, mac = h[:-16], h[-16:]
        hm = int.from_bytes(hashlib.blake2b(body.tobytes(), key=key, digest_size=2).digest(), 'little')
        return bool(np.all(((hm >> np.arange(16)) & 1) == mac))
    for h in copies + [(np.sum(copies, 0) * 2 > len(copies)).astype(np.uint8)]:
        if mac_ok(h): return h[:-16], True
    return copies[0][:-16], False

def embed(img, w, lam, key=KEY, cache=None):
    tau, omega, b, kr = lam
    pb = KA + kr
    n = img.shape[0] // BS
    bm = border_mask(n).ravel(); wf = w.ravel()
    # hosts: the lowest-relevance fraction omega of interior blocks (rank-based capacity)
    interior = np.flatnonzero(~bm)
    nh = int(round(omega * len(interior)))
    h_idx = np.sort(interior[np.argsort(wf[interior], kind='stable')[:nh]])
    H = np.zeros(n*n, bool); H[h_idx] = True
    conflict = bool(np.any(wf[h_idx] >= tau))          # a block that should be protected had to host
    S = (wf >= tau) & ~bm & ~H
    s_all = np.flatnonzero(S)
    cap = len(h_idx) * slots_per_host(b, kr)
    feasible = len(s_all) <= cap
    s_eff = s_all if feasible else s_all[np.argsort(-wf[s_all], kind='stable')[:cap]]
    blocks = to_blocks(img).copy()
    if cache is None: cache = {}
    if 'tags' not in cache: cache['tags'] = mac_bits(blocks, np.arange(n*n), key)
    if 'rec' not in cache: cache['rec'] = recovery_bits(blocks)
    s_ord, host, pix, plane = _assignment(s_eff, h_idx, b, kr, key)
    payload = np.concatenate([cache['tags'][s_ord], cache['rec'][s_ord][:, :kr]], 1).astype(np.int32)
    if len(s_ord):
        hh = np.repeat(host[:, None], pb, 1)
        for pl in range(b):          # one plane at a time: (host, pixel) unique within a plane
            m = plane == pl
            v = blocks[hh[m], pix[m]].astype(np.int32)
            v = (v & ~(1 << pl)) | (payload[m] << pl)
            blocks[hh[m], pix[m]] = v.astype(np.uint8)
    # header in border ring
    cls = np.zeros(n*n, np.uint8); cls[s_eff] = 1; cls[h_idx] = 2
    body = _header_bits(tau, omega, b, kr, cls[~bm])
    hm = int.from_bytes(hashlib.blake2b(body.tobytes(), key=key, digest_size=2).digest(), 'little')
    hdr = np.concatenate([body, (hm >> np.arange(16)) & 1]).astype(np.uint8)
    blocks, hdr_fits = _write_header(blocks, bm, hdr, key)
    out = from_blocks(blocks)
    return out, dict(S_req=S, S=np.isin(np.arange(n*n), s_eff), H=H, feasible=bool(feasible and hdr_fits),
                     payload_bits=int(len(s_all)*pb), capacity_bits=int(cap*pb), n_prot=len(s_eff))

def verify(rx, key=KEY, ref_blocks=None, ref_tags=None):
    """Blind verification. Returns dict: flags (n*n bool), S (decoded protected mask),
    header_ok, recovered image (flagged protected blocks replaced when host intact)."""
    n = rx.shape[0] // BS
    bm = border_mask(n).ravel()
    blocks = to_blocks(rx)
    ni = int((~bm).sum())
    L = 19 + 2*ni + 16
    body, ok = _read_header(blocks, bm, L, key)
    flags = np.zeros(n*n, bool)
    res = dict(flags=flags, S=np.zeros(n*n, bool), header_ok=ok, recovered=rx.copy())
    if not ok: return res
    b = int(body[16]) + 2*int(body[17])
    kr = KR if body[18] else 0
    pb = KA + kr
    cm = body[19:].reshape(-1, 2).astype(int)
    cls = np.zeros(n*n, int); cls[~bm] = cm[:, 0] + 2*cm[:, 1]
    S, H = cls == 1, cls == 2
    s_ord, host, pix, plane = _assignment(np.flatnonzero(S), np.flatnonzero(H), b, kr, key)
    if len(s_ord) == 0:
        res['S'] = S; return res
    hh = np.repeat(host[:, None], pb, 1)
    ext = ((blocks[hh, pix].astype(np.int32) >> plane) & 1).astype(np.uint8)
    if ref_blocks is None:
        tags_now = mac_bits(blocks, s_ord, key)
    else:  # exact shortcut: recompute MACs only for blocks that differ from the reference
        tags_now = ref_tags[s_ord].copy()
        diff = np.flatnonzero(np.any(blocks[s_ord] != ref_blocks[s_ord], axis=1))
        if len(diff): tags_now[diff] = mac_bits(blocks, s_ord[diff], key)
    bad = np.any(ext[:, :KA] != tags_now, axis=1)
    flags[s_ord[bad]] = True
    rec = blocks.astype(float).copy()
    for k in (np.flatnonzero(bad) if kr else []):
        rec[s_ord[k]] = recovered_block(ext[k, KA:]).ravel()
    res.update(flags=flags, S=S, recovered=from_blocks(np.clip(np.round(rec), 0, 255).astype(np.uint8)))
    return res
