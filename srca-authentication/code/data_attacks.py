"""Patch pool (bundled scikit-image test images) and the tampering threat model."""
import numpy as np
import skimage.data as d
from skimage.color import rgb2gray
from skimage.transform import rescale
from scipy import ndimage as ndi
from srca_core import BS

N = 256

def _gray(im):
    if im.ndim == 3: im = rgb2gray(im[..., :3])
    im = im.astype(float)
    if im.max() <= 1.0: im = im * 255
    return np.clip(im, 0, 255)

SOURCES = {
    'natural': ['camera', 'astronaut', 'coffee', 'chelsea', 'coins', 'moon', 'hubble_deep_field',
                'rocket', 'brick', 'grass', 'gravel', 'clock', 'colorwheel'],
    'biomedical': ['retina', 'immunohistochemistry', 'cell', 'shepp_logan_phantom'],
}

def load_pool(n_per_source=None, seed=0):
    rng = np.random.default_rng(seed)
    pool = []
    for dom, names in SOURCES.items():
        for name in names:
            im = _gray(getattr(d, name)())
            if name == 'retina': im = rescale(im, 0.5, anti_aliasing=True)
            if name == 'shepp_logan_phantom':
                im = rescale(im, 1.3, anti_aliasing=True)
                im = im + rng.normal(0, 2.0, im.shape)          # acquisition-like noise
            k = n_per_source[dom] if n_per_source else 40
            for t in range(k):
                y = rng.integers(0, im.shape[0] - N + 1); x = rng.integers(0, im.shape[1] - N + 1)
                p = im[y:y+N, x:x+N]
                p = np.rot90(p, rng.integers(4))
                if rng.random() < 0.5: p = p[:, ::-1]
                pool.append(dict(img=np.clip(np.round(p), 0, 255).astype(np.uint8), domain=dom, source=name))
    return pool

# ------------------------------------------------------------------ threat model
ATTACKS = ['copy_move', 'splice', 'inpaint_blur', 'lesion_inject', 'lesion_remove']

def sample_attack(w, rng, donor=None, targeted=1.0, area=(0.01, 0.08), n_regions=(1, 3)):
    """Returns a function f(img)->tampered img.  Region centres are drawn with probability
    proportional to w^(2*targeted) (targeted=0 -> uniform over interior blocks)."""
    n = w.shape[0]
    wf = w.ravel().copy()
    p = np.where(wf > 0, wf, 0) ** (2 * targeted) if targeted > 0 else (wf > 0).astype(float)
    p = p + 1e-12 * (wf > 0); p = p / p.sum()
    k = rng.integers(n_regions[0], n_regions[1] + 1)
    ops = []
    for _ in range(k):
        j = rng.choice(n*n, p=p)
        cy, cx = (j // n) * BS + BS // 2, (j % n) * BS + BS // 2
        a = rng.uniform(*area) * N * N
        asp = rng.uniform(0.6, 1.6)
        h = int(np.clip(np.sqrt(a * asp), 6, N // 2)); ww = int(np.clip(a / max(h, 1), 6, N // 2))
        y0 = int(np.clip(cy - h // 2, 0, N - h)); x0 = int(np.clip(cx - ww // 2, 0, N - ww))
        kind = ATTACKS[rng.integers(len(ATTACKS))]
        par = dict(y0=y0, x0=x0, h=h, w=ww, kind=kind,
                   sy=int(rng.integers(0, N - h)), sx=int(rng.integers(0, N - ww)),
                   amp=float(rng.uniform(6, 30)) * (1 if rng.random() < 0.5 else -1),
                   sig=float(rng.uniform(2.0, 4.0)))
        ops.append(par)
    def f(img):
        out = img.astype(float).copy()
        for o in ops:
            y0, x0, h, ww = o['y0'], o['x0'], o['h'], o['w']
            sl = (slice(y0, y0+h), slice(x0, x0+ww))
            if o['kind'] == 'copy_move':
                out[sl] = img[o['sy']:o['sy']+h, o['sx']:o['sx']+ww]
            elif o['kind'] == 'splice' and donor is not None:
                out[sl] = donor[o['sy']:o['sy']+h, o['sx']:o['sx']+ww]
            elif o['kind'] == 'splice':
                out[sl] = img[o['sy']:o['sy']+h, o['sx']:o['sx']+ww][::-1, ::-1]
            elif o['kind'] == 'inpaint_blur':
                pad = int(4 * o['sig'] * 2) + 1
                Y0, X0 = max(y0 - pad, 0), max(x0 - pad, 0)
                crop = ndi.gaussian_filter(out[Y0:y0+h+pad, X0:x0+ww+pad], o['sig'] * 2)
                out[sl] = crop[y0-Y0:y0-Y0+h, x0-X0:x0-X0+ww]
            else:  # lesion inject / remove: smooth elliptical blob added / subtracted
                yy, xx = np.mgrid[0:h, 0:ww]
                g = np.exp(-(((yy - h/2) / (h/3))**2 + ((xx - ww/2) / (ww/3))**2))
                amp = abs(o['amp']) * (1 if o['kind'] == 'lesion_inject' else -1)
                out[sl] = out[sl] + amp * g
        return np.clip(np.round(out), 0, 255).astype(np.uint8)
    return f, [o['kind'] for o in ops]
