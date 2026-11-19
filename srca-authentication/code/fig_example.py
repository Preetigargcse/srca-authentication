import os
from paths import LOSS, RESULTS, FIGS, DATA
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.colors import ListedColormap
from srca_core import *
from data_attacks import load_pool, sample_attack
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9})

# ---------------- Figure 2: qualitative example
pool = load_pool({'natural': 40, 'biomedical': 60}, seed=0)
P = [p for p in pool if p['source'] == 'retina'][3]
img = P['img']; w = relevance_map(img)
lam = (0.25, 0.3, 2, 24)
cache = {}
wm, meta = embed(img, w, lam, cache=cache)
rng = np.random.default_rng(7)
f, kinds = sample_attack(w, rng, donor=pool[5]['img'])
rx = f(wm); res = verify(rx)
n = img.shape[0] // BS
T = np.any(to_blocks(rx) != to_blocks(wm), 1)
cls = np.zeros(n*n); cls[meta['H']] = 1; cls[meta['S']] = 2; cls[border_mask(n).ravel()] = 3
up = lambda a: np.kron(a.reshape(n, n), np.ones((BS, BS)))
fig, axs = plt.subplots(1, 6, figsize=(13, 2.6))
axs[0].imshow(img, cmap='gray', vmin=0, vmax=255); axs[0].set_title('(a) host X')
im = axs[1].imshow(up(w), cmap='magma', vmin=0, vmax=1); axs[1].set_title('(b) relevance w')
axs[2].imshow(up(cls), cmap=ListedColormap(['#dddddd', '#4c9be8', '#e8774c', '#555555']), vmin=0, vmax=3)
axs[2].set_title('(c) S (orange), H (blue)')
axs[3].imshow(rx, cmap='gray', vmin=0, vmax=255); axs[3].set_title('(d) tampered Y')
ov = np.zeros((n*n, 3)); ov[T & res['flags']] = [0.1, 0.7, 0.2]; ov[T & ~res['flags']] = [0.9, 0.1, 0.1]; ov[~T & res['flags']] = [1, 0.8, 0]
axs[4].imshow(rx, cmap='gray', vmin=0, vmax=255); axs[4].imshow(np.kron(ov.reshape(n, n, 3), np.ones((BS, BS, 1))), alpha=0.55)
axs[4].set_title('(e) flags: TP green,\nmiss red, false alarm yellow', fontsize=8)
axs[5].imshow(res['recovered'], cmap='gray', vmin=0, vmax=255); axs[5].set_title('(f) coarse recovery')
for a in axs: a.set_xticks([]); a.set_yticks([])
plt.tight_layout(); plt.savefig(os.path.join(FIGS, 'fig2_example.png'), dpi=200); plt.close()
wpsnr = 10*np.log10(255**2 / max((w.ravel()*((to_blocks(wm).astype(float)-to_blocks(img))**2).mean(1)).sum()/w.sum(), 1e-9))
print('example', kinds, 'prot', meta['n_prot'], 'feasible', meta['feasible'], 'TP', (T & res['flags']).sum(),
      'miss blocks', (T & ~res['flags'] & meta['S']).sum(), 'FA', (~T & res['flags']).sum(),
      'PSNR %.2f' % (10*np.log10(255**2/np.mean((wm.astype(float)-img)**2))), 'wPSNR %.2f' % wpsnr)
