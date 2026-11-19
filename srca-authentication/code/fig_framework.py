"""Redesigned Figure 1: SRCA framework with real thumbnails (offline calibration lane on top,
online embedding / verification lane below)."""
import os
from paths import LOSS, RESULTS, FIGS, DATA
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.colors import ListedColormap
from srca_core import *
from data_attacks import load_pool, sample_attack

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7.5, 'mathtext.fontset': 'dejavusans'})
INK, MUTED = '#222222', '#666666'
BLUE, ORANGE, AQUA, YELLOW = '#2a78d6', '#eb6834', '#1baf7a', '#eda100'
OFF_BG, ON_BG = '#f4f7fb', '#fbf6f1'

# ---------- thumbnails from a real retinal patch
pool = load_pool({'natural': 40, 'biomedical': 60}, seed=0)
P = [p for p in pool if p['source'] == 'retina'][3]
img = P['img']; w = relevance_map(img); n = img.shape[0] // BS
lam = (0.15, 0.2, 1, 0)
wm, meta = embed(img, w, lam)
f, _ = sample_attack(w, np.random.default_rng(7), donor=pool[5]['img'])
rx = f(wm); res = verify(rx)
T = np.any(to_blocks(rx) != to_blocks(wm), 1)
up = lambda a: np.kron(a.reshape(n, n), np.ones((BS, BS)))
cls = np.zeros(n*n); cls[meta['H']] = 1; cls[meta['S']] = 2; cls[border_mask(n).ravel()] = 3
ov = np.zeros((n*n, 4))
ov[T & res['flags']] = [0.10, 0.69, 0.48, 0.9]; ov[T & ~res['flags']] = [0.82, 0.18, 0.18, 0.95]
ov[~T & res['flags']] = [0.93, 0.63, 0.0, 0.9]
ovimg = np.kron(ov.reshape(n, n, 4), np.ones((BS, BS, 1)))


RED = '#d12f2f'
fig = plt.figure(figsize=(7.16, 4.0))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 56); ax.axis('off')
Y2X = 56 / 4.0 * 7.16 / 100   # y-units per x-unit so thumbnails are square

def lane(y0, h, title, fc):
    ax.add_patch(FancyBboxPatch((0.6, y0), 98.8, h, boxstyle='round,pad=0,rounding_size=1.0', fc=fc, ec='#c9ced6', lw=0.6))
    ax.text(2.0, y0 + h - 1.3, title, fontsize=8.2, weight='bold', color=INK, va='top')

def box(x, y, w_, h, head, body, ec, num):
    ax.add_patch(FancyBboxPatch((x, y), w_, h, boxstyle='round,pad=0,rounding_size=0.8', fc='white', ec=ec, lw=0.9))
    ax.text(x + w_/2, y + h - 1.4, head, ha='center', va='top', fontsize=7.4, weight='bold', color=INK)
    ax.text(x + w_/2, y + (h - 3.2)/2, body, ha='center', va='center', fontsize=6.6, color=MUTED, linespacing=1.35)
    ax.add_patch(plt.Circle((x + 0.2, y + h - 0.2), 1.25, color=ec, zorder=5))
    ax.text(x + 0.2, y + h - 0.25, str(num), ha='center', va='center', fontsize=6.6, color='white', weight='bold', zorder=6)

def arrow(p, q, color=INK, ls='-', rad=0.0, lw=1.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle='-|>', mutation_scale=8, color=color, lw=lw, ls=ls,
                                 connectionstyle=f'arc3,rad={rad}', shrinkA=0, shrinkB=0, zorder=4))

def thumb(x, y, s, im, label, cmap='gray', vmin=0, vmax=255, overlay=None):
    a = fig.add_axes([x/100, y/56, s/100, s*Y2X/56])
    a.imshow(im, cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest')
    if overlay is not None: a.imshow(overlay, interpolation='nearest')
    a.set_xticks([]); a.set_yticks([])
    for sp in a.spines.values(): sp.set_color('#999999'); sp.set_linewidth(0.5)
    ax.text(x + s/2, y - 1.0, label, ha='center', va='top', fontsize=6.7, color=INK)

# offline lane
lane(37.2, 18.2, 'Offline: risk-calibrated parameter selection (once per image source and threat model)', OFF_BG)
yb, hb = 39.0, 11.6
box(3.0, yb, 15.0, hb, 'Calibration set', '$n$ images $X_i$ with\nattacks $A_i\\sim\\mathcal{A}$', BLUE, 1)
box(22.0, yb, 18.0, hb, 'Run every $\\lambda\\in\\Lambda$', 'embed, tamper, verify\n$\\lambda=(\\tau,\\omega,b,r)$, $|\\Lambda|=384$', BLUE, 2)
box(44.0, yb, 17.0, hb, 'Three losses', '$L_M$ weighted miss\n$L_D$: $D>d_{\\max}$\n$L_F$: payload > capacity', BLUE, 3)
box(65.0, yb, 15.0, hb, 'Certify (LTT)', '$p_\\lambda=\\max(p^{HB}_M,p_D,p_F)$\n$p_\\lambda\\leq\\delta/|\\Lambda|$', BLUE, 4)
box(84.0, yb, 14.0, hb, 'Select', '$\\hat\\lambda=\\arg\\min\\,\\bar D$\nor abstain', BLUE, 5)
for x1, x2 in [(18.0, 22.0), (40.0, 44.0), (61.0, 65.0), (80.0, 84.0)]:
    arrow((x1, yb + hb/2 - 1.2), (x2, yb + hb/2 - 1.2), BLUE)

# online lane
lane(0.6, 35.4, 'Online: selective embedding and blind verification (per image)', ON_BG)
s, ty = 11.0, 14.0
th = s * Y2X
thumb(2.6, ty, s, img, 'host $X$')
thumb(16.2, ty, s, up(w), 'relevance $w$', cmap='magma', vmin=0, vmax=1)
thumb(29.8, ty, s, up(cls), 'block classes', cmap=ListedColormap(['#e6e6e6', BLUE, ORANGE, '#555555']), vmin=0, vmax=3)
ax.add_patch(FancyBboxPatch((43.2, ty), 12.6, th, boxstyle='round,pad=0,rounding_size=0.8', fc='white', ec=ORANGE, lw=0.9))
ax.text(49.5, ty + th - 1.0, 'Embed payload', ha='center', va='top', fontsize=7.2, weight='bold', color=INK)
ax.text(49.5, ty + (th - 3.0)/2, 'MAC (+ $r$ bits) of S\n$\\downarrow$\nkeyed slots in $b$\nLSBs of H;\nheader in border',
        ha='center', va='center', fontsize=6.1, color=MUTED, linespacing=1.3)
thumb(59.0, ty, s, wm, 'watermarked $\\tilde X$')
thumb(73.6, ty, s, rx, 'received $Y$')
thumb(87.0, ty, s, rx, 'verification flags', overlay=ovimg)
yc = ty + th/2
for x1, x2 in [(13.6, 16.0), (27.2, 29.6), (40.8, 43.0), (55.8, 58.8), (84.6, 86.8)]:
    arrow((x1, yc), (x2, yc))
arrow((70.0, yc), (73.4, yc)); ax.text(71.7, yc + 1.0, '$A$', ha='center', fontsize=7, color=MUTED, style='italic')
def legend(x, items):
    for i, (c, t) in enumerate(items):
        y = 7.6 - i * 2.2
        ax.add_patch(Rectangle((x, y), 1.4, 1.4 * Y2X, fc=c, ec='none'))
        ax.text(x + 2.0, y + 0.7 * Y2X, t, va='center', fontsize=6.3, color=INK)
legend(29.8, [(ORANGE, 'protected S (bit-exact)'), (BLUE, 'carriers H'), ('#555555', 'border header')])
legend(87.0, [(AQUA, 'detected'), (RED, 'missed'), (YELLOW, 'false alarm')])
ax.text(64.5, 10.2, 'only H and border\nLSBs are changed', ha='center', va='top', fontsize=6.0, color=MUTED, style='italic')
# deployed parameters
yl = 30.2
ax.plot([91.0, 91.0, 35.3], [yb, yl, yl], color=BLUE, ls='--', lw=1.1, zorder=4)
arrow((35.3, yl), (35.3, ty + th + 0.3), BLUE, ls='--', lw=1.1)
ax.text(63.0, yl, 'deployed $\\hat\\lambda$ and key $K$', color=BLUE, fontsize=7, ha='center', va='center',
        bbox=dict(fc=ON_BG, ec='none', pad=1.0), zorder=5)
plt.savefig(os.path.join(FIGS, 'fig1_framework.png'), dpi=300)
plt.savefig(os.path.join(FIGS, 'fig1_framework.pdf'))
print('ok')
