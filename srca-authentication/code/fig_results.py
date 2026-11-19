import os
from paths import LOSS, RESULTS, FIGS, DATA
import json, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.color': '#e6e6e6', 'grid.linewidth': 0.6, 'axes.axisbelow': True})
C = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4']
INK = '#333333'
r = json.load(open(os.path.join(RESULTS, 'results.json')))
L = r['landscape']; grid = [tuple(g) for g in L['grid']]
miss, infe, diag = np.array(L['miss']), np.array(L['infe']), np.array(L['diag'])
taus = sorted(set(g[0] for g in grid))

# ---- Fig 3: risk landscape
fig, axs = plt.subplots(1, 3, figsize=(11, 3.2))
for ax, (b, rr, ttl) in zip(axs, [(1, 0, '(a) b = 1, r = 0'), (2, 0, '(b) b = 2, r = 0'), (1, 24, '(c) b = 1, r = 24')]):
    for k, om in enumerate([0.1, 0.2, 0.3, 0.4]):
        idx = [grid.index((t, om, b, rr)) for t in taus]
        ax.plot(taus, miss[idx], '-o', color=C[k], lw=2, ms=4, label=f'ω = {om}')
        ax.plot(taus, infe[idx], ':', color=C[k], lw=1.6)
    ax.axhline(0.15, color=INK, lw=0.9, ls='--'); ax.text(0.8, 0.165, 'α_M = 0.15', ha='right', fontsize=8, color=INK)
    ax.set_title(ttl); ax.set_xlabel('protection threshold τ'); ax.set_ylim(-0.02, 1.02)
axs[0].set_ylabel('mean risk (solid: miss, dotted: infeasible)')
axs[0].legend(frameon=False, fontsize=8, loc='upper left')
plt.tight_layout(); plt.savefig(os.path.join(FIGS, 'fig3_landscape.png'), dpi=200); plt.close()

# ---- Fig 4: test-risk distributions over 300 splits
S = r['main_samples']
names = ['SRCA', 'Plug-in', 'Fixed-H2', 'Fixed-H1', 'Full-fragile']
fig, axs = plt.subplots(1, 2, figsize=(10, 3.3))
for j, (ax, lab, tgt) in enumerate(zip(axs, ['test miss risk', 'test diagnostic-violation rate'], [0.15, 0.10])):
    data = [[x[j] for x in S[n] if x is not None] for n in names]
    bp = ax.boxplot(data, orientation='horizontal', widths=0.55, patch_artist=True, showfliers=False)
    for p, c in zip(bp['boxes'], [C[0], C[1], C[2], C[3], C[4]]):
        p.set_facecolor(c); p.set_alpha(0.75); p.set_edgecolor(INK)
    for m in bp['medians']: m.set_color(INK)
    ax.axvline(tgt, color=INK, ls='--', lw=1); ax.text(tgt, 5.55, f' target {tgt}', fontsize=8, color=INK)
    ax.set_yticks(range(1, 6)); ax.set_yticklabels(names if j == 0 else [''] * 5); ax.set_xlabel(lab)
plt.tight_layout(); plt.savefig(os.path.join(FIGS, 'fig4_splits.png'), dpi=200); plt.close()

# ---- Fig 5: alpha sweep + calibration size
A = r['alpha_sweep']; al = sorted(float(a) for a in A)
fig, axs = plt.subplots(1, 2, figsize=(10, 3.3))
ax = axs[0]
for k, (n, lab) in enumerate([('SRCA', 'SRCA'), ('Plug-in', 'Plug-in')]):
    xs, ys, ab = [], [], []
    for a in al:
        s = A[str(a)][n]
        if s.get('abstain', 0) >= 0.5: ab.append(a); continue
        xs.append(a); ys.append(s['miss'][0])
    ax.plot(xs, ys, '-o', color=C[k], lw=2, ms=5, label=lab)
    for a in ab: ax.plot(a, 0.01, 'x', color=C[k], ms=7)
ax.plot([0.07, 0.31], [0.07, 0.31], color=INK, lw=0.9, ls='--'); ax.text(0.29, 0.3, 'risk = α', fontsize=8, ha='right', va='bottom', color=INK)
ax.set_xlabel('miss-risk target α_M'); ax.set_ylabel('mean test miss risk'); ax.set_title('(a) achieved vs target (× = SRCA abstains)')
ax.legend(frameon=False, fontsize=8)
ax = axs[1]
NC = r['ncal_sweep']; ns = sorted(int(n) for n in NC)
ab = [NC[str(n)]['abstain'] for n in ns]
ax.bar([str(n) for n in ns], ab, color=C[0], width=0.6)
for i, n in enumerate(ns):
    s = NC[str(n)]
    txt = f"{s['miss'][0]:.3f}" if s['abstain'] < 1 else None
    ax.text(i, ab[i] + 0.03, f'miss {txt}' if txt else 'always abstains', ha='center', fontsize=8, color=INK)
ax.set_ylim(0, 1.15); ax.set_xlabel('calibration size n'); ax.set_ylabel('abstention rate'); ax.set_title('(b) abstention vs n (α_M = 0.15)')
plt.tight_layout(); plt.savefig(os.path.join(FIGS, 'fig5_sweeps.png'), dpi=200); plt.close()

# ---- Fig 6: shift
Sh = r['shift']
rows = [('nat→nat', 'nat->nat'), ('nat→bio', 'nat->bio'), ('bio→nat', 'bio->nat'), ('targeted→uniform', 'targeted->uniform')]
fig, ax = plt.subplots(figsize=(6.2, 2.8))
vals = [Sh[k]['SRCA']['miss'][0] for _, k in rows]
err = [Sh[k]['SRCA']['miss'][1] for _, k in rows]
cols = [C[0], C[0], C[0], C[1]]
ax.barh([l for l, _ in rows][::-1], vals[::-1], xerr=err[::-1], color=cols[::-1], height=0.55, error_kw=dict(ecolor=INK, lw=0.8))
ax.axvline(0.15, color=INK, ls='--', lw=1); ax.text(0.152, 3.35, 'α_M', fontsize=8, color=INK)
for i, (v, e) in enumerate(zip(vals[::-1], err[::-1])): ax.text(v + e + 0.004, i, f'{v:.3f}', va='center', fontsize=8, color=INK)
ax.set_xlabel('mean test miss risk of SRCA (calibration → deployment)'); ax.set_xlim(0, 0.26)
plt.tight_layout(); plt.savefig(os.path.join(FIGS, 'fig6_shift.png'), dpi=200); plt.close()
print('ok')
