"""Figures and tables of the report, generated from the CSVs in results/ and from the trained checkpoints.
Run from the project root:  python3 report/make_figures.py
Colours: first three slots of a validated categorical palette (blue, orange, aqua), RdBu for receptive fields."""
import json, os
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

BLUE, ORANGE, AQUA, INK = '#2a78d6', '#eb6834', '#1baf7a', '#0b0b0b'
plt.rcParams.update({'font.size': 8, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': '#52514e', 'axes.labelcolor': INK,
                     'xtick.color': '#52514e', 'ytick.color': '#52514e', 'axes.grid': True, 'grid.color': '#e6e5e0', 'grid.linewidth': 0.6,
                     'axes.axisbelow': True, 'legend.frameon': False, 'figure.dpi': 150, 'savefig.bbox': 'tight'})
OUT = 'report/figs'; os.makedirs(OUT, exist_ok=True)
DS = ['stl10', 'imagenette']; NAME = {'stl10': 'STL-10', 'imagenette': 'Imagenette'}
KEY = ['dataset', 'bottleneck', 'retina_layers', 'cortex_layers', 'seed']

# ---- data -------------------------------------------------------------------------------------------------------------
raw, rr = pd.read_csv('results/results.csv'), pd.read_csv('results/reruns.csv')
sweep = pd.concat([raw.merge(rr[KEY], on=KEY, how='left', indicator=True).query("_merge == 'left_only'").drop(columns='_merge'), rr.drop(columns='lr')], ignore_index=True)
dg = pd.read_csv('results/depth_grid.csv')
cols = KEY + ['test_acc']
acc = pd.concat([sweep[sweep.retina_layers == 2][cols], dg[cols]])
lin = pd.read_csv('results/linearity_info_layer1.csv'); lin = lin[lin.retina_layers == 2].merge(acc, on=KEY)
filt, ctrl_f = pd.read_csv('results/filters.csv'), pd.read_csv('results/control_filters.csv')
ctrl = pd.read_csv('results/control_cifar10.csv')

def savefig(name): plt.savefig(f'{OUT}/{name}.pdf'); plt.close()

# ---- Fig 1: accuracy vs depth and width ------------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4))
for ax, ds in zip(axes[:2], DS):
    for b, col in zip([1, 4, 32], [BLUE, ORANGE, AQUA]):
        g = lin[(lin.dataset == ds) & (lin.bottleneck == b)].groupby('cortex_layers').test_acc.agg(['mean', 'std'])
        ax.errorbar(g.index, g['mean'], g['std'], marker='o', ms=4, lw=1.5, capsize=2, color=col, label=f'bottleneck {b}')
    ax.set(title=NAME[ds], xlabel='cortex depth', xticks=[1, 2, 3, 4], ylabel='test accuracy' if ds == 'stl10' else '')
axes[0].legend(loc='lower right')
for ds, col in zip(DS, [BLUE, ORANGE]):
    g = lin[(lin.dataset == ds) & (lin.cortex_layers == 2)].groupby('bottleneck').test_acc.agg(['mean', 'std'])
    axes[2].errorbar(g.index, g['mean'], g['std'], marker='o', ms=4, lw=1.5, capsize=2, color=col, label=NAME[ds])
axes[2].set(title='cortex depth 2', xlabel='bottleneck channels', xscale='log', xticks=[1, 2, 4, 8, 16, 32], xticklabels=[1, 2, 4, 8, 16, 32])
axes[2].minorticks_off(); axes[2].legend()
plt.tight_layout(); savefig('fig_accuracy')

# ---- Fig 2: linearity and information vs depth -----------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(5.8, 4.2), sharex=True)
for r, (m, lab) in enumerate([('linearity', 'linearity of the retinal response'), ('info_r2', 'information kept (local $R^2$)')]):
    for c, ds in enumerate(DS):
        for b, col in zip([1, 4, 32], [BLUE, ORANGE, AQUA]):
            g = lin[(lin.dataset == ds) & (lin.bottleneck == b)].groupby('cortex_layers')[m].agg(['mean', 'std'])
            axes[r, c].errorbar(g.index, g['mean'], g['std'], marker='o', ms=4, lw=1.5, capsize=2, color=col, label=f'bottleneck {b}')
        axes[r, c].set(title=NAME[ds] if r == 0 else '', ylabel=lab if c == 0 else '', xticks=[1, 2, 3, 4])
        axes[r, c].set_xlabel('cortex depth' if r == 1 else '')
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.02))
plt.tight_layout(rect=(0, 0.04, 1, 1)); savefig('fig_linearity')

# ---- Fig 3: retina vs first cortical layer OSI ------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.4), sharey=True)
for ax, ds in zip(axes, DS):
    g = lin[(lin.dataset == ds) & (lin.cortex_layers == 2)].groupby('bottleneck')[['osi_retina_median', 'osi_cortex1_median']].mean()
    x = np.arange(len(g))
    ax.bar(x - 0.19, g.osi_retina_median, 0.36, color=BLUE, label='retina output', edgecolor='white'); ax.bar(x + 0.19, g.osi_cortex1_median, 0.36, color=ORANGE, label='first cortical layer', edgecolor='white')
    ax.axhline(1.56, ls='--', color=BLUE, lw=0.9); ax.axhline(3.05, ls=':', color=ORANGE, lw=1.1)
    ax.set(title=NAME[ds], xticks=x, xticklabels=g.index, xlabel='bottleneck channels'); ax.grid(False)
axes[0].set_ylabel('median OSI (live channels)')
axes[0].legend(loc='upper right', fontsize=6)
plt.tight_layout(); savefig('fig_layer1')

# ---- numbers printed for the text ---------------------------------------------------------------------------------------------
print('sweep label counts', filt.label.value_counts().to_dict(), 'live', int((filt.label != 'dead').sum()))
print('control accuracy by config'); print(ctrl.groupby(['bottleneck', 'cortex_layers']).test_acc.agg(['mean', 'std']).round(3))

# ---- LaTeX tables -----------------------------------------------------------------------------------------------------------------
def tex(df, name, **kw):
    open(f'report/{name}.tex', 'w').write(df.to_latex(escape=False, **kw))
# label counts of the retina channels, retina depth 2 and cortex depth 2
rows = []
for ds in DS:
    for b in [1, 2, 4, 8, 16, 32]:
        d = filt[(filt.dataset == ds) & (filt.bottleneck == b) & (filt.retina_layers == 2) & (filt.cortex_layers == 2)]; lv = d[d.label != 'dead']
        rows.append([NAME[ds], b, len(d), (d.label == 'center-surround').sum(), (d.label == 'oriented').sum(), (d.label == 'other').sum(), (d.label == 'dead').sum(), f'{lv.osi.median():.2f}', f'{(lv.osi < 2).sum()}/{len(lv)}'])
tex(pd.DataFrame(rows, columns=['Dataset', '$N_{BN}$', 'Channels', 'C-S', 'Orient.', 'Other', 'Dead', 'Median OSI', 'OSI$<$2']), 'tab_labels', index=False, column_format='lrrrrrrrr')
# positive control, natural-image probe
rows = []
for b in [1, 4, 32]:
    for cl in [2, 4]:
        a = ctrl[(ctrl.bottleneck == b) & (ctrl.cortex_layers == cl)]; f = ctrl_f[(ctrl_f.probe == 'natural') & (ctrl_f.bottleneck == b) & (ctrl_f.cortex_layers == cl)]; lv = f[f.label != 'dead']
        rows.append([b, cl, f'{a.test_acc.mean():.3f} $\\pm$ {a.test_acc.std():.3f}', int(a.first_val_acc.notna().sum()), f'{len(lv)}/{len(f)}', int((f.label == 'center-surround').sum()), f'{lv.osi.median():.2f}', f'{(lv.osi < 2).sum()}/{len(lv)}'])
tex(pd.DataFrame(rows, columns=['$N_{BN}$', 'Depth', 'Test acc.', 'Collapsed', 'Live/ch.', 'C-S', 'Median OSI', 'OSI$<$2']), 'tab_control', index=False, column_format='rrcrrrrr')

# bridge experiment: one factor at a time from the paper's setup (control) to mine (sweep); same rule as the notebook (Delta, 90% bootstrap over seeds)
bf = pd.read_csv('results/bridge_filters.csv')
BR = [('control', 'CIFAR-10', '45\\,000', 32, 'flatten + FC'), ('data', 'CIFAR-10', '4\\,500', 32, 'flatten + FC'), ('stl32', 'STL-10', '4\\,500', 32, 'flatten + FC'),
      ('stl32_pool', 'STL-10', '4\\,500', 32, 'pool + GAP'), ('sweep', 'STL-10', '4\\,500', 96, 'pool + GAP'), ('head', 'CIFAR-10', '45\\,000', 32, 'pool + GAP')]
def delta_median(d):
    nr, wd = d[d.bottleneck.isin([1, 4])].osi, d[d.bottleneck == 32].osi
    return wd.median() - nr.median() if len(nr) and len(wd) else np.nan
rng = np.random.default_rng(0); rows = []
for c, ds, n, res, head in BR:
    da, d = bf[bf.condition == c], bf[(bf.condition == c) & (bf.label != 'dead')]; seeds = sorted(da.seed.unique())
    bt = np.array([delta_median(pd.concat([d[d.seed == s] for s in rng.choice(seeds, len(seeds))])) for _ in range(2000)]); lo, hi = np.nanpercentile(bt, [5, 95])
    b1, w = d[d.bottleneck == 1], da[da.bottleneck == 32]
    rows.append([c.replace('_', '\\_'), ds, n, res, head, f'{(w.label == "dead").sum()}/{len(w)}', f'{b1.osi.median():.2f} ({(b1.osi < 2).sum()}/{len(b1)})', f'{delta_median(d):.2f} [{lo:.2f}, {hi:.2f}]'.replace('-', '$-$')])
tex(pd.DataFrame(rows, columns=['Condition', 'Images', 'Train', 'Res.', 'Head', 'Dead, $N_{BN}$=32', 'OSI, $N_{BN}$=1 ($<$2)', '$\\Delta$ [90\\% interval]']), 'tab_bridge', index=False, column_format='llrrlrrr')
print('tables written')

# ---- official recipe (the authors' code settings): table and receptive-field montage -----------------------------------------
of = pd.concat([pd.read_csv(p) for p in sorted(Path('results/official').glob('filters_*.csv'))], ignore_index=True)
orun = pd.concat([pd.read_csv(p) for p in sorted(Path('results/official').glob('runs_*.csv'))], ignore_index=True)
OFF_NAME = dict(NAME, cifar10='CIFAR-10')
def osi_cell(d, layer, probe):
    g = d[(d.layer == layer) & (d.probe == probe)]; v = g.loc[~g.zero_grad, 'osi_official']
    if len(v) == 0:
        return f'-- (0/{len(g)})'
    ci = f' $\\pm$ {1.96 * v.std(ddof=0) / np.sqrt(len(v)):.2f}' if len(v) > 1 else ''  # 95% normal interval over channels, as in the registered rule
    return f'{v.mean():.2f}{ci} ({len(v)}/{len(g)})'
rows = []
for ds, cls in [('cifar10', (2, 4)), ('stl10', (2,)), ('imagenette', (2,))]:
    for cl in cls:
        for b in (1, 4, 32):
            d = of[(of.dataset == ds) & (of.bottleneck == b) & (of.cortex_layers == cl)]; a = orun[(orun.dataset == ds) & (orun.bottleneck == b) & (orun.cortex_layers == cl)]
            nat = d[(d.layer == 'retina') & (d.probe == 'natural')]
            acc_s = f'{a.test_acc.mean():.3f}' + (f' $\\pm$ {a.test_acc.std():.3f}' if len(a) > 1 else '') + ('' if len(a) == 3 else f' ({len(a)})')
            rows.append([OFF_NAME[ds], b, cl, acc_s, osi_cell(d, 'retina', 'black'), osi_cell(d, 'retina', 'natural'),
                         f'{(nat.label == "center-surround").sum()} / {(nat.label == "oriented").sum()}', osi_cell(d, 'cortex1', 'black')])
tex(pd.DataFrame(rows, columns=['Data', '$N_{BN}$', 'D', 'Test acc.', 'Retina, black', 'Retina, natural', 'C-S / or.', 'Layer 1, black']), 'tab_official', index=False, column_format='lrrcllcl')
print(pd.DataFrame(rows).to_string())

# receptive-field montage of the code-recipe networks (needs the checkpoints and the notebook functions)
nb = {c['id']: ''.join(c['source']) for c in json.load(open('retina_bottleneck.ipynb'))['cells']}
ns = {'__name__': '__main__'}
for cid in ['e1e9680f', 'b6fa8b64', '3a5fc4e7', 'ff802086', 'ddf081a0', 'aea3507b', '0ff1c1a2']:  # setup, control model and data, code recipe
    src = nb[cid]
    for stop in ('CTRL_CKPT.mkdir(exist_ok=True)', 'OFF_GROUPS = {'):  # definitions only, no training loops
        src = src[:src.index(stop)] if stop in src else src
    exec(compile(src, cid, 'exec'), ns)
import torch
ncol, rows = 8, []
for ds in ('cifar10', 'stl10', 'imagenette'):
    size = 32 if ds == 'cifar10' else ns['IMG_SIZE']
    nat = ns['sample_probe']((ns['cifar_splits']() if ds == 'cifar10' else ns['raw_splits'](ds))[1][0])
    for bn in (1, 4, 32):
        cfg = dict(bottleneck=bn, retina_layers=2, cortex_layers=2)
        model = ns['load_ckpt'](ns['off_model'](ds, cfg), f"checkpoints_official/{ns['ckpt_tag'](ds, cfg, 0)}.pt")
        rows.append((f'{OFF_NAME[ds]}, bottleneck {bn}', ns['centre_crop'](ns['layer_rfs'](model, nat, size, centre=False), 2)))
fig, axes = plt.subplots(len(rows), ncol, figsize=(7.0, 0.72 * len(rows)))
for r, (title, rf) in enumerate(rows):
    liv = [x for x in rf if np.abs(x).max() > 1e-10][:ncol]
    for c in range(ncol):
        ax = axes[r, c]; ax.axis('off')
        if c < len(liv):
            m = np.abs(liv[c]).max(); ax.imshow(liv[c], cmap='RdBu', vmin=-m, vmax=m)
    axes[r, 0].text(-0.15, 0.5, title, transform=axes[r, 0].transAxes, ha='right', va='center', fontsize=7)
    axes[r, ncol - 1].text(1.1, 0.5, f'{sum(np.abs(x).max() > 1e-10 for x in rf)} of {len(rf)} live' + (f' (first {ncol} shown)' if len(liv) == ncol else ''), transform=axes[r, ncol - 1].transAxes, ha='left', va='center', fontsize=6, color='#52514e')
plt.subplots_adjust(left=0.30, right=0.88, wspace=0.05, hspace=0.12)
savefig('fig_rfs_official')
print('official recipe written')
