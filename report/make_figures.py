"""Figures and tables of the report, generated from the CSVs in results/ and from the trained checkpoints.
Run from the project root:  python3 report/make_figures.py
Colours: first three slots of a validated categorical palette (blue, orange, aqua), RdBu for receptive fields."""
import json, os, itertools
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

BLUE, ORANGE, AQUA, GRAY, INK = '#2a78d6', '#eb6834', '#1baf7a', '#9a9993', '#0b0b0b'
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

# ---- Fig 3: OSI of live retina channels: sweep (both datasets) and CIFAR-10 control ------------------------------------
live = filt[(filt.label != 'dead') & (filt.retina_layers == 2) & (filt.cortex_layers == 2)]
cl = ctrl_f[(ctrl_f.probe == 'natural') & (ctrl_f.label != 'dead') & (ctrl_f.cortex_layers == 2)]
panels = [(f'{NAME[d]} (sweep)', live[live.dataset == d], [1, 2, 4, 8, 16, 32]) for d in DS] + [('CIFAR-10 control (paper setup)', cl, [1, 4, 32])]
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5), sharey=True)
rng = np.random.default_rng(0)
for ax, (title, d, bns) in zip(axes, panels):
    for j, b in enumerate(bns):
        v = d[d.bottleneck == b].osi.values
        if len(v):
            ax.boxplot(v, positions=[j], widths=0.5, showfliers=False, boxprops=dict(color=BLUE), medianprops=dict(color=BLUE, lw=1.5), whiskerprops=dict(color=BLUE), capprops=dict(color=BLUE))
            ax.scatter(j + rng.uniform(-0.18, 0.18, len(v)), v, s=6, color=BLUE, alpha=0.45, lw=0, zorder=3)
        ax.text(j, 0.62, f'n={len(v)}', ha='center', fontsize=6, color='#52514e')
    ax.axhline(1.56, ls='--', color=ORANGE, lw=1); ax.axhline(3.05, ls=':', color=AQUA, lw=1.2)
    ax.set(title=title, xticks=range(len(bns)), xticklabels=bns, xlabel='bottleneck channels', yscale='log', ylim=(0.55, 16))
    ax.set_yticks([1, 2, 4, 8, 16]); ax.set_yticklabels([1, 2, 4, 8, 16]); ax.minorticks_off()
axes[0].set_ylabel('OSI of live retina channels')
axes[2].legend(handles=[Line2D([], [], color=ORANGE, ls='--', label='paper, retina output (1.56)'), Line2D([], [], color=AQUA, ls=':', label='paper, first cortical layer (3.05)')], loc='upper right', fontsize=6)
plt.tight_layout(); savefig('fig_osi')

# ---- Fig 4: label composition -----------------------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.4), sharey=True)
LAB = [('center-surround', BLUE), ('oriented', ORANGE), ('other', AQUA), ('dead', '#c9c8c2')]
for ax, ds in zip(axes, DS):
    d = filt[(filt.dataset == ds) & (filt.retina_layers == 2) & (filt.cortex_layers == 2)]
    c = d.groupby(['bottleneck', 'label']).size().unstack(fill_value=0).reindex(columns=[l for l, _ in LAB], fill_value=0)
    n = c.sum(axis=1); bottom = np.zeros(len(c))
    for lab, col in LAB:
        h = c[lab].values / n.values
        ax.bar(range(len(c)), h, bottom=bottom, color=col, label=lab, edgecolor='white', linewidth=1, width=0.72); bottom += h
    ax.set(title=NAME[ds], xticks=range(len(c)), xticklabels=[f'{b}\nn={k}' for b, k in zip(c.index, n.values)], xlabel='bottleneck channels')
    ax.tick_params(axis='x', labelsize=6)
    ax.grid(False)
axes[0].set_ylabel('share of retina channels')
axes[1].legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.tight_layout(); savefig('fig_labels')

# ---- Fig 5: retina vs first cortical layer OSI ------------------------------------------------------------------------
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

# ---- Fig 6: receptive-field montage (needs the trained models and the notebook functions) -----------------------------------
nb = json.load(open('retina_bottleneck.ipynb')); by = {c['id']: c for c in nb['cells']}
ns = {'__name__': '__main__'}
def run(cid, cut=None):
    s = ''.join(by[cid]['source']); s = s[:s.index(cut)] if cut else s
    exec(compile(s, cid, 'exec'), ns)
for cid in ['e1e9680f', 'b6fa8b64', '3a5fc4e7', 'ff802086', 'ddf081a0']: run(cid)
run('aea3507b', cut='CTRL_CKPT.mkdir(exist_ok=True)'); run('0c6f7040', cut='if CTRL_FILTERS_CSV.exists():')
import torch
rows = []
for ds, bn in [('imagenette', 1), ('imagenette', 4), ('imagenette', 32), ('stl10', 1), ('stl10', 4), ('stl10', 32)]:
    cfg = dict(bottleneck=bn, retina_layers=2, cortex_layers=2)
    rows.append((f'{NAME[ds]}, bottleneck {bn}', ns['retina_rfs'](ns['load_model'](ds, cfg, 0), ns['probe_images'](ds))))
cimgs = ns['cifar_splits']()[1][0][torch.randperm(10000, generator=torch.Generator().manual_seed(0))[:256]].float().div(255).to(ns['DEVICE'])
for bn in (1, 4, 32):
    rows.append((f'CIFAR-10 control, bottleneck {bn}', ns['retina_rfs'](ns['ctrl_model'](bn, 2, 0), cimgs, size=32)))
ncol = 8
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
savefig('fig_rfs')

# ---- numbers printed for the text ---------------------------------------------------------------------------------------------
print('sweep label counts', filt.label.value_counts().to_dict(), 'live', int((filt.label != 'dead').sum()))
print('control accuracy by config'); print(ctrl.groupby(['bottleneck', 'cortex_layers']).test_acc.agg(['mean', 'std']).round(3))
print('done')

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
# accuracy by cortex depth
rows = []
for ds in DS:
    for b in [1, 4, 32]:
        g = lin[(lin.dataset == ds) & (lin.bottleneck == b)].groupby('cortex_layers').test_acc.agg(['mean', 'std'])
        rows.append([NAME[ds], b] + [f'{g.loc[d, "mean"]:.3f} $\\pm$ {g.loc[d, "std"]:.3f}' for d in [1, 2, 3, 4]])
tex(pd.DataFrame(rows, columns=['Dataset', '$N_{BN}$', 'depth 1', 'depth 2', 'depth 3', 'depth 4']), 'tab_accuracy', index=False, column_format='lrcccc')
# positive control, natural-image probe
rows = []
for b in [1, 4, 32]:
    for cl in [2, 4]:
        a = ctrl[(ctrl.bottleneck == b) & (ctrl.cortex_layers == cl)]; f = ctrl_f[(ctrl_f.probe == 'natural') & (ctrl_f.bottleneck == b) & (ctrl_f.cortex_layers == cl)]; lv = f[f.label != 'dead']
        rows.append([b, cl, f'{a.test_acc.mean():.3f} $\\pm$ {a.test_acc.std():.3f}', int(a.first_val_acc.notna().sum()), f'{len(lv)}/{len(f)}', int((f.label == 'center-surround').sum()), f'{lv.osi.median():.2f}', f'{(lv.osi < 2).sum()}/{len(lv)}'])
tex(pd.DataFrame(rows, columns=['$N_{BN}$', 'Depth', 'Test acc.', 'Collapsed', 'Live/ch.', 'C-S', 'Median OSI', 'OSI$<$2']), 'tab_control', index=False, column_format='rrcrrrrr')
print('tables written')
