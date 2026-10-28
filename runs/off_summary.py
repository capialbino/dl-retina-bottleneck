# Print the official-recipe tables in full (run from the project root after the official runs).
import pandas as pd, numpy as np
from pathlib import Path
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
D = Path('results/official')
f = pd.concat([pd.read_csv(p) for p in sorted(D.glob('filters_*.csv'))], ignore_index=True)
r = pd.concat([pd.read_csv(p) for p in sorted(D.glob('runs_*.csv'))], ignore_index=True)
print(r.groupby(['dataset', 'bottleneck', 'cortex_layers']).test_acc.agg(['mean', 'std', 'count']).round(3))
def s(g):
    live = g.loc[~g.zero_grad, 'osi_official']
    return pd.Series(dict(n=len(g), zero=int(g.zero_grad.sum()), osi=live.mean(), ci=1.96 * live.std() / np.sqrt(max(len(live), 1)), lt_noise=int((live < 1.925).sum()),
                          mine=g.loc[~g.zero_grad, 'osi_mine'].mean() if 'osi_mine' in g and g.osi_mine.notna().any() else np.nan))
print(f.groupby(['dataset', 'bottleneck', 'cortex_layers', 'layer', 'probe']).apply(s, include_groups=False).round(2).to_string())
print(f[(f.layer == 'retina') & (f.probe == 'natural')].groupby(['dataset', 'bottleneck', 'cortex_layers']).label.value_counts().unstack(fill_value=0))
