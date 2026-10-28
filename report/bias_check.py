# Run from the project root: python3 report/bias_check.py <keras22|tf2> <bottlenecks, e.g. 32,1>
# Bias-initialisation check on the CIFAR-10 control: the paper draws weights AND biases from Glorot-uniform, my recipe uses zero biases.
# Paper architecture, grayscale CIFAR-10 32x32, cortex depth 2, seed 0; dead channels under three probes and OSI of the live ones.
import json, os, sys, time
os.environ.setdefault('MPLBACKEND', 'Agg')
nb = {c['id']: ''.join(c['source']) for c in json.load(open('retina_bottleneck.ipynb'))['cells']}
G = {'__name__': '__main__'}
for i in ['e1e9680f', 'b6fa8b64', '3a5fc4e7', '9519730b', 'ff802086', 'ddf081a0', 'aea3507b', 'b21d0a02']:  # setup, control, bridge loaders
    exec(compile(nb[i], i, 'exec'), G)
import numpy as np, torch, torch.nn as nn
from pathlib import Path
OUT, CK = Path('results/bias_check.jsonl'), Path('checkpoints_bias')
CK.mkdir(exist_ok=True)
variant, bns = sys.argv[1], [int(b) for b in sys.argv[2].split(',')]

def bias_limit(n):
    # Keras 2.2 glorot_uniform on a 1-D shape: fan_in = fan_out = sqrt(n) -> sqrt(3 / sqrt(n)); tf.keras (TF 2): fan_in = fan_out = n -> sqrt(3 / n)
    return np.sqrt(3 / np.sqrt(n)) if variant == 'keras22' else np.sqrt(3 / n)

def glorot_bias_net(b):
    model = G['PaperNet'](b, 2)
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            lim = bias_limit(m.bias.numel())
            nn.init.uniform_(m.bias, -lim, lim)
    return model

nat = G['sample_probe'](G['cifar_splits']()[1][0])
probes = {'natural': nat, 'gray': torch.full((1, 1, 32, 32), 0.5, device=G['DEVICE']), 'black': torch.zeros((1, 1, 32, 32), device=G['DEVICE'])}
for b in bns:
    t0 = time.time()
    # control networks of the paper recipe with Glorot biases; same collapse rule as the sweep
    (model, val, test), first = G['with_rerun'](lambda lr: G['fit'](0, lambda: G['bridge_loaders']('control', 0), lambda: glorot_bias_net(b), lr), 1e-3)
    torch.save(model.state_dict(), CK / f'{variant}_b{b}_s0.pt')
    model.eval()
    res = dict(variant=variant, bottleneck=b, seed=0, val_acc=val, test_acc=test, first_val_acc=first, time_s=time.time() - t0)
    for probe, imgs in probes.items():
        st = [G['filter_stats'](rf) for rf in G['retina_rfs'](model, imgs, size=32)]
        live = [s for s in st if s['label'] != 'dead']
        res[f'{probe}_live'] = len(live)
        res[f'{probe}_osi'] = [round(float(s['osi']), 2) for s in live]
        res[f'{probe}_labels'] = [s['label'] for s in live]
    with open(OUT, 'a') as f:
        f.write(json.dumps(res) + '\n')
    print(res, flush=True)
