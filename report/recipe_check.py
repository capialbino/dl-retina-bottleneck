# Run from the project root: python3 report/recipe_check.py
# Recipe-fidelity check on the CIFAR-10 control: which of our unstated choices (input centring, flip augmentation, best-val checkpoint) kills retina channels?
import sys, json, time, itertools
import os, json as _j
os.environ.setdefault('HF_HUB_OFFLINE', '1')
import matplotlib; matplotlib.use('Agg')
_nb = _j.load(open('retina_bottleneck.ipynb')); by_id = {c['id']: c for c in _nb['cells']}
ns = {'__name__': '__main__'}
def run(cid, cut=None):
    src = ''.join(by_id[cid]['source']); src = src[:src.index(cut)] if cut else src
    exec(compile(src, f'cell-{cid}', 'exec'), ns)
SETUP = ['e1e9680f', 'b6fa8b64', '3a5fc4e7', 'ff802086', 'ddf081a0']   # config, data, model, RF probe, labeller
CTRL_RUN = 'aea3507b'
for c in SETUP: run(c)
run(CTRL_RUN, cut='CTRL_CKPT.mkdir(exist_ok=True)')
globals().update(ns)
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, pandas as pd

class PaperNetV(PaperNet):
    def __init__(self, *a, centre=True, **k):
        super().__init__(*a, **k); self.centre = centre
    def forward(self, x):
        x = (x - 0.5) / 0.25 if self.centre else x
        return self.head(self.cortex(self.retina(x)))

def rfs(model, images, centre):
    n_conv = sum(isinstance(m, nn.Conv2d) for m in model.retina); c, h = 16, n_conv * 4
    x = images.clone().requires_grad_(True)
    out = model.retina((x - 0.5) / 0.25 if centre else x)
    return np.stack([torch.autograd.grad(out[:, ch, c, c].sum(), x, retain_graph=True)[0].mean(0).mean(0)[c-h:c+h+1, c-h:c+h+1].cpu().numpy() for ch in range(out.shape[1])])

def run_variant(b, cl, seed, centre, flip):
    set_seed(seed)
    (x_tr, y_tr), (x_te, y_te) = cifar_splits()
    n_val = len(y_tr) // 10
    perm = torch.randperm(len(y_tr), generator=torch.Generator().manual_seed(seed)); tr, va = perm[n_val:], perm[:n_val]
    train_dl, val_dl, test_dl = GPULoader(x_tr[tr], y_tr[tr], True, flip=flip), GPULoader(x_tr[va], y_tr[va], False), GPULoader(x_te, y_te, False)
    model = PaperNetV(b, cl, centre=centre).to(DEVICE); opt = make_optimizer(model)
    best, best_state, hist = -1, None, []
    for ep in range(EPOCHS):
        model.train()
        for x, y in train_dl:
            loss = F.cross_entropy(model(x), y); opt.zero_grad(); loss.backward(); opt.step()
        acc = evaluate(model, val_dl); hist.append(acc)
        if acc > best: best, best_state = acc, {k: v.detach().clone() for k, v in model.state_dict().items()}
    res = dict(b=b, cl=cl, seed=seed, centre=centre, flip=flip, val_curve=[round(a, 3) for a in hist])
    nat = cifar_splits()[1][0][torch.randperm(10000, generator=torch.Generator().manual_seed(0))[:256]].float().div(255).to(DEVICE)
    for which, state in (('final', {k: v.detach().clone() for k, v in model.state_dict().items()}), ('best', best_state)):
        model.load_state_dict(state); model.eval()
        r = rfs(model, nat, centre); live = [orientation_index(x / np.abs(x).max())[0] for x in r if np.abs(x).max() > 1e-10]
        res[which] = dict(val=round(evaluate(model, val_dl), 4), test=round(evaluate(model, test_dl), 4), dead=int(sum(np.abs(x).max() <= 1e-10 for x in r)), channels=len(r),
                          osi_median=round(float(np.median(live)), 2) if live else None)
    return res

out = open('results/recipe_check.jsonl', 'a')
for b, cl, centre, flip in [(32, 2, True, True), (32, 2, False, True), (32, 2, False, False), (4, 2, False, False)]:
    t0 = time.time(); r = run_variant(b, cl, 0, centre, flip); r['time_s'] = round(time.time() - t0)
    out.write(json.dumps(r) + '\n'); out.flush(); print(r, flush=True)
