# Execute notebook code cells by id in one namespace (used to run the long training cells detached).
# Usage from the project root: python3 runs/nbexec.py <cell ids, comma-separated> [NAME=python-expression ...] [--then <cell ids>]
import json, os, sys
os.environ.setdefault('MPLBACKEND', 'Agg')
cells = {c['id']: ''.join(c['source']) for c in json.load(open('retina_bottleneck.ipynb'))['cells']}
G = {'__name__': '__main__', 'display': print}
args = sys.argv[1:]
first, rest = args[0].split(','), args[1:]
then = []
if '--then' in rest:
    i = rest.index('--then'); then, rest = rest[i + 1].split(','), rest[:i]
for cid in first:
    exec(compile(cells[cid], cid, 'exec'), G)
for kv in rest:
    k, v = kv.split('=', 1); G[k] = eval(v, G)
for cid in then:
    exec(compile(cells[cid], cid, 'exec'), G)
print('done', flush=True)
