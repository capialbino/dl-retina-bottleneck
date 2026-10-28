# Execute notebook cells by id in a Jupyter kernel; only the outputs of the --store cells are written back to the notebook.
# Usage from the project root: python3 runs/nbstore.py <setup cell ids, comma-separated> --store <cell ids>
import copy, sys
import nbformat
from nbclient import NotebookClient
PATH = 'retina_bottleneck.ipynb'
nb = nbformat.read(PATH, as_version=4)
idx = {c.id: i for i, c in enumerate(nb.cells)}
setup, store = sys.argv[1].split(','), sys.argv[sys.argv.index('--store') + 1].split(',')
client = NotebookClient(nb, timeout=None, kernel_name='python3', resources={'metadata': {'path': '.'}})
with client.setup_kernel():
    for cid in setup:
        client.execute_cell(copy.deepcopy(nb.cells[idx[cid]]), idx[cid])
    for cid in store:
        client.execute_cell(nb.cells[idx[cid]], idx[cid])
        print('stored', cid, flush=True)
nbformat.write(nb, PATH)
