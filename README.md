# Retinal bottleneck in anatomically constrained CNNs

Deep Learning project (University of Bologna). A reduced-scale reproduction of

> J. Lindsey, S. A. Ocko, S. Ganguli, S. Deny. *A unified theory of early visual representations from retina to cortex through anatomically constrained deep CNNs.* ICLR 2019 ([arXiv:1901.00945](https://arxiv.org/abs/1901.00945)).

The paper's experiments are run on STL-10 and Imagenette (grayscale, 96×96), and on the paper's own setting (grayscale CIFAR-10) as a positive control. The project uses 3 seeds per configuration and a single laptop GPU.

**The report is [`report/report.pdf`](report/report.pdf)** (12 pages). All code is in the notebook [`retina_bottleneck.ipynb`](retina_bottleneck.ipynb).

## The question

The paper splits a CNN into a *retina* (two convolutions whose output has only $N_{BN}$ channels, the optic-nerve bottleneck) and a *cortex* of $D_{VVS}$ convolutions. It makes three claims:

1. A narrow bottleneck yields **center-surround** receptive fields in the retina and **oriented** ones in the first cortical layer. A wide bottleneck yields an oriented retina.
2. A deeper cortex makes the retina more **linear**: it keeps more information and extracts fewer features.
3. Accuracy grows with the bottleneck width and with the cortex depth.

The project tests these claims, and the assignment's hypotheses H1–H3, on STL-10 and Imagenette.

## Two training recipes

The paper's text leaves part of the training recipe unstated, most importantly the learning rate. Two recipes are therefore compared:

| | Paper recipe | Code recipe (main runs) |
|---|---|---|
| Source | text of the paper, Keras defaults for the gaps | [authors' code](https://github.com/ganguli-lab/RetinalResources), commit 3235142 |
| RMSProp learning rate | 1e-3 (Keras default) | 1e-4, decay 1e-6 |
| Batch size | 32 (as in the text) | 64 (as in the code) |
| Augmentation, input, checkpoint | flip, centred input, best validation epoch | shift + flip, uncentred input, last epoch |
| RF probe | natural images | black image (as in the code) + natural images |

## Main results (code recipe, 3 seeds)

| | CIFAR-10 (paper's setting) | STL-10 / Imagenette (96×96) |
|---|---|---|
| One-channel retina, OSI | **1.83 ± 0.38**, 3/3 center-surround (paper 1.56 ± 0.22) | **3.42 / 3.30** (natural probe), 0/6 center-surround |
| First cortical layer, OSI | **3.22 ± 0.62** (paper 3.05 ± 0.30) | oriented, but so is the retina |
| 32-channel retina | oriented (56/96 channels) | oriented |
| Accuracy, $N_{BN}$ = 1 → 32 | 0.676 → 0.705 | 0.409 → 0.508 / 0.473 → 0.595 |

- **CIFAR-10:** with the code recipe, the paper's central claim is reproduced in its own setting.
- **STL-10 and Imagenette:** with the same recipe and the paper's number of weight updates, the accuracy trend is reproduced but the center-surround retina is not.
- **Paper recipe:** the Keras-default learning rate kills most retinal channels (dying ReLU), even on CIFAR-10.

Details, the decision rule written before the main runs, and the limitations are in the report.

## Repository layout

```
retina_bottleneck.ipynb   all code: data, model, training, RF probes, OSI, labels, analyses, findings
requirements.txt          pinned Python dependencies
report/
  report.tex, report.pdf  the report
  make_figures.py         regenerates every figure (figs/) and table (tab_*.tex) from results/ and the checkpoints
  recipe_check.py         CIFAR-10 check: which unstated choice of the paper recipe kills retinal channels
  bias_check.py           CIFAR-10 check: Glorot-uniform biases (paper text) against zero biases (code)
  figs/, tab_*.tex        generated figures and tables (some are not used in the 12-page version)
runs/
  nbexec.py               runs notebook cells by id in a detached process (long training loops)
  nbstore.py              runs notebook cells in a Jupyter kernel and stores their outputs in the notebook
  check_osi.py            noise level of the ported OSI on Gaussian-noise RFs (paper: 1.97 ± 0.08)
  off_summary.py          prints the full code-recipe tables
results/
  official/               code recipe: runs_*.csv (accuracy), filters_*.csv (RF metrics per channel), linearity_info.csv
  official/pilot_20epochs first code-recipe runs with the code's 20 epochs, undertrained on the small datasets; superseded by the update-matched runs
  results.csv, filters*.csv           paper recipe: width × depth sweep on STL-10 and Imagenette
  depth_grid.csv, linearity_info_layer1.csv   paper recipe: cortex-depth grid, linearity, information, first layer
  control_*.csv           paper recipe: CIFAR-10 positive control
  bridge/, bridge_filters.csv         paper recipe: bridge experiment between CIFAR-10 and STL-10
  recipe_check.jsonl, bias_check.jsonl  outputs of the two checks in report/
  reruns.csv              collapsed runs retrained once with lr/3
  *_rawprobe.csv          first analysis with the un-centred probe, kept for the record
results_v1_adam/          first sweep with Adam (PyTorch defaults), not used; see the notebook's Model section
```

Datasets (`data/`) and checkpoints (`checkpoints*/`, about 10 GB) are not committed.

## Reproducing

```bash
pip install -r requirements.txt   # plus jupyter, nbclient, nbformat for the notebook and runs/nbstore.py
jupyter lab retina_bottleneck.ipynb
```

- **Data:** the datasets are downloaded by torchvision into `data/` on first use.
- **Training:** the loops resume from the CSVs in `results/` and skip the runs already recorded. To retrain from scratch, move the CSVs away.
- **Running the long cells detached:** use `python3 runs/nbexec.py <cell ids>`, from the project root.
- **Figures and tables:** `python3 report/make_figures.py` regenerates them. The receptive-field montages need the checkpoints, so they require retraining.
- **Compute:** about 8.5 hours for the code-recipe runs and about 10 hours for the paper-recipe runs, on one RTX PRO 2000 (8 GB).
