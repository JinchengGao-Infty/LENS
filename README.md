# LENS: selecting feature interactions via independent replication

Code and archived results for the paper *Beyond Interaction Scores: Selecting
Feature Interactions via Independent Replication* (ICDE 2027 submission).

LENS fits the same learner twice, on two disjoint parts of the data. For every
pair of features it removes the main effects from the pair's partial-dependence
table and asks whether the second model reproduces the residual table of the
first by more than a boundary. One Benjamini--Yekutieli correction over all
pairs sets the list.

## Layout

| Path | Contents |
|---|---|
| `lens_method/lens/` | The method: grids and residual tables (`grids.py`, `interaction_vectors.py`), the replication test (`model_inference.py`), BY (`calibration/multiple_testing.py`), splits and model wrappers |
| `lens/` | Thin import layer over `lens_method/lens`, plus the experiment runners in `lens/experiments/` |
| `experiments/` | Synthetic designs (`dgps.py`) and real-data loading (`real_data_loader.py`) |
| `scripts/` | One shell script per experiment |
| `figures/`, `tables/` | Scripts that turn results into the figures and tables of the paper |
| `results/` | Every run behind the paper, as JSON (about 120 MB) |

`lens/experiments/icde_main_comparison.py`, function `run_lens`, shows the whole
procedure on one data set: two fits, residual tables, boundary, test, BY.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=$PWD:$PWD/lens_method
```

The runs used NVIDIA RTX 5090 GPUs (CUDA 12.8). XGBoost and the MLPs run on
the GPU; `cupy` is used for batched prediction. The figure and table scripts
need only NumPy, SciPy and Matplotlib.

DIAMOND runs in its own environment (`requirements-diamond.txt`) with the
official source:

```bash
git clone https://github.com/batmen-lab/diamond third_party/diamond
git -C third_party/diamond checkout ea807f39cdcfc95c6f039be46f16c7e02d009ca3
```

## Rebuild the figures and tables from the archived results

These take seconds on a CPU.

```bash
python figures/make_figure_data.py   # figures/figure_data.json, tab_sensitivity.tex, tab_fremtpl2.tex
python figures/make_figures.py       # figures/fig_*.pdf
python figures/make_overview_tikz.py # figures/fig_overview.tex (Fig. 2)
python tables/make_ranking_table.py  # Table IV
python tables/make_appendix_synthetic.py
python tables/make_appendix_real.py
python tables/make_appendix_ablation.py
```

`make_figure_data.py` also checks that the LENS lists recomputed from the stored
per-pair statistics match the stored lists (440/440).

## Rerun the experiments

Each script reads `PY` (main environment), `DPY` (DIAMOND environment),
`DIAMOND_SRC`, `GPUS` (for example `"0 1 2 3"`) and `WORKERS` (cells run at the
same time; several per GPU work) from the environment, writes to `runs/`, and
skips cells that already exist.

| Paper | Script | Archived output |
|---|---|---|
| Table IV (ranking accuracy) | `scripts/run_ranking.sh` | `results/ranking/` |
| Fig. 4 ($H^2$ against energy) | `scripts/run_h2_diagnostic.sh` | `results/h2_diagnostic/` |
| Table V, Figs. 5--7, Table VI (lists on synthetic data, sensitivity) | `scripts/run_synthetic.sh` | `results/synthetic/` |
| Section IV-E (cost) | `scripts/run_timing.sh` | `results/timing/` |
| Table VII, Figs. 8--9, Table VIII (freMTPL2, HELOC, UCI credit) | `scripts/run_real.sh` | `results/real/` |
| Ablation of the selection rule (synthetic) | `scripts/run_synthetic.sh` (LENS cells) | `results/synthetic_ablation/` |

For example:

```bash
GPUS="0 1 2 3" WORKERS=12 PY=.venv/bin/python DPY=/path/to/diamond-env/bin/python \
  DIAMOND_SRC=third_party/diamond/src bash scripts/run_synthetic.sh
```

The real-data files hold every list compared in the paper and the ablation:
LENS, LENS without the scale floor, the energy ranking cut at the length of the
LENS list, the discovery model alone, both models above the floor without the
comparison, the point estimate above the boundary without the test, and all
pairs, each refitted on the same additive model. `data/README.md` says where
to get the three public data sets.

A LENS cell takes 36, 63 and 252 seconds at $p=10$, 30 and 50 on one GPU; a
freMTPL2 split with all 184,420 training policies takes 3.3--4.5 minutes.

### Baselines in Table IV

`scripts/run_ranking.sh` runs the reference implementations from their official
code, checked out under `third_party/official_baselines/`:

| Method | Source | Commit |
|---|---|---|
| NID | `mtsang/interaction_interpretability` | `02afd5b75b758e179f39c182a27de786b18be416` |
| Archipelago | `mtsang/archipelago` | `8ff437e5672809827d7daa6a5656aeedbc0e1094` |
| Faith-Shap | `chepingt/Faith-Shap` | `a5226d600a8ea5761a95da9e23bd58eee5aa85a5` |
| TreeSHAP-IQ, SHAP-IQ | `mmschlk/shapiq` | `a063be3056a480333281b13998d36107a4527329` |
| iRF | `Yu-Group/iterative-Random-Forest` | `7fa1724cf817d24c079a590acd2bf35fc4f95c91` |
| Integrated Hessians | `suinleelab/path_explain` | `93c75a5e9b3caeb17408297ea564e2f579ea15e3` |
| hstats (R) | `ModelOriented/hstats` | `5e062f829843f4203ee68ff23c0a1bef037c012c` (1.2.2) |

shapiq runs in a Python 3.12 environment (`SHAPIQ_PY`), iRF in a Python 3.7
environment (`IRF_PY`), hstats with `Rscript` (`RSCRIPT`). The scores computed
on the same fitted models as LENS (TreeSHAP, $H^2$, Integrated Hessians, NID
weights) are in `lens/synthetic_rankers.py` and
`lens/experiments/teacher_family_sensitivity.py`; the list baselines of Table V
($H^2$ with a refit null, iLOCO, FAST, DIAMOND) are in
`lens/experiments/icde_main_comparison.py` and `icde_diamond_adapter.py`.

## Results format

One JSON per cell. Synthetic cells (`results/synthetic/<design>_p<p>_seed<s>_<method>.json`)
store the true pairs and the list of every decision rule of that method; LENS
cells also store the energy ranking and, per pair, $a_u$, $\hat\mu_u$ and its
standard error. The cells in `results/synthetic_ablation/` were rerun with the
current code, which also stores the strength of the second model's table, so
that the ablation rules can be recomputed; their LENS lists equal those in
`results/synthetic/`. Real-data splits (`results/real/.../split<s>.json`) store the
lists, the per-pair statistics and the test loss of each list.
