# Community-Centric Graph Unlearning (CCGU / CGE)

Reference implementation of **Community-Centric Graph Unlearning**, published at
AAAI 2025. The framework performs graph unlearning by mapping the original graph
onto a compact *community-centric mapped graph* and operating on that mapped graph
in place of the original one. The implementation is based on **DGL**.

## Method summary

**Stage 1 — community-centric mapping.** Communities are detected on the original
graph and each community is contracted into one mapped node. Communities may
overlap, so a node can contribute to several mapped nodes. Four rules define the
mapped graph:

| Rule | Definition |
|---|---|
| Node | one mapped node per community |
| Feature | principal-component projection of the member features, then their mean (`--agg_feat pca`), or the community mean (`--agg_feat mean`) |
| Label | majority vote over the members closest to the mapped feature, cut at the largest gap in the sorted distances (`--agg_label th`) |
| Edge | a robustness score combining inter-community edge counts with degree normalisation and a community-union term (`--agg_edge rubost`) |

**Stage 2 — node-level unlearning.** A deletion request removes the requested nodes
from every community they belong to, after which the affected parts of the mapping
are updated, the mapped graph is rebuilt and a fresh backbone is trained on it.

```
Partition   original graph -> communities -> mapped features / labels / edges
Train       mapped graph   -> GCN | GAT | GraphSAGE
Unlearn     deletion request -> updated mapping -> mapped graph -> retrained model
```

Each stage is a separate invocation of `main.py` and writes its artifacts to disk, so
`Train` and `Unlearn` reuse the mapping produced by `Partition`. `--partition ccp`
provides the coarse-to-fine overlapping community process described in the paper;
the partitioning options and their parameters are documented in
[`exp/methods/README.md`](exp/methods/README.md).

## Installation

```bash
git clone https://github.com/liiiyi/CCGU.git
cd CCGU
conda create -n ccgu python=3.8.19
conda activate ccgu
pip install -r requirements.txt
```

The pinned dependencies are PyTorch 2.1.2, torchdata 0.7.1, DGL 2.1.0, PyTorch
Geometric 2.4.0, NumPy 1.24.4, SciPy 1.10.1, scikit-learn 1.3.2, networkx 3.1,
Matplotlib 3.7.5 and tqdm 4.66.5.

Alternatively, build the validated CUDA environment, which requires no administrator
rights:

```bash
bash reproduction/scripts/setup_env.sh
```

This creates a self-contained interpreter prefix at `./.conda-env` with Python
3.8.19, PyTorch 2.1.2+cu121 and DGL 2.2.1+cu121, together with the CUDA shared
libraries that recent DGL releases load at import time.

```bash
.conda-env/bin/python reproduction/scripts/report_env.py
```

This prints the resolved versions and runs one forward pass of each backbone.

## Datasets and artifacts

Datasets are obtained through DGL and downloaded automatically on first use, so no
data files are distributed here. `--dataset_name` selects the dataset (`cora`,
`citeseer`, `pubmed`, `cs`, `reddit`) and DGL caches the download under the directory
configured in `config.py`.

All generated artifacts — communities, mapped features and labels, mapped graphs,
diagnostics and trained weights — are written locally under `temp_data/`, or under
`CCGU_DATA_ROOT` when it is set, and are excluded from version control.

## Running the pipeline

```bash
# 1. detect communities and build the mapped graph
python main.py --exp Partition \
  --dataset_name cora --partition ccp --random_seed 4 --cuda 0

# 2. train a backbone on the mapped graph
python main.py --exp Train \
  --dataset_name cora --partition ccp --random_seed 4 --cuda 0 \
  --target_model GCN --train_lr 0.01 --train_weight_decay 0.001 --num_epochs 200

# 3. apply a node-deletion request and retrain on the updated mapping
python main.py --exp Unlearn \
  --dataset_name cora --partition ccp --random_seed 4 --cuda 0 \
  --target_model GCN --unlearn_task node --unlearn_ratio 0.005
```

`--unlearn_ratio` is read as a fraction of the eligible training nodes when it lies
in `(0, 1]`, and as an absolute node count when greater than `1`. The sampled request
is stored alongside the updated mapping so that it can be inspected or audited.

The three stages can also be run in order for one configuration, with a separate
artifact directory and per-stage metrics recorded as JSON:

```bash
python reproduction/scripts/run_experiment.py \
  --dataset cora --partition ccp --model GCN --seed 4
```

## Essential flags

| Flag | Values | Meaning |
|---|---|---|
| `--exp` | `Partition`, `Train`, `Unlearn` | pipeline stage |
| `--dataset_name` | `cora`, `citeseer`, `pubmed`, `cs`, `reddit` | DGL dataset |
| `--partition` | `ccp`, `gae`, `oslom`, `slpa`, `lpa`, `louvain`, `nikm`, `test`, `infomap` | community-detection method |
| `--target_model` | `GCN`, `GAT`, `SAGE` | backbone trained on the mapped graph |
| `--unlearn_ratio` | float | size of the deletion request |
| `--random_seed` | int | seeds every random source the pipeline uses |
| `--cuda` | int | CUDA device index |

`python main.py --help` lists the remaining flags, including the mapping rules, the
optimiser and split settings, and the partitioning parameters documented in
[`exp/methods/README.md`](exp/methods/README.md).

## Tests

```bash
python -m unittest discover -s tests -v
```

The suite runs on CPU and downloads nothing; a DGL-shaped stub graph stands in for a
dataset. Tests requiring torch or DGL are skipped when those packages are absent.

## Citation

```bibtex
@inproceedings{li2025ccgu,
  title     = {Community-Centric Graph Unlearning},
  author    = {Li, Yi and Zhang, Shichao and Zhang, Guixian and Cheng, Debo},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  volume    = {39},
  number    = {17},
  pages     = {18548--18556},
  year      = {2025},
  doi       = {10.1609/aaai.v39i17.34041}
}
```

## Contact

For questions about the paper or implementation, contact Yi Li at
[yi.li04@adelaide.edu.au](mailto:yi.li04@adelaide.edu.au) (preferred) or liiyi.xsjl@gmail.com.

## Validation note

This release was validated on an NVIDIA RTX 6000 Ada with CUDA 12.1, Python 3.8.19,
PyTorch 2.1.2+cu121 and DGL 2.2.1+cu121. That environment differs from the one
reported in the paper, which used an NVIDIA Tesla A800 with PyTorch 2.1 and DGL 2.1.
The checks covered Cora, Citeseer and Coauthor-CS; Reddit was not evaluated. Most
qualitative trends were consistent with those reported in the paper. These checks
support the reliability of the released pipeline, but they do not constitute exact
numerical reproduction: results remain sensitive to seeds, stochastic partitioning,
library versions and hardware.

---

This codebase was recently jointly organized and updated by Codex 5.6-sol@max and Claude Code claude-opus-5@ultracode.
