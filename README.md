# Community-Centric Graph Unlearning

**Paper:** [Community-Centric Graph Unlearning](https://doi.org/10.1609/aaai.v39i17.34041)  

---

## Overview
This repository provides the implementation of **Community-Centric Graph Unlearning (CCGU)** proposed in our AAAI 2025 paper.  

The main CCGU pipeline keeps the original **DGL** data and model interfaces. The reference environment for the paper is **Python 3.8.19**, **PyTorch 2.1**, and **DGL 2.1** on Linux. The exact Python dependencies used by this release are listed in `requirements.txt`.

---

## Installation

```bash
conda create -n ccgu python=3.8.19
conda activate ccgu
pip install -r requirements.txt
```

The full partition/train/unlearn workflow downloads a DGL dataset on first use. The default dataset is Reddit; use `--dataset_name cora` for a much smaller functional run.

## Workflow

### 1 Build and Download DGL Datasets

### 2 Preprocess the Original Dataset

```bash
python main.py --exp Partition --dataset_name cora
```

### 3 Generate Aggregated Graph and Train

```bash
python main.py --exp Train --dataset_name cora
```

### 4 Perform Node Unlearning

```bash
python main.py \
  --exp Unlearn \
  --dataset_name cora \
  --unlearn_task node \
  --unlearn_ratio 0.005 \
  --target_model GCN
```

`--unlearn_ratio` is interpreted as a fraction when it is in `(0, 1]` and as an absolute node count when it is greater than `1`.

The unlearning stage follows the paper's deterministic workflow:

1. sample the request only from training nodes;
2. remove each requested node from every community it belongs to;
3. remove empty communities and compact the community IDs;
4. recalculate the affected mapped features and labels, then rebuild mapped-edge scores from the remaining original nodes;
5. rebuild the mapped graph as a DGL graph;
6. initialize a fresh GCN, GAT, or GraphSAGE backbone and retrain it on the adjusted mapped graph.

The updated model is written below `temp_data/models/<dataset>/` with the suffix `_unlearned.pt`. The sampled request is also saved under `temp_data/processed_data/<dataset>/` for audit and repeatability.

## CPU Smoke Test

The smoke test does not download a dataset and does not train a GNN. It uses a tiny DGL-shaped graph to verify overlapping-community deletion, empty-community removal/reindexing, and mapped-edge reconstruction:

```bash
python -m unittest discover -s tests -v
```

## Notes

- CCGU node unlearning remains DGL-based; the smoke fixture is not used by the runtime pipeline.
- The unlearning command updates the stored community and mapped-graph artifacts so subsequent requests operate on the already-unlearned state. Keep a copy of `temp_data/` when an untouched deployment snapshot is required.
- Membership-inference attack (MIA) experiments are intentionally not included in this repository.
- Some ablation-related or baseline files were removed from the public release. They are not required by the CCGU node-unlearning path documented above.
- If you encounter unresolved issues or have academic questions for discussion, please feel free to contact: liiyi.xsjl@gmail.com

## Citation
If you find this repository useful in your research, please cite the following paper:
```bibtex
@inproceedings{li2025ccgu,
  title={Community-Centric Graph Unlearning},
  author={Li, Yi and Zhang, Shichao and Zhang, Guixian and Cheng, Debo},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={17},
  pages={18548--18556},
  year={2025},
  doi={10.1609/aaai.v39i17.34041}
}
```
(CCGU_Readme_2.0)
