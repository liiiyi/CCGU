# Community-Centric Graph Unlearning

**Paper:** [Community-Centric Graph Unlearning](https://doi.org/10.1609/aaai.v39i17.34041)  

---

## Overview
This repository provides the implementation of **Community-Centric Graph Unlearning (CCGU)** proposed in our AAAI 2025 paper.  

Most experiments were conducted with **Python 3.7** and **DGL**, while a small portion (e.g., GraphEraser comparisons) were based on **Python 3.6** and **PyTorch Geometric (PyG)**.

---

## Usage

### 1 Build and Download DGL Datasets

### 2 Preprocess the Original Dataset
```
python main.py --exp Partition
```

### 3 Generate Aggregated Graph and Train
```
python main.py --exp Train
```

### 4 Perform Unlearning and Evaluation
```
python main.py --exp Unlearn
```

## Notes
- Some ablation-related or redundant files have been removed or commented out in this public release to streamline the codebase.
- This may cause minor compilation or dependency issues, which might require small manual adjustments depending on your setup.
- If you encounter unresolved issues or have academic questions for discussion, please feel free to contact: liiyi.xsjl@gmail.com

## Citation
If you find this repository useful in your research, please cite the following paper:
```bibtex
@inproceedings{li2025ccgu,
  title={Community-Centric Graph Unlearning},
  author={Li, Yi and Zhang, Shuang and Zhang, Guimei and Cheng, Dawei},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={17},
  pages={18548--18556},
  year={2025},
  doi={10.1609/aaai.v39i17.34041}
}
```
(CCGU_Readme_1.0)
