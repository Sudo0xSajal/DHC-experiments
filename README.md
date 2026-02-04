## 🚀 DHC-experiments

<p align="center">
  <img src="./images/framework.png" width="800" />
</p>

<p align="center">
  <b>Experimental & Research-Oriented Fork of MICCAI 2023 DHC</b><br/>
  Dual-debiased Heterogeneous Co-training for Class-imbalanced Semi-supervised Medical Image Segmentation
</p>

---

### 📌 Overview

This repository is a **personal experimental fork** of the MICCAI 2023 paper:

> **DHC: Dual-debiased Heterogeneous Co-training Framework for Class-imbalanced Semi-supervised Medical Image Segmentation**
> *Haonan Wang, Xiaomeng Li — MICCAI 2023*

⚠️ **Note**: This is **NOT** the official implementation.
This repository is maintained for **research, experimentation, debugging, and controlled modifications** of the original DHC framework.

---

### 🔬 Motivation & Scope

This fork exists to explore:

* 🧪 **Loss function experimentation** (Dice, CE, class-balanced, noise-robust losses)
* 🔧 **Noise handling & robustness analysis**
* 📊 **Ablation studies** and metric behavior (Dice, ASD, HD)
* 🧠 Architectural tweaks without altering the upstream codebase

Results in this repository may **deviate from the published paper** and should not be treated as official benchmarks.

---

### 🏛️ Original Work

* **Conference**: MICCAI 2023
* **Paper**: DHC: Dual-debiased Heterogeneous Co-training Framework for Class-imbalanced Semi-supervised Medical Image Segmentation
* **Authors**: Haonan Wang, Xiaomeng Li

If you are looking for the **official and stable implementation**, please refer to the original authors’ repository.

---

### ⚙️ Environment Setup

This fork follows the original experimental setup unless otherwise stated.

Tested configuration:

* Python ≥ 3.6
* PyTorch 1.8
* torchvision 0.9.0
* CUDA 11.x
* Ubuntu 20.04 / Windows (Git Bash)

Set the `PYTHONPATH` before running:

```bash
export PYTHONPATH=$(pwd)/code:$PYTHONPATH
```

---

### 📂 Data Preparation

Data preprocessing strictly follows the **official DHC protocol**.

#### 🧠 Synapse Dataset

* Dataset: [https://www.synapse.org/#!Synapse:syn3193805/wiki/](https://www.synapse.org/#!Synapse:syn3193805/wiki/)
* Preprocessing:

```bash
python ./code/data/preprocess.py
```

#### 🧠 AMOS Dataset

* Dataset: [https://amos22.grand-challenge.org/Dataset/](https://amos22.grand-challenge.org/Dataset/)
* Preprocessing:

```bash
python ./code/data/preprocess_amos.py
```

Expected directory structure:

```text
synapse_data/
├── npy/
├── splits/
```

---

### 🚀 Training & Experiments

Example training command:

```bash
bash train3times_seeds_20p.sh -c 0 -t amos -m dhc -e '' -l 3e-2 -w 0.1
```

Key parameters:

* `-c` : GPU index
* `-t` : task (`synapse`, `amos`)
* `-m` : method (`dhc`, `cps`, `uamt`, etc.)
* `-e` : experiment identifier
* `-l` : learning rate
* `-w` : unsupervised loss weight

---

### 📊 Experimental Results

All reported results in this repository are:

* ⚠️ **Experimental**
* 🔄 Subject to change
* 🧪 Influenced by loss modifications and noise strategies

Do **NOT** use these results as a replacement for the official benchmarks.

---

### 📚 Citation

If this work helps your research, please cite the original paper:

```bibtex
@inproceedings{wang2023dhc,
  title={DHC: Dual-debiased Heterogeneous Co-training Framework for Class-imbalanced Semi-supervised Medical Image Segmentation},
  author={Wang, Haonan and Li, Xiaomeng},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={582--591},
  year={2023},
  organization={Springer}
}
```

---

### 📜 License

This repository follows the **MIT License** of the original DHC implementation.

---

### 🧪 Disclaimer

This repository is intended **solely for learning and research experimentation**.
For official results, production usage, or reproducibility claims, always refer to the original authors’ codebase.
