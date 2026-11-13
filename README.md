# AdaHCL: Adaptive Hypergraph Contrastive Learning for Online Course Recommendation

PyTorch implementation of **AdaHCL**.

> Shiyu Zhao, Lu Wang, Ruihai Dong, and Yong Zhang. *Adaptive Hypergraph Contrastive Learning for Online Course Recommendation.*

## Requirements

- **NVIDIA GPU with CUDA >= 11.3** (required, CPU not supported)
- Python >= 3.9
- PyTorch == 1.13.1 (with CUDA support)
- torch-sparse == 0.6.17 (must match your PyTorch and CUDA version)
- NumPy, SciPy, tqdm

## Installation

```bash
# 1. Create environment
conda create -n adahcl python=3.9 -y
conda activate adahcl

# 2. Install PyTorch with CUDA 11.3
pip install torch==1.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113

# 3. Install torch-sparse (must match torch + CUDA version)
pip install torch-sparse==0.6.17 -f https://data.pyg.org/whl/torch-1.13.1+cu113.html

# 4. Install other dependencies
pip install -r requirements.txt
```

## Datasets

| Dataset | #Students | #Courses | #Interactions | Sparsity |
|---------|-----------|----------|---------------|----------|
| Proprietary | 397 | 31,338 | 50,000 | 99.60% |
| MOOC | 82,534 | 1,301 | 458,454 | 99.57% |

- **Proprietary Dataset** (`student`): placed in `Datasets/student/`
- **MOOC Dataset** (`mooc`): placed in `Datasets/mooc/`, source: http://moocdata.cn

Each dataset contains pre-split `trnMat.pkl`, `valMat.pkl`, `tstMat.pkl` (7:2:1 ratio).

## How to Run

```bash
# Proprietary dataset
python Main.py --data student --gpu 0

# MOOC dataset
python Main.py --data mooc --gpu 0

# Or use the one-click script
bash run.sh
```

### Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--lr` | 1e-3 | Learning rate |
| `--latdim` | 128 | Embedding dimension |
| `--gnn_layer` | 2 | Number of GCN/HGCN layers |
| `--ssl_reg` | 0.1 | Weight for pairwise contrastive loss |
| `--ib_reg` | 1e-2 | Weight for information bottleneck |
| `--triple_reg` | 1e-6 | Weight for triple contrastive loss |
| `--temp` | 0.5 | Temperature in contrastive learning |
| `--margin` | 1.0 | Margin in triple loss |
| `--epoch` | 20 | Number of training epochs |
| `--seed` | 2026 | Random seed |

## Reproducibility

Reported results were obtained on a single **NVIDIA RTX 3090** with **CUDA 11.3** and **PyTorch 1.13.1**. Results may vary across different GPU architectures, CUDA versions, and hardware configurations. We do not guarantee identical metrics on other setups.

