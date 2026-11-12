# AdaHCL: Adaptive Hypergraph Contrastive Learning for Online Course Recommendation

PyTorch implementation of **AdaHCL**.

> Shiyu Zhao, Lu Wang, Ruihai Dong, and Yong Zhang. *Adaptive Hypergraph Contrastive Learning for Online Course Recommendation.*


## Requirements

- **NVIDIA GPU with CUDA >= 11.3** (required, CPU not supported)
- Python >= 3.9
- PyTorch >= 1.11.0 (with CUDA support)
- torch-sparse >= 0.6.17
- NumPy, SciPy
- tqdm

## Dataset

- **Proprietary Dataset**: [student]
- **MOOC Dataset**: [mooc]

## How to Run

```bash
# Proprietary dataset
python Main.py --data student --gpu 0

# MOOC dataset
python Main.py --data mooc --gpu 0
```


## Reproducibility

Reported results were obtained on a single **NVIDIA RTX 3090 (24GB)** with CUDA 11.3 and PyTorch 1.11.0. Results may vary across different GPU architectures, CUDA versions, and hardware configurations. We do not guarantee identical metrics on other setups.

