#!/bin/bash
# AdaHCL: Adaptive Hypergraph Contrastive Learning for Online Course Recommendation
# One-click run script for both datasets

# Proprietary dataset (student)
echo "========== Running AdaHCL on Proprietary Dataset =========="
python Main.py --data student --gpu 0

# MOOC dataset
echo "========== Running AdaHCL on MOOC Dataset =========="
python Main.py --data mooc --gpu 0
