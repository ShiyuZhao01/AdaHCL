"""
Data preprocessing pipeline for AdaHCL.
Cleans raw interaction data and generates train/val/test splits.

Usage:
    python Datasets/Preprocess.py --dataset student
    python Datasets/Preprocess.py --dataset mooc
"""

import argparse
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Datasets.generate_splits import load_interactions, split_data
import pickle


def preprocess_student():
    """Preprocess the proprietary e-learning (student) dataset."""
    raw_path = 'Datasets/student_raw.csv'
    output_dir = 'Datasets/student'

    if not os.path.exists(raw_path):
        print(f'Raw data not found at {raw_path}. Please place the raw CSV file there.')
        print('Expected format: CSV with columns [user_id, item_id]')
        return

    print('=== Preprocessing Proprietary Dataset (student) ===')
    users, items = load_interactions(raw_path)
    print(f'  Loaded {len(users)} interactions, {users.max()+1} students, {items.max()+1} courses')

    trn_mat, val_mat, tst_mat = split_data(users, items, ratio=(0.7, 0.2, 0.1), seed=2026)
    print(f'  Train: {trn_mat.nnz}, Val: {val_mat.nnz}, Test: {tst_mat.nnz}')

    os.makedirs(output_dir, exist_ok=True)
    for name, mat in [('trnMat', trn_mat), ('valMat', val_mat), ('tstMat', tst_mat)]:
        with open(os.path.join(output_dir, f'{name}.pkl'), 'wb') as f:
            pickle.dump(mat, f)
    print(f'  Saved to {output_dir}/')


def preprocess_mooc():
    """Preprocess the MOOC public dataset.
    Source: http://moocdata.cn
    """
    raw_path = 'Datasets/mooc_raw.csv'
    output_dir = 'Datasets/mooc'

    if not os.path.exists(raw_path):
        print(f'Raw data not found at {raw_path}. Please place the raw CSV file there.')
        print('Expected format: CSV with columns [user_id, item_id]')
        print('Download from: http://moocdata.cn')
        return

    print('=== Preprocessing MOOC Dataset ===')
    users, items = load_interactions(raw_path)
    print(f'  Loaded {len(users)} interactions, {users.max()+1} students, {items.max()+1} courses')

    trn_mat, val_mat, tst_mat = split_data(users, items, ratio=(0.7, 0.2, 0.1), seed=2026)
    print(f'  Train: {trn_mat.nnz}, Val: {val_mat.nnz}, Test: {tst_mat.nnz}')

    os.makedirs(output_dir, exist_ok=True)
    for name, mat in [('trnMat', trn_mat), ('valMat', val_mat), ('tstMat', tst_mat)]:
        with open(os.path.join(output_dir, f'{name}.pkl'), 'wb') as f:
            pickle.dump(mat, f)
    print(f'  Saved to {output_dir}/')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess datasets for AdaHCL')
    parser.add_argument('--dataset', type=str, required=True, choices=['student', 'mooc'],
                        help='Dataset to preprocess: student | mooc')
    args = parser.parse_args()

    if args.dataset == 'student':
        preprocess_student()
    elif args.dataset == 'mooc':
        preprocess_mooc()
