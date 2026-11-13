"""
Data preprocessing script for AdaHCL.
Splits raw interaction data into train/validation/test sets (7:2:1 ratio)
and saves as sparse matrices (trnMat.pkl, valMat.pkl, tstMat.pkl).

Usage:
    python Datasets/generate_splits.py --input Datasets/student_raw.csv --output Datasets/student/ --ratio 0.7 0.2 0.1
    python Datasets/generate_splits.py --input Datasets/mooc_raw.csv --output Datasets/mooc/ --ratio 0.7 0.2 0.1
"""

import argparse
import os
import pickle
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix


def load_interactions(filepath):
    """Load raw interaction data from CSV file.
    Expected columns: user_id, item_id (and optionally rating/timestamp).
    """
    df = pd.read_csv(filepath)
    # Expect at least user_id and item_id columns
    user_col = df.columns[0]
    item_col = df.columns[1]
    users = df[user_col].values
    items = df[item_col].values
    return users, items


def build_sparse_matrix(users, items, num_users, num_items):
    """Build a binary sparse interaction matrix."""
    data = np.ones(len(users), dtype=np.float32)
    mat = coo_matrix((data, (users, items)), shape=(num_users, num_items))
    return mat


def split_data(users, items, ratio=(0.7, 0.2, 0.1), seed=2026):
    """Split interactions into train/val/test by ratio.
    Each user's interactions are split independently.
    """
    rng = np.random.RandomState(seed)
    num_users = users.max() + 1
    num_items = items.max() + 1

    trn_users, trn_items = [], []
    val_users, val_items = [], []
    tst_users, tst_items = [], []

    # Group interactions by user
    user_interactions = {}
    for u, i in zip(users, items):
        if u not in user_interactions:
            user_interactions[u] = []
        user_interactions[u].append(i)

    for u, items_list in user_interactions.items():
        items_arr = np.array(items_list)
        rng.shuffle(items_arr)
        n = len(items_arr)
        n_trn = max(1, int(n * ratio[0]))
        n_val = max(0, int(n * ratio[1]))
        # Ensure at least 1 test sample if possible
        if n_trn + n_val >= n:
            n_val = max(0, n - n_trn - 1)
        n_tst = n - n_trn - n_val

        trn_items_list = items_arr[:n_trn]
        val_items_list = items_arr[n_trn:n_trn + n_val]
        tst_items_list = items_arr[n_trn + n_val:]

        for i in trn_items_list:
            trn_users.append(u)
            trn_items.append(i)
        for i in val_items_list:
            val_users.append(u)
            val_items.append(i)
        for i in tst_items_list:
            tst_users.append(u)
            tst_items.append(i)

    trn_mat = build_sparse_matrix(trn_users, trn_items, num_users, num_items)
    val_mat = build_sparse_matrix(val_users, val_items, num_users, num_items)
    tst_mat = build_sparse_matrix(tst_users, tst_items, num_users, num_items)

    return trn_mat, val_mat, tst_mat


def main():
    parser = argparse.ArgumentParser(description='Generate train/val/test splits for AdaHCL')
    parser.add_argument('--input', type=str, required=True, help='Path to raw interaction CSV file')
    parser.add_argument('--output', type=str, required=True, help='Output directory for processed data')
    parser.add_argument('--ratio', type=float, nargs=3, default=[0.7, 0.2, 0.1],
                        help='Train/val/test split ratio (default: 0.7 0.2 0.1)')
    parser.add_argument('--seed', type=int, default=2026, help='Random seed')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print(f'Loading data from {args.input}...')
    users, items = load_interactions(args.input)
    print(f'  {len(users)} interactions, {users.max()+1} users, {items.max()+1} items')

    print(f'Splitting data (ratio={args.ratio}, seed={args.seed})...')
    trn_mat, val_mat, tst_mat = split_data(users, items, args.ratio, args.seed)
    print(f'  Train: {trn_mat.nnz} interactions')
    print(f'  Val:   {val_mat.nnz} interactions')
    print(f'  Test:  {tst_mat.nnz} interactions')

    print(f'Saving to {args.output}...')
    with open(os.path.join(args.output, 'trnMat.pkl'), 'wb') as f:
        pickle.dump(trn_mat, f)
    with open(os.path.join(args.output, 'valMat.pkl'), 'wb') as f:
        pickle.dump(val_mat, f)
    with open(os.path.join(args.output, 'tstMat.pkl'), 'wb') as f:
        pickle.dump(tst_mat, f)

    print('Done!')


if __name__ == '__main__':
    main()
