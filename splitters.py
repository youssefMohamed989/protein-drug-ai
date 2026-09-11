"""Splitting strategies that avoid optimistic bias in affinity benchmarks."""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Dict, List, Tuple

import pandas as pd

from src.data.molecule_features import scaffold_smiles


def random_split(
    df: pd.DataFrame, frac_train: float = 0.8, frac_val: float = 0.1, seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    idx = list(df.index)
    random.Random(seed).shuffle(idx)
    n = len(idx)
    n_train = int(n * frac_train)
    n_val = int(n * frac_val)
    train_idx = idx[:n_train]
    val_idx = idx[n_train : n_train + n_val]
    test_idx = idx[n_train + n_val :]
    return df.loc[train_idx], df.loc[val_idx], df.loc[test_idx]


def scaffold_split(
    df: pd.DataFrame,
    smiles_col: str = "smiles",
    frac_train: float = 0.8,
    frac_val: float = 0.1,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Bemis-Murcko scaffold split: molecules sharing a scaffold stay together
    in the same split, giving a much harder / more realistic generalization
    test than a random split (standard practice for QSAR/affinity models).
    """
    scaffolds: Dict[str, List[int]] = defaultdict(list)
    for i, smi in df[smiles_col].items():
        scaf = scaffold_smiles(str(smi)) or "invalid"
        scaffolds[scaf].append(i)

    groups = list(scaffolds.values())
    random.Random(seed).shuffle(groups)

    n = len(df)
    n_train_target = int(n * frac_train)
    n_val_target = int(n * frac_val)

    train_idx, val_idx, test_idx = [], [], []
    for group in groups:
        if len(train_idx) < n_train_target:
            train_idx += group
        elif len(val_idx) < n_val_target:
            val_idx += group
        else:
            test_idx += group

    return df.loc[train_idx], df.loc[val_idx], df.loc[test_idx]


def protein_cluster_split(
    df: pd.DataFrame,
    protein_id_col: str = "protein_id",
    frac_train: float = 0.8,
    frac_val: float = 0.1,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Group by protein target so the same target never leaks across splits
    (approximates a protein-family / cluster split when combined with
    pre-clustered ids, e.g. via MMseqs2 cluster assignment upstream).
    """
    groups: Dict[str, List[int]] = defaultdict(list)
    for i, pid in df[protein_id_col].items():
        groups[pid].append(i)

    group_list = list(groups.values())
    random.Random(seed).shuffle(group_list)

    n = len(df)
    n_train_target = int(n * frac_train)
    n_val_target = int(n * frac_val)

    train_idx, val_idx, test_idx = [], [], []
    for group in group_list:
        if len(train_idx) < n_train_target:
            train_idx += group
        elif len(val_idx) < n_val_target:
            val_idx += group
        else:
            test_idx += group

    return df.loc[train_idx], df.loc[val_idx], df.loc[test_idx]
