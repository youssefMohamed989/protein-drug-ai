import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import pytest

from src.data.splitters import protein_cluster_split, random_split


def test_random_split_covers_all_rows_no_overlap():
    df = pd.DataFrame({"x": range(100)})
    train, val, test = random_split(df, frac_train=0.7, frac_val=0.15, seed=1)
    assert len(train) + len(val) + len(test) == 100
    assert set(train.index) & set(val.index) == set()
    assert set(train.index) & set(test.index) == set()
    assert set(val.index) & set(test.index) == set()


def test_protein_cluster_split_keeps_groups_together():
    df = pd.DataFrame(
        {
            "protein_id": ["A"] * 10 + ["B"] * 10 + ["C"] * 10,
            "x": range(30),
        }
    )
    train, val, test = protein_cluster_split(df, frac_train=0.6, frac_val=0.2, seed=3)

    train_ids = set(train["protein_id"])
    val_ids = set(val["protein_id"])
    test_ids = set(test["protein_id"])
    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)
