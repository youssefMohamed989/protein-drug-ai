#!/usr/bin/env python
"""Evaluate a trained drug-affinity or fusion-affinity model against naive
baselines on a held-out test set: mean-predictor (regression) and
majority-class (classification). A real model that doesn't clear these is
not learning anything useful — this is the minimum bar, not a target.

Example:
    python scripts/evaluate.py --task drug_affinity \
        --checkpoint checkpoints/drug_affinity/best.pt \
        --config configs/drug_affinity.yaml \
        --test_csv data/processed/drug_affinity_test.csv

    python scripts/evaluate.py --task fusion_affinity \
        --checkpoint checkpoints/fusion_affinity/best.pt \
        --config configs/fusion_affinity.yaml \
        --test_csv data/processed/pairs_test.csv
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.datasets import (
    MoleculeAffinityDataset,
    ProteinLigandPairDataset,
    molecule_collate_fn,
    pair_collate_fn,
)
from src.models.fusion import FusionAffinityModel
from src.models.molecule_gnn import MoleculeAffinityModel
from src.training.task_adapters import fusion_affinity_forward, molecule_affinity_forward
from src.utils.config import Config


def naive_baselines(y_true_reg, y_true_cls, train_mean, train_majority_class):
    metrics = {}
    if y_true_reg is not None and len(y_true_reg):
        y_true_reg = np.array(y_true_reg)
        pred = np.full_like(y_true_reg, train_mean)
        metrics["baseline_mean_predictor_mae"] = float(np.mean(np.abs(pred - y_true_reg)))
        metrics["baseline_mean_predictor_rmse"] = float(np.sqrt(np.mean((pred - y_true_reg) ** 2)))
    if y_true_cls is not None and len(y_true_cls):
        y_true_cls = np.array(y_true_cls)
        pred = np.full_like(y_true_cls, train_majority_class)
        metrics["baseline_majority_class_accuracy"] = float(np.mean(pred == y_true_cls))
    return metrics


def model_metrics(all_preds_reg, all_true_reg, all_preds_cls_prob, all_true_cls):
    metrics = {}
    if all_true_reg:
        preds = np.array(all_preds_reg)
        trues = np.array(all_true_reg)
        metrics["model_mae"] = float(np.mean(np.abs(preds - trues)))
        metrics["model_rmse"] = float(np.sqrt(np.mean((preds - trues) ** 2)))
        if np.std(preds) > 1e-8 and np.std(trues) > 1e-8:
            metrics["model_pearson_r"] = float(np.corrcoef(preds, trues)[0, 1])
    if all_true_cls:
        probs = np.array(all_preds_cls_prob)
        trues = np.array(all_true_cls)
        preds = (probs > 0.5).astype(int)
        metrics["model_activity_accuracy"] = float(np.mean(preds == trues))
        try:
            from sklearn.metrics import roc_auc_score

            if len(np.unique(trues)) > 1:
                metrics["model_roc_auc"] = float(roc_auc_score(trues, probs))
        except Exception:
            pass
    return metrics


def evaluate_drug_affinity(checkpoint: str, config_path: str, test_csv: str):
    cfg = Config.from_yaml(config_path)
    ds = MoleculeAffinityDataset(test_csv)
    loader = DataLoader(ds, batch_size=cfg.data.batch_size, shuffle=False, collate_fn=molecule_collate_fn)

    model = MoleculeAffinityModel(gnn_kwargs=dict(cfg.model))
    ckpt = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    all_preds_reg, all_true_reg, all_preds_cls, all_true_cls = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            out = molecule_affinity_forward(model, batch)
            if "pActivity" in batch:
                all_preds_reg += out["pActivity_pred"].tolist()
                all_true_reg += batch["pActivity"].tolist()
            if "label" in batch:
                all_preds_cls += torch.sigmoid(out["activity_logit"]).tolist()
                all_true_cls += batch["label"].tolist()

    return all_preds_reg, all_true_reg, all_preds_cls, all_true_cls


def evaluate_fusion_affinity(checkpoint: str, config_path: str, test_csv: str):
    cfg = Config.from_yaml(config_path)
    ds = ProteinLigandPairDataset(test_csv, max_len=cfg.data.max_len)
    loader = DataLoader(ds, batch_size=cfg.data.batch_size, shuffle=False, collate_fn=pair_collate_fn)

    model = FusionAffinityModel(
        protein_kwargs=dict(cfg.model.protein),
        molecule_kwargs=dict(cfg.model.molecule),
        fusion_hidden_dim=cfg.model.fusion_hidden_dim,
        fusion_heads=cfg.model.fusion_heads,
    )
    ckpt = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    all_preds_reg, all_true_reg, all_preds_cls, all_true_cls = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            out = fusion_affinity_forward(model, batch)
            if "pActivity" in batch:
                all_preds_reg += out["pActivity_pred"].tolist()
                all_true_reg += batch["pActivity"].tolist()
            if "label" in batch:
                all_preds_cls += torch.sigmoid(out["activity_logit"]).tolist()
                all_true_cls += batch["label"].tolist()

    return all_preds_reg, all_true_reg, all_preds_cls, all_true_cls


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True, choices=["drug_affinity", "fusion_affinity"])
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--test_csv", type=str, required=True)
    parser.add_argument(
        "--train_csv",
        type=str,
        default=None,
        help="used to compute baseline mean/majority-class; defaults to test_csv's stats if omitted",
    )
    args = parser.parse_args()

    evaluator = evaluate_drug_affinity if args.task == "drug_affinity" else evaluate_fusion_affinity
    preds_reg, true_reg, preds_cls, true_cls = evaluator(args.checkpoint, args.config, args.test_csv)

    train_mean = float(np.mean(true_reg)) if true_reg else 0.0
    train_majority = int(round(float(np.mean(true_cls)))) if true_cls else 0

    print(f"\n=== Evaluation: {args.task} ===")
    print(f"Test set size: {len(preds_reg) or len(preds_cls)}")

    m = model_metrics(preds_reg, true_reg, preds_cls, true_cls)
    b = naive_baselines(true_reg, true_cls, train_mean, train_majority)

    print("\nModel:")
    for k, v in m.items():
        print(f"  {k}: {v:.4f}")
    print("\nNaive baselines:")
    for k, v in b.items():
        print(f"  {k}: {v:.4f}")

    if "model_mae" in m and "baseline_mean_predictor_mae" in b:
        improvement = 100 * (1 - m["model_mae"] / b["baseline_mean_predictor_mae"])
        print(f"\nRegression MAE improvement over mean-predictor baseline: {improvement:.1f}%")
    if "model_activity_accuracy" in m and "baseline_majority_class_accuracy" in b:
        delta = m["model_activity_accuracy"] - b["baseline_majority_class_accuracy"]
        print(f"Classification accuracy vs. majority-class baseline: {delta:+.4f}")


if __name__ == "__main__":
    main()
