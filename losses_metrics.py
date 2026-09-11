"""Task losses and evaluation metrics."""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def structure_loss(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    ss8_weight: float = 1.0,
    contact_weight: float = 1.0,
) -> Dict[str, torch.Tensor]:
    losses = {}
    total = 0.0

    if "ss8_labels" in batch:
        ss8_logits = outputs["ss8_logits"].reshape(-1, outputs["ss8_logits"].size(-1))
        ss8_labels = batch["ss8_labels"].reshape(-1)
        loss_ss8 = F.cross_entropy(ss8_logits, ss8_labels, ignore_index=-100)
        losses["ss8_loss"] = loss_ss8
        total = total + ss8_weight * loss_ss8

    if "contact_map" in batch:
        pad_mask = outputs["pad_mask"]  # (B, L) True at pad
        valid = (~pad_mask).unsqueeze(1) & (~pad_mask).unsqueeze(2)  # (B, L, L)
        logits = outputs["contact_logits"][valid]
        targets = batch["contact_map"][valid]
        loss_contact = F.binary_cross_entropy_with_logits(logits, targets)
        losses["contact_loss"] = loss_contact
        total = total + contact_weight * loss_contact

    losses["total_loss"] = total
    return losses


def affinity_loss(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    regression_weight: float = 1.0,
    classification_weight: float = 1.0,
) -> Dict[str, torch.Tensor]:
    losses = {}
    total = 0.0

    if "pActivity" in batch:
        loss_reg = F.smooth_l1_loss(outputs["pActivity_pred"], batch["pActivity"])
        losses["regression_loss"] = loss_reg
        total = total + regression_weight * loss_reg

    if "label" in batch:
        loss_cls = F.binary_cross_entropy_with_logits(outputs["activity_logit"], batch["label"])
        losses["classification_loss"] = loss_cls
        total = total + classification_weight * loss_cls

    losses["total_loss"] = total
    return losses


@torch.no_grad()
def structure_metrics(outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
    metrics = {}
    if "ss8_labels" in batch:
        preds = outputs["ss8_logits"].argmax(-1)
        labels = batch["ss8_labels"]
        mask = labels != -100
        acc = (preds[mask] == labels[mask]).float().mean().item() if mask.any() else float("nan")
        metrics["ss8_accuracy"] = acc

    if "contact_map" in batch:
        pad_mask = outputs["pad_mask"]
        valid = (~pad_mask).unsqueeze(1) & (~pad_mask).unsqueeze(2)
        probs = torch.sigmoid(outputs["contact_logits"])[valid]
        targets = batch["contact_map"][valid]
        preds = (probs > 0.5).float()
        tp = (preds * targets).sum().item()
        fp = (preds * (1 - targets)).sum().item()
        fn = ((1 - preds) * targets).sum().item()
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        metrics["contact_precision"] = precision
        metrics["contact_recall"] = recall
        metrics["contact_f1"] = f1
    return metrics


@torch.no_grad()
def affinity_metrics(outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
    metrics = {}
    if "pActivity" in batch:
        preds = outputs["pActivity_pred"].cpu().numpy()
        targets = batch["pActivity"].cpu().numpy()
        mae = float(np.mean(np.abs(preds - targets)))
        rmse = float(np.sqrt(np.mean((preds - targets) ** 2)))
        if len(preds) > 1 and np.std(preds) > 1e-8 and np.std(targets) > 1e-8:
            pearson = float(np.corrcoef(preds, targets)[0, 1])
        else:
            pearson = float("nan")
        metrics.update({"mae": mae, "rmse": rmse, "pearson_r": pearson})

    if "label" in batch:
        probs = torch.sigmoid(outputs["activity_logit"]).cpu().numpy()
        targets = batch["label"].cpu().numpy()
        preds = (probs > 0.5).astype(np.float32)
        acc = float((preds == targets).mean())
        metrics["activity_accuracy"] = acc
        try:
            from sklearn.metrics import roc_auc_score

            if len(np.unique(targets)) > 1:
                metrics["roc_auc"] = float(roc_auc_score(targets, probs))
        except Exception:
            pass
    return metrics
