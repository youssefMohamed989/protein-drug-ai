#!/usr/bin/env python
"""Unified training entry point.

Usage:
    python scripts/train.py --config configs/protein_structure.yaml
    python scripts/train.py --config configs/drug_affinity.yaml
    python scripts/train.py --config configs/fusion_affinity.yaml
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader

from src.data.datasets import (
    MoleculeAffinityDataset,
    ProteinLigandPairDataset,
    ProteinStructureDataset,
    molecule_collate_fn,
    pair_collate_fn,
    protein_collate_fn,
)
from src.models.fusion import FusionAffinityModel
from src.models.molecule_gnn import MoleculeAffinityModel
from src.models.protein_encoder import ProteinStructureModel
from src.structure.losses import advanced_structure_loss, advanced_structure_metrics
from src.structure.structure_model import AdvancedStructureModel
from src.training.losses_metrics import (
    affinity_loss,
    affinity_metrics,
    structure_loss,
    structure_metrics,
)
from src.training.task_adapters import (
    fusion_affinity_forward,
    molecule_affinity_forward,
    protein_structure_forward,
)
from src.training.trainer import Trainer
from src.utils.common import set_seed
from src.utils.config import Config


def build_protein_structure(cfg: Config):
    train_ds = ProteinStructureDataset(cfg.data.train_csv, max_len=cfg.data.max_len)
    val_ds = ProteinStructureDataset(cfg.data.val_csv, max_len=cfg.data.max_len)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        collate_fn=protein_collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        collate_fn=protein_collate_fn,
    )

    model = ProteinStructureModel(encoder_kwargs=dict(cfg.model))
    return model, train_loader, val_loader, structure_loss, structure_metrics, protein_structure_forward


def build_drug_affinity(cfg: Config):
    train_ds = MoleculeAffinityDataset(cfg.data.train_csv)
    val_ds = MoleculeAffinityDataset(cfg.data.val_csv)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        collate_fn=molecule_collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        collate_fn=molecule_collate_fn,
    )

    model = MoleculeAffinityModel(gnn_kwargs=dict(cfg.model))
    return model, train_loader, val_loader, affinity_loss, affinity_metrics, molecule_affinity_forward


def build_fusion_affinity(cfg: Config):
    train_ds = ProteinLigandPairDataset(cfg.data.train_csv, max_len=cfg.data.max_len)
    val_ds = ProteinLigandPairDataset(cfg.data.val_csv, max_len=cfg.data.max_len)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        collate_fn=pair_collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        collate_fn=pair_collate_fn,
    )

    model = FusionAffinityModel(
        protein_kwargs=dict(cfg.model.protein),
        molecule_kwargs=dict(cfg.model.molecule),
        fusion_hidden_dim=cfg.model.fusion_hidden_dim,
        fusion_heads=cfg.model.fusion_heads,
    )

    warm = cfg.get("warm_start", {})
    protein_ckpt = warm.get("protein_checkpoint") if isinstance(warm, dict) else None
    molecule_ckpt = warm.get("molecule_checkpoint") if isinstance(warm, dict) else None

    if protein_ckpt and Path(protein_ckpt).exists():
        sd = torch.load(protein_ckpt, map_location="cpu")["model_state_dict"]
        sd = {k.replace("encoder.", ""): v for k, v in sd.items() if k.startswith("encoder.")}
        missing, unexpected = model.protein_encoder.load_state_dict(sd, strict=False)
        print(
            f"Warm-started protein encoder from {protein_ckpt} "
            f"(missing={len(missing)}, unexpected={len(unexpected)})"
        )

    if molecule_ckpt and Path(molecule_ckpt).exists():
        sd = torch.load(molecule_ckpt, map_location="cpu")["model_state_dict"]
        sd = {k.replace("gnn.", ""): v for k, v in sd.items() if k.startswith("gnn.")}
        missing, unexpected = model.molecule_gnn.load_state_dict(sd, strict=False)
        print(
            f"Warm-started molecule GNN from {molecule_ckpt} "
            f"(missing={len(missing)}, unexpected={len(unexpected)})"
        )

    return model, train_loader, val_loader, affinity_loss, affinity_metrics, fusion_affinity_forward


def build_structure_advanced(cfg: Config):
    train_ds = ProteinStructureDataset(cfg.data.train_csv, max_len=cfg.data.max_len)
    val_ds = ProteinStructureDataset(cfg.data.val_csv, max_len=cfg.data.max_len)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        collate_fn=protein_collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        collate_fn=protein_collate_fn,
    )

    encoder_kwargs = dict(cfg.model)
    esm_cfg = encoder_kwargs.pop("esm", None)
    if esm_cfg:
        from src.models.esm_backend import ESM2EmbeddingBackend

        encoder_kwargs["embedding_backend"] = ESM2EmbeddingBackend(
            model_name=esm_cfg.get("model_name", "facebook/esm2_t6_8M_UR50D"),
            pretrained=esm_cfg.get("pretrained", True),
            freeze=esm_cfg.get("freeze", True),
            num_layers_override=esm_cfg.get("num_layers_override"),
        )
        print(f"[structure_advanced] using ESM-2 embedding backend: {esm_cfg.get('model_name')}")

    model = AdvancedStructureModel(encoder_kwargs=encoder_kwargs)
    return (
        model,
        train_loader,
        val_loader,
        advanced_structure_loss,
        advanced_structure_metrics,
        protein_structure_forward,
    )


BUILDERS = {
    "protein_structure": build_protein_structure,
    "drug_affinity": build_drug_affinity,
    "fusion_affinity": build_fusion_affinity,
    "structure_advanced": build_structure_advanced,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)
    set_seed(cfg.training.get("seed", 42))

    builder = BUILDERS.get(cfg.task)
    if builder is None:
        raise ValueError(f"Unknown task '{cfg.task}'. Valid: {list(BUILDERS)}")

    model, train_loader, val_loader, loss_fn, metrics_fn, forward_fn = builder(cfg)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[{cfg.task}] model has {n_params:,} trainable parameters")
    print(
        f"[{cfg.task}] train examples: {len(train_loader.dataset)} | val examples: {len(val_loader.dataset)}"
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        metrics_fn=metrics_fn,
        forward_fn=forward_fn,
        run_dir=cfg.output.run_dir,
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
        epochs=cfg.training.epochs,
        grad_accum_steps=cfg.training.get("grad_accum_steps", 1),
        grad_clip_norm=cfg.training.get("grad_clip_norm", 1.0),
        early_stopping_patience=cfg.training.get("early_stopping_patience", 10),
        monitor_metric=cfg.training.get("monitor_metric", "total_loss"),
        monitor_mode=cfg.training.get("monitor_mode", "min"),
        device=cfg.training.get("device", "auto"),
        use_amp=cfg.training.get("use_amp", True),
    )

    best_metrics = trainer.fit()
    print(f"[{cfg.task}] best validation metrics: {best_metrics}")


if __name__ == "__main__":
    main()
