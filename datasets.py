"""PyTorch Dataset classes for the two training tasks + the fused task."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data.molecule_features import MoleculeGraph, smiles_to_graph
from src.data.protein_features import PAD_IDX, encode_sequence, encode_ss8


class ProteinStructureDataset(Dataset):
    """Reads a CSV with columns: protein_id, sequence, ss8[, contact_map].

    `contact_map`, if present, is stored as a flattened, semicolon-joined
    string of 0/1 (row-major) so the whole dataset stays a single CSV file
    for easy inspection/versioning; it is reshaped to (L, L) on load.
    """

    def __init__(self, csv_path: str, max_len: int = 512):
        self.df = pd.read_csv(csv_path)
        self.max_len = max_len
        required = {"protein_id", "sequence"}
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"{csv_path} missing required columns: {missing}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        sequence = str(row["sequence"])[: self.max_len]
        length = len(sequence)

        item = {
            "protein_id": row["protein_id"],
            "input_ids": encode_sequence(sequence, max_len=self.max_len),
            "length": torch.tensor(length, dtype=torch.long),
        }

        if "ss8" in self.df.columns and isinstance(row.get("ss8"), str):
            item["ss8_labels"] = encode_ss8(str(row["ss8"])[: self.max_len], max_len=self.max_len)

        if "contact_map" in self.df.columns and isinstance(row.get("contact_map"), str):
            flat = np.array(row["contact_map"].split(";"), dtype=np.float32)
            L = int(np.sqrt(len(flat)))
            cmap = flat.reshape(L, L)
            cmap = cmap[: self.max_len, : self.max_len]
            padded = np.zeros((self.max_len, self.max_len), dtype=np.float32)
            padded[: cmap.shape[0], : cmap.shape[1]] = cmap
            item["contact_map"] = torch.tensor(padded)

        if "phi" in self.df.columns and isinstance(row.get("phi"), str):
            item["phi"] = _parse_angle_series(row["phi"], self.max_len)
        if "psi" in self.df.columns and isinstance(row.get("psi"), str):
            item["psi"] = _parse_angle_series(row["psi"], self.max_len)

        return item


def _parse_angle_series(raw: str, max_len: int) -> torch.Tensor:
    """Parse a ';'-joined list of degree values (or 'nan') into a padded
    radians tensor, NaN for undefined/padding positions.
    """
    vals = []
    for tok in raw.split(";")[:max_len]:
        tok = tok.strip()
        vals.append(float("nan") if tok.lower() == "nan" else np.deg2rad(float(tok)))
    vals = vals + [float("nan")] * (max_len - len(vals))
    return torch.tensor(vals, dtype=torch.float32)


class MoleculeAffinityDataset(Dataset):
    """Reads a CSV with columns: smiles, pActivity (regression), label (0/1, optional).

    Invalid SMILES are dropped at load time with a warning count.
    """

    def __init__(self, csv_path: str, cache_graphs: bool = True):
        df = pd.read_csv(csv_path)
        required = {"smiles"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{csv_path} missing required columns: {missing}")

        self.records: List[dict] = []
        n_dropped = 0
        for _, row in df.iterrows():
            graph = smiles_to_graph(str(row["smiles"]))
            if graph is None:
                n_dropped += 1
                continue
            rec = {"graph": graph, "smiles": row["smiles"]}
            if "pActivity" in df.columns and not pd.isna(row.get("pActivity")):
                rec["pActivity"] = float(row["pActivity"])
            if "label" in df.columns and not pd.isna(row.get("label")):
                rec["label"] = int(row["label"])
            self.records.append(rec)

        if n_dropped:
            print(f"[MoleculeAffinityDataset] dropped {n_dropped} unparsable SMILES from {csv_path}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        return self.records[idx]


class ProteinLigandPairDataset(Dataset):
    """Reads a CSV with columns: protein_id, sequence, smiles, pActivity[, label].

    Used to train the fused cross-modal affinity model.
    """

    def __init__(self, csv_path: str, max_len: int = 512):
        df = pd.read_csv(csv_path)
        required = {"sequence", "smiles"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{csv_path} missing required columns: {missing}")

        self.max_len = max_len
        self.records: List[dict] = []
        n_dropped = 0
        for _, row in df.iterrows():
            graph = smiles_to_graph(str(row["smiles"]))
            if graph is None:
                n_dropped += 1
                continue
            rec = {
                "protein_id": row.get("protein_id", "unknown"),
                "sequence": str(row["sequence"])[:max_len],
                "graph": graph,
            }
            if "pActivity" in df.columns and not pd.isna(row.get("pActivity")):
                rec["pActivity"] = float(row["pActivity"])
            if "label" in df.columns and not pd.isna(row.get("label")):
                rec["label"] = int(row["label"])
            self.records.append(rec)

        if n_dropped:
            print(f"[ProteinLigandPairDataset] dropped {n_dropped} unparsable SMILES from {csv_path}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]
        return {
            "protein_id": rec["protein_id"],
            "input_ids": encode_sequence(rec["sequence"], max_len=self.max_len),
            "length": torch.tensor(len(rec["sequence"]), dtype=torch.long),
            "graph": rec["graph"],
            **({"pActivity": rec["pActivity"]} if "pActivity" in rec else {}),
            **({"label": rec["label"]} if "label" in rec else {}),
        }


def protein_collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    out = {
        "protein_id": [b["protein_id"] for b in batch],
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "length": torch.stack([b["length"] for b in batch]),
    }
    if "ss8_labels" in batch[0]:
        out["ss8_labels"] = torch.stack([b["ss8_labels"] for b in batch])
    if "contact_map" in batch[0]:
        out["contact_map"] = torch.stack([b["contact_map"] for b in batch])
    if "phi" in batch[0]:
        out["phi"] = torch.stack([b["phi"] for b in batch])
    if "psi" in batch[0]:
        out["psi"] = torch.stack([b["psi"] for b in batch])
    return out


def molecule_collate_fn(batch: List[Dict]) -> Dict:
    from src.data.molecule_features import batch_molecule_graphs

    graphs = [b["graph"] for b in batch]
    atom_feats, edge_index, edge_feats, batch_idx = batch_molecule_graphs(graphs)
    out = {
        "atom_features": atom_feats,
        "edge_index": edge_index,
        "edge_features": edge_feats,
        "batch_index": batch_idx,
        "num_graphs": len(graphs),
        "smiles": [b["smiles"] for b in batch] if "smiles" in batch[0] else None,
    }
    if "pActivity" in batch[0]:
        out["pActivity"] = torch.tensor([b["pActivity"] for b in batch], dtype=torch.float32)
    if "label" in batch[0]:
        out["label"] = torch.tensor([b["label"] for b in batch], dtype=torch.float32)
    return out


def pair_collate_fn(batch: List[Dict]) -> Dict:
    from src.data.molecule_features import batch_molecule_graphs

    graphs = [b["graph"] for b in batch]
    atom_feats, edge_index, edge_feats, batch_idx = batch_molecule_graphs(graphs)
    out = {
        "protein_id": [b["protein_id"] for b in batch],
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "length": torch.stack([b["length"] for b in batch]),
        "atom_features": atom_feats,
        "edge_index": edge_index,
        "edge_features": edge_feats,
        "batch_index": batch_idx,
        "num_graphs": len(graphs),
    }
    if "pActivity" in batch[0]:
        out["pActivity"] = torch.tensor([b["pActivity"] for b in batch], dtype=torch.float32)
    if "label" in batch[0]:
        out["label"] = torch.tensor([b["label"] for b in batch], dtype=torch.float32)
    return out
