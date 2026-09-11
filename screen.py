"""Docking-free virtual screening: rank a ligand library against a target
protein purely from the learned fusion embeddings, with an optional
Tanimoto-diversity re-ranking pass as a cheap sanity filter against
trivial duplicate hits.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from src.data.molecule_features import morgan_fingerprint
from src.data.protein_features import parse_fasta
from src.inference.predictor import AffinityPredictor
from src.utils.config import Config


def load_library(path: str) -> List[Tuple[str, str]]:
    """Loads a .smi (SMILES<TAB>id) or .csv (column `smiles`[, `compound_id`])."""
    path = Path(path)
    if path.suffix == ".csv":
        df = pd.read_csv(path)
        ids = df.get("compound_id", pd.Series([f"cand_{i}" for i in range(len(df))]))
        return list(zip(df["smiles"].tolist(), ids.tolist()))

    pairs = []
    with open(path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            smi = parts[0]
            cid = parts[1] if len(parts) > 1 else f"cand_{i}"
            pairs.append((smi, cid))
    return pairs


def tanimoto(fp_a: np.ndarray, fp_b: np.ndarray) -> float:
    inter = np.logical_and(fp_a, fp_b).sum()
    union = np.logical_or(fp_a, fp_b).sum()
    return float(inter / union) if union > 0 else 0.0


def diversity_filter(ranked: List[dict], max_similarity: float = 0.85, top_k: int = 50) -> List[dict]:
    """Greedy re-ranking: drop candidates near-identical to an already-kept
    higher-scoring hit, so the final list isn't dominated by one scaffold.
    """
    kept: List[dict] = []
    kept_fps: List[np.ndarray] = []
    for cand in ranked:
        fp = morgan_fingerprint(cand["smiles"])
        if fp is None:
            continue
        too_similar = any(tanimoto(fp, kfp) >= max_similarity for kfp in kept_fps)
        if not too_similar:
            kept.append(cand)
            kept_fps.append(fp)
        if len(kept) >= top_k:
            break
    return kept


def run_screen(
    target_fasta: str,
    library_path: str,
    checkpoint_path: str,
    config_path: str,
    top_k: int = 20,
    apply_diversity_filter: bool = True,
    device: str = "cpu",
) -> pd.DataFrame:
    records = parse_fasta(target_fasta)
    if not records:
        raise ValueError(f"No sequences found in {target_fasta}")
    target_id, target_seq = records[0]

    library = load_library(library_path)
    smiles_list = [s for s, _ in library]
    id_by_smiles = {s: cid for s, cid in library}

    cfg = Config.from_yaml(config_path)
    predictor = AffinityPredictor(checkpoint_path, cfg, device=device)

    scored = predictor.score(target_seq, smiles_list)
    for r in scored:
        r["compound_id"] = id_by_smiles.get(r["smiles"], "unknown")
        r["target_id"] = target_id

    ranked = sorted(scored, key=lambda r: r["pActivity_pred"], reverse=True)

    if apply_diversity_filter:
        ranked = diversity_filter(ranked, top_k=top_k)
    else:
        ranked = ranked[:top_k]

    return pd.DataFrame(ranked)
