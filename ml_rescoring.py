"""Two-stage funnel: fast ML pre-screen (the FusionAffinityModel from the
core ML pipeline) narrows a large ligand library down to a shortlist, then
physics-based AutoDock Vina docking is only run on that shortlist. This is
the standard way ML and docking are combined in real virtual screening
campaigns — ML makes physics-based docking tractable at library scale.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from src.inference.predictor import AffinityPredictor


@dataclass
class ConsensusResult:
    smiles: str
    compound_id: str
    ml_pActivity_pred: float
    ml_activity_prob: float
    vina_affinity_kcal_mol: Optional[float]
    consensus_score: float


def ml_prescreen(
    predictor: AffinityPredictor,
    target_sequence: str,
    candidates: List[Tuple[str, str]],  # (smiles, compound_id)
    top_k: int,
) -> List[dict]:
    """Rank the full library by the learned model and keep the top_k, so
    expensive physics-based docking only runs on the most promising subset.
    """
    smiles_list = [s for s, _ in candidates]
    id_by_smiles = {s: cid for s, cid in candidates}

    scored = predictor.score(target_sequence, smiles_list)
    for r in scored:
        r["compound_id"] = id_by_smiles.get(r["smiles"], "unknown")

    ranked = sorted(scored, key=lambda r: r["pActivity_pred"], reverse=True)
    return ranked[:top_k]


def vina_score_to_pseudo_pActivity(affinity_kcal_mol: float, temperature_k: float = 298.15) -> float:
    """Convert a Vina binding free energy (kcal/mol, more negative = tighter)
    to a pKd-like scale, via the standard dG = -RT * ln(Kd) relation, so it
    lives on roughly the same numeric scale as the ML model's pActivity
    output and the two can be sensibly averaged in the consensus score.
    """
    R = 1.987204e-3  # kcal / (mol*K)
    kd = pow(2.718281828, affinity_kcal_mol / (R * temperature_k))
    if kd <= 0:
        return float("nan")
    import math

    return -math.log10(kd)


def consensus_rank(
    ml_results: List[dict],
    vina_results: dict,  # compound_id -> best affinity_kcal_mol
    ml_weight: float = 0.5,
    vina_weight: float = 0.5,
) -> List[ConsensusResult]:
    """Combine ML pActivity prediction with Vina's docking score into a
    single consensus ranking. Both terms are z-scored across the shortlist
    before combining, so neither dominates just because of its numeric
    scale.
    """
    import statistics

    ml_vals = [r["pActivity_pred"] for r in ml_results]
    ml_mean, ml_std = statistics.fmean(ml_vals), (statistics.pstdev(ml_vals) or 1.0)

    vina_pseudo = {
        cid: vina_score_to_pseudo_pActivity(aff) for cid, aff in vina_results.items() if aff is not None
    }
    vina_vals = list(vina_pseudo.values())
    vina_mean, vina_std = (
        (statistics.fmean(vina_vals), statistics.pstdev(vina_vals) or 1.0) if vina_vals else (0.0, 1.0)
    )

    consensus = []
    for r in ml_results:
        cid = r["compound_id"]
        ml_z = (r["pActivity_pred"] - ml_mean) / ml_std
        vina_aff = vina_results.get(cid)
        if cid in vina_pseudo:
            vina_z = (vina_pseudo[cid] - vina_mean) / vina_std
            score = ml_weight * ml_z + vina_weight * vina_z
        else:
            score = ml_z  # no docking result available, fall back to ML-only

        consensus.append(
            ConsensusResult(
                smiles=r["smiles"],
                compound_id=cid,
                ml_pActivity_pred=r["pActivity_pred"],
                ml_activity_prob=r["activity_prob"],
                vina_affinity_kcal_mol=vina_aff,
                consensus_score=score,
            )
        )

    consensus.sort(key=lambda c: c.consensus_score, reverse=True)
    return consensus
