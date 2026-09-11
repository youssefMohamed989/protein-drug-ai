"""Per-residue backbone torsion angle (phi/psi) prediction head.

Angles are regressed via their (sin, cos) representation rather than raw
radians, which avoids the angular wraparound discontinuity at +-pi and is
the standard trick used by real structure-prediction torsion heads.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


class TorsionHead(nn.Module):
    """residue_embeddings (B, L, D) -> phi/psi as unit (sin, cos) vectors.

    Output shape (B, L, 2, 2): dim -2 indexes {phi, psi}, dim -1 is (sin, cos).
    The vectors are L2-normalized so downstream `atan2(sin, cos)` gives a
    valid angle even though the raw linear output isn't normalized.
    """

    def __init__(self, in_dim: int, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 4),  # phi_sin, phi_cos, psi_sin, psi_cos
        )

    def forward(self, residue_embeddings: torch.Tensor) -> Dict[str, torch.Tensor]:
        raw = self.net(residue_embeddings)  # (B, L, 4)
        raw = raw.view(*raw.shape[:-1], 2, 2)  # (B, L, {phi,psi}, {sin,cos})
        norm = raw.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        unit = raw / norm

        phi = torch.atan2(unit[..., 0, 0], unit[..., 0, 1])
        psi = torch.atan2(unit[..., 1, 0], unit[..., 1, 1])

        return {
            "torsion_unit_vectors": unit,  # (B, L, 2, 2) — for training loss
            "phi": phi,  # (B, L) radians, for export/inference
            "psi": psi,
        }


def torsion_angle_loss(
    pred_unit_vectors: torch.Tensor,  # (B, L, 2, 2)
    true_phi: torch.Tensor,  # (B, L) radians, NaN where unknown
    true_psi: torch.Tensor,  # (B, L) radians, NaN where unknown
) -> torch.Tensor:
    """L2 loss between predicted unit (sin,cos) vectors and the true angle's
    unit vector, only over positions with a defined ground-truth angle
    (the N-terminal residue has no phi, the C-terminal has no psi).
    """
    true = torch.stack(
        [
            torch.stack([torch.sin(true_phi), torch.cos(true_phi)], dim=-1),
            torch.stack([torch.sin(true_psi), torch.cos(true_psi)], dim=-1),
        ],
        dim=-2,
    )  # (B, L, 2, 2)

    mask = torch.stack([~torch.isnan(true_phi), ~torch.isnan(true_psi)], dim=-1)  # (B, L, 2)
    true = torch.nan_to_num(true, nan=0.0)

    diff2 = ((pred_unit_vectors - true) ** 2).sum(dim=-1)  # (B, L, 2)
    if mask.any():
        return diff2[mask].mean()
    return diff2.sum() * 0.0  # no valid targets in this batch
