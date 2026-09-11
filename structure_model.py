"""Advanced structure prediction model: extends the base ProteinStructureModel
with a backbone torsion head, so a trained checkpoint can produce actual 3D
coordinates (via NeRF reconstruction), not just 2D contact/SS8 maps.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from src.models.protein_encoder import ContactMapHead, ProteinEncoder, SecondaryStructureHead
from src.structure.nerf import BackboneCoordinates, build_backbone
from src.structure.torsion_head import TorsionHead


class AdvancedStructureModel(nn.Module):
    """ProteinEncoder + SS8 head + contact-map head + torsion head, trained
    jointly with `structure.losses.advanced_structure_loss`. At inference
    time, `predict_structure()` turns the torsion predictions into an
    actual 3D backbone trace.
    """

    def __init__(self, encoder_kwargs: Optional[dict] = None):
        super().__init__()
        self.encoder = ProteinEncoder(**(encoder_kwargs or {}))
        self.ss8_head = SecondaryStructureHead(self.encoder.residue_out_dim)
        self.contact_head = ContactMapHead(self.encoder.residue_out_dim)
        self.torsion_head = TorsionHead(self.encoder.residue_out_dim)

    def forward(self, input_ids: torch.LongTensor) -> Dict[str, torch.Tensor]:
        enc_out = self.encoder(input_ids)
        ss8_logits = self.ss8_head(enc_out["residue_embeddings"])
        contact_logits = self.contact_head(enc_out["residue_embeddings"])
        torsion_out = self.torsion_head(enc_out["residue_embeddings"])
        return {
            **enc_out,
            "ss8_logits": ss8_logits,
            "contact_logits": contact_logits,
            **torsion_out,
        }

    @torch.no_grad()
    def predict_structure(self, input_ids: torch.LongTensor, sequence: str) -> BackboneCoordinates:
        """Run the model on a single sequence and reconstruct 3D backbone
        coordinates from the predicted phi/psi angles via NeRF.
        """
        self.eval()
        out = self.forward(input_ids)
        L = len(sequence)

        phi_rad = out["phi"][0, :L].cpu().numpy()
        psi_rad = out["psi"][0, :L].cpu().numpy()

        phi_deg: List[Optional[float]] = [None] + list(np.degrees(phi_rad[1:]))
        psi_deg: List[Optional[float]] = list(np.degrees(psi_rad[:-1])) + [None]

        return build_backbone(sequence, phi_deg, psi_deg)
