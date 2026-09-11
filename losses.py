"""Joint loss for the advanced structure model (SS8 + contact + torsion),
and a Biopython-based extractor for real phi/psi angles from a PDB file
(used to build real training labels, complementing the toy-data generator).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from src.structure.torsion_head import torsion_angle_loss


def advanced_structure_loss(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    ss8_weight: float = 1.0,
    contact_weight: float = 1.0,
    torsion_weight: float = 1.0,
) -> Dict[str, torch.Tensor]:
    losses = {}
    total = 0.0

    if "ss8_labels" in batch:
        logits = outputs["ss8_logits"].reshape(-1, outputs["ss8_logits"].size(-1))
        labels = batch["ss8_labels"].reshape(-1)
        loss_ss8 = F.cross_entropy(logits, labels, ignore_index=-100)
        losses["ss8_loss"] = loss_ss8
        total = total + ss8_weight * loss_ss8

    if "contact_map" in batch:
        pad_mask = outputs["pad_mask"]
        valid = (~pad_mask).unsqueeze(1) & (~pad_mask).unsqueeze(2)
        logits = outputs["contact_logits"][valid]
        targets = batch["contact_map"][valid]
        loss_contact = F.binary_cross_entropy_with_logits(logits, targets)
        losses["contact_loss"] = loss_contact
        total = total + contact_weight * loss_contact

    if "phi" in batch and "psi" in batch:
        loss_torsion = torsion_angle_loss(outputs["torsion_unit_vectors"], batch["phi"], batch["psi"])
        losses["torsion_loss"] = loss_torsion
        total = total + torsion_weight * loss_torsion

    losses["total_loss"] = total
    return losses


@torch.no_grad()
def advanced_structure_metrics(
    outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
) -> Dict[str, float]:
    from src.training.losses_metrics import structure_metrics

    metrics = structure_metrics(outputs, batch)
    if "phi" in batch and "psi" in batch:
        pred_phi = outputs["phi"]
        pred_psi = outputs["psi"]
        for name, pred, true in (("phi", pred_phi, batch["phi"]), ("psi", pred_psi, batch["psi"])):
            mask = ~torch.isnan(true)
            if mask.any():
                # circular (wrap-aware) mean absolute error, in degrees
                diff = torch.atan2(torch.sin(pred[mask] - true[mask]), torch.cos(pred[mask] - true[mask]))
                mae_deg = diff.abs().mean().item() * 180.0 / 3.14159265
                metrics[f"{name}_mae_deg"] = mae_deg
    return metrics


def extract_phi_psi_from_pdb(
    path: str, chain_id: Optional[str] = None
) -> Tuple[str, List[Optional[float]], List[Optional[float]]]:
    """Extract the real sequence + phi/psi (in degrees) for every residue
    of a chain in a PDB file, via Biopython's internal_coords module.
    Returns (sequence, phi_list, psi_list); undefined angles are None.
    """
    from Bio.PDB import PDBParser
    from Bio.PDB.internal_coords import IC_Chain
    from Bio.PDB.Polypeptide import index_to_one, three_to_index

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", path)
    model = next(structure.get_models())

    chains = list(model.get_chains())
    if chain_id is not None:
        chains = [c for c in chains if c.id == chain_id]
    chain = chains[0]

    chain.atom_to_internal_coordinates()

    seq_chars, phi_list, psi_list = [], [], []
    for residue in chain.get_residues():
        if not residue.internal_coord:
            continue
        try:
            one_letter = index_to_one(three_to_index(residue.get_resname()))
        except Exception:
            one_letter = "X"
        seq_chars.append(one_letter)
        phi = residue.internal_coord.get_angle("phi")
        psi = residue.internal_coord.get_angle("psi")
        phi_list.append(float(phi) if phi is not None else None)
        psi_list.append(float(psi) if psi is not None else None)

    return "".join(seq_chars), phi_list, psi_list
