"""Cross-modal co-attention fusion of protein residues and molecule atoms,
feeding a final affinity/activity prediction head.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from src.models.molecule_gnn import AffinityHead, MoleculeGNN
from src.models.protein_encoder import ProteinEncoder


class CoAttention(nn.Module):
    """Bidirectional attention: molecule atoms attend to protein residues
    and vice versa, producing attention-pooled context vectors for each
    modality (a lightweight version of the cross-attention used in
    structure-aware binding-affinity models).
    """

    def __init__(self, protein_dim: int, molecule_dim: int, hidden_dim: int = 256, heads: int = 4):
        super().__init__()
        self.protein_proj = nn.Linear(protein_dim, hidden_dim)
        self.molecule_proj = nn.Linear(molecule_dim, hidden_dim)

        self.mol_to_prot_attn = nn.MultiheadAttention(hidden_dim, heads, batch_first=True)
        self.prot_to_mol_attn = nn.MultiheadAttention(hidden_dim, heads, batch_first=True)

    def forward(
        self,
        residue_embeddings: torch.Tensor,  # (B, Lp, protein_dim)
        residue_pad_mask: torch.Tensor,  # (B, Lp) True at pad
        atom_embeddings: torch.Tensor,  # (B, La, molecule_dim) — dense-padded per graph
        atom_pad_mask: torch.Tensor,  # (B, La) True at pad
    ) -> Dict[str, torch.Tensor]:
        p = self.protein_proj(residue_embeddings)
        m = self.molecule_proj(atom_embeddings)

        # molecule queries attend over protein residues -> ligand-context-of-target
        mol_ctx, _ = self.mol_to_prot_attn(m, p, p, key_padding_mask=residue_pad_mask)
        # protein queries attend over molecule atoms -> target-context-of-ligand
        prot_ctx, _ = self.prot_to_mol_attn(p, m, m, key_padding_mask=atom_pad_mask)

        atom_valid = (~atom_pad_mask).unsqueeze(-1).float()
        residue_valid = (~residue_pad_mask).unsqueeze(-1).float()

        mol_pooled = (mol_ctx * atom_valid).sum(1) / atom_valid.sum(1).clamp(min=1e-6)
        prot_pooled = (prot_ctx * residue_valid).sum(1) / residue_valid.sum(1).clamp(min=1e-6)

        return {"molecule_context": mol_pooled, "protein_context": prot_pooled}


def dense_batch_from_scatter(
    node_features: torch.Tensor, batch_index: torch.LongTensor, num_graphs: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert a scatter-style batched graph (N_total, D) into a dense
    (num_graphs, max_nodes, D) tensor + pad mask, so it can be fed to
    nn.MultiheadAttention alongside the (already-dense) protein tensor.
    """
    device = node_features.device
    counts = torch.bincount(batch_index, minlength=num_graphs)
    max_nodes = int(counts.max().item()) if len(counts) else 0
    D = node_features.size(-1)

    dense = node_features.new_zeros((num_graphs, max_nodes, D))
    pad_mask = torch.ones((num_graphs, max_nodes), dtype=torch.bool, device=device)

    # position-within-graph index for each node
    pos = torch.zeros_like(batch_index)
    running = torch.zeros(num_graphs, dtype=torch.long, device=device)
    for i in range(batch_index.size(0)):
        g = batch_index[i]
        pos[i] = running[g]
        running[g] += 1

    dense[batch_index, pos] = node_features
    pad_mask[batch_index, pos] = False
    return dense, pad_mask


class FusionAffinityModel(nn.Module):
    """Full pipeline: ProteinEncoder + MoleculeGNN + CoAttention + AffinityHead.

    Designed to optionally warm-start from independently pretrained
    protein-structure and molecule-affinity checkpoints (see
    `scripts/train.py --config configs/fusion_affinity.yaml`).
    """

    def __init__(
        self,
        protein_kwargs: Optional[dict] = None,
        molecule_kwargs: Optional[dict] = None,
        fusion_hidden_dim: int = 256,
        fusion_heads: int = 4,
        head_kwargs: Optional[dict] = None,
    ):
        super().__init__()
        self.protein_encoder = ProteinEncoder(**(protein_kwargs or {}))
        self.molecule_gnn = MoleculeGNN(**(molecule_kwargs or {}))

        self.co_attention = CoAttention(
            protein_dim=self.protein_encoder.residue_out_dim,
            molecule_dim=self.molecule_gnn.input_proj.out_features,
            hidden_dim=fusion_hidden_dim,
            heads=fusion_heads,
        )

        fused_dim = fusion_hidden_dim * 2 + self.protein_encoder.pooled_dim + self.molecule_gnn.pooled_dim
        self.fuse_proj = nn.Sequential(
            nn.Linear(fused_dim, fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.head = AffinityHead(fusion_hidden_dim, **(head_kwargs or {}))

    def forward(
        self,
        input_ids: torch.LongTensor,
        atom_features: torch.Tensor,
        edge_index: torch.LongTensor,
        edge_features: torch.Tensor,
        batch_index: torch.LongTensor,
        num_graphs: int,
    ) -> Dict[str, torch.Tensor]:
        prot_out = self.protein_encoder(input_ids)

        # molecule node-level embeddings, needed for co-attention (pre-pool)
        mol_hidden = self.molecule_gnn.input_proj(atom_features)
        for conv, ln in zip(self.molecule_gnn.layers, self.molecule_gnn.layer_norms):
            mol_hidden = torch.relu(ln(conv(mol_hidden, edge_index, edge_features)))
        mol_out = self.molecule_gnn(atom_features, edge_index, edge_features, batch_index, num_graphs)

        atom_dense, atom_pad_mask = dense_batch_from_scatter(mol_hidden, batch_index, num_graphs)

        attn_out = self.co_attention(
            residue_embeddings=prot_out["residue_embeddings"],
            residue_pad_mask=prot_out["pad_mask"],
            atom_embeddings=atom_dense,
            atom_pad_mask=atom_pad_mask,
        )

        fused = torch.cat(
            [
                attn_out["molecule_context"],
                attn_out["protein_context"],
                prot_out["pooled_embedding"],
                mol_out["pooled_embedding"],
            ],
            dim=-1,
        )
        fused = self.fuse_proj(fused)
        head_out = self.head(fused)

        return {
            "protein_pooled": prot_out["pooled_embedding"],
            "molecule_pooled": mol_out["pooled_embedding"],
            "fused_embedding": fused,
            **head_out,
        }
