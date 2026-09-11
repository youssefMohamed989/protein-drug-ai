"""Graph Isomorphism Network (GIN) encoder for molecules, built from scratch
(no torch_geometric dependency) using scatter-add message passing.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.molecule_features import ATOM_FEATURE_DIM, BOND_FEATURE_DIM


def scatter_add(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    """Minimal scatter-add: sum `src` rows into `dim_size` buckets by `index`."""
    shape = (dim_size,) + src.shape[1:]
    out = src.new_zeros(shape)
    out.index_add_(0, index, src)
    return out


def scatter_mean(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    summed = scatter_add(src, index, dim_size)
    counts = scatter_add(torch.ones(src.size(0), 1, device=src.device), index, dim_size).clamp(min=1)
    return summed / counts


class GINConv(nn.Module):
    """One GIN message-passing layer with edge-feature-aware messages."""

    def __init__(
        self, node_dim: int, edge_dim: int, hidden_dim: int, eps: float = 0.0, train_eps: bool = True
    ):
        super().__init__()
        self.edge_proj = nn.Linear(edge_dim, node_dim)
        self.mlp = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_dim),
        )
        if train_eps:
            self.eps = nn.Parameter(torch.tensor(eps, dtype=torch.float32))
        else:
            self.register_buffer("eps", torch.tensor(eps, dtype=torch.float32))

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.LongTensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        src, dst = edge_index[0], edge_index[1]
        edge_embed = self.edge_proj(edge_attr)
        messages = x[src] + edge_embed  # additive edge-conditioned message
        aggregated = scatter_add(messages, dst, dim_size=x.size(0))
        out = (1 + self.eps) * x + aggregated
        return self.mlp(out)


class MoleculeGNN(nn.Module):
    """Molecule -> pooled embedding via stacked GIN layers + graph pooling."""

    def __init__(
        self,
        atom_feature_dim: int = ATOM_FEATURE_DIM,
        edge_feature_dim: int = BOND_FEATURE_DIM,
        hidden_dim: int = 128,
        num_layers: int = 4,
        pooled_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(atom_feature_dim, hidden_dim)
        self.layers = nn.ModuleList(
            [GINConv(hidden_dim, edge_feature_dim, hidden_dim * 2) for _ in range(num_layers)]
        )
        self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

        # jumping-knowledge: concat all layer outputs before pooling
        self.jk_proj = nn.Sequential(
            nn.Linear(hidden_dim * (num_layers + 1), pooled_dim),
            nn.GELU(),
            nn.LayerNorm(pooled_dim),
        )
        self.pooled_dim = pooled_dim

    def forward(
        self,
        atom_features: torch.Tensor,
        edge_index: torch.LongTensor,
        edge_features: torch.Tensor,
        batch_index: torch.LongTensor,
        num_graphs: int,
    ) -> Dict[str, torch.Tensor]:
        x = self.input_proj(atom_features)
        layer_outputs = [x]

        for conv, ln in zip(self.layers, self.layer_norms):
            x = conv(x, edge_index, edge_features)
            x = ln(x)
            x = F.relu(x)
            x = self.dropout(x)
            layer_outputs.append(x)

        node_repr = torch.cat(layer_outputs, dim=-1)  # (N, hidden*(L+1))

        # combined mean + max readout, standard for molecular property prediction
        mean_pool = scatter_mean(node_repr, batch_index, dim_size=num_graphs)
        max_pool = node_repr.new_full((num_graphs, node_repr.size(-1)), float("-inf"))
        max_pool = max_pool.scatter_reduce(
            0, batch_index.unsqueeze(-1).expand_as(node_repr), node_repr, reduce="amax", include_self=True
        )
        max_pool = torch.nan_to_num(max_pool, neginf=0.0)

        pooled = (mean_pool + max_pool) / 2.0
        pooled = self.jk_proj(pooled)

        return {"node_embeddings": x, "pooled_embedding": pooled}


class AffinityHead(nn.Module):
    """Joint regression (pKd/pIC50) + classification (active/inactive) head."""

    def __init__(self, in_dim: int, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.regression_head = nn.Linear(hidden_dim, 1)
        self.classification_head = nn.Linear(hidden_dim, 1)

    def forward(self, embedding: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.shared(embedding)
        return {
            "pActivity_pred": self.regression_head(h).squeeze(-1),
            "activity_logit": self.classification_head(h).squeeze(-1),
        }


class MoleculeAffinityModel(nn.Module):
    """Ligand-only baseline: SMILES -> GNN -> affinity/activity (no target)."""

    def __init__(self, gnn_kwargs: Optional[dict] = None, head_kwargs: Optional[dict] = None):
        super().__init__()
        self.gnn = MoleculeGNN(**(gnn_kwargs or {}))
        self.head = AffinityHead(self.gnn.pooled_dim, **(head_kwargs or {}))

    def forward(
        self, atom_features, edge_index, edge_features, batch_index, num_graphs
    ) -> Dict[str, torch.Tensor]:
        gnn_out = self.gnn(atom_features, edge_index, edge_features, batch_index, num_graphs)
        head_out = self.head(gnn_out["pooled_embedding"])
        return {**gnn_out, **head_out}
