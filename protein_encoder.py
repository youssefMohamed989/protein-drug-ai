"""BiLSTM + Transformer protein encoder with structure prediction heads."""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.protein_features import PAD_IDX, SS8_CLASSES, VOCAB


class PositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int = 2048):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float32) * (-torch.log(torch.tensor(10000.0)) / dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class ProteinEncoder(nn.Module):
    """Sequence -> per-residue embeddings + pooled embedding.

    Architecture: learned token embedding -> BiLSTM -> Transformer encoder
    (captures both local motifs via recurrence and long-range dependencies
    via self-attention, echoing the local+global design used in real
    protein language models, at a fraction of the parameter count).
    """

    def __init__(
        self,
        vocab_size: int = len(VOCAB),
        embed_dim: int = 128,
        lstm_hidden: int = 128,
        transformer_layers: int = 4,
        transformer_heads: int = 8,
        pooled_dim: int = 256,
        dropout: float = 0.1,
        pad_idx: int = PAD_IDX,
        embedding_backend: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.pad_idx = pad_idx
        self.embedding_backend = embedding_backend  # e.g. frozen ESM-2 wrapper

        if embedding_backend is None:
            self.token_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        else:
            embed_dim = embedding_backend.output_dim  # type: ignore[attr-defined]
            self.token_embed = None

        self.pos_encoding = PositionalEncoding(embed_dim)

        self.bilstm = nn.LSTM(
            embed_dim, lstm_hidden, batch_first=True, bidirectional=True, num_layers=2, dropout=dropout
        )
        lstm_out_dim = lstm_hidden * 2

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=lstm_out_dim,
            nhead=transformer_heads,
            dim_feedforward=lstm_out_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)

        self.pool_proj = nn.Sequential(
            nn.Linear(lstm_out_dim, pooled_dim),
            nn.GELU(),
            nn.LayerNorm(pooled_dim),
        )
        self.residue_out_dim = lstm_out_dim
        self.pooled_dim = pooled_dim

    def forward(self, input_ids: torch.LongTensor) -> Dict[str, torch.Tensor]:
        pad_mask = input_ids.eq(self.pad_idx)  # (B, L) True at pad positions

        if self.embedding_backend is not None:
            x = self.embedding_backend(input_ids)
        else:
            x = self.token_embed(input_ids)
        x = self.pos_encoding(x)

        lengths = (~pad_mask).sum(dim=1).clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_out, _ = self.bilstm(packed)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out, batch_first=True, total_length=input_ids.size(1)
        )

        residue_embeds = self.transformer(lstm_out, src_key_padding_mask=pad_mask)

        # mean-pool over valid (non-pad) residues for a fixed-size protein embedding
        mask = (~pad_mask).unsqueeze(-1).float()
        pooled = (residue_embeds * mask).sum(1) / mask.sum(1).clamp(min=1e-6)
        pooled = self.pool_proj(pooled)

        return {
            "residue_embeddings": residue_embeds,  # (B, L, lstm_out_dim)
            "pooled_embedding": pooled,  # (B, pooled_dim)
            "pad_mask": pad_mask,  # (B, L)
        }


class SecondaryStructureHead(nn.Module):
    """Per-residue Q8 secondary structure classifier."""

    def __init__(self, in_dim: int, num_classes: int = len(SS8_CLASSES), dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_dim, num_classes),
        )

    def forward(self, residue_embeddings: torch.Tensor) -> torch.Tensor:
        return self.net(residue_embeddings)  # (B, L, num_classes)


class ContactMapHead(nn.Module):
    """Predicts an L x L residue-residue contact map from pairwise residue
    embedding outer concatenation (symmetrized), similar in spirit to the
    pairwise representation used by real structure predictors, just much
    smaller.
    """

    def __init__(self, in_dim: int, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.pair_proj = nn.Sequential(
            nn.Linear(in_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, residue_embeddings: torch.Tensor) -> torch.Tensor:
        B, L, D = residue_embeddings.shape
        ei = residue_embeddings.unsqueeze(2).expand(B, L, L, D)
        ej = residue_embeddings.unsqueeze(1).expand(B, L, L, D)
        pair = torch.cat([ei, ej], dim=-1)
        logits = self.pair_proj(pair).squeeze(-1)  # (B, L, L)
        logits = (logits + logits.transpose(1, 2)) / 2.0  # symmetrize
        return logits


class ProteinStructureModel(nn.Module):
    """Full model: encoder + SS8 head + contact-map head, trained jointly."""

    def __init__(self, encoder_kwargs: Optional[dict] = None):
        super().__init__()
        self.encoder = ProteinEncoder(**(encoder_kwargs or {}))
        self.ss8_head = SecondaryStructureHead(self.encoder.residue_out_dim)
        self.contact_head = ContactMapHead(self.encoder.residue_out_dim)

    def forward(self, input_ids: torch.LongTensor) -> Dict[str, torch.Tensor]:
        enc_out = self.encoder(input_ids)
        ss8_logits = self.ss8_head(enc_out["residue_embeddings"])
        contact_logits = self.contact_head(enc_out["residue_embeddings"])
        return {
            **enc_out,
            "ss8_logits": ss8_logits,
            "contact_logits": contact_logits,
        }
