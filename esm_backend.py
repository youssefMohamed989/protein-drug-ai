"""ESM-2 embedding backend: plugs a real pretrained protein language model
into `ProteinEncoder` via its `embedding_backend` hook, replacing the
learned from-scratch token embedding with per-residue ESM-2 embeddings.

Uses HuggingFace `transformers`' `EsmModel` rather than Meta's original
`fair-esm` package, since it needs no extra tokenizer plumbing and the
weights (`facebook/esm2_t6_8M_UR50D` through `esm2_t36_3B_UR50D`) are
hosted on the Hugging Face Hub under the same model names.

Network note: downloading pretrained weights requires internet access to
huggingface.co, which this pipeline does not assume. `ESM2EmbeddingBackend`
is exercised in this repo's tests with `pretrained=False` (a randomly
initialized ESM-2 architecture, built entirely offline via `EsmConfig`) —
this validates the tensor-shape contract with `ProteinEncoder` without
needing real weights. Set `pretrained=True` (the default) wherever you do
have network access to Hugging Face; the download happens once and is
cached under `~/.cache/huggingface`.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from src.data.protein_features import PAD_IDX, VOCAB

# ESM-2 alphabet order differs from this repo's VOCAB (see
# src/data/protein_features.py), so tokens must be remapped before being
# fed to the HF model rather than reusing our vocab indices directly.
_ESM_ALPHABET = [
    "<cls>",
    "<pad>",
    "<eos>",
    "<unk>",
    "L",
    "A",
    "G",
    "V",
    "S",
    "E",
    "R",
    "T",
    "I",
    "D",
    "P",
    "K",
    "Q",
    "N",
    "F",
    "Y",
    "M",
    "H",
    "W",
    "C",
    "X",
    "B",
    "U",
    "Z",
    "O",
    ".",
    "-",
    "<null_1>",
    "<mask>",
]
_ESM_TOKEN_TO_IDX = {tok: i for i, tok in enumerate(_ESM_ALPHABET)}
_ESM_PAD_IDX = _ESM_TOKEN_TO_IDX["<pad>"]

# Map this repo's VOCAB index -> ESM-2 vocab index, precomputed once.
_OUR_IDX_TO_ESM_IDX = torch.tensor(
    [_ESM_TOKEN_TO_IDX.get(tok, _ESM_TOKEN_TO_IDX["<unk>"]) for tok in VOCAB], dtype=torch.long
)

MODEL_NAME_TO_HIDDEN_DIM = {
    "facebook/esm2_t6_8M_UR50D": 320,
    "facebook/esm2_t12_35M_UR50D": 480,
    "facebook/esm2_t30_150M_UR50D": 640,
    "facebook/esm2_t33_650M_UR50D": 1280,
    "facebook/esm2_t36_3B_UR50D": 2560,
}


class ESM2EmbeddingBackend(nn.Module):
    """Drop-in `embedding_backend` for `ProteinEncoder`: takes the same
    `(B, L)` LongTensor of this repo's vocab indices that the learned
    `nn.Embedding` would, and returns `(B, L, output_dim)` per-residue
    embeddings from ESM-2 instead.

        encoder = ProteinEncoder(embedding_backend=ESM2EmbeddingBackend())

    `ProteinEncoder` reads `output_dim` off this module to size its
    downstream BiLSTM, so no other config changes are needed.
    """

    def __init__(
        self,
        model_name: str = "facebook/esm2_t6_8M_UR50D",
        pretrained: bool = True,
        freeze: bool = True,
        num_layers_override: Optional[int] = None,
    ):
        super().__init__()
        from transformers import EsmConfig, EsmModel

        if pretrained:
            self.model = EsmModel.from_pretrained(model_name, add_pooling_layer=False)
            self.output_dim = self.model.config.hidden_size
        else:
            # Fully offline path: build the real ESM-2 architecture (or a
            # scaled-down variant via num_layers_override, for fast tests)
            # with randomly initialized weights. Useful for validating the
            # integration/tensor-shape contract without network access;
            # NOT a substitute for real pretrained weights on real data.
            base_hidden = MODEL_NAME_TO_HIDDEN_DIM.get(model_name, 320)
            cfg = EsmConfig(
                vocab_size=len(_ESM_ALPHABET),
                hidden_size=base_hidden,
                num_hidden_layers=num_layers_override or 6,
                num_attention_heads=max(1, base_hidden // 64),
                intermediate_size=base_hidden * 4,
                max_position_embeddings=1026,
                pad_token_id=_ESM_PAD_IDX,
            )
            self.model = EsmModel(cfg, add_pooling_layer=False)
            self.output_dim = base_hidden

        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False
            self.model.eval()
        self.freeze = freeze

        self.register_buffer("_idx_map", _OUR_IDX_TO_ESM_IDX, persistent=False)

    def forward(self, input_ids: torch.LongTensor) -> torch.Tensor:
        """input_ids: (B, L) in this repo's vocab -> (B, L, output_dim)."""
        esm_ids = self._idx_map[input_ids.clamp(min=0, max=len(VOCAB) - 1)]
        attention_mask = (input_ids != PAD_IDX).long()

        ctx = torch.no_grad() if self.freeze else torch.enable_grad()
        with ctx:
            out = self.model(input_ids=esm_ids, attention_mask=attention_mask)
        return out.last_hidden_state  # (B, L, output_dim)
