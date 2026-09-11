import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

transformers = pytest.importorskip("transformers", reason="transformers not installed")

import torch

from src.data.protein_features import encode_sequence
from src.models.esm_backend import ESM2EmbeddingBackend
from src.models.protein_encoder import ProteinEncoder


def _offline_backend():
    # pretrained=False builds the real ESM-2 architecture with randomly
    # initialized weights, entirely offline -- see esm_backend.py docstring
    # for why real pretrained weights aren't downloaded in this test suite.
    return ESM2EmbeddingBackend(
        model_name="facebook/esm2_t6_8M_UR50D",
        pretrained=False,
        num_layers_override=2,
    )


def test_backend_output_dim_matches_config():
    backend = _offline_backend()
    assert backend.output_dim == 320  # esm2_t6_8M hidden size


def test_backend_forward_shape():
    backend = _offline_backend()
    ids = encode_sequence("ACDEFGHIKL", max_len=16).unsqueeze(0)
    out = backend(ids)
    assert out.shape == (1, 16, 320)


def test_backend_frozen_by_default_no_grad():
    backend = _offline_backend()
    assert backend.freeze is True
    assert all(not p.requires_grad for p in backend.model.parameters())


def test_protein_encoder_uses_backend_output_dim():
    backend = _offline_backend()
    encoder = ProteinEncoder(
        embedding_backend=backend, lstm_hidden=16, transformer_layers=1, transformer_heads=2, pooled_dim=32
    )
    assert encoder.token_embed is None  # replaced by the backend

    ids = encode_sequence("ACDEFGHIKL", max_len=16).unsqueeze(0)
    out = encoder(ids)
    assert out["pooled_embedding"].shape == (1, 32)


def test_protein_encoder_backward_trains_downstream_not_esm():
    backend = _offline_backend()
    encoder = ProteinEncoder(
        embedding_backend=backend, lstm_hidden=16, transformer_layers=1, transformer_heads=2, pooled_dim=32
    )
    ids = encode_sequence("ACDEFGHIKL", max_len=16).unsqueeze(0)
    out = encoder(ids)
    out["pooled_embedding"].sum().backward()

    assert all(p.grad is None for p in backend.model.parameters())
    assert any(p.grad is not None for p in encoder.bilstm.parameters())
