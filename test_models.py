import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
import torch

from src.data.protein_features import encode_sequence
from src.models.protein_encoder import ProteinStructureModel

rdkit = pytest.importorskip("rdkit", reason="RDKit not installed in this environment")
from src.data.molecule_features import batch_molecule_graphs, smiles_to_graph
from src.models.fusion import FusionAffinityModel
from src.models.molecule_gnn import MoleculeAffinityModel

ASPIRIN = "CC(=O)OC1=CC=CC=C1C(=O)O"
IBUPROFEN = "CC(C)Cc1ccc(cc1)C(C)C(=O)O"


def make_toy_batch_protein(batch_size=3, seq_len=32):
    seqs = ["ACDEFGHIKLMNPQRSTVWY" * 3][:seq_len]
    input_ids = torch.stack([encode_sequence(s, max_len=seq_len) for s in seqs * batch_size])
    return input_ids


def test_protein_structure_model_forward_shapes():
    model = ProteinStructureModel(
        encoder_kwargs=dict(
            embed_dim=32, lstm_hidden=32, transformer_layers=2, transformer_heads=4, pooled_dim=64
        )
    )
    input_ids = make_toy_batch_protein(batch_size=2, seq_len=40)
    out = model(input_ids)

    B, L = input_ids.shape
    assert out["residue_embeddings"].shape[:2] == (B, L)
    assert out["pooled_embedding"].shape == (B, 64)
    assert out["ss8_logits"].shape[:2] == (B, L)
    assert out["contact_logits"].shape == (B, L, L)


def test_molecule_affinity_model_forward_shapes():
    graphs = [smiles_to_graph(ASPIRIN), smiles_to_graph(IBUPROFEN)]
    atom_feats, edge_index, edge_feats, batch_idx = batch_molecule_graphs(graphs)

    model = MoleculeAffinityModel(gnn_kwargs=dict(hidden_dim=32, num_layers=2, pooled_dim=64))
    out = model(atom_feats, edge_index, edge_feats, batch_idx, num_graphs=2)

    assert out["pooled_embedding"].shape == (2, 64)
    assert out["pActivity_pred"].shape == (2,)
    assert out["activity_logit"].shape == (2,)


def test_fusion_model_forward_shapes():
    graphs = [smiles_to_graph(ASPIRIN), smiles_to_graph(IBUPROFEN)]
    atom_feats, edge_index, edge_feats, batch_idx = batch_molecule_graphs(graphs)

    seq_len = 30
    input_ids = torch.stack([encode_sequence("ACDEFGHIKLMNPQRSTVWY" * 2, max_len=seq_len) for _ in range(2)])

    model = FusionAffinityModel(
        protein_kwargs=dict(
            embed_dim=32, lstm_hidden=32, transformer_layers=2, transformer_heads=4, pooled_dim=64
        ),
        molecule_kwargs=dict(hidden_dim=32, num_layers=2, pooled_dim=64),
        fusion_hidden_dim=64,
        fusion_heads=4,
    )
    out = model(
        input_ids=input_ids,
        atom_features=atom_feats,
        edge_index=edge_index,
        edge_features=edge_feats,
        batch_index=batch_idx,
        num_graphs=2,
    )

    assert out["pActivity_pred"].shape == (2,)
    assert out["activity_logit"].shape == (2,)
    assert out["fused_embedding"].shape == (2, 64)
