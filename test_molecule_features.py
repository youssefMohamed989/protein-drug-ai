import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

rdkit = pytest.importorskip("rdkit", reason="RDKit not installed in this environment")

from src.data.molecule_features import (
    ATOM_FEATURE_DIM,
    BOND_FEATURE_DIM,
    batch_molecule_graphs,
    morgan_fingerprint,
    scaffold_smiles,
    smiles_to_graph,
)

ASPIRIN = "CC(=O)OC1=CC=CC=C1C(=O)O"
INVALID = "not_a_smiles!!!"


def test_smiles_to_graph_valid():
    graph = smiles_to_graph(ASPIRIN)
    assert graph is not None
    assert graph.atom_features.shape[1] == ATOM_FEATURE_DIM
    assert graph.edge_features.shape[1] == BOND_FEATURE_DIM
    assert graph.num_atoms == graph.atom_features.shape[0]
    assert graph.edge_index.shape[0] == 2


def test_smiles_to_graph_invalid_returns_none():
    assert smiles_to_graph(INVALID) is None


def test_batch_molecule_graphs():
    g1 = smiles_to_graph(ASPIRIN)
    g2 = smiles_to_graph("CC(C)Cc1ccc(cc1)C(C)C(=O)O")  # ibuprofen
    atom_feats, edge_index, edge_feats, batch_idx = batch_molecule_graphs([g1, g2])
    assert atom_feats.shape[0] == g1.num_atoms + g2.num_atoms
    assert batch_idx.max().item() == 1
    assert edge_index.max().item() < atom_feats.shape[0]


def test_morgan_fingerprint_shape():
    fp = morgan_fingerprint(ASPIRIN, n_bits=1024)
    assert fp.shape == (1024,)


def test_scaffold_smiles_runs():
    scaffold = scaffold_smiles(ASPIRIN)
    assert isinstance(scaffold, str)
