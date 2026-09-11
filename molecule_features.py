"""SMILES <-> molecular graph featurization for the GNN drug-discovery module."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem

    _RDKIT_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only when RDKit missing
    _RDKIT_AVAILABLE = False

ATOM_LIST = [
    "C",
    "N",
    "O",
    "S",
    "F",
    "Cl",
    "Br",
    "I",
    "P",
    "B",
    "Si",
    "Se",
    "H",
    "*",
]
HYBRIDIZATIONS = ["SP", "SP2", "SP3", "SP3D", "SP3D2", "UNSPECIFIED"]
BOND_TYPES = ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"]

# _one_hot() below always returns len(choices) + 1 dims (extra slot for
# "unknown"), so every one-hot block here is len(choices) + 1, not len(choices).
ATOM_FEATURE_DIM = (
    (len(ATOM_LIST) + 1)  # atom type (one-hot + unknown)
    + (6 + 1)  # degree 0-5 one-hot (+ unknown)
    + (len(HYBRIDIZATIONS) + 1)  # hybridization one-hot (+ unknown)
    + 1  # formal charge (scalar)
    + 1  # aromatic flag
    + 1  # in ring flag
    + 1  # num H
)
BOND_FEATURE_DIM = (len(BOND_TYPES) + 1) + 2  # bond type one-hot (+unknown) + conjugated + in-ring


@dataclass
class MoleculeGraph:
    """Sparse graph representation ready for a PyG-style or custom GNN."""

    smiles: str
    atom_features: torch.Tensor  # (N, ATOM_FEATURE_DIM)
    edge_index: torch.LongTensor  # (2, E) source/target atom indices (both directions)
    edge_features: torch.Tensor  # (E, BOND_FEATURE_DIM)
    num_atoms: int


def _one_hot(value, choices) -> List[float]:
    vec = [0.0] * (len(choices) + 1)
    if value in choices:
        vec[choices.index(value)] = 1.0
    else:
        vec[-1] = 1.0
    return vec


def _require_rdkit() -> None:
    if not _RDKIT_AVAILABLE:
        raise ImportError(
            "RDKit is required for molecule featurization. "
            "Install with `pip install rdkit` (or `rdkit-pypi`)."
        )


def smiles_to_graph(smiles: str, add_hs: bool = False) -> Optional[MoleculeGraph]:
    """Convert a SMILES string into a MoleculeGraph. Returns None if invalid."""
    _require_rdkit()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    if add_hs:
        mol = Chem.AddHs(mol)

    atom_feats = []
    for atom in mol.GetAtoms():
        feats = []
        feats += _one_hot(atom.GetSymbol(), ATOM_LIST)
        feats += _one_hot(atom.GetDegree(), [0, 1, 2, 3, 4, 5])
        feats += _one_hot(str(atom.GetHybridization()), HYBRIDIZATIONS)
        feats += [float(atom.GetFormalCharge())]
        feats += [1.0 if atom.GetIsAromatic() else 0.0]
        feats += [1.0 if atom.IsInRing() else 0.0]
        feats += [float(atom.GetTotalNumHs())]
        atom_feats.append(feats)

    src, dst, bond_feats = [], [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        feat = _one_hot(str(bond.GetBondType()), BOND_TYPES)
        feat += [1.0 if bond.GetIsConjugated() else 0.0]
        feat += [1.0 if bond.IsInRing() else 0.0]
        # undirected -> add both directions for message passing
        src += [i, j]
        dst += [j, i]
        bond_feats += [feat, feat]

    if len(bond_feats) == 0:
        # single-atom molecule: add a self-loop so the GNN doesn't choke
        src, dst = [0], [0]
        bond_feats = [[0.0] * BOND_FEATURE_DIM]

    return MoleculeGraph(
        smiles=smiles,
        atom_features=torch.tensor(atom_feats, dtype=torch.float32),
        edge_index=torch.tensor([src, dst], dtype=torch.long),
        edge_features=torch.tensor(bond_feats, dtype=torch.float32),
        num_atoms=mol.GetNumAtoms(),
    )


def batch_molecule_graphs(graphs: List[MoleculeGraph]):
    """Collate a list of MoleculeGraph into one disjoint-union batch graph.

    Returns (atom_features, edge_index, edge_features, batch_index) where
    batch_index[i] gives the graph id that atom i belongs to (standard
    PyTorch-Geometric-style batching, implemented from scratch so this repo
    has no hard dependency on `torch_geometric`).
    """
    atom_feats, edge_indices, edge_feats, batch_index = [], [], [], []
    node_offset = 0
    for gi, g in enumerate(graphs):
        atom_feats.append(g.atom_features)
        edge_indices.append(g.edge_index + node_offset)
        edge_feats.append(g.edge_features)
        batch_index += [gi] * g.num_atoms
        node_offset += g.num_atoms

    return (
        torch.cat(atom_feats, dim=0),
        torch.cat(edge_indices, dim=1),
        torch.cat(edge_feats, dim=0),
        torch.tensor(batch_index, dtype=torch.long),
    )


def morgan_fingerprint(smiles: str, radius: int = 2, n_bits: int = 2048) -> Optional[np.ndarray]:
    """Classic ECFP fingerprint, used for the Tanimoto sanity-filter in screening."""
    _require_rdkit()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.uint8)
    Chem.DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def scaffold_smiles(smiles: str) -> Optional[str]:
    """Bemis-Murcko scaffold, used by the scaffold splitter to avoid leakage."""
    _require_rdkit()
    from rdkit.Chem.Scaffolds import MurckoScaffold

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaffold)
    except Exception:
        return None
