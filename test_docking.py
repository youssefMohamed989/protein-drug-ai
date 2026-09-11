import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

rdkit = pytest.importorskip("rdkit", reason="RDKit not installed")
meeko = pytest.importorskip("meeko", reason="Meeko not installed")

from src.docking.ml_rescoring import consensus_rank, vina_score_to_pseudo_pActivity
from src.docking.prepare import prepare_ligand

ASPIRIN = "CC(=O)OC1=CC=CC=C1C(=O)O"


def test_prepare_ligand_valid():
    lig = prepare_ligand(ASPIRIN)
    assert lig is not None
    assert "ROOT" in lig.pdbqt_string
    assert lig.num_rotatable_bonds >= 1


def test_prepare_ligand_invalid_returns_none():
    lig = prepare_ligand("not_a_smiles!!!")
    assert lig is None


def test_vina_score_to_pseudo_pActivity_monotonic():
    # more negative (tighter) Vina affinity -> higher pseudo-pActivity
    weak = vina_score_to_pseudo_pActivity(-5.0)
    strong = vina_score_to_pseudo_pActivity(-10.0)
    assert strong > weak


def test_consensus_rank_combines_scores():
    ml_results = [
        {"smiles": "A", "compound_id": "c1", "pActivity_pred": 7.0, "activity_prob": 0.8},
        {"smiles": "B", "compound_id": "c2", "pActivity_pred": 5.0, "activity_prob": 0.3},
    ]
    vina_results = {"c1": -9.0, "c2": -6.0}

    ranked = consensus_rank(ml_results, vina_results)
    assert ranked[0].compound_id == "c1"  # both ML and Vina favor c1
    assert len(ranked) == 2


def test_consensus_rank_handles_missing_vina_score():
    ml_results = [{"smiles": "A", "compound_id": "c1", "pActivity_pred": 7.0, "activity_prob": 0.8}]
    ranked = consensus_rank(ml_results, {})
    assert len(ranked) == 1
    assert ranked[0].vina_affinity_kcal_mol is None
