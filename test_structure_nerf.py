import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from src.structure.nerf import BOND_LEN_C_N, BOND_LEN_CA_C, BOND_LEN_N_CA, build_backbone, place_next_atom
from src.structure.pdb_writer import write_backbone_pdb
from src.structure.torsion_head import TorsionHead, torsion_angle_loss


def test_build_backbone_bond_lengths():
    seq = "ACDEFGHIKL"
    phi = [None] + [-60.0] * (len(seq) - 1)
    psi = [-45.0] * (len(seq) - 1) + [None]
    bb = build_backbone(seq, phi, psi)

    n_ca = np.linalg.norm(bb.ca_coords - bb.n_coords, axis=1)
    ca_c = np.linalg.norm(bb.c_coords - bb.ca_coords, axis=1)
    c_n_next = np.linalg.norm(bb.c_coords[:-1] - bb.n_coords[1:], axis=1)

    assert np.allclose(n_ca, BOND_LEN_N_CA, atol=1e-3)
    assert np.allclose(ca_c, BOND_LEN_CA_C, atol=1e-3)
    assert np.allclose(c_n_next, BOND_LEN_C_N, atol=1e-3)


def test_build_backbone_shapes():
    seq = "GATTACA"
    phi = [None] + [-60.0] * (len(seq) - 1)
    psi = [-45.0] * (len(seq) - 1) + [None]
    bb = build_backbone(seq, phi, psi)
    L = len(seq)
    assert bb.n_coords.shape == (L, 3)
    assert bb.ca_coords.shape == (L, 3)
    assert bb.c_coords.shape == (L, 3)
    assert bb.o_coords.shape == (L, 3)


def test_write_backbone_pdb(tmp_path):
    seq = "AC"
    bb = build_backbone(seq, [None, -60.0], [-45.0, None])
    out_path = tmp_path / "test.pdb"
    write_backbone_pdb(bb, str(out_path))
    content = out_path.read_text()
    assert content.startswith("ATOM")
    assert "END" in content
    assert content.count("ATOM") == 8  # 2 residues * 4 backbone atoms


def test_torsion_head_output_shapes():
    head = TorsionHead(in_dim=32)
    x = torch.randn(2, 10, 32)
    out = head(x)
    assert out["torsion_unit_vectors"].shape == (2, 10, 2, 2)
    assert out["phi"].shape == (2, 10)
    assert out["psi"].shape == (2, 10)


def test_torsion_angle_loss_zero_when_perfect():
    # construct unit vectors that exactly match given angles -> loss ~ 0
    phi_true = torch.tensor([[0.5, 1.0]])
    psi_true = torch.tensor([[float("nan"), -0.3]])

    unit_vec = torch.stack(
        [
            torch.stack([torch.sin(phi_true), torch.cos(phi_true)], dim=-1),
            torch.stack(
                [torch.sin(torch.nan_to_num(psi_true)), torch.cos(torch.nan_to_num(psi_true))], dim=-1
            ),
        ],
        dim=-2,
    )

    loss = torsion_angle_loss(unit_vec, phi_true, psi_true)
    assert loss.item() < 1e-5
