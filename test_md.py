import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

openmm = pytest.importorskip("openmm", reason="OpenMM not installed")
mdtraj = pytest.importorskip("mdtraj", reason="mdtraj not installed")

from src.structure.nerf import build_backbone
from src.structure.pdb_writer import write_backbone_pdb


def _make_toy_pdb(tmp_path) -> str:
    seq = "A" * 8
    phi = [None] + [-120.0] * (len(seq) - 1)
    psi = [120.0] * (len(seq) - 1) + [None]
    bb = build_backbone(seq, phi, psi)
    path = str(tmp_path / "toy.pdb")
    write_backbone_pdb(bb, path)
    return path


def _fix_pdb(raw_pdb: str, out_pdb: str) -> str:
    from openmm.app import PDBFile
    from pdbfixer import PDBFixer

    fixer = PDBFixer(filename=raw_pdb)
    fixer.findMissingResidues()
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.removeHeterogens(keepWater=False)
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(7.4)
    with open(out_pdb, "w") as f:
        PDBFile.writeFile(fixer.topology, fixer.positions, f, keepIds=True)
    return out_pdb


def test_minimize_reduces_energy(tmp_path):
    from src.md.simulation import MDSimulation

    raw = _make_toy_pdb(tmp_path)
    fixed = _fix_pdb(raw, str(tmp_path / "fixed.pdb"))

    sim = MDSimulation(fixed, solvent="implicit")
    e0 = sim.get_potential_energy_kj_mol()
    sim.minimize(max_iterations=200)
    e1 = sim.get_potential_energy_kj_mol()

    assert e1 < e0


def test_run_full_pipeline_minimize_only(tmp_path):
    from src.md.simulation import MDSimulation

    raw = _make_toy_pdb(tmp_path)
    fixed = _fix_pdb(raw, str(tmp_path / "fixed.pdb"))

    sim = MDSimulation(fixed, solvent="implicit")
    result = sim.run_full_pipeline(out_dir=str(tmp_path / "out"), minimize_only=True)

    assert Path(result.minimized_pdb).exists()
    assert result.trajectory_dcd is None
