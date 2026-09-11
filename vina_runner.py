"""Thin, typed wrapper around the `vina` Python bindings (AutoDock Vina 1.2+).

Handles: loading a prepared receptor PDBQT, setting the search box, running
the Vina scoring function's local optimization/global search, and parsing
the resulting poses + binding-affinity estimates (kcal/mol).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple


@dataclass
class DockingPose:
    rank: int
    affinity_kcal_mol: float  # Vina's predicted binding free energy
    rmsd_lower_bound: float
    rmsd_upper_bound: float
    pose_pdbqt: str  # PDBQT block for this single pose


@dataclass
class DockingResult:
    ligand_smiles: str
    poses: List[DockingPose] = field(default_factory=list)

    @property
    def best_affinity(self) -> Optional[float]:
        return self.poses[0].affinity_kcal_mol if self.poses else None


class VinaDocker:
    """One receptor, many ligands: load the receptor + maps once, dock
    repeatedly (this amortizes the (comparatively) expensive receptor grid
    computation across a whole library).
    """

    def __init__(
        self,
        receptor_pdbqt: str,
        center: Tuple[float, float, float],
        box_size: Tuple[float, float, float],
        exhaustiveness: int = 8,
        n_poses: int = 9,
        scoring_function: str = "vina",
        cpu: int = 0,  # 0 = let Vina auto-detect
        seed: int = 42,
    ):
        from vina import Vina

        self.center = center
        self.box_size = box_size
        self.exhaustiveness = exhaustiveness
        self.n_poses = n_poses

        self.v = Vina(sf_name=scoring_function, cpu=cpu, seed=seed, verbosity=0)
        self.v.set_receptor(receptor_pdbqt)
        self.v.compute_vina_maps(center=list(center), box_size=list(box_size))

    def dock_ligand_pdbqt(self, ligand_pdbqt_string: str, smiles: str = "") -> DockingResult:
        """Dock a single ligand (already prepared as a PDBQT string)."""
        self.v.set_ligand_from_string(ligand_pdbqt_string)
        self.v.dock(exhaustiveness=self.exhaustiveness, n_poses=self.n_poses)

        energies = self.v.energies(
            n_poses=self.n_poses
        )  # (n_poses, 4): total, inter, intra, torsion (varies by version)
        poses_pdbqt = self.v.poses(n_poses=self.n_poses).split("ENDMDL")

        result = DockingResult(ligand_smiles=smiles)
        for i, energy_row in enumerate(energies):
            pose_block = poses_pdbqt[i] if i < len(poses_pdbqt) else ""
            result.poses.append(
                DockingPose(
                    rank=i + 1,
                    affinity_kcal_mol=float(energy_row[0]),
                    rmsd_lower_bound=float(energy_row[1]) if len(energy_row) > 1 else 0.0,
                    rmsd_upper_bound=float(energy_row[2]) if len(energy_row) > 2 else 0.0,
                    pose_pdbqt=pose_block,
                )
            )
        return result

    def score_only_pdbqt(self, ligand_pdbqt_string: str) -> float:
        """Score a ligand pose in-place (no search) — useful for rescoring
        externally generated poses, or scoring the output of a fast ML
        pre-screen before committing to full docking.
        """
        self.v.set_ligand_from_string(ligand_pdbqt_string)
        energies = self.v.score()
        return float(energies[0])

    def write_best_pose(self, result: DockingResult, output_pdbqt: str) -> Optional[str]:
        if not result.poses:
            return None
        Path(output_pdbqt).parent.mkdir(parents=True, exist_ok=True)
        with open(output_pdbqt, "w") as f:
            f.write(result.poses[0].pose_pdbqt)
        return output_pdbqt
