#!/usr/bin/env python
"""Full docking pipeline: ML pre-screen -> receptor/ligand prep -> AutoDock
Vina docking -> consensus rescoring.

Example:
    python scripts/dock.py \
        --receptor_pdb data/raw/receptors/target.pdb \
        --library data/raw/candidates.smi \
        --target_fasta data/raw/target.fasta \
        --ml_checkpoint checkpoints/fusion_affinity/best.pt \
        --ml_config configs/fusion_affinity.yaml \
        --prescreen_top_k 10 \
        --out_csv checkpoints/docking_results.csv

If --ml_checkpoint is omitted, every ligand in the library is docked
directly (no ML pre-screen) — fine for small libraries, slow for large ones.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.data.protein_features import parse_fasta
from src.docking.ml_rescoring import consensus_rank, ml_prescreen
from src.docking.prepare import estimate_binding_box, prepare_ligand, prepare_receptor, receptor_pdb_to_pdbqt
from src.docking.vina_runner import VinaDocker
from src.inference.predictor import AffinityPredictor
from src.inference.screen import load_library
from src.utils.config import Config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--receptor_pdb", type=str, required=True)
    parser.add_argument("--library", type=str, required=True, help=".smi or .csv")
    parser.add_argument("--target_fasta", type=str, default=None, help="required if using --ml_checkpoint")
    parser.add_argument("--ml_checkpoint", type=str, default=None)
    parser.add_argument("--ml_config", type=str, default="configs/fusion_affinity.yaml")
    parser.add_argument("--prescreen_top_k", type=int, default=20)
    parser.add_argument(
        "--ligand_resname",
        type=str,
        default=None,
        help="co-crystallized ligand resname to center the search box on, if known",
    )
    parser.add_argument("--box_center", type=float, nargs=3, default=None)
    parser.add_argument("--box_size", type=float, nargs=3, default=None)
    parser.add_argument("--exhaustiveness", type=int, default=8)
    parser.add_argument("--n_poses", type=int, default=9)
    parser.add_argument(
        "--skip_minimize_warning",
        action="store_true",
        help="suppress the reminder to MD-minimize predicted (non-experimental) structures first",
    )
    parser.add_argument("--out_csv", type=str, default="checkpoints/docking_results.csv")
    parser.add_argument("--out_poses_dir", type=str, default="checkpoints/docking_poses")
    args = parser.parse_args()

    if not args.skip_minimize_warning:
        print(
            "[dock.py] NOTE: if --receptor_pdb is a *predicted* structure "
            "(e.g. from scripts/predict_structure.py) rather than an "
            "experimental one, run scripts/run_md.py --minimize_only on it "
            "first — Meeko's docking prep expects relaxed, non-clashing "
            "geometry. See docs/docking_and_md.md."
        )

    # 1. Receptor prep
    fixed_pdb = str(Path(args.out_poses_dir) / "receptor_fixed.pdb")
    prepare_receptor(args.receptor_pdb, fixed_pdb)
    receptor_pdbqt = str(Path(args.out_poses_dir) / "receptor.pdbqt")
    receptor_pdb_to_pdbqt(fixed_pdb, receptor_pdbqt)

    center = tuple(args.box_center) if args.box_center else None
    size = tuple(args.box_size) if args.box_size else None
    if center is None or size is None:
        est_center, est_size = estimate_binding_box(fixed_pdb, ligand_resname=args.ligand_resname)
        center = center or est_center
        size = size or est_size
    print(f"[dock.py] search box center={center} size={size}")

    # 2. Load candidate library
    library = load_library(args.library)  # list of (smiles, compound_id)
    print(f"[dock.py] loaded {len(library)} candidates from {args.library}")

    # 3. Optional ML pre-screen
    shortlist = library
    ml_results_by_id = {}
    if args.ml_checkpoint:
        if not args.target_fasta:
            raise ValueError("--target_fasta is required when using --ml_checkpoint")
        target_id, target_seq = parse_fasta(args.target_fasta)[0]
        cfg = Config.from_yaml(args.ml_config)
        predictor = AffinityPredictor(args.ml_checkpoint, cfg, device="cpu")
        ranked = ml_prescreen(predictor, target_seq, library, top_k=args.prescreen_top_k)
        shortlist = [(r["smiles"], r["compound_id"]) for r in ranked]
        ml_results_by_id = {r["compound_id"]: r for r in ranked}
        print(f"[dock.py] ML pre-screen shortlisted {len(shortlist)} / {len(library)} candidates")

    # 4. Dock the shortlist with real AutoDock Vina
    docker = VinaDocker(
        receptor_pdbqt,
        center=center,
        box_size=size,
        exhaustiveness=args.exhaustiveness,
        n_poses=args.n_poses,
    )

    vina_best_by_id = {}
    Path(args.out_poses_dir).mkdir(parents=True, exist_ok=True)
    for smiles, compound_id in shortlist:
        lig = prepare_ligand(smiles)
        if lig is None:
            print(f"[dock.py] skipping unparsable SMILES for {compound_id}")
            continue
        result = docker.dock_ligand_pdbqt(lig.pdbqt_string, smiles=smiles)
        vina_best_by_id[compound_id] = result.best_affinity
        if result.poses:
            docker.write_best_pose(result, str(Path(args.out_poses_dir) / f"{compound_id}_best.pdbqt"))
        print(f"[dock.py] {compound_id}: best Vina affinity = {result.best_affinity} kcal/mol")

    # 5. Consensus rescoring (only meaningful if ML pre-screen was used)
    if ml_results_by_id:
        consensus = consensus_rank(list(ml_results_by_id.values()), vina_best_by_id)
        df = pd.DataFrame([c.__dict__ for c in consensus])
    else:
        df = pd.DataFrame(
            [{"compound_id": cid, "vina_affinity_kcal_mol": aff} for cid, aff in vina_best_by_id.items()]
        ).sort_values("vina_affinity_kcal_mol")

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(df.to_string(index=False))
    print(f"\nSaved full results to {args.out_csv}; best poses in {args.out_poses_dir}")


if __name__ == "__main__":
    main()
