#!/usr/bin/env python
"""CLI: minimize (and optionally simulate) a structure with OpenMM, then
report standard trajectory analysis (RMSD/RMSF/Rg[/DSSP]).

Example (full pipeline):
    python scripts/run_md.py --pdb data/raw/receptors/target_fixed.pdb \
        --out_dir checkpoints/md_run --production_ps 50

Example (minimize-only, e.g. to prep a predicted structure for docking):
    python scripts/run_md.py --pdb checkpoints/predicted_structure_fixed.pdb \
        --out_dir checkpoints/md_run --minimize_only
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.md.analysis import analyze_trajectory, summarize
from src.md.simulation import MDSimulation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdb", type=str, required=True, help="a PDBFixer-cleaned, all-atom PDB")
    parser.add_argument("--out_dir", type=str, default="checkpoints/md_run")
    parser.add_argument("--run_name", type=str, default="run")
    parser.add_argument("--solvent", type=str, default="implicit", choices=["vacuum", "implicit", "explicit"])
    parser.add_argument("--temperature_k", type=float, default=300.0)
    parser.add_argument("--minimize_only", action="store_true")
    parser.add_argument("--equilibration_ps", type=float, default=10.0)
    parser.add_argument("--production_ps", type=float, default=50.0)
    parser.add_argument("--timestep_fs", type=float, default=2.0)
    parser.add_argument("--report_interval_steps", type=int, default=500)
    parser.add_argument(
        "--platform_name",
        type=str,
        default=None,
        choices=["Reference", "CPU", "CUDA", "OpenCL"],
        help="explicit OpenMM platform (default: auto-select the fastest available)",
    )
    parser.add_argument(
        "--platform_property",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="platform-specific property, e.g. --platform_property Precision=mixed "
        "(repeatable; only used with --platform_name)",
    )
    args = parser.parse_args()

    platform_properties = None
    if args.platform_property:
        platform_properties = dict(kv.split("=", 1) for kv in args.platform_property)

    sim = MDSimulation(
        args.pdb,
        solvent=args.solvent,
        temperature_k=args.temperature_k,
        timestep_fs=args.timestep_fs,
        platform_name=args.platform_name,
        platform_properties=platform_properties,
    )
    print(f"[run_md.py] running on OpenMM platform: {sim.platform_name}")

    eq_steps = int(args.equilibration_ps * 1000 / args.timestep_fs)
    prod_steps = int(args.production_ps * 1000 / args.timestep_fs)

    result = sim.run_full_pipeline(
        out_dir=args.out_dir,
        minimize_only=args.minimize_only,
        equilibration_steps=eq_steps,
        production_steps=prod_steps,
        report_interval=args.report_interval_steps,
        run_name=args.run_name,
    )

    print(f"[run_md.py] minimized structure: {result.minimized_pdb}")
    print(f"[run_md.py] final potential energy: {result.final_potential_energy_kj_mol:.2f} kJ/mol")

    if result.trajectory_dcd:
        print(f"[run_md.py] trajectory: {result.trajectory_dcd}")
        print(f"[run_md.py] state log: {result.log_csv}")

        analysis = analyze_trajectory(result.minimized_pdb, result.trajectory_dcd)
        summary = summarize(analysis)
        print("[run_md.py] trajectory analysis summary:")
        for k, v in summary.items():
            print(f"    {k}: {v}")


if __name__ == "__main__":
    main()
