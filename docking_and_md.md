# Docking and molecular dynamics pipelines

This extends the core ML pipeline (protein structure prediction + drug
discovery) with two physics-based modules — **molecular docking**
(AutoDock Vina) and **molecular dynamics** (OpenMM) — plus the glue that
lets them talk to the ML models and to each other.

```
sequence ──▶ predict_structure.py ──▶ backbone-only PDB (NeRF)
                                              │
                                    PDBFixer (add sidechains/H)
                                              │
                                run_md.py --minimize_only  <-- REQUIRED, see below
                                              │
                                     relaxed, all-atom PDB
                                              │
                    ┌─────────────────────────┴─────────────────────────┐
                    │                                                     │
            scripts/dock.py                                      scripts/run_md.py
      (ML pre-screen + Vina docking                          (full production MD run:
       + consensus rescoring)                                  minimize→equilibrate→production)
                    │                                                     │
           ranked hits + poses                          trajectory + RMSD/RMSF/Rg/DSSP
                                                                          │
                                                        src/md/feature_extraction.py
                                                     (time-averaged contact map, in the
                                                      same CSV format the structure model
                                                      trains on — real physics-derived
                                                      labels to fine-tune on)
```

## Why minimize before docking prep

Meeko (the tool that converts a receptor/ligand into Vina's PDBQT format)
perceives chemical bonds from interatomic distances. That's fine for real
crystal structures, but a structure straight out of `predict_structure.py`
+ PDBFixer can have small geometric strain (from idealized bond
angles/lengths compounding over a long chain, or from PDBFixer guessing
sidechain rotamers) that confuses Meeko's bond perception and makes it
fail outright.

We hit this directly while building this pipeline: docking prep failed on
a raw predicted+PDBFixer'd structure with `AtomValenceException`/"unable to
build rdkit mol" errors, and succeeded immediately after running it through
one round of OpenMM energy minimization (potential energy dropped from
~+2,900,000 kJ/mol to about -490 kJ/mol — the geometry was badly strained,
not just cosmetically off). So:

**Always run `scripts/run_md.py --minimize_only` between structure
prediction and docking**, unless your receptor PDB is already an
experimental (X-ray/cryo-EM) structure or has already been through some
other relaxation. `scripts/dock.py` prints a reminder of this every run
(silence it with `--skip_minimize_warning` once you've internalized it).

## Docking (`src/docking/`, `scripts/dock.py`)

- `prepare.py` — `prepare_receptor()` (PDBFixer: fix missing atoms/H,
  strip waters/heterogens), `receptor_pdb_to_pdbqt()` /
  `prepare_ligand()` (Meeko: PDBQT with Gasteiger charges + AD4 atom
  types), `estimate_binding_box()` (centers on a co-crystallized ligand
  HETATM if you give its residue name, else falls back to the whole
  structure's centroid with a capped box — replace with a real pocket
  center for anything that matters).
- `vina_runner.py` — `VinaDocker`: loads the receptor + precomputes the
  scoring grid once, then docks/scores many ligands against it.
- `ml_rescoring.py` — `ml_prescreen()` ranks a whole library with the
  trained `FusionAffinityModel` and keeps only the top-K for real Vina
  docking (the standard way to make physics-based docking tractable at
  library scale); `consensus_rank()` combines the ML prediction and the
  Vina score (z-scored, so neither dominates by numeric scale alone) into
  one ranking.
- `configs/docking.yaml` — box/exhaustiveness/consensus-weight defaults.

Run it:
```bash
python scripts/dock.py \
    --receptor_pdb data/raw/receptors/target_fixed.pdb \
    --library data/raw/candidates.smi \
    --target_fasta data/raw/target.fasta \
    --ml_checkpoint checkpoints/fusion_affinity/best.pt \
    --prescreen_top_k 20 \
    --out_csv checkpoints/docking_results.csv
```
Omit `--ml_checkpoint` to dock every library member directly (fine for
small libraries).

## Molecular dynamics (`src/md/`, `scripts/run_md.py`)

- `system_builder.py` — Amber14 force field; `implicit` (GBn2, fast,
  no box/equilibration needed — the default), `explicit` (TIP3P +
  padded water box + ions, for production-quality runs), or `vacuum`.
- `simulation.py` — `MDSimulation`: Langevin integrator, energy
  minimization, NVT equilibration, production run with DCD trajectory +
  CSV energy/temperature logging, checkpoint PDBs at each stage.
- `analysis.py` — mdtraj-based RMSD (to frame 0), per-residue RMSF,
  radius of gyration, and simplified 3-class DSSP (helix/sheet/coil)
  fraction over the trajectory.
- `feature_extraction.py` — `time_averaged_contact_map()` (the MD-ensemble
  generalization of the binary contact map used in `protein_structure.csv`
  — this is real physics-derived data you can retrain the structure model
  on), per-residue RMSF/SASA, and `extract_representative_frame()`
  (medoid structure, for downstream docking against a "typical"
  conformation rather than one arbitrary frame).

Run it:
```bash
# Minimize only (e.g. to relax a predicted structure before docking prep)
python scripts/run_md.py --pdb checkpoints/predicted_structure_fixed.pdb \
    --out_dir checkpoints/md_run --minimize_only

# Full run: minimize -> 10 ps equilibration -> 50 ps production
python scripts/run_md.py --pdb data/raw/receptors/target_fixed.pdb \
    --out_dir checkpoints/md_run --equilibration_ps 10 --production_ps 50
```

The bundled defaults (implicit solvent, tens of picoseconds) are sized to
actually finish on a laptop CPU in this repo's spirit of "runs end to end
without a cluster" — they're a smoke-test/refinement scale, not a
production sampling budget. For real conformational sampling, switch to
`--solvent explicit`, extend `--production_ps` by 3-4 orders of magnitude,
and run on a GPU (OpenMM's CUDA/OpenCL platforms need no code changes here,
just `--platform_name CUDA` wiring — see `MDSimulation.__init__`).

## Closing the loop: MD -> new training data

`src/md/feature_extraction.contact_map_to_csv_field()` serializes a
time-averaged MD contact map into exactly the `;`-joined format
`ProteinStructureDataset` reads (see `docs/data_schema.md`). A minimal
loop for turning MD refinement into new supervision:

```python
from src.md.feature_extraction import time_averaged_contact_map, contact_map_to_csv_field

cmap = time_averaged_contact_map("run_minimized.pdb", "run_production.dcd")
csv_field = contact_map_to_csv_field(cmap)
# append {"protein_id": ..., "sequence": ..., "contact_map": csv_field} as a
# new row to data/processed/structure_advanced_train.csv, then re-run
# scripts/train.py --config configs/structure_advanced.yaml
```

This is genuinely useful when you don't trust a single static structure
(predicted or even crystallographic) as ground truth: the MD ensemble
average is a better-justified contact-map label than any single frame.

## What's verified vs. what to expect on real targets

Everything above was run and its output inspected while building this
repo (not just written and assumed to work): NeRF bond geometry is exact
to 1e-3 Å, minimization drops potential energy by orders of magnitude on
a strained toy structure, receptor/ligand PDBQT prep succeeds post-
minimization, and AutoDock Vina returns physically reasonable affinities
(around -2 to -3 kcal/mol for small, weakly-binding-by-default toy
ligands against an untuned pocket) for a small toy receptor.

None of this used a real experimental protein structure or a real binding
pocket (no network access to RCSB in this environment) — the toy receptor
here is a short synthetic polypeptide, and the "pocket" is just its
geometric center. For a real target: supply an experimental PDB (or a
well-refined predicted one), a real co-crystallized-ligand resname or
known pocket coordinates for `estimate_binding_box`, and treat
`exhaustiveness`/`n_poses` in `configs/docking.yaml` as the first knobs to
raise for production-quality docking.
