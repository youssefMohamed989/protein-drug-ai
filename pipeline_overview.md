# Pipeline overview

A full-stack computational pipeline connecting protein structure
prediction, drug discovery, docking, and molecular dynamics end-to-end:

```
SEQUENCE → STRUCTURE → SCREENING → DOCKING → MD REFINEMENT → (loop back into training)
```

## 1. Protein structure prediction
`src/models/protein_encoder.py`, `src/structure/`

A BiLSTM + Transformer encoder reads an amino-acid sequence and predicts:
- secondary structure (8-class Q8, per residue)
- a residue-residue contact map (L×L binary)
- backbone torsion angles (phi/psi, per residue)

The torsion angles are converted into an actual 3D backbone (N/CA/C/O)
via NeRF geometric reconstruction and exported as a PDB file —
`scripts/predict_structure.py`.

## 2. Drug discovery (affinity prediction)
`src/models/molecule_gnn.py`

SMILES strings are converted to molecular graphs (RDKit) and passed
through a Graph Isomorphism Network (GIN) that predicts binding affinity
(pKd/pIC50, regression) and activity (active/inactive, classification)
for a candidate compound.

## 3. Cross-modal fusion
`src/models/fusion.py`

A co-attention layer lets protein residues and ligand atoms attend to
each other, fusing both encoders into one model that predicts
protein-ligand binding affinity directly from sequence + SMILES — no
docking required for this step.

## 4. Virtual screening
`src/inference/screen.py`

The fusion model ranks a whole candidate library against a target
sequence in one batched pass, with Tanimoto-similarity filtering to
avoid redundant hits — a fast, physics-free pre-filter before anything
more expensive runs.

## 5. Molecular docking
`src/docking/`

The top-ranked candidates from ML screening are docked for real with
AutoDock Vina (via PDBFixer + Meeko for receptor/ligand prep). ML
predictions and Vina binding energies are combined into one consensus
score — ML narrows a huge library down to a tractable shortlist; physics-
based docking validates the finalists. See `docs/docking_and_md.md` for
the required minimize-before-dock-prep step.

## 6. Molecular dynamics
`src/md/`

OpenMM minimizes, equilibrates, and simulates the receptor (or a
predicted structure) to relax strain and generate a conformational
ensemble. Trajectories are analyzed (RMSD/RMSF/Rg/DSSP) and can be
converted back into contact-map training labels — closing the loop from
simulation back into model retraining.

## Also included
- A FastAPI REST server (`src/api/`, `scripts/serve.py`) exposing
  structure prediction and affinity scoring over HTTP.
- Docker packaging (`Dockerfile`, `docker-compose.yml`).
- A pytest suite (31 tests) and CI (lint, format check, coverage,
  smoke-training, Docker build).
- `scripts/evaluate.py` to check a trained model against naive baselines
  before trusting it.

Every stage above is runnable as code — see the Quickstart in `README.md`
for the exact commands, `docs/architecture.md` for model internals, and
`docs/data_schema.md` for the expected file formats at each stage.
