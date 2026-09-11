# Architecture

## 1. Protein structure module

`sequence` → `nn.Embedding` (or pluggable pretrained backend, e.g. ESM-2) →
sinusoidal positional encoding → 2-layer BiLSTM → 4-layer Transformer
encoder (8 heads) → per-residue embeddings.

Two auxiliary heads are trained jointly on top of the residue embeddings:

- **Secondary structure (Q8)**: per-residue linear classifier over 8 DSSP
  classes, cross-entropy loss (padding ignored via `ignore_index=-100`).
- **Contact map**: pairwise concatenation of residue embeddings →
  2-layer MLP → symmetrized L×L logits, trained with BCE against a binary
  C-alpha distance-threshold (8 Å) contact map.

Mean-pooling the residue embeddings (over non-pad positions) and projecting
gives a fixed-size 256-d protein embedding, reused by the fusion model.

Rationale: BiLSTM captures strong local sequential motifs (helix/sheet
propensities are locally correlated) cheaply, while the Transformer layers
give the O(1)-hop long-range mixing needed for contact prediction between
residues that are far apart in sequence but close in 3D space — the same
intuition behind real structure-prediction architectures, just without the
MSA/template pipeline or the huge parameter count.

## 2. Drug discovery module

`SMILES` → RDKit parse → atom/bond feature graph (atom type, degree,
hybridization, charge, aromaticity, ring membership, H-count; bond type,
conjugation, ring membership) → 4-layer **Graph Isomorphism Network (GIN)**
with edge-conditioned messages, implemented with a from-scratch
`scatter_add`/`scatter_mean` (no `torch_geometric` dependency) →
jumping-knowledge concatenation of all layer outputs → combined mean+max
graph readout → 256-d molecule embedding.

An `AffinityHead` (shared MLP trunk + two linear heads) predicts:
- `pActivity` (regression: pKd/pIC50/pKi), Smooth-L1 loss
- `activity_logit` (binary classification: active/inactive), BCE-with-logits

GIN is used because it's provably as expressive as the Weisfeiler-Lehman
graph isomorphism test — a strong default for molecular property prediction
where distinguishing subtly different substructures (e.g. stereoisomers'
2D graph, functional group placement) matters.

## 3. Cross-modal fusion (protein-ligand binding)

Given a target sequence and a ligand SMILES:

1. Encode both independently (`ProteinEncoder`, `MoleculeGNN`, sharing
   architecture with modules 1–2 above, optionally warm-started from their
   independently pretrained checkpoints).
2. **Co-attention**: molecule atom embeddings attend over protein residue
   embeddings (`mol_to_prot_attn`) and vice versa (`prot_to_mol_attn`),
   each an `nn.MultiheadAttention` block, producing attention-pooled
   context vectors for each modality — this lets the model learn which
   residues are most relevant to which atoms without explicit
   docking/pocket supervision.
3. Concatenate `[molecule_context, protein_context, protein_pooled,
   molecule_pooled]` → project → `AffinityHead` → final pActivity/activity
   prediction.

This is a lightweight version of the cross-attention design used in
several published structure-aware and sequence-only binding-affinity
models (e.g. DeepDTA-style baselines extended with attention, or
MONN/GraphDTA-style graph+sequence fusion), scaled down to run on CPU/laptop
GPU for the bundled toy data.

## 4. Virtual screening

`src/inference/screen.py` scores every candidate ligand in a library
against one target sequence in batched forward passes through the trained
`FusionAffinityModel`, ranks by predicted `pActivity`, and (optionally)
applies a greedy Tanimoto-similarity diversity filter (ECFP4 fingerprints)
so the top-K hit list isn't dominated by near-duplicate scaffolds. No
docking engine (AutoDock/Glide/etc.) is required — this is a fast
pre-filter to shortlist candidates for downstream physics-based docking or
wet-lab validation, not a replacement for either.

## Design choices & limitations (be upfront about these)

- This is **not** a from-scratch AlphaFold/RoseTTAFold reimplementation —
  no MSA, no templates, no full 3D coordinate output (only Cα-based binary
  contact maps + Q8 secondary structure). It's a realistic, extensible
  scaffold sized for a single GPU/CPU and small-to-medium datasets.
- The bundled toy data is synthetic-but-structurally-valid (random walk
  backbones for contact maps, real drug SMILES for molecules) purely so the
  full pipeline runs end-to-end offline; **do not** draw scientific
  conclusions from a model trained on it. Swap in ChEMBL/PDBbind/PDB data
  via the schema in `docs/data_schema.md` for anything real.
- `embedding_backend` in `ProteinEncoder` is a hook, not a bundled
  implementation — plug in a frozen ESM-2/ProtT5 wrapper there for a
  meaningful accuracy jump on real data.

## Pretrained ESM-2 backend

`src/models/esm_backend.py` implements the `embedding_backend` hook
above with a real pretrained protein language model: `ESM2EmbeddingBackend`
wraps HuggingFace `transformers`' `EsmModel` (`facebook/esm2_t6_8M_UR50D`
through `esm2_t36_3B_UR50D`), remaps this repo's token vocabulary to
ESM-2's alphabet, and returns per-residue embeddings in place of the
learned `nn.Embedding` — `ProteinEncoder` needs no other changes, since it
reads `output_dim` off whatever backend it's given.

```python
from src.models.esm_backend import ESM2EmbeddingBackend
from src.models.protein_encoder import ProteinEncoder

encoder = ProteinEncoder(embedding_backend=ESM2EmbeddingBackend(
    model_name="facebook/esm2_t6_8M_UR50D", pretrained=True, freeze=True,
))
```

Or via config — see `configs/structure_advanced_esm2.yaml` and
`scripts/train.py`'s `build_structure_advanced`, which pops an `esm:`
block out of `model:` and constructs the backend automatically. Frozen by
default (`freeze: true`): only the downstream BiLSTM/Transformer/heads
train, ESM-2's weights stay fixed and run under `torch.no_grad()`.

Downloading real weights needs network access to huggingface.co, which
this repo doesn't assume. `pretrained: false` builds the real ESM-2
architecture with random weights instead (offline, via `EsmConfig`) —
useful for validating the integration end-to-end without network access,
*not* a substitute for real weights on real data. This repo's tests
(`tests/test_esm_backend.py`) and the smoke-training path both use
`pretrained: false` for exactly that reason; set `pretrained: true`
wherever you have Hub access.

## GPU platform selection (molecular dynamics)

`src/md/platform_utils.py` resolves an explicit OpenMM platform
(`Reference` / `CPU` / `CUDA` / `OpenCL`) by name, failing with a clear
error listing what's actually available rather than silently falling back
to something much slower. Wired into `MDSimulation` (`platform_name` /
`platform_properties` constructor args) and exposed on the CLI:

```bash
python scripts/run_md.py --pdb receptor_fixed.pdb --platform_name CUDA \
    --platform_property Precision=mixed
```

Omit `--platform_name` (the default) to let OpenMM auto-select the
fastest platform actually present on the machine — CUDA/OpenCL when a
GPU + matching OpenMM build are available, CPU otherwise. This repo's
sandbox has no GPU, so `CPU`/`Reference` are what's been exercised in
testing here; the `CUDA`/`OpenCL` paths use the same OpenMM APIs and
differ only in which platform string is resolved, but haven't been run
against real GPU hardware as part of this repo's own testing.
