
<img width="1698" height="926" alt="Protein AI pipline" src="https://github.com/user-attachments/assets/7bf06bf8-167a-4bf6-99fe-8bdbb81cd6f0" />
<img width="6420" height="3920" alt="integrated_deep_learning_MD_workflow_high_res (1)" src="https://github.com/user-attachments/assets/3ee189b9-6177-4ef8-9a41-2022f9b72cf3" />
<img width="1360" height="1190" alt="07_contact_map_surface_3d" src="https://github.com/user-attachments/assets/dd67977b-4cd8-42ff-b8fb-61e63c26e336" />
<img width="1190" height="1105" alt="10_ligand_conformer_3d" src="https://github.com/user-attachments/assets/e8f58ffa-c855-4dae-91bf-6f0bd9ace078" />
<img width="1360" height="1190" alt="08_residue_embedding_pca_3d" src="https://github.com/user-attachments/assets/209cf8e5-9240-4e21-bcb0-bec4bb0739ca" />
<img width="1200" height="750" alt="19_vina_ranking" src="https://github.com/user-attachments/assets/a2cf9f01-423c-4627-9633-c4d0bfe92484" /><img width="1121" height="944" alt="docking" src="https://github.com/user-attachments/assets/9f5e9826-0a08-46b6-8d27-213e9e1e381c" />
# protein-drug-ai
End-to-end ML pipeline for protein structure prediction, drug discovery, molecular docking (AutoDock Vina), and molecular dynamics (OpenMM) trainable models, a REST API, and Docker packaging, connected sequence → structure → screening → docking → MD refinement.
This is not a reimplementation of AlphaFold, and it does not claim production-grade accuracy out of the box. It is a realistic, fully-wired engineering scaffold — the kind a research engineering team would actually build and hand off — that you train on your own data, extend module by module, and deploy. Every script in this README has been run and its output inspected; see Project status & honesty below for exactly what that does and doesn't mean.

Table of contents
Why this exists
Pipeline at a glance
Features
Repository layout
Requirements
Installation
Quickstart
Command-line reference
Configuration reference
REST API
Docker
Testing
Evaluating a trained model against naive baselines
Bringing your own data
Documentation index
Project status & honesty
Roadmap
Contributing
Acknowledgments
License
Why this exists

Most "AI for drug discovery" repos on GitHub are one of two things: a single Colab notebook wrapping one pretrained model, or a thin fork of someone else's research code with the training loop stripped out. Neither gives you something you can actually build a product or a research pipeline on top of.

ProteinDrugAI is different in three ways:

It's connected, not four separate demos. The structure model's output feeds docking prep. Docking's ML pre-screen uses the affinity model. MD refinement is a required step before docking (not optional — see docs/docking_and_md.md for why, including the actual failure we hit and debugged). MD trajectories can be converted back into training labels for the structure model, closing the loop.
It's trainable, not just inference-only. Every model — the protein encoder, the molecular GNN, the fusion model — has a real training loop, real losses, real metrics, and a config file. You're not stuck with someone else's frozen weights.
It's engineered like production code, not a notebook: a test suite, CI, Docker packaging, a REST API, structured logging, config-driven experiments, and docs that tell you what's actually been verified to work versus what's a documented limitation.
Pipeline at a glance
                         ┌─ secondary structure (Q8)
protein sequence ──▶ ProteinEncoder ──┼─ residue contact map
                                       ├─ phi/psi torsion angles ──▶ NeRF ──▶ 3D backbone PDB
                                       └─ 256-d protein embedding             │
                                                    │                  PDBFixer + OpenMM
SMILES ──▶ MoleculeGraph ──▶ GNN ──▶ 256-d molecule embedding          minimization
                                                    │                         │
                              protein embedding  molecule embedding   AutoDock Vina
                                                    │                    docking
                                    Affinity/Activity MLP head                │
                                                    │                  ranked poses
                                          pKd / pIC50 / binding prob.  + consensus score
                                                    │                         │
                                        ML pre-screen (top-K) ────────────────┘
                                        for docking at library scale

See docs/pipeline_overview.md for a short, stage-by-stage walkthrough, and docs/architecture.md for the full model math and design rationale.

Features
Protein structure prediction

src/models/protein_encoder.py, src/structure/

BiLSTM + multi-head self-attention encoder over amino-acid sequences.
Three auxiliary heads trained jointly: 8-class secondary structure (Q8), residue-residue contact map (binary, L×L), and backbone torsion angles (phi/psi) regressed via their (sin, cos) representation to avoid the angular wraparound discontinuity.
Real 3D structure export: predicted torsion angles are reconstructed into an actual N/CA/C/O backbone trace via NeRF (Natural Extension Reference Frame) — exact idealized bond geometry (verified to 1e-3 Å), pure NumPy, written straight to a PDB file (scripts/predict_structure.py).
Pluggable embedding_backend interface for pretrained embeddings: src/models/esm_backend.py implements this with a real HuggingFace transformers ESM-2 model, frozen by default, with an offline random-init path for testing without Hugging Face Hub access.
Drug discovery (affinity prediction)

src/models/molecule_gnn.py

SMILES → molecular graph (RDKit) → Graph Isomorphism Network (GIN), implemented from scratch (no torch_geometric dependency).
Predicts binding affinity (regression: pKd/pIC50) and activity (classification: active/inactive) jointly from one shared trunk.
Cross-modal fusion

src/models/fusion.py

Co-attention between protein residues and molecule atoms — each modality attends over the other before pooling.
Late-fusion MLP head for the final affinity/activity prediction, trained end to end (optionally warm-started from the independently pretrained protein and molecule encoders).
Molecular docking

src/docking/, scripts/dock.py

Receptor prep (PDBFixer: missing atoms/hydrogens, strip heterogens) and ligand prep (RDKit 3D embedding + MMFF/UFF optimization + Meeko PDBQT conversion) feeding real AutoDock Vina docking runs.
ML pre-screen: the fusion affinity model ranks the whole candidate library first, so expensive physics-based docking only runs on the most promising top-K — the standard way ML and docking are combined at library scale.
Consensus rescoring: z-scored combination of the ML prediction and the Vina binding free energy into one ranking.
Molecular dynamics

src/md/, scripts/run_md.py

OpenMM-based minimization, NVT equilibration, and production runs (implicit-solvent GBn2 by default for CPU-friendly speed; explicit TIP3P solvation available for production-quality runs).
mdtraj-based analysis: RMSD, per-residue RMSF, radius of gyration, simplified DSSP secondary-structure fractions over the trajectory.
Closes the loop with the ML pipeline: MD trajectories can be turned into time-averaged contact maps in the exact CSV format the structure-prediction dataset reads — real physics-derived training labels, not just static single-frame targets.
Also the required relaxation step before docking prep: predicted (or otherwise strained) structures need one minimization pass before Meeko's distance-based bond perception can process them. See docs/docking_and_md.md — we hit this failure directly while building the pipeline and documented the fix.
Explicit GPU platform selection (src/md/platform_utils.py, --platform_name CUDA) that fails with a clear error listing what's actually available, instead of silently falling back to something much slower.
Virtual screening

src/inference/screen.py

Ranks a library of candidate molecules (SDF/CSV/SMILES) against a target protein purely from learned embeddings — no docking engine required for this fast pre-filter stage.
Optional Tanimoto-similarity re-ranking (ECFP4 fingerprints) so a top-K hit list isn't dominated by near-duplicate scaffolds.
Training infrastructure

src/training/

One generic Trainer class drives every task (not four near-duplicate training loops) — see CONTRIBUTING.md for the pattern used to add a new trainable task.
Config-driven (configs/*.yaml) via a small dataclass-style config loader — no framework lock-in.
Mixed precision, gradient accumulation, early stopping, best/last checkpointing.
CSV + stdout experiment logging (swap in Weights & Biases by implementing the same log() interface).
Reproducible, leakage-aware splits: random, scaffold split (Bemis–Murcko), and protein-family cluster split.
Data pipeline

src/data/

Parsers for FASTA, PDB (contact-map extraction via Biopython C-alpha distances, phi/psi extraction via Biopython's internal-coordinates module), and SMILES/CSV bioactivity tables (ChEMBL/BindingDB/PDBbind-style schema).
Dataset/DataLoader classes with on-the-fly featurization.
REST API

src/api/, scripts/serve.py

FastAPI server exposing structure prediction and affinity scoring over HTTP, with proper error handling (503 for a missing checkpoint, 422 for invalid input) and in-memory model caching across requests.
Testing & CI

tests/, .github/workflows/ci.yml

41 pytest tests covering data parsing, featurization, every model's forward pass, NeRF geometry, splitters, docking prep, MD minimization, the ESM-2 backend, GPU platform resolution, and the REST API.
Optional-dependency modules (docking, md, esm, api) use pytest.importorskip so the suite degrades gracefully without those extras installed.
CI runs lint (flake8), format checks (black/isort), the full test suite with coverage, console-entry-point verification, two smoke-training jobs, and a Docker build check.
Packaging
pyproject.toml with optional dependency groups (chem, docking, md, esm, api, dev, all) and console scripts (pdai-train, pdai-dock, pdai-run-md, pdai-predict-structure, pdai-screen).
Dockerfile + docker-compose.yml.
Makefile with install/test/lint/format/train/dock/run-md/serve/docker targets.
Pre-commit hooks, issue/PR templates, CONTRIBUTING.md, CHANGELOG.md.
Repository layout
protein-drug-ai/
├── src/
│   ├── data/            # parsers, datasets, featurizers, splitters
│   ├── models/           # protein encoder, ESM-2 backend, molecule GNN, fusion
│   ├── structure/         # torsion head, NeRF 3D reconstruction, PDB export
│   ├── docking/            # receptor/ligand prep, AutoDock Vina, ML rescoring
│   ├── md/                  # OpenMM simulation, platform selection, analysis
│   ├── api/                  # FastAPI inference server
│   ├── training/               # generic trainer, losses, metrics, task adapters
│   ├── inference/                # affinity predictor, virtual screening
│   └── utils/                     # config loader, logging, seeding
├── configs/                # one YAML per trainable task/pipeline
├── scripts/                # CLIs: train, predict_structure, dock, run_md, serve, evaluate, screen
├── tests/                  # 41 pytest tests; optional-dep modules skip cleanly
├── notebooks/              # exploratory analysis
├── data/                   # raw/ and processed/ (gitignored; generated by make_toy_data.py)
├── checkpoints/            # saved model weights (gitignored)
├── docs/                   # architecture.md, data_schema.md, docking_and_md.md, pipeline_overview.md
├── .github/                 # CI workflow, issue/PR templates
├── pyproject.toml           # packaging + tool configs (black/isort/pytest)
├── Dockerfile, docker-compose.yml
├── Makefile                  # install/test/lint/format/train/serve/docker targets
├── CONTRIBUTING.md, CHANGELOG.md, LICENSE
└── README.md                  # you are here
Requirements
Python: 3.10 or 3.11 (CI tests both).
OS: Linux/macOS recommended. Windows works via WSL2; native Windows is untested.
Disk: a few GB free — PyTorch, RDKit, OpenMM, and transformers together are the bulk of it.
RAM: the default configs assume a reasonably provisioned machine. On memory-constrained environments (under ~4GB free RAM), reduce batch_size and max_len in the relevant configs/*.yaml — the protein_structure and structure_advanced tasks are the most memory-hungry because the contact-map head materializes an L×L tensor per example. This is a real, tested limit, not a hypothetical caveat.
GPU: fully optional. Training and inference all run on CPU by default (that's what this repo has been tested on); pass CUDA device handling is standard PyTorch (.to(device)) and --platform_name CUDA for the MD pipeline specifically.
Optional heavy dependencies: AutoDock Vina, OpenMM, and RDKit are real scientific libraries with C++ extensions — they install cleanly via pip on Linux/macOS but budget a few extra minutes for the first install.
Installation
bash
git clone https://github.com/<you>/protein-drug-ai.git
cd protein-drug-ai
python -m venv .venv && source .venv/bin/activate

Install the pieces you need — everything is behind optional dependency groups so a structure-prediction-only use case doesn't pull in AutoDock Vina, and vice versa:

bash
pip install -e .                # core only (structure/drug-discovery ML, no chem/docking/MD)
pip install -e ".[chem]"         # + RDKit — needed for anything molecule-related
pip install -e ".[docking]"       # + AutoDock Vina, Meeko, PDBFixer
pip install -e ".[md]"             # + OpenMM, PDBFixer, mdtraj
pip install -e ".[esm]"             # + transformers, for the pretrained ESM-2 backend
pip install -e ".[api]"              # + FastAPI, uvicorn
pip install -e ".[dev]"               # + pytest, black, isort, flake8, pre-commit
pip install -e ".[all]"                # everything above

This also installs five console scripts, equivalent to python scripts/<name>.py:

Command	Equivalent to
pdai-train	python scripts/train.py
pdai-predict-structure	python scripts/predict_structure.py
pdai-dock	python scripts/dock.py
pdai-run-md	python scripts/run_md.py
pdai-screen	python scripts/screen.py

Or use make (see the full target list in Makefile):

bash
make install-all   # editable install, every extra
make toy-data        # generate synthetic data so everything runs offline
make test              # run the full test suite
make format              # black + isort
make lint                  # flake8 + format check — exactly what CI runs

Plain requirements.txt is also provided if you'd rather not use the package extras (pip install -r requirements.txt installs everything); requirements-dev.txt adds the dev tools on top.

Quickstart
bash
# 1. Prepare toy data (small, synthetic-but-valid samples so the whole
#    pipeline runs end-to-end offline, with no external downloads)
python scripts/make_toy_data.py

# 2. Train the protein structure encoder (SS8 + contact map)
python scripts/train.py --config configs/protein_structure.yaml

# 3. Train the molecule affinity GNN
python scripts/train.py --config configs/drug_affinity.yaml

# 4. Train the fused affinity model (warm-starts from both encoders above)
python scripts/train.py --config configs/fusion_affinity.yaml

# 4b. (optional) Train the advanced structure model with 3D torsion prediction
python scripts/train.py --config configs/structure_advanced.yaml

# 5. Run virtual screening: rank a ligand library against a target sequence
python scripts/screen.py \
    --target_fasta data/raw/target.fasta \
    --library data/raw/candidates.smi \
    --checkpoint checkpoints/fusion_affinity/best.pt \
    --top_k 20

# 6. Predict a 3D structure and export a PDB
python scripts/predict_structure.py \
    --sequence MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKV \
    --checkpoint checkpoints/structure_advanced/best.pt \
    --out_pdb checkpoints/predicted_structure.pdb

# 7. Relax the predicted structure with MD before docking (REQUIRED — see
#    docs/docking_and_md.md for why this isn't optional)
python scripts/run_md.py --pdb checkpoints/predicted_structure_fixed.pdb \
    --out_dir checkpoints/md_run --minimize_only

# 8. Dock a candidate library against it: ML pre-screen -> AutoDock Vina ->
#    consensus rescoring
python scripts/dock.py \
    --receptor_pdb checkpoints/md_run/run_minimized.pdb \
    --library data/raw/candidates.smi \
    --target_fasta data/raw/target.fasta \
    --ml_checkpoint checkpoints/fusion_affinity/best.pt \
    --prescreen_top_k 20

Every command above has actually been run against the bundled toy data; see Project status & honesty below for exactly what that verification covered.

Command-line reference

All scripts live in scripts/ and are also installed as pdai-* console commands. Run any of them with --help for the full flag list.

Script	Purpose
make_toy_data.py	Generate small synthetic datasets for every task, so the whole repo runs offline with zero setup.
train.py	Unified training entry point for all four trainable tasks (protein_structure, drug_affinity, fusion_affinity, structure_advanced) — task is read from the config file's task: field.
predict_structure.py	Sequence → trained model → phi/psi angles → NeRF → backbone-only PDB.
screen.py	Rank a SMILES library against a target sequence using the fusion model; optional Tanimoto diversity filter.
dock.py	Full docking pipeline: receptor/ligand prep → optional ML pre-screen → real AutoDock Vina docking → consensus rescoring.
run_md.py	OpenMM minimize / equilibrate / production run, with mdtraj trajectory analysis (RMSD/RMSF/Rg/DSSP).
serve.py	Launch the FastAPI inference server.
evaluate.py	Compare a trained affinity model against naive baselines (mean-predictor, majority-class) on a held-out test set.
Configuration reference

Every trainable task and pipeline is driven by a YAML file in configs/; nothing is hardcoded in the scripts. The loader (src/utils/config.py) supports both dict-style and attribute-style access (cfg.model.hidden_dim or cfg["model"]["hidden_dim"]).

Config	Drives
protein_structure.yaml	SS8 + contact-map training (ProteinStructureModel).
drug_affinity.yaml	Molecule GNN affinity/activity training.
fusion_affinity.yaml	Cross-modal fusion training, with optional warm_start: checkpoints for the protein encoder and molecule GNN.
structure_advanced.yaml	SS8 + contact + torsion-angle training (AdvancedStructureModel), the model behind predict_structure.py.
structure_advanced_esm2.yaml	Same as above, with a pretrained ESM-2 embedding_backend swapped in — see the esm: block.
docking.yaml	Default box padding, Vina exhaustiveness/pose count, ML-prescreen top-K, consensus weights for dock.py.
md.yaml	Default solvent model, integrator settings, equilibration/production length, analysis options for run_md.py.

Every config's data:, model:, and training: blocks (and warm_start: for fusion) are documented inline with comments — open one directly for the fastest way to understand the available knobs.

REST API

A FastAPI server exposes structure prediction and affinity scoring over HTTP, for integrating this pipeline into a larger service:

bash
python scripts/serve.py --port 8000
# or: make serve
Endpoint	Method	Description
/health	GET	Reports server status and which model checkpoints are currently cached in memory.
/predict_structure	POST	{"sequence": "..."} → predicted phi/psi angles + a full backbone-only PDB (as text).
/score_affinity	POST	{"sequence": "...", "smiles": [...]} → predicted pActivity + activity probability per valid SMILES.
bash
curl -X POST http://localhost:8000/predict_structure \
    -H "Content-Type: application/json" \
    -d '{"sequence": "ACDEFGHIKLMNPQRSTVWY"}'

curl -X POST http://localhost:8000/score_affinity \
    -H "Content-Type: application/json" \
    -d '{"sequence": "ACDEFGHIKLMNPQRSTVWY", "smiles": ["CC(=O)OC1=CC=CC=C1C(=O)O"]}'

Both prediction endpoints return 503 with a clear message if the requested checkpoint doesn't exist yet (train one first), and 422 on malformed input (FastAPI/Pydantic validation). Model checkpoints are loaded lazily on first request and cached in memory per checkpoint path — GET /health shows what's currently cached. Checkpoint/config paths can be overridden per-request in the JSON body, or globally via PDAI_STRUCTURE_CHECKPOINT, PDAI_STRUCTURE_CONFIG, PDAI_FUSION_CHECKPOINT, PDAI_FUSION_CONFIG environment variables. See src/api/server.py for the full Pydantic request/response schemas.

Docker
bash
docker build -t protein-drug-ai .
docker run --rm -p 8000:8000 \
    -v $(pwd)/checkpoints:/app/checkpoints \
    protein-drug-ai
# or: docker compose up --build

The default CMD launches the API server; override it to run training, docking, or MD scripts inside the container instead:

bash
docker run --rm -v $(pwd)/data:/app/data -v $(pwd)/checkpoints:/app/checkpoints \
    protein-drug-ai python scripts/train.py --config configs/protein_structure.yaml

Note: the Docker build has been reviewed for correctness (base image, layer ordering, system deps for RDKit/OpenMM wheels) but has not been build-tested in this project's own verification pass, since no Docker daemon was available during development. The CI workflow does include a docker build job on every push, so once it's on GitHub you'll have a live signal.

Testing
bash
make test              # pytest, verbose
make test-cov            # + coverage report (terminal + htmlcov/)

41 tests across data featurization, every model's forward pass, NeRF bond geometry, dataset splitters, docking preparation, MD minimization, the ESM-2 backend, GPU platform resolution, and the REST API. Tests for optional-dependency modules (docking, md, esm, api) use pytest.importorskip, so the suite still passes cleanly on a minimal install — but if you're modifying that code, install the relevant extra and confirm the tests actually run rather than skip.

Evaluating a trained model against naive baselines
bash
python scripts/evaluate.py --task fusion_affinity \
    --checkpoint checkpoints/fusion_affinity/best.pt \
    --config configs/fusion_affinity.yaml \
    --test_csv data/processed/pairs_test.csv

Reports model MAE/RMSE/Pearson-r/accuracy/ROC-AUC against a mean-predictor and majority-class baseline. A model that doesn't clear these baselines by a solid margin isn't learning anything useful yet — treat this as the minimum bar for trusting a checkpoint, not a benchmark to brag about. (On a 1-epoch smoke-train over the bundled ~27-example toy test set, the model does not yet beat the mean-predictor baseline — exactly the kind of honest signal this script exists to surface. Train longer, and on real data, before drawing conclusions.)

Bringing your own data

The bundled scripts/make_toy_data.py output exists purely so the whole pipeline runs offline with zero setup — it is synthetic and should never be used to draw scientific conclusions. To use real data, match the same CSV/FASTA schema documented in docs/data_schema.md:

Modality	Format	Suggested public source
Protein sequence + structure	FASTA / PDB	RCSB PDB, CATH, SCOPe
Secondary structure labels	DSSP-derived Q8/Q3	PDB + mkdssp
Bioactivity (affinity/activity)	CSV (SMILES, target, pKd/pIC50, label)	ChEMBL, BindingDB, PDBbind

None of this is downloaded automatically — the pipeline assumes no network access at run time by default. Point the data: block in the relevant configs/*.yaml at your own CSV/FASTA files once you have them in the documented schema.

Documentation index
Doc	Covers
docs/pipeline_overview.md	Short, stage-by-stage description of the full pipeline.
docs/architecture.md	Model internals and design rationale for every module, including the ESM-2 backend and GPU platform selection.
docs/data_schema.md	Exact CSV/FASTA column formats expected by each Dataset.
docs/docking_and_md.md	The full docking/MD pipeline, why the minimize-before-dock-prep step is required (with real numbers from the debugging session), and how to feed MD trajectories back into training data.
CONTRIBUTING.md	Dev setup, the pattern for adding a new trainable task, PR checklist.
CHANGELOG.md	What was added in each stage of this repo's development.
Project status & honesty

This section exists because "it works" claims are cheap and this repo tries not to make them without backing.

What has actually been run and verified, not just written:

Every model's forward pass, at real batch sizes, with real gradients.
Full training loops for all four tasks, including warm-starting the fusion model from independently pretrained protein/molecule checkpoints (verified zero missing/unexpected state-dict keys).
NeRF backbone reconstruction — bond lengths exact to 1e-3 Å against the idealized targets.
Real AutoDock Vina docking runs (via the actual vina Python bindings, not mocked), producing physically reasonable binding-affinity numbers.
Real OpenMM minimize → equilibrate → production runs, with mdtraj trajectory analysis (RMSD/RMSF/Rg/DSSP) confirmed against expected behavior (energy dropping by orders of magnitude on minimization of a strained structure).
The ESM-2 backend, end to end through the training loop, in offline (random-init) mode.
GPU platform resolution correctly succeeding for CPU and correctly raising a clear error for CUDA when no GPU is present.
The FastAPI server: started, and every endpoint hit directly, including error paths (missing checkpoint → 503, invalid input → 422).
The full test suite (41 tests) from a completely fresh pip install -e . of the exact packaged repo, not just the development copy.
Every console entry point, make lint/make format/make test.

What has not been verified, stated plainly rather than implied away:

Docker build: reviewed for correctness, not build-tested (no Docker daemon in the environment this was developed in). CI includes a build job, so this will get a real signal once pushed.
Real experimental data: everything above was tested against small synthetic toy data and a synthetic toy receptor — never a real PDB structure or real ChEMBL/PDBbind/BindingDB data (no network access to fetch either during development). The pipeline wiring is verified; the scientific results you'd get on real data are not.
GPU/CUDA execution: the platform-selection code path is correct and tested against the "unavailable" branch, but has never run against real GPU hardware.
Pretrained ESM-2 weights: the offline random-init path is tested; downloading and running real facebook/esm2_* weights requires Hugging Face Hub access that wasn't available during development.
Model quality: the bundled example checkpoints (if any ship with a given release) come from 1–2 epoch smoke-training runs meant to verify the pipeline runs, not to represent a trained model — evaluate.py honestly shows this by not beating a naive baseline on toy data.

If you hit something in the unverified list that doesn't work as documented, that's a genuine bug report, not user error — please open an issue.

Roadmap

Reasonable next steps, roughly in order of expected value:

Validate the full pipeline against a real PDB structure and real ChEMBL/PDBbind data once available.
Run the GPU (CUDA/OpenCL) MD path against real GPU hardware.
Download and evaluate real pretrained ESM-2 weights.
Add a pocket-detection step (e.g. fpocket) for estimate_binding_box instead of falling back to whole-structure centroid when no co-crystallized ligand is available.
Extend the structure model toward full side-chain placement, not just backbone.
Add explicit-solvent MD as the CI-tested default for production runs (implicit solvent is the current default, chosen for CPU speed).

Contributions toward any of these are very welcome — see below.

Contributing

See CONTRIBUTING.md for the full guide, including dev environment setup and the checklist pattern for adding a new trainable task (dataset → model → loss/metrics → task adapter → builder function → config → toy-data generator). Quick reference before opening a PR:

bash
make format     # black + isort
make lint         # flake8 + format check (what CI runs)
make test          # full pytest suite

Bug reports and feature requests use the templates in .github/ISSUE_TEMPLATE/.

Acknowledgments

Built on top of, and would not exist without: PyTorch, RDKit, AutoDock Vina, Meeko, OpenMM, PDBFixer, MDTraj, Biopython, Hugging Face transformers / ESM-2, and FastAPI.

License

MIT — see LICENSE.
