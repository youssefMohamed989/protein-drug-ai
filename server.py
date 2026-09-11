"""REST API for the trained pipeline: structure prediction, protein-ligand
affinity scoring, and library screening, over HTTP.

Run with:
    uvicorn src.api.server:app --host 0.0.0.0 --port 8000
or:
    python scripts/serve.py --port 8000

Models are loaded lazily on first request per checkpoint path and cached in
memory (`_MODEL_CACHE`), so repeated calls against the same checkpoint don't
pay reload cost.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.data.protein_features import encode_sequence
from src.inference.predictor import AffinityPredictor
from src.structure.pdb_writer import write_backbone_pdb
from src.structure.structure_model import AdvancedStructureModel
from src.utils.config import Config

app = FastAPI(
    title="ProteinDrugAI API",
    description="Structure prediction, protein-ligand affinity scoring, and virtual screening.",
    version="0.2.0",
)

_MODEL_CACHE: Dict[str, object] = {}
DEFAULT_STRUCTURE_CHECKPOINT = os.environ.get(
    "PDAI_STRUCTURE_CHECKPOINT", "checkpoints/structure_advanced/best.pt"
)
DEFAULT_STRUCTURE_CONFIG = os.environ.get("PDAI_STRUCTURE_CONFIG", "configs/structure_advanced.yaml")
DEFAULT_FUSION_CHECKPOINT = os.environ.get("PDAI_FUSION_CHECKPOINT", "checkpoints/fusion_affinity/best.pt")
DEFAULT_FUSION_CONFIG = os.environ.get("PDAI_FUSION_CONFIG", "configs/fusion_affinity.yaml")


def _get_structure_model(checkpoint: str, config_path: str) -> tuple:
    key = f"structure::{checkpoint}::{config_path}"
    if key not in _MODEL_CACHE:
        cfg = Config.from_yaml(config_path)
        model = AdvancedStructureModel(encoder_kwargs=dict(cfg.model))
        ckpt = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        _MODEL_CACHE[key] = (model, cfg)
    return _MODEL_CACHE[key]


def _get_affinity_predictor(checkpoint: str, config_path: str) -> AffinityPredictor:
    key = f"affinity::{checkpoint}::{config_path}"
    if key not in _MODEL_CACHE:
        cfg = Config.from_yaml(config_path)
        _MODEL_CACHE[key] = AffinityPredictor(checkpoint, cfg, device="cpu")
    return _MODEL_CACHE[key]


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    status: str
    cached_models: List[str]


class StructurePredictRequest(BaseModel):
    sequence: str = Field(..., min_length=1, max_length=2000)
    checkpoint: Optional[str] = None
    config: Optional[str] = None


class StructurePredictResponse(BaseModel):
    sequence: str
    num_residues: int
    phi_degrees: List[Optional[float]]
    psi_degrees: List[Optional[float]]
    pdb: str  # full backbone-only PDB text


class AffinityScoreRequest(BaseModel):
    sequence: str = Field(..., min_length=1)
    smiles: List[str] = Field(..., min_length=1, max_length=500)
    checkpoint: Optional[str] = None
    config: Optional[str] = None


class AffinityScoreItem(BaseModel):
    smiles: str
    pActivity_pred: float
    activity_prob: float


class AffinityScoreResponse(BaseModel):
    target_length: int
    results: List[AffinityScoreItem]
    num_invalid_smiles: int


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", cached_models=list(_MODEL_CACHE.keys()))


@app.post("/predict_structure", response_model=StructurePredictResponse)
def predict_structure(req: StructurePredictRequest) -> StructurePredictResponse:
    checkpoint = req.checkpoint or DEFAULT_STRUCTURE_CHECKPOINT
    config_path = req.config or DEFAULT_STRUCTURE_CONFIG

    if not Path(checkpoint).exists():
        raise HTTPException(
            status_code=503,
            detail=f"No checkpoint at '{checkpoint}'. Train one first: "
            f"python scripts/train.py --config {config_path}",
        )

    model, cfg = _get_structure_model(checkpoint, config_path)
    max_len = cfg.data.get("max_len", 512)
    seq = req.sequence.strip().upper()[:max_len]

    input_ids = encode_sequence(seq, max_len=max_len).unsqueeze(0)
    with torch.no_grad():
        backbone = model.predict_structure(input_ids, seq)
        out = model(input_ids)

    L = len(seq)
    phi_deg = [None] + [round(float(v), 2) for v in torch.rad2deg(out["phi"][0, 1:L]).tolist()]
    psi_deg = [round(float(v), 2) for v in torch.rad2deg(out["psi"][0, : L - 1]).tolist()] + [None]

    with tempfile.NamedTemporaryFile(suffix=".pdb", mode="w", delete=False) as tmp:
        write_backbone_pdb(backbone, tmp.name)
        pdb_text = Path(tmp.name).read_text()
    os.unlink(tmp.name)

    return StructurePredictResponse(
        sequence=seq,
        num_residues=L,
        phi_degrees=phi_deg,
        psi_degrees=psi_deg,
        pdb=pdb_text,
    )


@app.post("/score_affinity", response_model=AffinityScoreResponse)
def score_affinity(req: AffinityScoreRequest) -> AffinityScoreResponse:
    checkpoint = req.checkpoint or DEFAULT_FUSION_CHECKPOINT
    config_path = req.config or DEFAULT_FUSION_CONFIG

    if not Path(checkpoint).exists():
        raise HTTPException(
            status_code=503,
            detail=f"No checkpoint at '{checkpoint}'. Train one first: "
            f"python scripts/train.py --config {config_path}",
        )

    predictor = _get_affinity_predictor(checkpoint, config_path)
    results = predictor.score(req.sequence, req.smiles)

    return AffinityScoreResponse(
        target_length=len(req.sequence),
        results=[AffinityScoreItem(**r) for r in results],
        num_invalid_smiles=len(req.smiles) - len(results),
    )
