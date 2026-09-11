import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

fastapi = pytest.importorskip("fastapi", reason="FastAPI not installed")
from fastapi.testclient import TestClient

from src.api.server import app

client = TestClient(app)


def test_health_endpoint():
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert isinstance(body["cached_models"], list)


def test_predict_structure_missing_checkpoint_returns_503():
    resp = client.post(
        "/predict_structure",
        json={"sequence": "ACDEFGHIKL", "checkpoint": "checkpoints/does_not_exist/best.pt"},
    )
    assert resp.status_code == 503


def test_predict_structure_empty_sequence_rejected():
    resp = client.post("/predict_structure", json={"sequence": ""})
    assert resp.status_code == 422


def test_score_affinity_missing_checkpoint_returns_503():
    resp = client.post(
        "/score_affinity",
        json={
            "sequence": "ACDEFGHIKL",
            "smiles": ["CCO"],
            "checkpoint": "checkpoints/does_not_exist/best.pt",
        },
    )
    assert resp.status_code == 503


def test_score_affinity_requires_at_least_one_smiles():
    resp = client.post("/score_affinity", json={"sequence": "ACDEFGHIKL", "smiles": []})
    assert resp.status_code == 422
