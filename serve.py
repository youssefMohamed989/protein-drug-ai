#!/usr/bin/env python
"""CLI: launch the REST inference server.

Example:
    python scripts/serve.py --port 8000
    python scripts/serve.py --structure_checkpoint checkpoints/structure_advanced/best.pt \
        --fusion_checkpoint checkpoints/fusion_affinity/best.pt

Then:
    curl -X POST http://localhost:8000/predict_structure \
        -H "Content-Type: application/json" \
        -d '{"sequence": "ACDEFGHIKLMNPQRSTVWY"}'
"""

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--structure_checkpoint", type=str, default=None)
    parser.add_argument("--structure_config", type=str, default=None)
    parser.add_argument("--fusion_checkpoint", type=str, default=None)
    parser.add_argument("--fusion_config", type=str, default=None)
    parser.add_argument("--reload", action="store_true", help="auto-reload on code changes (dev only)")
    args = parser.parse_args()

    if args.structure_checkpoint:
        os.environ["PDAI_STRUCTURE_CHECKPOINT"] = args.structure_checkpoint
    if args.structure_config:
        os.environ["PDAI_STRUCTURE_CONFIG"] = args.structure_config
    if args.fusion_checkpoint:
        os.environ["PDAI_FUSION_CHECKPOINT"] = args.fusion_checkpoint
    if args.fusion_config:
        os.environ["PDAI_FUSION_CONFIG"] = args.fusion_config

    import uvicorn

    uvicorn.run("src.api.server:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
