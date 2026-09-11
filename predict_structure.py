#!/usr/bin/env python
"""CLI: predict a 3D backbone structure (N/CA/C/O trace) from a sequence.

Example:
    python scripts/predict_structure.py \
        --sequence MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWELVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWSTPSELGHAGLNGDILVWNPVLEDAFELSSMGIRVDADTLKHQLALTGDEDRLELEWHQALLRGEMPQTIGGGIGQSRLTMLLLQLPHIGQVQAGVWPAAVRESVPSLL \
        --checkpoint checkpoints/structure_advanced/best.pt \
        --config configs/structure_advanced.yaml \
        --out_pdb checkpoints/predicted_structure.pdb

The output is a backbone-only PDB (N/CA/C/O per residue). Feed it into
scripts/run_md.py to add sidechains/hydrogens (via PDBFixer) and refine
with a short molecular dynamics simulation.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from src.data.protein_features import encode_sequence
from src.structure.pdb_writer import write_backbone_pdb
from src.structure.structure_model import AdvancedStructureModel
from src.utils.config import Config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/structure_advanced.yaml")
    parser.add_argument("--out_pdb", type=str, default="checkpoints/predicted_structure.pdb")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)
    model = AdvancedStructureModel(encoder_kwargs=dict(cfg.model))
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(args.device)
    model.eval()

    max_len = cfg.data.get("max_len", 512)
    seq = args.sequence.strip().upper()[:max_len]
    input_ids = encode_sequence(seq, max_len=max_len).unsqueeze(0).to(args.device)

    backbone = model.predict_structure(input_ids, seq)

    Path(args.out_pdb).parent.mkdir(parents=True, exist_ok=True)
    write_backbone_pdb(backbone, args.out_pdb)
    print(f"Wrote {len(seq)}-residue backbone-only structure to {args.out_pdb}")


if __name__ == "__main__":
    main()
