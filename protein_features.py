"""Protein sequence <-> tensor featurization, plus optional PDB/DSSP parsing.

The pipeline is designed to accept a pluggable `embedding_backend`
(e.g. an ESM-2 wrapper) for residue-level embeddings; if none is supplied
it falls back to a learned nn.Embedding over the 20 standard amino acids
(+ pad/unk/mask tokens), which keeps the whole pipeline runnable offline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
SPECIAL_TOKENS = ["<pad>", "<unk>", "<cls>", "<eos>"]
VOCAB = SPECIAL_TOKENS + AMINO_ACIDS
AA_TO_IDX: Dict[str, int] = {aa: i for i, aa in enumerate(VOCAB)}
PAD_IDX = AA_TO_IDX["<pad>"]
UNK_IDX = AA_TO_IDX["<unk>"]

# 8-class secondary structure (DSSP Q8) label set, used by the aux head.
SS8_CLASSES = ["H", "G", "I", "E", "B", "T", "S", "C"]  # C = coil/loop/other
SS8_TO_IDX = {c: i for i, c in enumerate(SS8_CLASSES)}


def encode_sequence(sequence: str, max_len: Optional[int] = None) -> torch.LongTensor:
    """Map an amino-acid string to a LongTensor of vocab indices."""
    sequence = sequence.strip().upper()
    ids = [AA_TO_IDX.get(aa, UNK_IDX) for aa in sequence]
    if max_len is not None:
        ids = ids[:max_len]
        ids = ids + [PAD_IDX] * (max_len - len(ids))
    return torch.tensor(ids, dtype=torch.long)


def encode_ss8(labels: str, max_len: Optional[int] = None) -> torch.LongTensor:
    """Map a DSSP-style Q8 secondary-structure string to class indices.

    Unknown / lowercase / '-' characters map to coil (C). Padding position
    are filled with -100 (ignored by CrossEntropyLoss default ignore_index).
    """
    ids = [SS8_TO_IDX.get(c.upper(), SS8_TO_IDX["C"]) for c in labels]
    if max_len is not None:
        ids = ids[:max_len]
        pad_needed = max_len - len(ids)
        ids = ids + [-100] * pad_needed
    return torch.tensor(ids, dtype=torch.long)


def contact_map_from_coords(ca_coords: np.ndarray, threshold: float = 8.0) -> np.ndarray:
    """Build a binary L x L contact map from C-alpha coordinates (Angstrom)."""
    diff = ca_coords[:, None, :] - ca_coords[None, :, :]
    dist = np.sqrt((diff**2).sum(-1))
    contacts = (dist <= threshold).astype(np.float32)
    np.fill_diagonal(contacts, 0.0)
    return contacts


@dataclass
class ProteinRecord:
    """One protein sample: sequence + optional structural labels."""

    protein_id: str
    sequence: str
    ss8: Optional[str] = None  # secondary structure string, len == len(sequence)
    contact_map: Optional[np.ndarray] = None  # (L, L) binary
    rsa: Optional[np.ndarray] = None  # (L,) relative solvent accessibility in [0, 1]


def parse_fasta(path: str) -> List[Tuple[str, str]]:
    """Parse a (possibly multi-record) FASTA file -> list of (id, sequence)."""
    records: List[Tuple[str, str]] = []
    header, chunks = None, []
    with open(path, "r") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    records.append((header, "".join(chunks)))
                header = line[1:].split()[0]
                chunks = []
            else:
                chunks.append(line.strip())
    if header is not None:
        records.append((header, "".join(chunks)))
    return records


def parse_pdb_structure(path: str, chain_id: Optional[str] = None) -> ProteinRecord:
    """Extract sequence + C-alpha-based contact map from a PDB file via Biopython.

    Secondary structure (ss8) is left as None here since DSSP requires the
    external `mkdssp` binary; if available, wire it in via Bio.PDB.DSSP.
    """
    from Bio.PDB import PDBParser
    from Bio.PDB.Polypeptide import index_to_one, three_to_index

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", path)
    model = next(structure.get_models())

    chains = list(model.get_chains())
    if chain_id is not None:
        chains = [c for c in chains if c.id == chain_id]
    if not chains:
        raise ValueError(f"No matching chain found in {path}")
    chain = chains[0]

    seq_chars: List[str] = []
    ca_coords: List[List[float]] = []
    for residue in chain:
        if "CA" not in residue:
            continue
        try:
            one_letter = index_to_one(three_to_index(residue.get_resname()))
        except Exception:
            one_letter = "X"
        seq_chars.append(one_letter)
        ca_coords.append(residue["CA"].get_coord().tolist())

    sequence = "".join(seq_chars)
    coords = np.array(ca_coords, dtype=np.float32)
    contacts = contact_map_from_coords(coords) if len(coords) > 0 else None

    return ProteinRecord(
        protein_id=Path_stem(path),
        sequence=sequence,
        ss8=None,
        contact_map=contacts,
    )


def Path_stem(path: str) -> str:
    from pathlib import Path as _P

    return _P(path).stem
