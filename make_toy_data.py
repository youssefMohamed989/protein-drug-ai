#!/usr/bin/env python
"""Generates small, valid, synthetic datasets so every command in the README
runs immediately without network access. Replace these files with real
ChEMBL/PDB-derived data (matching the same schema) for real experiments.

Schema documented in docs/data_schema.md.
"""

import random
from pathlib import Path

import numpy as np
import pandas as pd

random.seed(42)
np.random.seed(42)

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
RAW.mkdir(parents=True, exist_ok=True)
PROCESSED.mkdir(parents=True, exist_ok=True)

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
SS8 = list("HGIEBTSC")

# A handful of real, valid drug-like SMILES (well-known small molecules) so
# RDKit featurization exercises realistic chemistry, plus programmatic
# decoys for volume.
SEED_SMILES = [
    "CC(=O)OC1=CC=CC=C1C(=O)O",  # aspirin
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",  # ibuprofen
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # caffeine
    "CC(=O)Nc1ccc(O)cc1",  # paracetamol
    "COc1ccc2nc(sc2c1)S(=O)Cc1ncc(C)c(OC)c1C",  # omeprazole-like
    "CN1CCC[C@H]1c1cccnc1",  # nicotine
    "OC(=O)c1ccccc1O",  # salicylic acid
    "CC1=CC(=O)C=CC1=O",  # simple quinone
    "CCN(CC)CCNC(=O)c1ccc(N)cc1",  # procainamide-like
    "c1ccc2c(c1)ccc1ccccc12",  # anthracene-like scaffold
]


def random_protein_sequence(min_len=60, max_len=200) -> str:
    length = random.randint(min_len, max_len)
    return "".join(random.choice(AMINO_ACIDS) for _ in range(length))


def random_ss8(length: int) -> str:
    # bias toward runs (helix/sheet segments) rather than iid noise, to look
    # a bit more like real secondary structure strings
    out = []
    cur = random.choice(SS8)
    for _ in range(length):
        if random.random() < 0.15:
            cur = random.choice(SS8)
        out.append(cur)
    return "".join(out)


def random_contact_map(length: int) -> str:
    coords = np.cumsum(np.random.randn(length, 3), axis=0)  # fake random-walk backbone
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt((diff**2).sum(-1))
    contacts = (dist <= 8.0).astype(np.float32)
    np.fill_diagonal(contacts, 0.0)
    return ";".join(str(int(v)) for v in contacts.flatten())


def perturb_smiles_pool(n: int) -> list:
    pool = []
    for _ in range(n):
        pool.append(random.choice(SEED_SMILES))
    return pool


def make_protein_structure_data(n_train=120, n_val=20, n_test=20):
    def make_split(n, prefix):
        rows = []
        for i in range(n):
            seq = random_protein_sequence()
            rows.append(
                {
                    "protein_id": f"{prefix}_{i:04d}",
                    "sequence": seq,
                    "ss8": random_ss8(len(seq)),
                    "contact_map": random_contact_map(len(seq)),
                }
            )
        return pd.DataFrame(rows)

    make_split(n_train, "train").to_csv(PROCESSED / "protein_structure_train.csv", index=False)
    make_split(n_val, "val").to_csv(PROCESSED / "protein_structure_val.csv", index=False)
    make_split(n_test, "test").to_csv(PROCESSED / "protein_structure_test.csv", index=False)


def make_drug_affinity_data(n_train=200, n_val=40, n_test=40):
    def make_split(n, prefix):
        smiles = perturb_smiles_pool(n)
        rows = []
        for i, smi in enumerate(smiles):
            # synthetic pActivity correlated with molecule length as a
            # learnable-but-not-trivial toy signal
            base = 5.0 + 0.05 * len(smi) + np.random.randn() * 0.5
            rows.append(
                {
                    "compound_id": f"{prefix}_{i:04d}",
                    "smiles": smi,
                    "pActivity": round(float(base), 3),
                    "label": int(base > 6.0),
                }
            )
        return pd.DataFrame(rows)

    make_split(n_train, "train").to_csv(PROCESSED / "drug_affinity_train.csv", index=False)
    make_split(n_val, "val").to_csv(PROCESSED / "drug_affinity_val.csv", index=False)
    make_split(n_test, "test").to_csv(PROCESSED / "drug_affinity_test.csv", index=False)


def make_pair_data(n_targets=6, n_ligands_each=30):
    rows = []
    targets = [(f"target_{t:02d}", random_protein_sequence(80, 150)) for t in range(n_targets)]
    for pid, seq in targets:
        for i in range(n_ligands_each):
            smi = random.choice(SEED_SMILES)
            base = 5.0 + 0.03 * len(seq) + 0.05 * len(smi) + np.random.randn() * 0.4
            rows.append(
                {
                    "protein_id": pid,
                    "sequence": seq,
                    "smiles": smi,
                    "pActivity": round(float(base), 3),
                    "label": int(base > 6.2),
                }
            )
    df = pd.DataFrame(rows).sample(frac=1.0, random_state=42).reset_index(drop=True)

    n = len(df)
    n_train = int(n * 0.75)
    n_val = int(n * 0.10)
    df.iloc[:n_train].to_csv(PROCESSED / "pairs_train.csv", index=False)
    df.iloc[n_train : n_train + n_val].to_csv(PROCESSED / "pairs_val.csv", index=False)
    df.iloc[n_train + n_val :].to_csv(PROCESSED / "pairs_test.csv", index=False)


def make_structure_advanced_data(n_train=100, n_val=20, n_test=20):
    """Toy phi/psi labels sampled from realistic Ramachandran-like clusters
    (alpha-helix / beta-sheet basins) rather than uniform noise, so the
    torsion head has actual structure to learn on this synthetic set.
    """
    basins = [
        (-60, -45, 25),  # alpha helix (phi, psi, spread degrees)
        (-120, 120, 25),  # beta sheet
        (60, 30, 20),  # left-handed helix (rare, adds variety)
    ]

    def sample_phi_psi(length):
        phis, psis = [], []
        for i in range(length):
            phi_c, psi_c, spread = random.choice(basins)
            phis.append(phi_c + np.random.randn() * spread * 0.3)
            psis.append(psi_c + np.random.randn() * spread * 0.3)
        phis[0] = None
        psis[-1] = None
        return phis, psis

    def make_split(n, prefix):
        rows = []
        for i in range(n):
            seq = random_protein_sequence(40, 100)
            ss8 = random_ss8(len(seq))
            cmap = random_contact_map(len(seq))
            phis, psis = sample_phi_psi(len(seq))
            rows.append(
                {
                    "protein_id": f"{prefix}_{i:04d}",
                    "sequence": seq,
                    "ss8": ss8,
                    "contact_map": cmap,
                    "phi": ";".join("nan" if p is None else f"{p:.2f}" for p in phis),
                    "psi": ";".join("nan" if p is None else f"{p:.2f}" for p in psis),
                }
            )
        return pd.DataFrame(rows)

    make_split(n_train, "train").to_csv(PROCESSED / "structure_advanced_train.csv", index=False)
    make_split(n_val, "val").to_csv(PROCESSED / "structure_advanced_val.csv", index=False)
    make_split(n_test, "test").to_csv(PROCESSED / "structure_advanced_test.csv", index=False)


def make_screening_inputs():
    target_seq = random_protein_sequence(100, 140)
    with open(RAW / "target.fasta", "w") as f:
        f.write(">demo_target\n")
        f.write(target_seq + "\n")

    with open(RAW / "candidates.smi", "w") as f:
        for i, smi in enumerate(SEED_SMILES * 3):
            f.write(f"{smi}\tcand_{i:03d}\n")


if __name__ == "__main__":
    make_protein_structure_data()
    make_drug_affinity_data()
    make_pair_data()
    make_structure_advanced_data()
    make_screening_inputs()
    print(f"Toy data written to {PROCESSED} and {RAW}")
