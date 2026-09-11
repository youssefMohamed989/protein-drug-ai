# Data schema

All datasets are plain CSV/FASTA/SMILES files so they're easy to inspect,
diff, and swap for real sources. `scripts/make_toy_data.py` generates
matching synthetic files.

## 1. Protein structure task (`data/processed/protein_structure_*.csv`)

| column | type | required | description |
|---|---|---|---|
| `protein_id` | str | yes | unique identifier |
| `sequence` | str | yes | amino-acid sequence, one-letter codes |
| `ss8` | str | no | DSSP Q8 secondary structure string, same length as `sequence` |
| `contact_map` | str | no | flattened `L*L` binary matrix, values joined with `;`, row-major |

To derive `ss8`/`contact_map` from real PDB files, see
`src/data/protein_features.parse_pdb_structure` (contact map from
C-alpha coordinates) and wire in `mkdssp` via `Bio.PDB.DSSP` for `ss8`.

## 2. Drug affinity task (`data/processed/drug_affinity_*.csv`)

| column | type | required | description |
|---|---|---|---|
| `compound_id` | str | no | identifier |
| `smiles` | str | yes | canonical or as-provided SMILES |
| `pActivity` | float | no* | e.g. pIC50/pKd/pKi (regression target) |
| `label` | int (0/1) | no* | active/inactive (classification target) |

*At least one of `pActivity`/`label` should be present per row to be useful,
but rows with neither still parse (just contribute no loss term).

This schema matches a ChEMBL/BindingDB export with minimal renaming.

## 3. Protein-ligand pairs, for the fused model (`data/processed/pairs_*.csv`)

| column | type | required | description |
|---|---|---|---|
| `protein_id` | str | no | target identifier, used for `protein_cluster` split |
| `sequence` | str | yes | target protein sequence |
| `smiles` | str | yes | ligand SMILES |
| `pActivity` | float | no* | regression target |
| `label` | int (0/1) | no* | classification target |

This is the standard shape of a PDBbind- or BindingDB-derived
target-ligand-affinity table.

## 4. Virtual screening inputs

- `--target_fasta`: standard FASTA, first record is used as the screening target.
- `--library`: either
  - a `.smi` file: one `SMILES<TAB>id` pair per line (id optional), or
  - a `.csv` file with a `smiles` column (and optional `compound_id`).
