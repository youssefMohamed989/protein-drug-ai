"""Small glue functions mapping a batch dict -> model(**kwargs), so the
generic Trainer can drive any of the three models without special-casing.
"""

from __future__ import annotations

from typing import Dict

import torch


def protein_structure_forward(model, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return model(batch["input_ids"])


def molecule_affinity_forward(model, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return model(
        atom_features=batch["atom_features"],
        edge_index=batch["edge_index"],
        edge_features=batch["edge_features"],
        batch_index=batch["batch_index"],
        num_graphs=batch["num_graphs"],
    )


def fusion_affinity_forward(model, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return model(
        input_ids=batch["input_ids"],
        atom_features=batch["atom_features"],
        edge_index=batch["edge_index"],
        edge_features=batch["edge_features"],
        batch_index=batch["batch_index"],
        num_graphs=batch["num_graphs"],
    )
