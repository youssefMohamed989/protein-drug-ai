"""Load a trained FusionAffinityModel checkpoint and score protein-ligand pairs."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Tuple

import torch

from src.data.molecule_features import batch_molecule_graphs, smiles_to_graph
from src.data.protein_features import encode_sequence
from src.models.fusion import FusionAffinityModel
from src.utils.config import Config


class AffinityPredictor:
    def __init__(self, checkpoint_path: str, config: Config, device: str = "cpu"):
        self.device = torch.device(device)
        self.model = FusionAffinityModel(
            protein_kwargs=dict(config.model.protein),
            molecule_kwargs=dict(config.model.molecule),
            fusion_hidden_dim=config.model.fusion_hidden_dim,
            fusion_heads=config.model.fusion_heads,
        )
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()
        self.max_len = config.data.get("max_len", 512)

    @torch.no_grad()
    def score(self, sequence: str, smiles_list: List[str]) -> List[dict]:
        """Score one protein sequence against a list of candidate SMILES.
        Returns a list of dicts with predicted pActivity and activity probability,
        aligned to the (validity-filtered) input order.
        """
        graphs, valid_smiles = [], []
        for smi in smiles_list:
            g = smiles_to_graph(smi)
            if g is not None:
                graphs.append(g)
                valid_smiles.append(smi)

        if not graphs:
            return []

        input_ids = encode_sequence(sequence, max_len=self.max_len).unsqueeze(0)
        input_ids = input_ids.repeat(len(graphs), 1).to(self.device)

        atom_feats, edge_index, edge_feats, batch_idx = batch_molecule_graphs(graphs)
        atom_feats = atom_feats.to(self.device)
        edge_index = edge_index.to(self.device)
        edge_feats = edge_feats.to(self.device)
        batch_idx = batch_idx.to(self.device)

        # NOTE: we replicate the protein across the batch so each ligand is
        # scored against the same target in one forward pass. For very large
        # libraries, chunk this call (see scripts/screen.py).
        out = self.model(
            input_ids=input_ids,
            atom_features=atom_feats,
            edge_index=edge_index,
            edge_features=edge_feats,
            batch_index=batch_idx,
            num_graphs=len(graphs),
        )

        pActivity = out["pActivity_pred"].cpu().tolist()
        activity_prob = torch.sigmoid(out["activity_logit"]).cpu().tolist()

        results = []
        for smi, pa, prob in zip(valid_smiles, pActivity, activity_prob):
            results.append({"smiles": smi, "pActivity_pred": pa, "activity_prob": prob})
        return results
