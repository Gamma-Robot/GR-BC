"""Strict precomputed sample weights and DDP-correct weighted SFT reduction."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch


class GRBCWeights:
    def __init__(self, path, column="graph_transition_weight", reduction="mean", device="cpu"):
        if reduction not in ("mean", "self_normalized"):
            raise ValueError("grbc_reduction must be mean or self_normalized")
        table = pd.read_parquet(path)
        if {"index", column} - set(table):
            raise ValueError(f"Weights require index and {column}")
        indices = table["index"].to_numpy()
        if not np.isfinite(indices).all() or not np.equal(indices, indices.astype(np.int64)).all():
            raise ValueError("Weight indices must be finite integers")
        weights = table[column].to_numpy(dtype=np.float32)
        if table["index"].duplicated().any() or not np.isfinite(weights).all() or (weights < 0).any():
            raise ValueError("Duplicate indices, nonfinite or negative weights")
        self.lookup = pd.Series(weights, index=indices.astype(np.int64))
        self.table, self.device, self.reduction = table, device, reduction
        summary = Path(path).with_name("summary.json")
        self.metadata = json.loads(summary.read_text()) if summary.exists() else None

    def validate_dataset(self, dataset):
        # Explicitly validate coverage before expensive backbone initialization/training.
        indices = np.asarray(dataset.hf_dataset["index"], dtype=np.int64)
        if not np.isin(indices, self.lookup.index).all():
            raise ValueError("Weight table does not cover the selected dataset indices")
        if "episode_index" in self.table and "frame_index" in self.table:
            lookup = self.table.set_index("index").loc[indices]
            for key in ("episode_index", "frame_index"):
                if not np.array_equal(lookup[key], np.asarray(dataset.hf_dataset[key])):
                    raise ValueError(f"Weights belong to a different dataset: {key} mismatch")
        if self.metadata and getattr(dataset, "root", None):
            from .dataset import load_dataset, table_identity

            expected = self.metadata.get("dataset_table_identity")
            if expected and expected != table_identity(
                load_dataset(dataset.root, self.metadata["cameras"])[0]
            ):
                raise ValueError("Dataset changed since weight generation; regenerate weights")

    def compute_batch_weights(self, batch):
        if "index" not in batch:
            raise KeyError("GRBC requires global sample index; the preprocessor dropped it")
        ids = torch.as_tensor(batch["index"]).detach().cpu().numpy().reshape(-1)
        if not np.equal(ids, ids.astype(np.int64)).all():
            raise ValueError("Batch indices must be integers")
        selected = self.lookup.reindex(ids.astype(np.int64))
        if selected.isna().any():
            raise KeyError(f"No weight for indices: {ids[selected.isna()][:10]}")
        weights = torch.tensor(selected.to_numpy(), dtype=torch.float32, device=self.device)
        return weights, {
            "grbc_mean_weight": weights.mean().item(),
            "grbc_zero_fraction": (weights == 0).float().mean().item(),
        }


def weighted_loss(per_sample, weights, accelerator=None, reduction="mean"):
    if per_sample.ndim != 1 or per_sample.shape != weights.shape:
        raise ValueError(f"Expected matching (B,) losses/weights, got {per_sample.shape}, {weights.shape}")
    if reduction not in ("mean", "self_normalized"):
        raise ValueError("Unknown GRBC reduction")
    weights = weights.to(device=per_sample.device, dtype=torch.float32)
    denominator = weights.sum() if reduction == "self_normalized" else weights.new_tensor(weights.numel())
    world_size = accelerator.num_processes if accelerator is not None else 1
    if accelerator is not None:
        denominator = accelerator.reduce(denominator.detach(), reduction="sum")
    # DDP averages local gradients, so compensate once for the global denominator.
    return (per_sample.float() * weights).sum() * world_size / denominator.clamp_min(1e-12)
