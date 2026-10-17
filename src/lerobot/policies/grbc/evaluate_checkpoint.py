"""Stochastic flow-matching validation loss, not environment success rate.

Use TrainPipelineConfig flags, including --policy.path and --dataset.episodes.
All selected episodes are evaluated once; select held-out episodes explicitly.
"""

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.utils.random_utils import set_seed

from .training import GRBCWeights


def checkpoint_rename_map(checkpoint, override):
    """Recover the camera contract saved with a local training checkpoint."""
    if override or checkpoint is None:
        return override
    manifest = Path(checkpoint) / "policy_preprocessor.json"
    if not manifest.is_file():
        return override
    for step in json.loads(manifest.read_text())["steps"]:
        if step.get("registry_name") == "rename_observations_processor":
            return step.get("config", {}).get("rename_map", {})
    return override


@parser.wrap()
def evaluate(cfg: TrainPipelineConfig):
    cfg.validate()
    if cfg.policy.type not in ("smolvla", "xvla", "pi05"):
        raise ValueError("Supported validation backbones: smolvla, xvla, pi05")
    if not cfg.dataset.episodes:
        raise ValueError("Specify --dataset.episodes='[...]' with an explicit validation split")
    set_seed(cfg.seed or 0)
    dataset = make_dataset(cfg)
    cfg.rename_map = checkpoint_rename_map(cfg.policy.pretrained_path, cfg.rename_map)
    policy = make_policy(cfg.policy, ds_meta=dataset.meta, rename_map=cfg.rename_map)
    overrides = {"device_processor": {"device": cfg.policy.device}}
    if cfg.rename_map:
        overrides["rename_observations_processor"] = {"rename_map": cfg.rename_map}
    processor, _ = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=cfg.policy.pretrained_path,
        preprocessor_overrides=overrides,
    )
    weights = (
        GRBCWeights(cfg.grbc_weights_path, cfg.grbc_weight_column, device=cfg.policy.device)
        if cfg.use_grbc
        else None
    )
    if weights:
        weights.validate_dataset(dataset)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, num_workers=cfg.num_workers, shuffle=False)
    total_loss, weighted_sum, weight_sum, count = 0.0, 0.0, 0.0, 0
    policy.eval()
    with torch.inference_mode():
        for batch in loader:
            batch = processor(batch)
            per_sample, _ = policy(batch, reduction="none")
            if per_sample.ndim != 1:
                raise ValueError("Validation requires per-sample losses (B,)")
            total_loss += per_sample.float().sum().item()
            count += len(per_sample)
            if weights:
                w, _ = weights.compute_batch_weights(batch)
                weighted_sum += (w * per_sample.float()).sum().item()
                weight_sum += w.sum().item()
    result = {
        "interpretation": "Offline stochastic flow-matching loss, not policy success",
        "seed": cfg.seed,
        "validation_episodes": cfg.dataset.episodes,
        "frames": count,
        "unweighted_loss": total_loss / max(count, 1),
    }
    if weights:
        result.update(
            weighted_mean_loss=weighted_sum / max(count, 1),
            self_normalized_loss=weighted_sum / weight_sum if weight_sum else None,
        )
    cfg.output_dir.mkdir(parents=True, exist_ok=False)
    (cfg.output_dir / "validation_loss.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    evaluate()
