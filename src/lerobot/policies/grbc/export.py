"""Build a graph and export per-transition weights from LeRobot v3.0 data."""

import argparse
import json
from dataclasses import asdict, fields
from pathlib import Path

import numpy as np

from .config import PRESETS, GraphConfig
from .dataset import CAMERAS, dataset_fingerprint, history_rows, load_dataset, sample_nodes, table_identity
from .encoder import extract_features
from .graph import build_multi_view_adaptive_similarity_edges, build_temporal_edges, reverse_dijkstra
from .progress import dense_progress
from .weights import compute_graph_transition_weights


def run(
    dataset,
    output,
    config,
    checkpoint,
    cameras=CAMERAS,
    cache=None,
    reuse=False,
    batch_size=48,
    device="cuda",
    selected=None,
):
    config.validate()
    output = Path(output)
    if (output / "summary.json").exists() or (output / "transition_weights.parquet").exists():
        raise FileExistsError("Choose a fresh output directory; completed runs are not overwritten")
    output.mkdir(parents=True, exist_ok=True)
    data, episodes, info = load_dataset(dataset, cameras)
    nodes = sample_nodes(data, config)
    if not (nodes.next_reward > 0).any():
        raise ValueError("No positive reward frames; supply true outcome labels, not invented success")
    features, cache_info = extract_features(
        dataset,
        nodes,
        episodes,
        info,
        cameras,
        checkpoint,
        cache or output / "cache",
        batch_size,
        device,
        reuse,
    )
    mapping = history_rows(nodes, config.history)
    temporal = build_temporal_edges(nodes)
    similarity, edge_stats = build_multi_view_adaptive_similarity_edges(
        features,
        nodes,
        nodes.gripper_closed.to_numpy(),
        config.mutual_k,
        config.stride,
        config.adaptive_edge_alpha,
        config.adaptive_base_scale,
        config.sim_cost_scale,
        camera_count=3,
        node_to_rows=mapping,
        history=config.history,
        min_camera_hits=config.min_camera_hits,
    )
    if config.cross_trajectory_only:
        ep = nodes.episode_index.to_numpy()
        similarity = [edge for edge in similarity if ep[edge[0]] != ep[edge[1]]]
    goals = set(nodes.loc[nodes.next_reward > 0, "node_id"].astype(int))
    distances, next_node, next_kind = reverse_dijkstra(len(nodes), temporal, similarity, goals)
    dense, rmax = dense_progress(data, nodes, distances, config.rmax_quantile, config.interpolation)
    weighted = compute_graph_transition_weights(
        dense,
        chunk=config.chunk,
        n_action_step=config.n_action_step,
        progress_dip_tolerance=config.progress_dip_tolerance,
        min_delta_progress=config.min_delta_progress,
        flat_weight=config.flat_weight,
        weight_alpha=config.weight_alpha,
        weight_gamma=config.weight_gamma,
        weight_min=config.weight_min,
        weight_max=config.weight_max,
    )
    columns = [
        key
        for key in weighted
        if key not in ("action", "observation.state") and not key.startswith("observation.images.")
    ]
    table = weighted[columns].copy()
    table.to_parquet(output / "transition_weights.parquet", index=False)
    nodes.drop(columns=["action", "observation.state"], errors="ignore").to_parquet(
        output / "nodes.parquet", index=False
    )
    np.save(output / "layout_features.npy", features)
    np.savez_compressed(
        output / "graph.npz",
        raw_distances=distances,
        next_node=next_node,
        next_kind=np.asarray(next_kind),
        history_rows=mapping,
        temporal=np.asarray(temporal).reshape(-1, 3),
        similarity=np.asarray(similarity).reshape(-1, 4),
    )
    from .visualize import plot_outputs

    chosen = plot_outputs(output, nodes, features, table, distances, temporal, similarity, selected)
    summary = {
        "schema_version": 1,
        "implementation": "grbc",
        "config": asdict(config),
        "dataset_fingerprint": dataset_fingerprint(dataset),
        "dataset_table_identity": table_identity(data),
        "encoder": "frozen_mcr_resnet50",
        "features": cache_info,
        "cameras": cameras,
        "frames": len(data),
        "episodes": len(episodes),
        "nodes": len(nodes),
        "success_nodes": len(goals),
        "temporal_edges": len(temporal),
        "similarity_edges": len(similarity),
        "reachable_ratio": float(np.isfinite(distances).mean()),
        "rmax": rmax,
        "zero_weight_fraction": float((table.graph_transition_weight == 0).mean()),
        "mean_weight": float(table.graph_transition_weight.mean()),
        "selected_episodes": chosen,
        "edge_filter": edge_stats,
        "weight_column": "graph_transition_weight",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mcr-ckpt", type=Path, required=True)
    parser.add_argument("--preset", choices=PRESETS, default="paper")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--reuse-cache", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size-mcr", type=int, default=48)
    parser.add_argument("--cameras", nargs=3, default=CAMERAS)
    parser.add_argument("--selected-episodes", nargs="+", type=int)
    defaults = GraphConfig()
    for field in fields(GraphConfig):
        default = getattr(defaults, field.name)
        key = "--" + field.name.replace("_", "-")
        if isinstance(default, bool):
            parser.add_argument(key, action=argparse.BooleanOptionalAction, default=None)
        else:
            parser.add_argument(key, type=type(default), default=None)
    args = parser.parse_args()
    values = {**asdict(defaults), **PRESETS[args.preset]}
    if args.config:
        overrides = json.loads(args.config.read_text())
        if overrides.keys() - values.keys():
            raise ValueError(f"Unknown configuration keys: {overrides.keys() - values.keys()}")
        values.update(overrides)
    for field in fields(GraphConfig):
        if getattr(args, field.name) is not None:
            values[field.name] = getattr(args, field.name)
    summary = run(
        args.dataset,
        args.output_dir,
        GraphConfig(**values),
        args.mcr_ckpt,
        args.cameras,
        args.cache_dir,
        args.reuse_cache,
        args.batch_size_mcr,
        args.device,
        args.selected_episodes,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
