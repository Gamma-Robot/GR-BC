"""Demo-only graph with isolated full-trajectory insertion, without query success edges.

This is offline trajectory evaluation: the whole query sequence (including future
observations) is available. It is not an online single-state success detector.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import GraphConfig
from .dataset import history_rows
from .graph import build_multi_view_adaptive_similarity_edges, build_temporal_edges, reverse_dijkstra


def similarity_for(features, nodes, config):
    edges, _ = build_multi_view_adaptive_similarity_edges(
        features,
        nodes,
        nodes.gripper_closed.to_numpy(),
        config.mutual_k,
        config.stride,
        config.adaptive_edge_alpha,
        config.adaptive_base_scale,
        config.sim_cost_scale,
        camera_count=3,
        node_to_rows=history_rows(nodes, config.history),
        history=config.history,
        min_camera_hits=config.min_camera_hits,
    )
    if config.cross_trajectory_only:
        ep = nodes.episode_index.to_numpy()
        edges = [edge for edge in edges if ep[edge[0]] != ep[edge[1]]]
    return edges


def evaluate(nodes, features, config, base_episodes, query_episodes):
    base_episodes, query_episodes = set(base_episodes), set(query_episodes)
    if not base_episodes or not query_episodes or base_episodes & query_episodes:
        raise ValueError("Supply nonempty, disjoint base/query episode splits")
    if (base_episodes | query_episodes) - set(nodes.episode_index):
        raise ValueError("Unknown episode IDs")
    base_mask = nodes.episode_index.isin(base_episodes).to_numpy()
    base = nodes[base_mask].reset_index(drop=True).copy()
    for ep, group in base.groupby("episode_index"):
        if not (group.next_reward > 0).any():
            raise ValueError(f"Reference episode {ep} has no success annotation")
    base.node_id = np.arange(len(base))
    base_features = features[base_mask]
    temporal = build_temporal_edges(base)
    similarity = similarity_for(base_features, base, config)
    goals = set(base.loc[base.next_reward > 0, "node_id"].astype(int))
    raw, _, _ = reverse_dijkstra(len(base), temporal, similarity, goals)
    rmax = max(float(np.quantile(raw[np.isfinite(raw)], config.rmax_quantile)), 1e-8)
    values, records = [], []
    for episode in sorted(query_episodes):
        query_mask = (nodes.episode_index == episode).to_numpy()
        query = nodes[query_mask].reset_index(drop=True).copy()
        query_features = features[query_mask]
        actual_success = bool(query.episode_success.iloc[0])
        # Do not use reward-only forced samples as a hidden query outcome hint.
        selected = (
            (query.frame_index % config.stride == 0) | (query.frame_index == query.episode_length - 1)
        ).to_numpy()
        if config.include_intervention_boundaries:
            raise ValueError("Held-out mode currently supports regular/terminal sampling only")
        query, query_features = query[selected].copy(), query_features[selected]
        query.next_reward = 0.0
        query["next.reward"] = 0.0
        query.episode_success = False
        combined = pd.concat([base, query], ignore_index=True)
        combined.node_id = np.arange(len(combined))
        combined_features = np.concatenate([base_features, query_features])
        candidate = similarity_for(combined_features, combined, config)
        additions = [edge for edge in candidate if edge[0] >= len(base) or edge[1] >= len(base)]
        query_temporal = build_temporal_edges(combined.iloc[len(base) :])
        distances, _, _ = reverse_dijkstra(
            len(combined), temporal + query_temporal, similarity + additions, goals
        )
        query_dist = distances[len(base) :]
        progress = np.clip(1.0 - query_dist / rmax, 0, 1)
        part = query[["index", "episode_index", "frame_index"]].copy()
        part["progress"] = progress
        part["episode_success"] = actual_success
        values.append(part)
        records.append(
            {
                "episode_index": int(episode),
                "success": actual_success,
                "nodes": len(query),
                "end_progress": float(progress[-1]),
                "mean_progress": float(progress.mean()),
                "low_progress_ratio": float((progress < 0.2).mean()),
                "reachable_ratio": float(np.isfinite(query_dist).mean()),
                "query_similarity_edges": len(additions),
                "regression_sum": float(np.maximum(-np.diff(progress), 0).sum()),
            }
        )
    return (
        pd.concat(values, ignore_index=True),
        records,
        {
            "base_nodes": len(base),
            "base_similarity_edges": len(similarity),
            "base_rmax": rmax,
            "base_episodes": sorted(base_episodes),
            "query_episodes": sorted(query_episodes),
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path, help="Run containing frozen MCR node features")
    parser.add_argument("--base-episodes", required=True, nargs="+", type=int)
    parser.add_argument("--query-episodes", required=True, nargs="+", type=int)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("Choose a new held-out output directory")
    summary = json.loads((args.run / "summary.json").read_text())
    config = GraphConfig(**summary["config"])
    config.validate()
    nodes = pd.read_parquet(args.run / "nodes.parquet")
    features = np.load(args.run / "layout_features.npy", allow_pickle=False)
    values, records, base = evaluate(nodes, features, config, args.base_episodes, args.query_episodes)
    args.output_dir.mkdir(parents=True)
    values.to_parquet(args.output_dir / "heldout_node_progress.parquet", index=False)
    result = {
        "protocol": "isolated full-query trajectory insertion into a fixed demo-only graph",
        "uses_future_query_observations": True,
        "query_success_edges": False,
        "query_reward_only_samples_removed": True,
        "config": summary["config"],
        **base,
        "episodes": records,
    }
    (args.output_dir / "heldout_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    from .visualize import plt

    fig, axes = plt.subplots(len(records), 1, figsize=(12, len(records) * 2.3), squeeze=False)
    for ax, record in zip(axes[:, 0], records, strict=True):
        ep = values[values.episode_index == record["episode_index"]]
        ax.plot(ep.frame_index, ep.progress, color="#2563eb" if record["success"] else "#dc4c40")
        ax.set(
            title=f"Held-out ep{record['episode_index']} | success={record['success']}",
            ylim=(-0.03, 1.03),
            ylabel="Progress",
        )
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(args.output_dir / "heldout_progress_timelines.png", dpi=160)
    plt.close(fig)
    print(
        json.dumps(
            {key: value for key, value in result.items() if key not in ("episodes", "config")}, indent=2
        )
    )


if __name__ == "__main__":
    main()
