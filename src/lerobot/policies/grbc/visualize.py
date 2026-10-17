"""Static diagnostics and camera montages. These are not policy success-rate evaluations."""

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from sklearn.decomposition import PCA

from .dataset import decode_requested, load_dataset, video_requests


def episode_metrics(table):
    records = []
    for episode, rows in table.groupby("episode_index", sort=True):
        rows = rows.sort_values("frame_index")
        p = rows.progress.to_numpy(dtype=float)
        delta = np.diff(p)
        records.append(
            {
                "episode_index": int(episode),
                "frames": len(rows),
                "success": bool(rows.episode_success.iloc[0]),
                "start_progress": float(p[0]),
                "end_progress": float(p[-1]),
                "regression_sum": float(np.maximum(-delta, 0).sum()),
                "max_progress_drop": float(np.maximum(-delta, 0).max(initial=0)),
                "zero_weight_fraction": float((rows.graph_transition_weight == 0).mean()),
            }
        )
    return pd.DataFrame(records)


def plot_outputs(output, nodes, features, table, distances, temporal, similarity, selected=None):
    output = Path(output)
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    metrics = episode_metrics(table)
    if selected is None:
        selected = (
            metrics.sort_values(["regression_sum", "episode_index"], ascending=[False, True])
            .head(6)
            .episode_index.tolist()
        )
    missing = set(selected) - set(metrics.episode_index)
    if missing:
        raise ValueError(f"Unknown episodes {missing}")
    fig, axes = plt.subplots(len(selected), 1, figsize=(15, 2.8 * len(selected)), squeeze=False)
    for ax, episode in zip(axes[:, 0], selected, strict=True):
        rows = table[table.episode_index == episode].sort_values("frame_index")
        ax.plot(rows.frame_index, rows.progress, color="#2563eb", lw=1.6)
        ax.set(
            ylim=(-0.03, 1.03),
            ylabel="Progress",
            title=f"Episode {episode} | frames={len(rows)} | success={bool(rows.episode_success.iloc[0])}",
        )
        ax.grid(alpha=0.18)
        ax.tick_params(axis="y", colors="#2563eb")
        right = ax.twinx()
        right.plot(rows.frame_index, rows.graph_transition_weight, color="#dc4c40", lw=1, alpha=0.85)
        right.set(ylim=(-0.04, max(1.55, table.graph_transition_weight.max() + 0.05)), ylabel="Weight")
        right.tick_params(axis="y", colors="#dc4c40")
    axes[-1, 0].set_xlabel("Frame index")
    fig.tight_layout()
    fig.savefig(figures / "progress_weight_timelines.png", dpi=160)
    plt.close(fig)
    xy = PCA(n_components=2, random_state=23).fit_transform(features) if len(nodes) > 1 else np.zeros((1, 2))
    np.save(output / "layout.npy", xy)
    fig, ax = plt.subplots(figsize=(11, 8))
    rng = np.random.default_rng(23)
    for edges, color in ((temporal, "#94a3b8"), (similarity, "#eea740")):
        if len(edges):
            subset = np.asarray(edges)[
                rng.choice(len(edges), min(len(edges), 12000), replace=False), :2
            ].astype(int)
            ax.add_collection(LineCollection(xy[subset], colors=color, linewidths=0.35, alpha=0.2))
    finite = np.isfinite(distances)
    ax.scatter(*xy[~finite].T, s=4, c="#bbc0c7", label="Unreachable")
    plotted = ax.scatter(*xy[finite].T, c=distances[finite], s=6, cmap="viridis_r")
    goals = nodes.next_reward.to_numpy() > 0
    ax.scatter(*xy[goals].T, c="#e23e3e", marker="*", s=90, edgecolors="white", label="Success")
    starts = nodes.frame_index.to_numpy() == 0
    ax.scatter(*xy[starts].T, facecolors="none", edgecolors="#111827", s=32, label="Start")
    ax.set(title="Current-state MCR PCA (display only; not the neighbor metric)", xlabel="PC1", ylabel="PC2")
    ax.legend()
    fig.colorbar(plotted, ax=ax, label="Raw graph remaining steps")
    fig.tight_layout()
    fig.savefig(figures / "graph_overview.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    axes[0].hist(table.graph_transition_weight, bins=40, color="#db7351")
    axes[0].set(xlabel="Transition weight", ylabel="Frames")
    for success, color in ((True, "#279780"), (False, "#db7351")):
        subset = metrics[metrics.success == success]
        axes[1].scatter(subset.frames, subset.zero_weight_fraction, color=color, label=f"Success={success}")
    axes[1].set(xlabel="Episode length", ylabel="Zero-weight fraction")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(figures / "weight_diagnostics.png", dpi=160)
    plt.close(fig)
    metrics.to_json(output / "episode_metrics.json", orient="records", indent=2)
    return [int(value) for value in selected]


def montage(dataset, output, episode, camera="observation.images.cam_1", start=0, end=None, every=15):
    from PIL import Image

    if every < 1:
        raise ValueError("Frame interval must be positive")
    data, episodes, info = load_dataset(dataset)
    rows = data[data.episode_index == episode].sort_values("frame_index")
    if rows.empty:
        raise ValueError(f"Episode {episode} not found")
    end = len(rows) - 1 if end is None else min(end, len(rows) - 1)
    if start < 0 or end < start:
        raise ValueError("Invalid frame range")
    rows = rows[(rows.frame_index >= start) & (rows.frame_index <= end)].iloc[::every].reset_index(drop=True)
    if len(rows) > 120:
        raise ValueError("Request at most 120 montage frames; narrow the window or increase --every")
    images = {}
    for path, requests in video_requests(dataset, rows, episodes, info, camera).items():
        for ids, rgb in decode_requested(path, requests):
            for position in ids:
                images[position] = Image.fromarray(rgb)
    columns = min(8, len(rows))
    fig, axes = plt.subplots(
        int(np.ceil(len(rows) / columns)),
        columns,
        figsize=(columns * 2.5, np.ceil(len(rows) / columns) * 2.1),
        squeeze=False,
    )
    for i, ax in enumerate(axes.flat):
        ax.axis("off")
        if i < len(rows):
            ax.imshow(images[i])
            ax.set_title(f"ep{episode} / frame {rows.frame_index.iloc[i]}", fontsize=9)
    fig.tight_layout()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--episodes", nargs="+", type=int)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--camera", default="observation.images.cam_1")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int)
    parser.add_argument("--every", type=int, default=15)
    args = parser.parse_args()
    artifacts = np.load(args.run / "graph.npz", allow_pickle=False)
    nodes, table = (
        pd.read_parquet(args.run / "nodes.parquet"),
        pd.read_parquet(args.run / "transition_weights.parquet"),
    )
    selected = plot_outputs(
        args.run,
        nodes,
        np.load(args.run / "layout_features.npy"),
        table,
        artifacts["raw_distances"],
        artifacts["temporal"],
        artifacts["similarity"],
        args.episodes,
    )
    if args.dataset:
        for episode in selected:
            name = args.camera.rsplit(".", 1)[-1]
            montage(
                args.dataset,
                args.run / f"figures/ep{episode}_{name}_{args.start}_{args.end}.png",
                episode,
                args.camera,
                args.start,
                args.end,
                args.every,
            )
    print(json.dumps({"figures": str(args.run / "figures"), "episodes": selected}))


if __name__ == "__main__":
    main()
