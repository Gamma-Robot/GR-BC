"""Inspect the actual stored shortest path, including trajectory switches."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .dataset import decode_requested, load_dataset, video_requests
from .visualize import plt


def trace_path(next_node, start):
    if start < 0 or start >= len(next_node):
        raise ValueError("Invalid start node ID")
    path, seen = [], set()
    current = start
    while 0 <= current < len(next_node):
        if current in seen:
            raise ValueError("Cycle detected in shortest-path pointers")
        seen.add(current)
        path.append(current)
        current = int(next_node[current])
    return path, current == len(next_node)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--episode", required=True, type=int)
    parser.add_argument("--frame", required=True, type=int, help="Exact sampled frame, see nodes.parquet")
    parser.add_argument("--camera", default="observation.images.cam_1")
    parser.add_argument("--max-nodes", type=int, default=32)
    args = parser.parse_args()
    if not 1 <= args.max_nodes <= 120:
        raise ValueError("max-nodes must be between 1 and 120")
    nodes = pd.read_parquet(args.run / "nodes.parquet")
    graph = np.load(args.run / "graph.npz", allow_pickle=False)
    selected = nodes[(nodes.episode_index == args.episode) & (nodes.frame_index == args.frame)]
    if len(selected) != 1:
        raise ValueError("Select an exact graph node by episode/frame")
    path, reached = trace_path(graph["next_node"], int(selected.node_id.iloc[0]))
    _, episodes, info = load_dataset(args.dataset)
    visible = nodes.iloc[path[: args.max_nodes]].reset_index(drop=True)
    images = {}
    for video, requests in video_requests(args.dataset, visible, episodes, info, args.camera).items():
        for ids, rgb in decode_requested(video, requests):
            for i in ids:
                images[i] = rgb
    columns = min(6, len(visible))
    fig, axes = plt.subplots(
        int(np.ceil(len(visible) / columns)),
        columns,
        figsize=(columns * 3, np.ceil(len(visible) / columns) * 2.6),
        squeeze=False,
    )
    for i, ax in enumerate(axes.flat):
        ax.axis("off")
        if i < len(visible):
            row = visible.iloc[i]
            ax.imshow(images[i])
            ax.set_title(
                f"hop {i}: ep{int(row.episode_index)} / f{int(row.frame_index)}\n"
                f"R={graph['raw_distances'][path[i]]:.1f}; next={graph['next_kind'][path[i]]}",
                fontsize=8,
            )
    fig.suptitle(f"Actual shortest path | reaches sink={reached} | showing {len(visible)}/{len(path)} nodes")
    fig.tight_layout()
    stem = args.run / f"figures/path_ep{args.episode}_f{args.frame}_{args.camera.rsplit('.', 1)[-1]}"
    fig.savefig(stem.with_suffix(".png"), dpi=150)
    plt.close(fig)
    records = nodes.iloc[path][["node_id", "episode_index", "frame_index"]].to_dict(orient="records")
    stem.with_suffix(".json").write_text(
        json.dumps({"reaches_sink": reached, "full_path": records}, indent=2) + "\n"
    )
    print(stem.with_suffix(".png"))


if __name__ == "__main__":
    main()
