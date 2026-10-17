"""Keep raw graph distances; interpolation policy is explicit and serialized."""

import numpy as np
import pandas as pd


def dense_progress(data, nodes, distances, quantile=0.95, mode="paper"):
    finite = np.isfinite(distances)
    if not finite.any():
        raise ValueError("Graph has no reachable success nodes; verify real reward labels")
    rmax = max(float(np.quantile(distances[finite], quantile)), 1e-8)
    if mode not in ("paper", "legacy"):
        raise ValueError("Unknown interpolation policy")
    parts = []
    for episode, rows in data.groupby("episode_index", sort=True):
        part = rows.sort_values("frame_index").copy()
        sampled = nodes[nodes.episode_index == episode].sort_values("frame_index")
        x = sampled.frame_index.to_numpy(dtype=int)
        raw = distances[sampled.node_id.to_numpy(dtype=int)]
        frames = part.frame_index.to_numpy(dtype=int)
        if mode == "legacy":
            values = np.minimum(np.where(np.isfinite(raw), raw, rmax), rmax).astype(np.float32)
            remaining = np.interp(frames, x, values).astype(np.float32)
        else:
            remaining = np.full(len(frames), np.inf, dtype=np.float64)
            for left in range(len(x) - 1):
                if np.isfinite(raw[left : left + 2]).all():
                    mask = (frames >= x[left]) & (frames <= x[left + 1])
                    remaining[mask] = np.interp(frames[mask], x[left : left + 2], raw[left : left + 2])
            remaining[x] = raw
        part["remaining_hat"] = remaining
        part["progress"] = np.clip(1.0 - remaining.astype(float) / rmax, 0.0, 1.0)
        part["episode_success"] = bool((part["next.reward"] > 0).any())
        parts.append(part)
    return pd.concat(parts, ignore_index=True), rmax
