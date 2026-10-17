"""Multi-view graph construction and shortest-path remaining-step retrieval."""

from __future__ import annotations

import heapq

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


def l2_normalize(features: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    return (features / np.maximum(norms, 1e-8)).astype(np.float32)


def _split_camera_features(features: np.ndarray, camera_count: int) -> list[np.ndarray]:
    if features.ndim != 2:
        raise ValueError(f"Expected a 2D feature matrix, got shape={features.shape}")
    if camera_count < 1:
        raise ValueError("camera_count must be >= 1")
    if features.shape[1] % camera_count != 0:
        raise ValueError(
            f"Feature dimension {features.shape[1]} is not divisible by camera_count={camera_count}"
        )
    camera_dim = features.shape[1] // camera_count
    pieces = []
    for camera_idx in range(camera_count):
        start = camera_idx * camera_dim
        end = start + camera_dim
        pieces.append(l2_normalize(features[:, start:end].astype(np.float32)))
    return pieces


def _split_camera_history_features(
    features: np.ndarray,
    node_to_rows: np.ndarray,
    camera_count: int,
) -> list[np.ndarray]:
    """Return one history-concat embedding matrix per camera.

    For each node and each camera c, the row is:
    [z_t,c, z_{t-1},c, ..., z_{t-k},c], with episode starts padded by the first
    sampled node through ``node_to_rows``.
    """
    camera_features = _split_camera_features(features, camera_count)
    history_features = []
    for camera_feature in camera_features:
        pieces = [camera_feature[node_to_rows[:, hist_step]] for hist_step in range(node_to_rows.shape[1])]
        history_features.append(l2_normalize(np.concatenate(pieces, axis=1).astype(np.float32)))
    return history_features


def _camera_mutual_pairs(
    features: np.ndarray, mutual_k: int
) -> tuple[set[tuple[int, int]], dict[tuple[int, int], float]]:
    n_nodes = len(features)
    nn = NearestNeighbors(n_neighbors=min(mutual_k + 1, n_nodes), metric="euclidean", n_jobs=-1)
    nn.fit(features)
    distances, indices = nn.kneighbors(features)
    # The first result is not necessarily self when identical embeddings tie.
    neighbors = [
        [(int(j), float(d)) for d, j in zip(ds, ids, strict=True) if int(j) != i][:mutual_k]
        for i, (ds, ids) in enumerate(zip(distances, indices, strict=True))
    ]
    neighbor_sets = [{j for j, _ in row} for row in neighbors]
    distance_lookup: dict[tuple[int, int], float] = {}
    for i in range(n_nodes):
        for j, distance in neighbors[i]:
            distance_lookup[(i, j)] = distance

    pairs: set[tuple[int, int]] = set()
    pair_distances: dict[tuple[int, int], float] = {}
    for i in range(n_nodes):
        for j, _ in neighbors[i]:
            if i >= j:
                continue
            if i not in neighbor_sets[j]:
                continue
            pair = (int(i), int(j))
            d_ij = distance_lookup.get((i, j), np.nan)
            d_ji = distance_lookup.get((j, i), d_ij)
            pairs.add(pair)
            pair_distances[pair] = float(np.nanmean([d_ij, d_ji]))
    return pairs, pair_distances


def _camera_temporal_scales(
    camera_features: list[np.ndarray],
    nodes: pd.DataFrame,
) -> np.ndarray:
    """Estimate local temporal feature scale sigma_i,c for every node and camera."""
    n_nodes = len(nodes)
    camera_count = len(camera_features)
    scales = np.zeros((n_nodes, camera_count), dtype=np.float32)

    for _episode_index, ep_nodes in nodes.groupby("episode_index", sort=True):
        ep_nodes = ep_nodes.sort_values("frame_index").reset_index(drop=True)
        node_ids = ep_nodes["node_id"].to_numpy(dtype=int)
        for pos, node_id in enumerate(node_ids):
            for camera_idx, features in enumerate(camera_features):
                neighbor_distances = []
                if pos > 0:
                    prev_id = int(node_ids[pos - 1])
                    neighbor_distances.append(float(np.linalg.norm(features[node_id] - features[prev_id])))
                if pos + 1 < len(node_ids):
                    next_id = int(node_ids[pos + 1])
                    neighbor_distances.append(float(np.linalg.norm(features[node_id] - features[next_id])))
                scales[node_id, camera_idx] = (
                    float(np.mean(neighbor_distances)) if neighbor_distances else 0.0
                )
    return scales


def build_multi_view_adaptive_similarity_edges(
    features: np.ndarray,
    nodes: pd.DataFrame,
    gripper_closed: np.ndarray,
    mutual_k: int,
    stride: int,
    adaptive_alpha: float,
    adaptive_base_scale: float,
    cost_scale: float,
    retrieval_label: str = "three_camera_mcr_per_view_history_adaptive",
    camera_count: int = 3,
    node_to_rows: np.ndarray | None = None,
    history: int = 0,
    min_camera_hits: int = 3,
) -> tuple[list[tuple[int, int, float, float]], dict]:
    """Build edges from per-camera history mutual-kNN and adaptive length.

    Candidate pairs must appear in at least ``min_camera_hits`` camera-specific
    mutual-kNN sets.  ``min_camera_hits=1`` is the original union behavior, and
    ``min_camera_hits=camera_count`` is the all-camera intersection.  A pair
    becomes a similarity edge only if gripper state matches and

    ``ell(i,j) = max_c ||z_i,c - z_j,c|| / (eps_c + (sigma_i,c + sigma_j,c) / 2)``

    is at most ``adaptive_alpha``.  The kept edge cost is ``stride * ell(i,j)``
    times ``cost_scale``.
    """
    if adaptive_alpha <= 0:
        raise ValueError(f"adaptive_alpha must be > 0, got {adaptive_alpha}")
    if adaptive_base_scale <= 0:
        raise ValueError(f"adaptive_base_scale must be > 0, got {adaptive_base_scale}")
    if cost_scale <= 0:
        raise ValueError(f"cost_scale must be > 0, got {cost_scale}")
    if min_camera_hits < 1 or min_camera_hits > camera_count:
        raise ValueError(f"min_camera_hits must be in [1, {camera_count}], got {min_camera_hits}")

    n_nodes = len(nodes)
    if node_to_rows is None:
        node_to_rows = np.arange(n_nodes, dtype=np.int64)[:, None]
    if len(node_to_rows) != n_nodes:
        raise ValueError(f"node_to_rows length {len(node_to_rows)} does not match nodes length {n_nodes}")
    camera_features = _split_camera_history_features(features, node_to_rows, camera_count)
    temporal_scales = _camera_temporal_scales(camera_features, nodes)

    camera_pair_sets: list[set[tuple[int, int]]] = []
    camera_pair_distances: list[dict[tuple[int, int], float]] = []
    for camera_feature in camera_features:
        pairs, distances = _camera_mutual_pairs(camera_feature, mutual_k)
        camera_pair_sets.append(pairs)
        camera_pair_distances.append(distances)

    union_pairs = set().union(*camera_pair_sets) if camera_pair_sets else set()
    pair_hit_count = {pair: int(sum(pair in pairs for pairs in camera_pair_sets)) for pair in union_pairs}
    candidate_pairs = {pair for pair, hits in pair_hit_count.items() if hits >= min_camera_hits}

    edges: list[tuple[int, int, float, float]] = []
    blocked_by_gripper = 0
    blocked_by_adaptive_length = 0
    adaptive_lengths = []
    camera_hit_counts = []
    camera_distance_values: list[list[float]] = [[] for _ in range(camera_count)]
    camera_scale_values: list[list[float]] = [[] for _ in range(camera_count)]

    for i, j in sorted(candidate_pairs):
        if bool(gripper_closed[i]) != bool(gripper_closed[j]):
            blocked_by_gripper += 1
            continue

        distances = np.asarray(
            [
                float(np.linalg.norm(camera_features[camera_idx][i] - camera_features[camera_idx][j]))
                for camera_idx in range(camera_count)
            ],
            dtype=np.float32,
        )
        pair_scales = adaptive_base_scale + 0.5 * (temporal_scales[i] + temporal_scales[j])
        per_view_lengths = distances / np.maximum(pair_scales, 1e-8)
        adaptive_length = float(np.max(per_view_lengths))
        camera_hit_counts.append(pair_hit_count[(i, j)])
        for camera_idx, value in enumerate(distances):
            camera_distance_values[camera_idx].append(float(value))
            camera_scale_values[camera_idx].append(float(pair_scales[camera_idx]))
        if adaptive_length > adaptive_alpha:
            blocked_by_adaptive_length += 1
            continue

        dynamic_cost = float(max(float(stride) * float(cost_scale) * adaptive_length, 1e-6))
        edges.append((int(i), int(j), dynamic_cost, adaptive_length))
        adaptive_lengths.append(adaptive_length)

    total_blocked = blocked_by_gripper + blocked_by_adaptive_length
    stats = {
        "retrieval_embedding": retrieval_label,
        "fact_check": "gripper_hard_filter_and_multi_view_adaptive_length",
        "motion_check_enabled": False,
        "mutual_k": int(mutual_k),
        "history": int(history),
        "history_embedding": f"per_camera_concat_current_plus_previous_{int(history)}_nodes",
        "camera_count": int(camera_count),
        "camera_candidate_pairs": [int(len(pairs)) for pairs in camera_pair_sets],
        "candidate_mknn_pairs": int(len(candidate_pairs)),
        "candidate_min_camera_hits": int(min_camera_hits),
        "candidate_union_pairs": int(len(candidate_pairs)),
        "candidate_full_union_pairs": int(len(union_pairs)),
        "candidate_full_intersection_pairs": int(
            sum(1 for hits in pair_hit_count.values() if hits >= camera_count)
        ),
        "candidate_pair_camera_hit_mean": float(np.mean(camera_hit_counts)) if camera_hit_counts else 0.0,
        "candidate_pair_camera_hit_p50": float(np.quantile(camera_hit_counts, 0.50))
        if camera_hit_counts
        else 0.0,
        "candidate_pair_camera_hit_p95": float(np.quantile(camera_hit_counts, 0.95))
        if camera_hit_counts
        else 0.0,
        "blocked_by_gripper": int(blocked_by_gripper),
        "blocked_by_adaptive_length": int(blocked_by_adaptive_length),
        "blocked_by_motion": 0,
        "passed_unreliable_motion": 0,
        "similarity_edges": int(len(edges)),
        "avg_aug_degree": float(2 * len(edges) / max(n_nodes, 1)),
        "blocked_ratio": float(total_blocked / max(len(candidate_pairs), 1)),
        "gripper_blocked_ratio": float(blocked_by_gripper / max(len(candidate_pairs), 1)),
        "adaptive_length_blocked_ratio": float(
            blocked_by_adaptive_length / max(len(candidate_pairs) - blocked_by_gripper, 1)
        ),
        "motion_blocked_ratio": 0.0,
        "adaptive_alpha": float(adaptive_alpha),
        "adaptive_base_scale": float(adaptive_base_scale),
        "dynamic_cost_rule": "stride * sim_cost_scale * adaptive_length",
        "dynamic_cost_min": float(min((edge[2] for edge in edges), default=0.0)),
        "dynamic_cost_mean": float(np.mean([edge[2] for edge in edges])) if edges else 0.0,
        "dynamic_cost_p95": float(np.quantile([edge[2] for edge in edges], 0.95)) if edges else 0.0,
        "adaptive_length_mean": float(np.mean(adaptive_lengths)) if adaptive_lengths else 0.0,
        "adaptive_length_p95": float(np.quantile(adaptive_lengths, 0.95)) if adaptive_lengths else 0.0,
        "camera_candidate_distance_mean": [
            float(np.mean(values)) if values else 0.0 for values in camera_distance_values
        ],
        "camera_pair_scale_mean": [
            float(np.mean(values)) if values else 0.0 for values in camera_scale_values
        ],
        "eta_dir": None,
        "dir_eps": None,
        "cos_dir_mean": None,
        "cos_dir_p10": None,
        "cos_dir_p50": None,
    }
    return edges, stats


def build_temporal_edges(nodes: pd.DataFrame) -> list[tuple[int, int, float]]:
    edges = []
    for _episode_index, ep_nodes in nodes.groupby("episode_index", sort=True):
        ep_nodes = ep_nodes.sort_values("frame_index")
        ids = ep_nodes["node_id"].to_numpy(dtype=int)
        frames = ep_nodes["frame_index"].to_numpy(dtype=int)
        for src, dst, f0, f1 in zip(ids[:-1], ids[1:], frames[:-1], frames[1:], strict=True):
            edges.append((int(src), int(dst), float(max(1, int(f1) - int(f0)))))
    return edges


def reverse_dijkstra(
    n_nodes: int,
    temporal_edges: list[tuple[int, int, float]],
    similarity_edges: list[tuple[int, int, float, float]],
    success_nodes: set[int],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    sink = n_nodes
    reverse_adj: list[list[tuple[int, float, str]]] = [[] for _ in range(n_nodes + 1)]
    for src, dst, cost in temporal_edges:
        reverse_adj[dst].append((src, float(cost), "temporal"))
    for src, dst, cost, _distance in similarity_edges:
        reverse_adj[src].append((dst, float(cost), "similarity"))
        reverse_adj[dst].append((src, float(cost), "similarity"))
    for node_id in success_nodes:
        reverse_adj[sink].append((int(node_id), 0.0, "success_sink"))

    dist = np.full(n_nodes + 1, np.inf, dtype=np.float64)
    next_node = np.full(n_nodes, -1, dtype=np.int64)
    next_kind = [""] * n_nodes
    dist[sink] = 0.0
    heap: list[tuple[float, int]] = [(0.0, sink)]
    while heap:
        cost_to_sink, node = heapq.heappop(heap)
        if cost_to_sink > dist[node] + 1e-9:
            continue
        for prev, edge_cost, kind in reverse_adj[node]:
            new_cost = cost_to_sink + edge_cost
            if new_cost + 1e-9 < dist[prev]:
                dist[prev] = new_cost
                if prev < n_nodes:
                    next_node[prev] = node
                    next_kind[prev] = kind
                heapq.heappush(heap, (new_cost, prev))
    return dist[:n_nodes], next_node, next_kind
