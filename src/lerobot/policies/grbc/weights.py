"""Residual weights with minimum-progress and regression checks."""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_WEIGHT_COLUMN = "graph_transition_weight"


def compute_graph_transition_weights(
    frame_values: pd.DataFrame,
    chunk: int,
    target_delta_progress: float | None = None,
    progress_dip_tolerance: float | None = None,
    target_progress_gradient: float = 0.005,
    weight_beta: float = 5.0,
    progress_column: str = "progress",
    weight_column: str = DEFAULT_WEIGHT_COLUMN,
    weight_alpha: float = 0.5,
    weight_gamma: float = 20.0,
    weight_min: float = 0.5,
    weight_max: float = 1.5,
    n_action_step: int = 30,
    endpoint_regression_tolerance: float = 0.05,
    min_delta_progress: float = 0.005,
    flat_weight: float = 0.0,
    eps: float = 1e-9,
) -> pd.DataFrame:
    """Assign tanh-residual transition weights to every frame.

    The policy trains on an action chunk of length ``chunk``.  The progress
    judgement uses the state reached after ``n_action_step`` executed actions:

    ``delta = P(s_{t+n_action_step}) - P(s_t)``.

    A transition receives zero weight if the minimum progress inside the judged
    window dips below the starting progress by more than
    ``progress_dip_tolerance``.  If the endpoint improvement does not exceed
    ``min_delta_progress``, it receives ``flat_weight``.  Otherwise it receives:

    ``clip(1 + alpha * tanh(gamma * delta), weight_min, weight_max)``.

    ``target_delta_progress``, ``target_progress_gradient``, ``weight_beta``,
    and ``endpoint_regression_tolerance`` are accepted for compatibility with
    older callers but are intentionally unused by this rule.
    """
    del target_delta_progress, target_progress_gradient, weight_beta, endpoint_regression_tolerance
    dip_tolerance = 0.0 if progress_dip_tolerance is None else float(progress_dip_tolerance)
    if chunk < 1:
        raise ValueError(f"chunk must be >= 1, got {chunk}")
    if n_action_step < 1:
        raise ValueError(f"n_action_step must be >= 1, got {n_action_step}")
    if n_action_step > chunk:
        raise ValueError(f"n_action_step ({n_action_step}) must be <= chunk ({chunk})")
    if dip_tolerance < 0:
        raise ValueError(f"progress_dip_tolerance must be >= 0, got {dip_tolerance}")
    if min_delta_progress < 0:
        raise ValueError(f"min_delta_progress must be >= 0, got {min_delta_progress}")
    if flat_weight < 0:
        raise ValueError(f"flat_weight must be >= 0, got {flat_weight}")
    if weight_min > weight_max:
        raise ValueError(f"weight_min ({weight_min}) must be <= weight_max ({weight_max})")

    out_parts = []
    for _episode_index, ep in frame_values.groupby("episode_index", sort=True):
        ep = ep.sort_values("frame_index").reset_index(drop=True).copy()
        progress = ep[progress_column].to_numpy(dtype=float)
        n = len(ep)
        weights = np.zeros(n, dtype=np.float32)
        delta_progress = np.zeros(n, dtype=np.float32)
        window_min = np.zeros(n, dtype=np.float32)
        progress_dip = np.zeros(n, dtype=np.float32)
        judged_endpoint_offset = np.zeros(n, dtype=np.int32)
        action_chunk_endpoint_offset = np.zeros(n, dtype=np.int32)
        reject_reason = np.full(n, "accepted", dtype=object)

        for start in range(n):
            judged_end = min(start + n_action_step, n - 1)
            action_end = min(start + chunk, n - 1)
            judged_endpoint_offset[start] = judged_end - start
            action_chunk_endpoint_offset[start] = action_end - start

            p_start = float(progress[start])
            p_end = float(progress[judged_end])
            delta = p_end - p_start
            min_p = float(np.min(progress[start : judged_end + 1]))
            dip = max(0.0, p_start - min_p)
            delta_progress[start] = delta
            window_min[start] = min_p
            progress_dip[start] = dip

            if dip > dip_tolerance + eps:
                weights[start] = 0.0
                reject_reason[start] = "window_dip_below_start"
            elif delta <= min_delta_progress + eps:
                weights[start] = float(flat_weight)
                reject_reason[start] = "delta_not_above_min_delta_progress"
            else:
                residual = 1.0 + weight_alpha * np.tanh(weight_gamma * delta)
                weights[start] = float(np.clip(residual, weight_min, weight_max))
                reject_reason[start] = "accepted_residual_gain"

        ep[weight_column] = weights
        ep["chunk_delta_progress"] = delta_progress
        ep["chunk_window_min_progress"] = window_min
        ep["chunk_progress_dip"] = progress_dip
        ep["chunk_reject_reason"] = reject_reason
        ep["judged_endpoint_offset"] = judged_endpoint_offset
        ep["action_chunk_endpoint_offset"] = action_chunk_endpoint_offset
        ep["progress_dip_tolerance"] = float(dip_tolerance)
        ep["min_delta_progress"] = float(min_delta_progress)
        ep["flat_weight"] = float(flat_weight)
        out_parts.append(ep)

    return pd.concat(out_parts, ignore_index=True)
