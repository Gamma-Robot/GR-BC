"""Graph weighting configurations for current-state and history-aware retrieval."""

import math
from dataclasses import asdict, dataclass


@dataclass
class GraphConfig:
    stride: int = 15
    history: int = 0
    mutual_k: int = 20
    adaptive_edge_alpha: float = 1.0
    adaptive_base_scale: float = 0.0001
    sim_cost_scale: float = 1.0
    min_camera_hits: int = 3
    chunk: int = 30
    n_action_step: int = 30
    progress_dip_tolerance: float = 0.03
    min_delta_progress: float = 0.002
    flat_weight: float = 0.0
    weight_alpha: float = 0.5
    weight_gamma: float = 20.0
    weight_min: float = 0.5
    weight_max: float = 1.5
    rmax_quantile: float = 0.95
    interpolation: str = "paper"
    gripper_state_index: int = -1
    gripper_threshold: float = 0.5
    include_intervention_boundaries: bool = False
    cross_trajectory_only: bool = False

    def validate(self):
        for name, value in asdict(self).items():
            if isinstance(value, (float, int)) and not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        for name in ("stride", "mutual_k", "chunk", "n_action_step"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(self.history, int) or self.history < 0:
            raise ValueError("history must be a nonnegative integer")
        if self.n_action_step > self.chunk:
            raise ValueError("n_action_step must not exceed chunk")
        if not 1 <= self.min_camera_hits <= 3 or not 0 < self.rmax_quantile <= 1:
            raise ValueError("Invalid camera hits or Rmax quantile")
        for name in ("adaptive_edge_alpha", "adaptive_base_scale", "sim_cost_scale", "weight_gamma"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in (
            "progress_dip_tolerance",
            "min_delta_progress",
            "flat_weight",
            "weight_alpha",
            "weight_min",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be nonnegative")
        if self.weight_max < self.weight_min or self.interpolation not in ("legacy", "paper"):
            raise ValueError("Invalid weight bounds or interpolation mode")


PRESETS = {
    "paper": {},
    "current": {
        "adaptive_base_scale": 0.05,
        "min_delta_progress": 0.005,
        "interpolation": "legacy",
    },
    "history": {
        "stride": 10,
        "history": 3,
        "adaptive_base_scale": 0.05,
        "progress_dip_tolerance": 0.06,
        "interpolation": "legacy",
    },
}
