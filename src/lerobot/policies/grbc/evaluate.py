"""Evaluate graph diagnostics or compare fixed episodes between existing runs."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .visualize import episode_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--compare-run", type=Path)
    parser.add_argument("--episodes", nargs="+", type=int)
    parser.add_argument(
        "--annotations", type=Path, help="Optional parquet: episode_index, frame_index, reference_progress"
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    table = pd.read_parquet(args.run / "transition_weights.parquet")
    summary = json.loads((args.run / "summary.json").read_text())
    metrics = episode_metrics(table)
    result = {
        "interpretation": "In-dataset graph diagnostics, not closed-loop policy success or held-out accuracy",
        "nodes": summary["nodes"],
        "reachable_ratio": summary["reachable_ratio"],
        "zero_weight_fraction": summary["zero_weight_fraction"],
        "episodes": json.loads(metrics.to_json(orient="records")),
    }
    if args.annotations:
        from scipy.stats import spearmanr

        reference = pd.read_parquet(args.annotations)
        required = {"episode_index", "frame_index", "reference_progress"}
        if required - set(reference) or reference.duplicated(["episode_index", "frame_index"]).any():
            raise ValueError("Annotations require unique episode/frame and reference_progress")
        joined = table.merge(
            reference[list(required)], on=["episode_index", "frame_index"], validate="one_to_one"
        )
        if len(joined) < 3 or not np.isfinite(joined.reference_progress).all():
            raise ValueError("Need at least three matched, finite reference progress annotations")
        correlation = spearmanr(joined.progress, joined.reference_progress).statistic
        result["annotation_evaluation"] = {
            "matched_frames": len(joined),
            "spearman": float(correlation) if np.isfinite(correlation) else None,
            "mae": float(np.abs(joined.progress - joined.reference_progress).mean()),
        }
    if args.compare_run:
        other = pd.read_parquet(args.compare_run / "transition_weights.parquet")
        joined = table.merge(
            other, on=["index", "episode_index", "frame_index"], suffixes=("_a", "_b"), validate="one_to_one"
        )
        if len(joined) != len(table) or len(joined) != len(other):
            raise ValueError("Comparison requires identical frame IDs; do not compare unmatched datasets")
        result["comparison"] = {
            "progress_mae": float(abs(joined.progress_a - joined.progress_b).mean()),
            "weight_mae": float(
                abs(joined.graph_transition_weight_a - joined.graph_transition_weight_b).mean()
            ),
            "zero_decision_disagreement": float(
                ((joined.graph_transition_weight_a == 0) != (joined.graph_transition_weight_b == 0)).mean()
            ),
        }
        import matplotlib.pyplot as plt

        selected = (
            args.episodes
            or metrics.sort_values("regression_sum", ascending=False).head(6).episode_index.tolist()
        )
        if set(selected) - set(metrics.episode_index):
            raise ValueError("Unknown comparison episode IDs")
        fig, axes = plt.subplots(len(selected), 1, figsize=(13, len(selected) * 2.4), squeeze=False)
        for ax, episode in zip(axes[:, 0], selected, strict=True):
            rows = joined[joined.episode_index == episode].sort_values("frame_index")
            ax.plot(rows.frame_index, rows.progress_a, label=args.run.name)
            ax.plot(rows.frame_index, rows.progress_b, label=args.compare_run.name, linestyle="--")
            ax.set(title=f"Episode {episode}", ylim=(-0.03, 1.03), ylabel="Progress")
            ax.grid(alpha=0.2)
            ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(args.run / "figures/comparison_fixed_episodes.png", dpi=160)
        plt.close(fig)
    output = args.output or args.run / "evaluation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {"evaluation": str(output), **{k: v for k, v in result.items() if k != "episodes"}}, indent=2
        )
    )


if __name__ == "__main__":
    main()
