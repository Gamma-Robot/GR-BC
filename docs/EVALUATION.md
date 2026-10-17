# Visualization and Evaluation

There are three distinct levels. Keep them separate in reports: graph diagnostics, offline checkpoint validation, and closed-loop policy evaluation.

## Graph Diagnostics

```bash
grbc-evaluate --run outputs/task_history
grbc-visualize --run outputs/task_history --episodes 0 1 6
grbc-visualize --run outputs/task_history --episodes 6 \
  --dataset /absolute/path/to/task_v3 --camera observation.images.cam_2 \
  --start 720 --end 840 --every 5
```

`evaluation.json` records reachability, rejected-frame fractions, per-episode total regression, endpoint progress, and largest one-frame progress drop. Curves are **dense per-frame values**, including interpolation; plotted episode IDs are explicit or deterministically selected by regression sum, not random. Camera montages decode actual recorded frames and include episode/frame IDs; choose a short window (at most 120 images) for inspection.

For parameter comparisons use the **same episode IDs**:

```bash
grbc-evaluate --run outputs/task_paper --compare-run outputs/task_history --episodes 0 1 6
```

This requires identical global and episode/frame IDs. It reports progress MAE, weight MAE, zero-weight decision disagreement, and a fixed-episode overlay. A different sampling stride or interpolation mode is supported; a different underlying dataset is not a valid matched comparison.

Optional human progress annotations can be supplied as parquet with `episode_index`, `frame_index`, `reference_progress`:

```bash
grbc-evaluate --run outputs/task_paper --annotations /absolute/path/to/annotations.parquet
```

This reports matched-frame Spearman correlation and MAE. Manually annotate difficult regressions, plateau intervals, start/end visual aliases, and physically implausible cross-trajectory jumps. High graph connectivity or smooth increasing curves alone do not establish correct progress. A PCA graph overview uses current features solely for display, even when graph retrieval uses history.

Inspect an actual shortest path and its cross-trajectory jumps:

```bash
python -m lerobot.policies.grbc.visualize_path --run outputs/task_history \
  --dataset /absolute/path/to/task_v3 --episode 6 --frame 150 \
  --camera observation.images.cam_1 --max-nodes 32
```

The starting frame must be a sampled graph node. Every displayed image is a consecutive graph hop, not an evenly subsampled montage that could hide intervening edges. Full node IDs are saved alongside the figure; a long path is visibly marked as truncated rather than falsely displayed as one jump.

**Leakage warning:** the regular exporter builds a graph on all provided episodes, using their success annotations. Its diagnostics are transductive/in-dataset. Do not call them a zero-shot or held-out success-recognition benchmark.

### Isolated Held-Out Trajectory Evaluation

The held-out evaluator inserts query trajectories into a graph built from successful reference episodes. First encode the dataset with the exporter, then specify disjoint reference and query splits:

```bash
python -m lerobot.policies.grbc.evaluate_holdout --run outputs/task_history \
  --base-episodes 0 1 2 3 4 --query-episodes 6 7 8 \
  --output-dir outputs/heldout_check
```

Only frozen raw MCR features and the chosen graph configuration are reused from the original run; its already-connected graph and its progress values are **not** reused. The tool rebuilds a reference graph on successful base episodes, fixes its edges and Rmax, and temporarily inserts one query episode at a time. Query positive-reward edges are never added; reward-only forced query samples are removed. Query episodes cannot help one another. Actual query outcomes are used only to label evaluation records/figures. Intervention-boundary sampling is currently rejected in this mode rather than given an ambiguous query sampling rule.

Reports include held-out per-node progress, terminal/mean progress, reachable fraction, and regression. This is **offline full-trajectory insertion**: future query images and query temporal edges are available, and attached query edges can participate in graph routes. It is not a causal, single-frame online progress predictor, and is not identical to a frozen one-way nearest-neighbor projection. Fix splits, thresholds, and hyperparameters before assessing held-out outcomes; do not choose a configuration based on the test episodes. Node-level held-out metrics should not be mixed with dense interpolated frame metrics without noting the difference.

## Offline Checkpoint Validation

Hold out full episodes from policy training, then evaluate a local checkpoint:

```bash
python -m lerobot.policies.grbc.evaluate_checkpoint \
  --policy.path=/absolute/path/to/checkpoint/pretrained_model \
  --dataset.repo_id=local/task --dataset.root=/absolute/path/to/task_v3 \
  --dataset.episodes='[2,7,11]' --dataset.video_backend=pyav \
  --output_dir=outputs/validation_seed42 --batch_size=4 --num_workers=2 \
  --policy.push_to_hub=false --seed=42
```

Add `--use_grbc=true --grbc_weights_path=...` to also report weighted mean and self-normalized losses. Always retain the raw unweighted validation loss: weighting can hide errors by ignoring hard samples. Use the same validation episodes, processor settings, and random seeds for all checkpoints. The evaluator restores a local checkpoint's saved camera rename map unless `--rename_map` supplies an override. For a different dataset schema, update camera rename/normalization processors explicitly rather than applying incompatible statistics.

Flow-matching/diffusion losses are stochastic. Repeat across seeds, report variation, and never interpret them as rollout success. This utility performs one pass over the selected validation episodes; it does not choose or verify the training/validation split for you. It also does not provide an unbiased progress benchmark if those episodes were included in the graph.

## Closed-Loop Evaluation

For a supported simulation task, the bundled upstream `lerobot-eval` evaluates actual rollouts and saves rendered videos:

```bash
lerobot-eval --policy.path=/absolute/path/to/checkpoint/pretrained_model \
  --env.type=pusht --eval.n_episodes=50 --eval.batch_size=10 \
  --output_dir=outputs/simulation_eval
```

This is an upstream interface example, **not a claim that a real-robot insertion policy can solve PushT**. Install the matching environment extra, use a checkpoint trained for that environment's observation/action schema, and inspect `lerobot-eval --help` for task-specific options. Simulation extras are optional and not installed by the GRBC extra.

For the paper's real-world tasks, report repeated physical trials under fixed initial conditions, raw success/failure counts, seeds/checkpoints, intervention rules, and uncertainty. Hardware deployment and dataset collection are not shipped here, so the real-robot success-rate experiments cannot be reproduced solely with this repository.
