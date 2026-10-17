# Method and Configuration

## Graph Construction

1. Sample every `stride` frames in each episode, always include its last frame and each positive reward frame. Optionally include intervention boundaries.
2. Independently encode each camera with frozen MCR ResNet50: resize shortest side to 256, center crop 224, RGB tensor in [0, 1], **no ImageNet mean/std normalization**. Per-view vectors are L2-normalized for graph distances.
3. Current-state retrieval uses current-node features only. History-aware retrieval concatenates current and the previous H **sampled graph nodes** separately per camera, then normalizes each concatenation. History is equal-weighted and padded with the episode's first node. It is not H consecutive raw frames or a motion-vector gate, and does not cross episode boundaries. Forced reward/boundary nodes make actual temporal offsets irregular.
4. Find mutual K nearest neighbors independently in all three view spaces. Default `min_camera_hits=3` means intersection of the three mutual-pair sets. `1` requests a union ablation. Self-neighbors are excluded by ID, including exact-feature ties.
5. Require matching binary gripper categories. For each view, sigma is the mean embedding distance to the immediate previous/next sampled temporal neighbors. With history enabled these are distances in the per-camera **history-concatenated** space.

```text
ell(i,j) = max_c distance(z_i,c, z_j,c) / [epsilon_c + (sigma_i,c + sigma_j,c)/2]
keep if ell <= alpha_edge
similarity_cost = max(stride * sim_cost_scale * ell, 1e-6)
```

Similarity edges are bidirectional. Temporal edges follow observed time only and cost the actual frame difference. Positive reward nodes connect to a zero-cost virtual success sink; reverse Dijkstra computes raw remaining distances. Success nodes can occupy several visual modes: they share the sink but are not collapsed into one visual point.

Within-episode similarity edges are allowed by default. Use `--cross-trajectory-only` to disallow them. Neither history nor the gripper filter proves physical reachability; visually aliased/palindromic tasks can still produce shortcuts.

## Remaining Steps and Interpolation

Rmax is the 95th percentile of **finite raw node distances**, with a tiny numerical floor for degenerate graphs. Unreachable node distance remains infinite in `graph.npz`.

- `paper`: interpolate raw remaining steps only between two finite nodes. Otherwise interval interiors have zero progress. An exact finite sampled node retains its own value. Apply progress clipping after interpolation.
- `legacy`: replace infinite distances with Rmax, cap finite node distances at Rmax, convert to float32, then interpolate. This is an alternative capped-distance boundary rule.

In either case, `P = clip(1 - Rhat/Rmax, 0, 1)`. Capping nodes before interpolation is not algebraically equivalent to clipping progress afterward. This can change the length of flat segments and which chunks pass the minimum-improvement test. Set the mode explicitly for comparisons. Exact sampled-node values and interval interiors are intentionally distinguished in the paper mode.

## Weighting

For starting frame t, judge through `end = min(t + n_action_step, last_frame)`:

```text
delta = P[end] - P[t]
dip = P[t] - min(P[t:end+1])
if dip > progress_dip_tolerance + 1e-9:
    weight = 0
elif delta <= min_delta_progress + 1e-9:
    weight = flat_weight                # default 0
else:
    weight = clip(1 + weight_alpha * tanh(weight_gamma * delta), weight_min, weight_max)
```

Tail windows shorten at the episode boundary, never continue into another episode, and are **not** horizon-rescaled. At the final frame delta=0, so its weight is zero with the default flat weight. A predicted chunk may be longer than the judged horizon (e.g. SmolVLA 50 versus 30); the weight multiplies the full existing per-sample supervised loss. Each backbone retains its action-padding/reduction behavior; tail masking is not identical across architectures.

For positive accepted delta and nonnegative alpha, the residual starts above 1. A lower bound below 1 therefore does not downweight slow accepted steps: rejection thresholds do the filtering. Changing `weight_min` alone will not make weights discriminate flat segments.

## Configurations

The default `paper` preset follows the paper settings. Use `--preset current` for current-state retrieval or `--preset history` for history-aware retrieval. Complete configurations are in `configs/grbc/`; these presets are not identical experiments. For a controlled comparison, change only the desired setting rather than switching several settings at once.

Use `--config path/to/config.json` for overrides, followed by explicit CLI options if needed. `--history H` selects the number of previous sampled nodes; zero disables history. The resolved configuration is saved in `summary.json`.

## Training Reduction

`grbc_reduction=mean` optimizes `sum(w_i * loss_i) / B`; rejected items still count in B, as in the paper's sample-weighted expectation. `self_normalized` divides by sum(w). This changes the batch-dependent gradient scale, especially when most items are rejected. Neither mode normalizes the stored weight labels. Distributed reductions use global denominators and account for DDP gradient averaging. A globally all-zero batch skips optimizer/scheduler updates to avoid learning through momentum or weight decay; it still consumes a loader iteration.
