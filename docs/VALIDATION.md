# Validation

## Automated Checks

The focused CPU suite covers graph directions/costs, camera intersection,
gripper filtering, history padding, exact-feature neighbor ties, interpolation,
weighting, tail windows, packed-video offsets, strict training label lookup,
all-zero update suppression, and launcher parsing for the three backbones.
A two-process CPU DDP check covers both reductions and unequal local batch sizes.

On 2026-10-16, all **49 tests passed** in a fresh Conda environment with
Python 3.11.17, PyTorch 2.7.1+cu128, Transformers 4.57.6, and LeRobot installed
from a clean clone. `pip check` also passed.

```bash
python -m unittest discover -s tests/grbc -v
OMP_NUM_THREADS=2 python -m torch.distributed.run \
  --standalone --nproc_per_node=2 tests/grbc/ddp_smoke.py
bash -n scripts/train_common.sh scripts/train_smolvla.sh scripts/train_xvla.sh scripts/train_pi05.sh
python -m pip wheel --no-deps . --wheel-dir /tmp/grbc_wheels
```

The unit suite does not download model checkpoints. Optional upstream robot,
policy, and simulation modules are not exhaustively tested.

## Data Checks

Frozen MCR inference, all-shard dataset reading, current-state and history-aware
graphs, dense progress, parquet export, camera montages, and shortest-path
visualization have been exercised on local LeRobot v3.0 data with 20 episodes,
7,364 frames, and 518 sampled nodes. Dataset videos and checkpoints are not
included in the repository.

Both graph modes produced per-frame progress/weight plots, graph overviews,
camera montages, and a fixed-episode comparison. A native LeRobot loader
validated all 7,364 exported labels against the dataset indices.

## SmolVLA Smoke Test

A two-step CPU run loaded the official pretrained SmolVLA checkpoint, applied
the exported graph weights, completed real forward/backward updates, and saved
a reloadable checkpoint. All 23,040 entries of the checked action-output weight
matrix changed and remained finite. This is an installation/integration check,
not a policy-quality experiment.

A separate **five-step RTX 5080 run** with batch size 1 and bf16 mixed precision
also completed. It used a fresh GPU MCR feature extraction, applied nonuniform
graph weights, and saved the final model, optimizer, and processors. The saved
checkpoint was reloaded without a manual camera-map override and evaluated over
all 281 frames of one episode. Losses were finite. That episode was part of the
same dataset, so this is not a held-out generalization result.

CPU and GPU feature extraction can produce small numerical differences near
neighbor/edge thresholds. In this fixture, the mean absolute progress difference
was 0.00071 and 3 of 7,364 zero-weight decisions differed. Do not assume bitwise
graph equivalence across devices; record environment and precision settings.

The first-run checks found and fixed missing default camera mapping, stale
state-dimension metadata, and missing checkpoint camera-map restoration in
offline validation. The launcher also avoids redundant VLM-weight downloads
and exposes warmup length for short smoke tests. CUDA installation instructions
now specify a build that supports RTX 50-series hardware.

An isolated-query smoke check used separate base and query episodes. This checks
the evaluation protocol, not held-out progress accuracy or policy success rates.

## Scope

Graph diagnostics and short training checks do not reproduce the paper's full
training budget or real-robot results. X-VLA and pi0.5 require separate model and
hardware validation before claiming end-to-end reproduction. Pi0.5 uses a
separate custom-Transformers environment.
