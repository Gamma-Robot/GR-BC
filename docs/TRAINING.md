# VLA Training

GR-BC adds four options to `TrainPipelineConfig`: `use_grbc`, `grbc_weights_path`, `grbc_weight_column`, and `grbc_reduction`. It reads precomputed graph weights and does not require a reward model or SARM network. Pure SFT with `use_grbc=false` remains available.

## Launchers

Activate the matching environment, set `DATASET_ROOT`, `OUTPUT_DIR`, and, for weighted SFT, `WEIGHTS_PATH`. Launch from any directory:

```bash
export DATASET_ROOT=/absolute/path/to/task_v3
export WEIGHTS_PATH=/absolute/path/to/run/transition_weights.parquet
export OUTPUT_DIR=/absolute/path/to/new_training_run
export PRETRAINED_PATH=/absolute/path/to/compatible_backbone_checkpoint
bash scripts/train_smolvla.sh --dry-run
# Or train_xvla.sh / train_pi05.sh; omit --dry-run to execute.
```

Additional CLI arguments are forwarded after the built-in arguments. Use environment variables for repeated settings. No fixed GPU device IDs are embedded; `CUDA_VISIBLE_DEVICES` and Accelerate handle device selection.

| Variable | Default / role |
|---|---|
| `PYTHON_BIN` | `python` in the activated environment |
| `NUM_PROCESSES` | 1; one process per GPU |
| `BATCH_SIZE` | 8 **per process**, a portable starting point, not paper compute |
| `MIXED_PRECISION` | bf16; change to `no` if unsupported |
| `N_ACTION_STEPS` | 30; must match exported scoring metadata |
| `CHUNK_SIZE` | SmolVLA 50, X-VLA/pi0.5 30 |
| `STEPS` | SmolVLA/X-VLA 20000, pi0.5 40000 |
| `WARMUP_STEPS` | 1000; set a smaller value for smoke tests |
| `GRBC_REDUCTION` | mean; optional self_normalized reduction |
| `USE_GRBC` | true; set false for matched uniform SFT |
| `RENAME_MAP` | Override JSON camera mapping when needed |
| `ACTION_MODE` | X-VLA `auto`; explicit robot/action modes are dataset-dependent |

SmolVLA's scoring horizon remains 30 even though it predicts 50 actions, matching Appendix C.2's interface. This does **not** truncate its supervised target to 30.

SmolVLA defaults map `cam_0/1/2` to `camera1/2/3`; X-VLA maps them to `image/image2/image3`. Adapt `RENAME_MAP` for other dataset names and custom checkpoints. Graph camera names and policy camera names need not be identical. X-VLA's `max_action_dim=20` and automatic action adapter are not a universal calibration. Input checkpoints must be compatible with the bundled policy/config classes and task action representation.

## Paper Table 10 Settings

The launchers explicitly set Adam betas (0.9, 0.95), learning rates 1e-4 / 1e-4 / 2.5e-5, and weight decay 1e-10 / 1e-4 / 1e-2 for SmolVLA / X-VLA / pi0.5.

To request the paper's effective batch sizes on suitable hardware:

```bash
# Four GPUs: 32 per GPU for SmolVLA/X-VLA, 16 for pi0.5.
NUM_PROCESSES=4 BATCH_SIZE=32 bash scripts/train_xvla.sh
# In a separate pi environment:
NUM_PROCESSES=4 BATCH_SIZE=16 bash scripts/train_pi05.sh
```

The paper used four H100 GPUs. These launchers do not reproduce that budget on one smaller GPU or guarantee the reported wall time/success rate. Smaller batches and different loss normalization are meaningful changes; record them. Pi0.5 uses expert-only training and gradient checkpointing. Match checkpoint, trainable modules, input preprocessing, data split, and rounds before asserting full policy reproduction.

## Environments

SmolVLA / X-VLA: `python -m pip install -e '.[grbc,smolvla,xvla]'`.

Pi0.5: create a separate Python environment and install `python -m pip install -e '.[grbc,pi]'`. This uses the upstream custom Transformers branch. Do not combine it with the normal Transformers extras. If loading an existing checkpoint's processors, retain the tokenizer and processor files with the checkpoint directory; model weights alone are insufficient.

Training labels are read once and looked up by global index. All three bundled model forwards support `reduction="none"`; GRBC expects one loss per sample. This release does not alter VLA architectures or substitute a reward loss. Weight lookup failures are errors, never silently uniform weights. The trainer rejects streaming datasets, for which this fixed indexed export contract is not implemented.

At each logging interval, the trainer prints the main process's local mean
weight, zero-weight fraction, and all-zero skip flag. These help verify that
labels are actually being applied without requiring a W&B account.

## Baselines

For baseline comparisons use the same seed, pretrained initialization, training episodes, batch size, optimizer, and training budget; change only `USE_GRBC=false`. Comparing a weighted training objective alone does not measure policy quality.
