# First-Run Check

Use a fresh environment and install this repository as shown in the README.
Do not install a second LeRobot checkout into the same environment. Check that
`python -c 'import lerobot; print(lerobot.__file__)'` points to this checkout.

## Check CUDA

```bash
python -c 'import torch; print(torch.__version__); print((torch.ones(8, device="cuda") + 1).sum().item())'
python -m pip check
```

`torch.cuda.is_available()` alone does not prove that a wheel supports your GPU.
On RTX 50-series hardware, a CUDA 12.6 wheel can detect the GPU but fail with
`no kernel image is available`. Install the CUDA 12.8 build from the README.

## Build and Visualize

Set `DATASET_ROOT` to your LeRobot v3.0 dataset and `MCR_CKPT` to the official
MCR ResNet50 checkpoint. This does not train or fine-tune MCR.

```bash
grbc-export --dataset "$DATASET_ROOT" --mcr-ckpt "$MCR_CKPT" \
  --output-dir outputs/graph
grbc-visualize --run outputs/graph --episodes 0 1
grbc-evaluate --run outputs/graph
```

For history-aware retrieval, use `--history 3` in a fresh output directory.
Raw image features may be reused with `--cache-dir outputs/graph/cache
--reuse-cache`; the graph and labels are rebuilt. Omit cache reuse for fresh
feature extraction.

Inspect `figures/progress_weight_timelines.png`, `figures/graph_overview.png`,
and `summary.json`. Choose episode IDs that exist in your dataset. To inspect
actual recorded images, add `--dataset "$DATASET_ROOT" --camera
observation.images.cam_1 --start 0 --end 90 --every 5` to the visualization command.

## Short SmolVLA Training

```bash
export WEIGHTS_PATH="$PWD/outputs/graph/transition_weights.parquet"
export OUTPUT_DIR="$PWD/outputs/smolvla_smoke"
export PRETRAINED_PATH=lerobot/smolvla_base
BATCH_SIZE=1 STEPS=5 WARMUP_STEPS=1 NUM_WORKERS=0 \
  bash scripts/train_smolvla.sh --log_freq=1
```

This runs real pretrained forward/backward updates and saves the final checkpoint.
It is a wiring check, not a convergence or policy-quality experiment. Use a fresh
output directory for every run. Increase training budget and batch size only after
the smoke check succeeds.

The launcher maps the documented `cam_0/1/2` views to SmolVLA's `camera1/2/3`.
Set `RENAME_MAP` explicitly for different camera names or custom checkpoints.
`CHUNK_SIZE` controls the predicted action sequence; `N_ACTION_STEPS` must match
the weight scoring horizon. The checkpoint includes VLM weights, so the launcher
does not separately download VLM weights; its configuration and tokenizer are
still needed. For offline use, cache those assets first, or set
`--policy.vlm_model_name=/path/to/local/vlm_assets`.

## Tests

```bash
python -m unittest discover -s tests/grbc -v
OMP_NUM_THREADS=2 python -m torch.distributed.run \
  --standalone --nproc_per_node=2 tests/grbc/ddp_smoke.py
```
