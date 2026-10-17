#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
: "${DATASET_ROOT:?Set DATASET_ROOT to a LeRobot v3.0 dataset}"
: "${OUTPUT_DIR:?Set a fresh OUTPUT_DIR}"
PYTHON_BIN="${PYTHON_BIN:-python}"
USE_GRBC="${USE_GRBC:-true}"
if [[ "$USE_GRBC" != true && "$USE_GRBC" != false ]]; then
  printf 'USE_GRBC must be true or false\n' >&2; exit 2
fi
if [[ "$USE_GRBC" == true ]]; then
  : "${WEIGHTS_PATH:?Set WEIGHTS_PATH to transition_weights.parquet}"
fi
case "${GRBC_BACKBONE:?Use a backbone-specific training script}" in
  smolvla) MODEL="${PRETRAINED_PATH:-lerobot/smolvla_base}"; CHUNK="${CHUNK_SIZE:-50}"; DEFAULT_STEPS=20000; LR=0.0001; WD=1e-10 ;;
  xvla) MODEL="${PRETRAINED_PATH:-lerobot/xvla-base}"; CHUNK="${CHUNK_SIZE:-30}"; DEFAULT_STEPS=20000; LR=0.0001; WD=0.0001 ;;
  pi05) MODEL="${PRETRAINED_PATH:-lerobot/pi05_base}"; CHUNK="${CHUNK_SIZE:-30}"; DEFAULT_STEPS=40000; LR=0.000025; WD=0.01 ;;
  *) printf 'Unknown backbone\n' >&2; exit 2 ;;
esac
args=(--policy.path="$MODEL" --policy.push_to_hub=false
  --policy.chunk_size="$CHUNK" --policy.n_action_steps="${N_ACTION_STEPS:-30}"
  --policy.optimizer_lr="${LEARNING_RATE:-$LR}" --policy.scheduler_warmup_steps="${WARMUP_STEPS:-1000}"
  --policy.optimizer_betas='[0.9,0.95]' --policy.optimizer_weight_decay="${WEIGHT_DECAY:-$WD}"
  --policy.scheduler_decay_steps="${STEPS:-$DEFAULT_STEPS}"
  --dataset.repo_id="${DATASET_REPO_ID:-local/task}" --dataset.root="$DATASET_ROOT"
  --dataset.video_backend=pyav --output_dir="$OUTPUT_DIR"
  --batch_size="${BATCH_SIZE:-8}" --steps="${STEPS:-$DEFAULT_STEPS}"
  --num_workers="${NUM_WORKERS:-4}" --save_freq="${SAVE_FREQ:-5000}"
  --eval_freq=0 --wandb.enable="${WANDB_ENABLE:-false}" --seed="${SEED:-42}"
  --use_grbc="$USE_GRBC")
if [[ "$USE_GRBC" == true ]]; then
  args+=(--grbc_weights_path="$WEIGHTS_PATH" --grbc_weight_column=graph_transition_weight
         --grbc_reduction="${GRBC_REDUCTION:-mean}")
fi
if [[ "$GRBC_BACKBONE" == xvla ]]; then
  args+=(--policy.dtype=bfloat16 --policy.action_mode="${ACTION_MODE:-auto}"
    --policy.max_action_dim="${MAX_ACTION_DIM:-20}" --policy.num_image_views=3
    --policy.freeze_vision_encoder=false --policy.freeze_language_encoder=false
    --policy.train_policy_transformer=true --policy.train_soft_prompts=true)
fi
if [[ "$GRBC_BACKBONE" == smolvla ]]; then
  # A full policy checkpoint already contains the VLM parameters.
  args+=(--policy.load_vlm_weights=false)
fi
if [[ "$GRBC_BACKBONE" == pi05 ]]; then
  args+=(--policy.dtype=bfloat16 --policy.gradient_checkpointing=true --policy.train_expert_only=true)
fi
if [[ -n "${RENAME_MAP:-}" ]]; then
  args+=(--rename_map="$RENAME_MAP")
elif [[ "$GRBC_BACKBONE" == xvla ]]; then
  args+=(--rename_map='{"observation.images.cam_0":"observation.images.image","observation.images.cam_1":"observation.images.image2","observation.images.cam_2":"observation.images.image3"}')
elif [[ "$GRBC_BACKBONE" == smolvla ]]; then
  args+=(--rename_map='{"observation.images.cam_0":"observation.images.camera1","observation.images.cam_1":"observation.images.camera2","observation.images.cam_2":"observation.images.camera3"}')
fi
if [[ "${1:-}" == --dry-run ]]; then
  shift
  printf '%q ' "$PYTHON_BIN" -m accelerate.commands.launch --num_processes="${NUM_PROCESSES:-1}" --mixed_precision="${MIXED_PRECISION:-bf16}" -m lerobot.scripts.lerobot_train "${args[@]}" "$@"
  printf '\n'
  exit 0
fi
[[ -f "$DATASET_ROOT/meta/info.json" ]] || { printf 'Missing dataset meta/info.json\n' >&2; exit 2; }
if [[ "$USE_GRBC" == true ]]; then
  [[ -f "$WEIGHTS_PATH" ]] || { printf 'Missing weights parquet\n' >&2; exit 2; }
fi
exec "$PYTHON_BIN" -m accelerate.commands.launch --num_processes="${NUM_PROCESSES:-1}" \
  --mixed_precision="${MIXED_PRECISION:-bf16}" -m lerobot.scripts.lerobot_train "${args[@]}" "$@"
