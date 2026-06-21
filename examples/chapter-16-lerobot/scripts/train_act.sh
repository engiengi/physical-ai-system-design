#!/usr/bin/env bash
set -euo pipefail

DATASET_REPO_ID="${DATASET_REPO_ID:-username/sim_pick_cube}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/train/act_sim_pick_cube}"
JOB_NAME="${JOB_NAME:-act_sim_pick_cube}"
DEVICE="${DEVICE:-cuda}"

if [[ "${DATASET_REPO_ID}" == username/* ]]; then
  echo "주의: DATASET_REPO_ID가 기본값입니다."
  echo "예: DATASET_REPO_ID=myname/sim_pick_cube bash scripts/train_act.sh"
fi

lerobot-train \
  --dataset.repo_id="${DATASET_REPO_ID}" \
  --policy.type=act \
  --output_dir="${OUTPUT_DIR}" \
  --job_name="${JOB_NAME}" \
  --policy.device="${DEVICE}" \
  --policy.push_to_hub=false \
  --wandb.enable=false
