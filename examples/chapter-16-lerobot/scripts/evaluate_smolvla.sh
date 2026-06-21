#!/usr/bin/env bash
set -euo pipefail

export MUJOCO_GL="${MUJOCO_GL:-egl}"

lerobot-eval \
  --policy.path=HuggingFaceVLA/smolvla_libero \
  --env.type=libero \
  --env.task=libero_object \
  --eval.batch_size=1 \
  --eval.n_episodes=3
