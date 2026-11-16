#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# Dual-Level DiT Distributed Launch Script (torchrun)
#
# Usage:
#   Single-node (auto-detects GPU count, defaults to 1):
#     bash run.sh
#
#   Specify GPU count / GPUs / custom YAML config / CLI overrides:
#     CUDA_VISIBLE_DEVICES=0,1 NPROC_PER_NODE=2 bash run.sh
#     CONFIG=configs/train_dog.yaml bash run.sh --total_steps 3000
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

CONFIG="${CONFIG:-configs/train_dog.yaml}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"

if [[ -z "${NPROC_PER_NODE:-}" ]]; then
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    IFS=',' read -ra _VISIBLE_GPUS <<< "${CUDA_VISIBLE_DEVICES}"
    NPROC_PER_NODE="${#_VISIBLE_GPUS[@]}"
  elif command -v nvidia-smi >/dev/null 2>&1; then
    GPU_COUNT="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
    NPROC_PER_NODE="${GPU_COUNT:-1}"
  else
    NPROC_PER_NODE=1
  fi
  if [[ "${NPROC_PER_NODE}" -lt 1 ]]; then
    NPROC_PER_NODE=1
  fi
fi

echo "============================================================"
echo " Launching Dual-Level DiT Training via torchrun"
echo "   Config         : ${CONFIG}"
echo "   Nodes          : ${NNODES} (rank ${NODE_RANK})"
echo "   Proc per node  : ${NPROC_PER_NODE}"
echo "   Master         : ${MASTER_ADDR}:${MASTER_PORT}"
echo "============================================================"

torchrun \
  --nnodes="${NNODES}" \
  --node_rank="${NODE_RANK}" \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  train.py \
  --config "${CONFIG}" \
  "$@"
