#!/usr/bin/env bash
# Serve the Unsloth Qwen3.8-27B-NVFP4 workload for Max-Q vs WS SKU comparison.
# vLLM 0.27.1 / SM120. Required: --kv-cache-dtype bfloat16 (checkpoint FP8 KV + flash_attn fails on SM120).
# Thinking off. No --speculative-config (AR only; accept noise would hide SKU deltas).
set -euo pipefail

: "${MODEL:?set MODEL to the local Unsloth checkpoint directory}"
PORT="${PORT:-8000}"
SERVED_NAME="${SERVED_NAME:-qwen38-nvfp4}"
VLLM="${VLLM:-vllm}"

CHAT_TEMPLATE="${MODEL}/chat_template.jinja"
if [[ ! -f "${CHAT_TEMPLATE}" ]]; then
  echo "missing chat template: ${CHAT_TEMPLATE}" >&2
  exit 1
fi

exec "${VLLM}" serve "${MODEL}" \
  --host 127.0.0.1 \
  --port "${PORT}" \
  --dtype bfloat16 \
  --trust-remote-code \
  --language-model-only \
  --max-model-len 8192 \
  --max-num-seqs 8 \
  --gpu-memory-utilization 0.85 \
  --mamba-cache-dtype float32 \
  --mamba-ssm-cache-dtype float32 \
  --attention-backend flash_attn \
  --kv-cache-dtype bfloat16 \
  --served-model-name "${SERVED_NAME}" \
  --chat-template "${CHAT_TEMPLATE}" \
  --default-chat-template-kwargs '{"enable_thinking": false}'
