#!/usr/bin/env bash
# Official vllm bench serve matrix for the Max-Q vs WS SKU comparison in this repo.
# Warmup num_prompts=1 is discarded. Timed n=3, num_prompts=max(16, 4*C).
# --tokenizer $MODEL --trust-remote-code is required (served alias is not an HF id).
set -euo pipefail

: "${MODEL:?set MODEL to the local Unsloth checkpoint directory}"
: "${RESULT:?set RESULT to an output directory}"
BASE="${BASE:-http://127.0.0.1:8000}"
SERVED_NAME="${SERVED_NAME:-qwen38-nvfp4}"
VLLM="${VLLM:-vllm}"

mkdir -p "${RESULT}"

run_one() {
  local input_len="$1" output_len="$2" conc="$3" num_prompts="$4" filename="$5"
  "${VLLM}" bench serve \
    --backend openai \
    --base-url "${BASE}" \
    --model "${SERVED_NAME}" \
    --dataset-name random \
    --seed 0 \
    --temperature 0 \
    --request-rate inf \
    --ignore-eos \
    --input-len "${input_len}" \
    --output-len "${output_len}" \
    --max-concurrency "${conc}" \
    --num-prompts "${num_prompts}" \
    --tokenizer "${MODEL}" \
    --trust-remote-code \
    ${filename:+--save-result --result-dir "${RESULT}" --result-filename "${filename}"}
}

for spec in "1024 128" "256 1024"; do
  # shellcheck disable=SC2086
  set -- ${spec}
  input_len="$1"
  output_len="$2"
  for conc in 1 2 4 8; do
    timed_prompts=$((4 * conc))
    if (( timed_prompts < 16 )); then
      timed_prompts=16
    fi
    cell="random-in${input_len}-out${output_len}-c${conc}"
    echo "WARMUP ${cell}"
    run_one "${input_len}" "${output_len}" "${conc}" 1 ""
    for run in 1 2 3; do
      echo "TIMED ${cell} run=${run} num_prompts=${timed_prompts}"
      run_one "${input_len}" "${output_len}" "${conc}" "${timed_prompts}" \
        "${cell}-run${run}.json"
    done
  done
done
