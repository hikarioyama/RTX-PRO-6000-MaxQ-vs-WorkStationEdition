# Qwen3.8-27B NVFP4 (Unsloth) — WS Edition, spec off

Date: 2026-08-20  
Serve: already-running `http://127.0.0.1:8041` model `qwen38-nvfp4-ws`  
Weights: `$MODEL` — **Unsloth NVFP4, not NVIDIA official**  
SKU: RTX PRO 6000 Blackwell **Workstation Edition** (full WS, not Max-Q)  
Speculative decoding: **off** (`speculative_config=None`). `/metrics` had **zero** `spec_decode` counters before, after every official cell, and after single-stream.

## Conditions (all numbers)

- n=3 timed runs per cell / metric, **median** reported
- same-condition: same prompt/dataset, seed=0, temperature=0, warm state
- official warmup: `num_prompts=1` discarded per cell
- single-stream warmup: one discarded non-stream POST
- official client: `$VLLM bench serve` (`--dataset-name random --request-rate inf --ignore-eos`)
- timed `num_prompts = max(16, 4*conc)`
- single-stream tokens from `usage.completion_tokens / wall` (not SSE chunk counts)
- single-stream TTFT from first streamed content delta (separate n=3)

## Official `vllm bench serve` (n=3, same-condition)

Failed requests: 0 in every timed run. `spec_decode` keys in result JSON: none.

### Lane (input=1024, output=128)

| conc | num_prompts | median output tok/s | median TTFT (ms) | run tok/s | run TTFT (ms) |
| ---: | ---: | ---: | ---: | --- | --- |
| 1 | 16 | **63.50** | **81.7** | 63.53 / 63.48 / 63.50 | 81.3 / 82.4 / 81.7 |
| 2 | 16 | **113.26** | **147.0** | 112.30 / 113.27 / 113.26 | 160.1 / 146.5 / 147.0 |
| 4 | 16 | **207.05** | **318.1** | 207.12 / 207.05 / 207.00 | 318.1 / 318.1 / 318.8 |
| 8 | 32 | **368.42** | **656.6** | 368.50 / 368.42 / 368.24 | 655.0 / 656.6 / 657.5 |

### Lane (input=256, output=1024)

| conc | num_prompts | median output tok/s | median TTFT (ms) | run tok/s | run TTFT (ms) |
| ---: | ---: | ---: | ---: | --- | --- |
| 1 | 16 | **65.50** | **55.5** | 65.50 / 65.50 / 65.50 | 55.5 / 53.7 / 55.8 |
| 2 | 16 | **120.50** | **103.3** | 120.55 / 120.47 / 120.50 | 103.2 / 113.5 / 103.3 |
| 4 | 16 | **235.26** | **116.6** | 235.26 / 235.26 / 235.28 | 116.7 / 116.6 / 115.2 |
| 8 | 32 | **475.83** | **193.4** | 475.83 / 475.84 / 475.65 | 193.4 / 193.2 / 194.5 |

Decode-heavy lane scales roughly with concurrency through c=8 (65.5 → 120.5 → 235.3 → 475.8 tok/s). Prefill-heavy lane also scales (63.5 → 113.3 → 207.0 → 368.4) with TTFT growing as expected under batching.

## Single-stream bench (n=3, same-condition)

`POST /v1/chat/completions` model `qwen38-nvfp4-ws`  
Prompt: `Write a complete, production-quality Python LRU cache with per-key TTL and a pytest suite.`  
`max_tokens=1024 temperature=0 seed=0`

| metric | median | runs |
| --- | ---: | --- |
| output tok/s (`usage.completion_tokens / wall`) | **65.54** | 65.54 / 65.53 / 65.54 |
| TTFT (first content delta) | **53.35 ms** | 53.35 / 55.41 / 52.81 |
| completion_tokens | 1024 | all 1024 (`finish_reason=length`) |

Warmup discarded: 1024 tok in 15.75 s (65.01 tok/s).  
Single-stream 65.54 tok/s matches official random c=1 / out=1024 (65.50 tok/s) to <0.1%.

Headline: **qwen38-nvfp4-ws Unsloth NVFP4, WS Edition, spec off : 65.54 tok/s (n=3, same-condition) | TTFT 53.35 ms | accept n/a (no spec counters)**

## Files

- `summary.json` — n=3 medians + per-run official cells
- `single-stream.json` — single-stream bench
- `progress.log` — LAUNCH line preserved; every RUN / CELL_DONE appended
- `random-in*-out*-c*-run{1,2,3}.json` — official client dumps (24 files)

