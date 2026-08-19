# Max-Q vs Workstation Edition

このリポジトリは、**同じ checkpoint・同じ software・投機デコードなし（no speculative decoding）** で **RTX PRO 6000 Blackwell Max-Q** と **Workstation Edition** を比較した記録です。

Same weights (`unsloth/Qwen3.8-27B-NVFP4` revision `7d6f8d4d72f56b92b3cdbf22f156b90e1bab0108`), same vLLM 0.27.1 flags, same official client matrix. Spec off. Thinking off. `--kv-cache-dtype bfloat16`.

SKU の話は C=1 decode 差（~+6%）対 long-prefill の TTFT 差（~0.73× TTFT / derived prefill ~+38%）である。モデルのスコアではない。

All official cells: n=3 median, same-condition, warmup `num_prompts=1` discarded, timed `num_prompts=max(16, 4*C)`, failed=0.

`Δ%` is `(WS − Max-Q) / Max-Q`. Prefill columns are **derived** (`input_len / median_ttft_s`). C>1 derived prefill includes queueing; it is not a clean kernel rate.

## Output tok/s

### Lane (input=1024, output=128)

| C | num_prompts | Max-Q | WS | Δ% | Max-Q runs | WS runs |
| ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 16 | **59.43** | **63.50** | +6.8 | 59.80 / 59.43 / 59.30 | 63.53 / 63.48 / 63.50 |
| 2 | 16 | **105.13** | **113.26** | +7.7 | 104.49 / 105.13 / 105.13 | 112.30 / 113.27 / 113.26 |
| 4 | 16 | **189.04** | **207.05** | +9.5 | 188.98 / 189.21 / 189.04 | 207.12 / 207.05 / 207.00 |
| 8 | 32 | **324.48** | **368.42** | +13.5 | 324.74 / 324.45 / 324.48 | 368.50 / 368.42 / 368.24 |

### Lane (input=256, output=1024)

| C | num_prompts | Max-Q | WS | Δ% | Max-Q runs | WS runs |
| ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 16 | **61.64** | **65.50** | +6.3 | 61.68 / 61.64 / 61.63 | 65.50 / 65.50 / 65.50 |
| 2 | 16 | **114.10** | **120.50** | +5.6 | 114.15 / 114.10 / 114.10 | 120.55 / 120.47 / 120.50 |
| 4 | 16 | **222.28** | **235.26** | +5.8 | 222.35 / 222.28 / 222.27 | 235.26 / 235.26 / 235.28 |
| 8 | 32 | **445.72** | **475.83** | +6.8 | 445.72 / 445.75 / 445.55 | 475.83 / 475.84 / 475.65 |

## TTFT (ms)

Median of the three run `median_ttft_ms` values (same run that is the median output tok/s). Run column is each run's `median_ttft_ms`.

### Lane (input=1024, output=128)

| C | Max-Q | WS | Max-Q runs | WS runs |
| ---: | ---: | ---: | --- | --- |
| 1 | **112.6** | **81.7** | 105.1 / 112.6 / 113.3 | 81.3 / 82.4 / 81.7 |
| 2 | **196.4** | **147.0** | 216.4 / 195.9 / 196.4 | 160.1 / 146.5 / 147.0 |
| 4 | **434.9** | **318.1** | 436.4 / 434.6 / 434.9 | 318.1 / 318.1 / 318.8 |
| 8 | **910.8** | **656.6** | 910.5 / 911.4 / 910.8 | 655.0 / 656.6 / 657.5 |

### Lane (input=256, output=1024)

| C | Max-Q | WS | Max-Q runs | WS runs |
| ---: | ---: | ---: | --- | --- |
| 1 | **55.5** | **55.5** | 56.0 / 55.5 / 55.3 | 55.5 / 53.7 / 55.8 |
| 2 | **107.9** | **103.3** | 107.9 / 106.7 / 110.4 | 103.2 / 113.5 / 103.3 |
| 4 | **142.3** | **116.6** | 141.8 / 142.5 / 142.3 | 116.7 / 116.6 / 115.2 |
| 8 | **247.8** | **193.4** | 245.9 / 247.8 / 249.8 | 193.4 / 193.2 / 194.5 |

## Derived prefill (tok/s)

`derived = input_len / (median_ttft_s)`. Labeled derived. Not a profiler measurement.

| lane | C | Max-Q derived | WS derived |
| --- | ---: | ---: | ---: |
| in=1024, out=128 | 1 | 9089 | 12533 |
| in=1024, out=128 | 2 | 5215 | 6968 |
| in=1024, out=128 | 4 | 2354 | 3219 |
| in=1024, out=128 | 8 | 1124 | 1560 |
| in=256, out=1024 | 1 | 4615 | 4610 |
| in=256, out=1024 | 2 | 2373 | 2479 |
| in=256, out=1024 | 4 | 1799 | 2195 |
| in=256, out=1024 | 8 | 1033 | 1324 |

C=1 decode-heavy derived prefill is level (4615 vs 4610). C=1 prefill-heavy is not (9089 vs 12533). That matches the TTFT split: short-prompt TTFT is the same, long-prompt TTFT is not.

## WS single-stream (no Max-Q pair)

`POST /v1/chat/completions`, prompt fixed, `max_tokens=1024`, temperature 0, seed 0. Tokens from `usage.completion_tokens / wall`. TTFT from first streamed content delta. n=3, one discarded warmup.

| metric | median | runs |
| --- | ---: | --- |
| output tok/s | **65.54** | 65.54 / 65.53 / 65.54 |
| TTFT (ms) | **53.35** | 53.35 / 55.41 / 52.81 |

Official random C=1 / out=1024 on the same serve: 65.50 tok/s.
