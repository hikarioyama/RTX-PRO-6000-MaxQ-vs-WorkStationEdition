# RTX PRO 6000 Blackwell Max-Q vs Workstation Edition

このリポジトリは、**同じ checkpoint・同じ software・投機デコードなし（no speculative decoding）** で **RTX PRO 6000 Blackwell Max-Q** と **Workstation Edition** を比較した記録です。

ワークロードは `unsloth/Qwen3.8-27B-NVFP4` を **vLLM 0.27.1 / SM120** で serve し、公式クライアント `vllm bench serve` で測ったものです。モデルの tok/s を出すことが目的ではありません。

**この重みは Unsloth です。NVIDIA 公式の 3.8 NVFP4 ではありません。** 測定時点で NVIDIA 公式の Qwen3.8 NVFP4 は未公開です。revision は `7d6f8d4d72f56b92b3cdbf22f156b90e1bab0108` に固定しています。

## なぜ speculative decoding を切るか

speculative decoding の accept は prompt と sampling / 確率で揺れる。SKU 差（帯域 vs 計算、300W vs 600W）を見るにはその項をゼロに固定した。AR（投機なし）だけが同じ重みを同じ回数読む比較になる。

## Headline（n=3, same-condition）

公式 `vllm bench serve` random、seed 0、temperature 0、request-rate inf、ignore-eos。warmup `num_prompts=1` は捨て、timed は n=3 の median。

| SKU | lane | C | output tok/s | TTFT (ms) | derived prefill tok/s |
| --- | --- | ---: | ---: | ---: | ---: |
| Max-Q | in=256, out=1024 | 1 | **61.64** | **55.5** | 4615 (derived) |
| Max-Q | in=1024, out=128 | 1 | **59.43** | **112.6** | 9089 (derived) |
| Workstation Edition | in=256, out=1024 | 1 | **65.50** | **55.5** | 4610 (derived) |
| Workstation Edition | in=1024, out=128 | 1 | **63.50** | **81.7** | 12533 (derived) |
| Workstation Edition | single-stream chat | 1 | **65.54** | **53.35** | — |

`derived prefill` は `input_len / (median_ttft_s)` です。TTFT には first decode token も含まれるので **kernel 実測ではなく derived** と書いてあります。C>1 は待ち行列が混ざるので `results/COMPARE.md` を見てください。

decode-heavy の C=1 は両 SKU とも TTFT ≈ 55.5 ms。差はほぼ decode レート（Max-Q 61.64 vs WS 65.50、**+6.3%**）。prefill-heavy の C=1 は TTFT が Max-Q 112.6 ms vs WS 81.7 ms と開き、derived prefill も 9089 vs 12533 です。

全セルの表と Δ% は [`results/COMPARE.md`](results/COMPARE.md) にあります。

## 条件（すべての数字）

- クライアント: 公式 `vllm bench serve`（`recipes/bench.sh`）
- dataset: `random`、seed `0`、temperature `0`、`--request-rate inf`、`--ignore-eos`
- 行列: `(input, output) = (1024, 128)` と `(256, 1024)` × `C = 1, 2, 4, 8`
- warmup: 各セル `num_prompts=1` を捨てる
- timed: n=3 median、`num_prompts = max(16, 4*C)`
- `--tokenizer $MODEL --trust-remote-code` 必須（`--served-model-name` の alias は HF id ではない）
- serve: `--language-model-only`、thinking **off**、`--speculative-config` **なし**
- `--kv-cache-dtype bfloat16` **必須**。checkpoint の FP8 KV + `flash_attn` は SM120 で落ちます
- 失敗リクエスト: 全 timed run で 0。result JSON に `spec_decode` キーなし

WS だけ追加で single-stream（`POST /v1/chat/completions`、`usage.completion_tokens / wall`、TTFT は first content delta）。公式 random C=1 / out=1024 の 65.50 tok/s と 65.54 で一致（差 < 0.1%）。

## ハードウェア

どちらも **RTX PRO 6000 Blackwell**、1 GPU、TP=1。

- **Max-Q Workstation Edition** — `results/2026-08-20-maxq/`
- **Workstation Edition**（full WS、Max-Q ではない）— `results/2026-08-20-ws/`

vLLM **0.27.1**、attention `flash_attn`、KV `bfloat16`、quantization auto-detect `compressed-tensors`。

## 重みの取得

```bash
export MODEL="${PWD}/Qwen3.8-27B-NVFP4-unsloth"
hf download unsloth/Qwen3.8-27B-NVFP4 \
  --revision 7d6f8d4d72f56b92b3cdbf22f156b90e1bab0108 \
  --local-dir "${MODEL}"
```

NVIDIA 公式 NVFP4 では再現しません。

## 再現手順

vLLM 0.27.1 を `PATH` に置くか、`VLLM=` でバイナリを渡します。

```bash
# serve（別端末。thinking off / spec off / bfloat16 KV）
MODEL=... PORT=8000 SERVED_NAME=qwen38-nvfp4 VLLM=vllm ./recipes/serve.sh

# 公式行列
MODEL=... BASE=http://127.0.0.1:8000 SERVED_NAME=qwen38-nvfp4 RESULT=./out ./recipes/bench.sh
```

ホームディレクトリはスクリプトに書いていません。`MODEL` / `VLLM` / `PORT` / `BASE` / `RESULT` だけ見て動きます。

Max-Q 測定時の serve argv は `results/2026-08-20-maxq/meta.json` の `serve_argv` と同じです。

## レイアウト

```
recipes/serve.sh          # $MODEL $PORT $SERVED_NAME $VLLM
recipes/bench.sh          # official vllm bench serve matrix
results/2026-08-20-maxq/  # sanitized JSON + progress + meta
results/2026-08-20-ws/    # sanitized JSON + progress + REPORT + single-stream
results/COMPARE.md        # Max-Q vs WS
```

`serve.log` は入れていません。
