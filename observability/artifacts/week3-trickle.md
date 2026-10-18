# Week 3 trickle run, 2026-10-18

`BASE_URL=<modal host> ./trickle.sh 2 120`, from the Pi, ending 17:51:04 local. Not a benchmark: 2 closed-loop curl workers, 1 s sleep between requests, `max_tokens: 128`, thinking disabled via `chat_template_kwargs`.

Target: vLLM 0.21.0, `Qwen/Qwen3-8B` BF16, one Modal A10, `--max-num-seqs 16 --enforce-eager --max-model-len 8192`.

## Server side (vLLM `/metrics` counters after the run)

Counters include 1 earlier cold-start request (TTFT 0.818 s).

| Counter | Value |
|---|---|
| `request_success_total{finished_reason="stop"}` | 62 (`length`, `error`: 0) |
| `time_to_first_token_seconds` sum / count | 5.796 s / 62 |
| `inter_token_latency_seconds` sum / count | 152.06 s / 4154 |
| `generation_tokens_total` | 4216 |

Derived, trickle requests only:

- Mean TTFT: (5.796 - 0.818) / 61 = **82 ms**
- Mean inter-token latency: 152.06 / 4154 = **36.6 ms** (about 27 tokens/s per user)
- Mean output length: about 68 tokens; every request stopped on EOS before the 128 limit

## Client side (curl `time_total`)

61/61 requests returned 200. Sampled `time_total` values were 3.2-4.2 s; per-request times were not saved for this run.

Rerun at 18:18:53, same command, saved to `week3-trickle-client.txt`: 60/60 returned 200, `time_total` min 2.58 s, p50 2.98 s, p95 3.42 s, max 4.55 s. Server-side mean queue time for that run: 11.5 µs (0.69 ms over 60 requests), which the fixed Queue Time panel now shows correctly.

Server-side time per request is about 82 ms + 68 x 36.6 ms = 2.6 s. The 0.6-1.6 s gap is network and TLS from the Pi to Modal us-east, because each curl opens a new connection.

## Bandwidth check

Decode reads all weights once per step: 16 GB / 600 GB/s (A10) = 27 ms minimum. Observed 36.6 ms, about 73% of the bandwidth limit, ignoring KV cache reads.
