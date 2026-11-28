# Shared-prefix TTFT on vLLM and SGLang, Modal A10s, 2026-11-27

The Week 4 prefix-reuse comparison. Each engine ran on its own A10 with prefix caching on. Every prompt was a 2,048-token prefix plus a 64-token suffix, with 32 output tokens. The test compared prompts whose prefix had never been seen against prompts reusing one of a few prefixes.

Reuse cut time to first token (TTFT) for one request at a time from 533 to 73 ms on vLLM and from 607 to 170 ms on SGLang (p50). At 2 req/s, the same rate overloaded both engines with distinct prefixes but was easy with reused ones. Both engines completed every request in all 12 stages, and a second benchmark client agreed with the first.

## Setup

| Item | Value |
|---|---|
| Engines | vLLM 0.21.0 and SGLang 0.5.9, one Modal NVIDIA A10 (24 GB) each, us-east |
| Launchers | [`vllm_qwen3_8b.py`](../../deploy/modal/vllm_qwen3_8b.py) and [`sglang_qwen3_8b.py`](../../deploy/modal/sglang_qwen3_8b.py), default `A10` profile |
| Engine limits | BF16, 8,192-token context, 16 running requests, CUDA graphs off, prefix caching on (vLLM `--enable-prefix-caching`, SGLang RadixAttention by default). vLLM reserves 90% of GPU memory, SGLang 85% static |
| Model | `Qwen/Qwen3-8B` at `b968826d9c46dd6066d109eabc6255188de91218`, served as `qwen3-8b` |
| Client | [`prefix_modal.py`](../prefix_modal.py) in a 4-CPU Modal container in us-east, one per engine, started together. Tokenizer loaded from the pinned snapshot |
| Request | Raw prompt to `/v1/completions`, streaming, temperature 0, `ignore_eos` |
| Workload | 2,048-token prefix + 64-token suffix = 2,112 prompt tokens, 32 output tokens |

## Is the cache actually used?

Before the benchmark, [`prefix_cache.py`](../prefix_cache.py) sent 10 pairs of chat requests from the Pi. Each pair was a new prefix and then the same prefix with a different question. It used its own prompts, so it primed nothing the benchmark measured.

| Engine | Cached tokens, first request | Cached tokens, repeat | TTFT p50, first / repeat |
|---|---:|---:|---:|
| vLLM | 0 of about 2,360 | 2,336 to 2,352 | 752 / 264 ms |
| SGLang | 3 to 6 | 2,347 to 2,352 | 840 / 348 ms |

Both engines report reused prompt tokens and both get faster on reuse. These TTFTs include the Pi to us-east network, so they are higher than the benchmark's. [Probe output](raw/prefix-cache-probe.json)

## One request at a time

`vllm bench serve` with the `prefix_repetition` dataset, 100 requests per stage. In the repeated stage, 5 prefixes are each used 20 times in shuffled order, so 5 requests are the first, uncached use of their prefix.

| | vLLM | SGLang |
|---|---:|---:|
| Distinct prefixes, TTFT p50 / p95 / p99 | 533 / 541 / 544 ms | 607 / 615 / 888 ms |
| Repeated prefixes, all 100 requests, p50 / p95 / p99 | 73 / 108 / 529 ms | 170 / 456 / 607 ms |
| Repeated, 95 fastest (cache hits), p50 / p95 | 73 / 79 ms | 170 / 172 ms |
| Repeated, 5 slowest (first use of each prefix), p50 | 519 ms | 605 ms |
| TTFT saved by a hit, p50 | 460 ms (86%) | 437 ms (72%) |
| Inter-token latency p50, distinct / repeated | 34.6 / 34.8 ms | 34.4 / 34.4 ms |

On vLLM the split is unambiguous: exactly 5 requests took about 520 ms and the other 95 were under 90 ms. On SGLang one repeated request took 448 ms and five took 603 to 638 ms. The table treats the five slowest as first uses.

The p95 and p99 of the full repeated stage are set by those first uses: 5 misses in 100 requests sit right at the 95th percentile. Report hit latency and first-use latency separately, as above.

Reuse does not change inter-token latency, because decode does not depend on how the prompt was prefilled.

## At 2 req/s

Same workloads at a fixed 2 req/s with near-uniform arrivals, up to 32 requests in flight. The repeated stage used 6 prefixes, each used 20 times.

| | vLLM distinct | vLLM repeated | SGLang distinct | SGLang repeated |
|---|---:|---:|---:|---:|
| Completed req/s | 1.69 | 1.96 | 1.61 | 1.96 |
| Peak requests in flight | 32 (cap) | 8 | 33 | 9 |
| TTFT p50 / p95 / p99 | 4,863 / 9,360 / 9,696 ms | 151 / 233 / 650 ms | 7,334 / 13,378 / 13,452 ms | 214 / 276 / 669 ms |
| TTFT p50 / p95, hits only | | 149 / 167 ms | | 213 / 231 ms |
| Inter-token latency p50 | 67.9 ms | 38.9 ms | 44.2 ms | 38.8 ms |
| Failed requests | 0 | 0 | 0 | 0 |

With distinct prefixes, 2 req/s is more than either engine can prefill on an A10. Completed throughput fell short of the offered rate, the client hit its 32-request cap, and TTFT grew to several seconds as requests queued. I chose 2 req/s expecting it to be near the limit; it was past it. Each request needs about 2,100 tokens of prefill, and both engines topped out around 1.6 to 1.7 req/s, about 3,400 to 3,600 prefill tokens per second.

With reused prefixes, the same 2 req/s was easy. Each request only prefills its 64-token suffix, so both engines kept up, held at most 9 requests in flight, and kept p50 TTFT near 150 to 210 ms. **Prefix reuse raised capacity, not just one-request latency.** On this workload it moved 2 req/s from overloaded to comfortable.

vLLM's inter-token latency rose to 68 ms in the overloaded distinct stage. With up to 16 requests running, long prefills share each step with ongoing decodes. SGLang's rose less, to 44 ms, while its TTFT grew more. These were single runs, so treat that difference as a lead, not a result.

## Cross-check with SGLang's client

`sglang.bench_serving` with the `generated-shared-prefix` dataset, 100 requests one at a time against both engines. Its generator overshoots the requested lengths slightly: prefixes averaged about 2,155 tokens and questions about 67.

| TTFT p50 | vLLM | SGLang |
|---|---:|---:|
| 100 distinct prefixes | 554 ms | 671 ms |
| 5 prefixes × 20 requests | 84 ms | 215 ms |

Same verdict as the vLLM client: reuse cuts TTFT by several hundred milliseconds on both engines, and vLLM is lower in both conditions. The absolute values are 4 to 26% higher, consistent with the longer prompts.

## What this shows

1. **Both engines reuse a shared prefix, and it is the biggest TTFT lever measured so far.** A hit removed 72 to 86% of one-request TTFT on 2,112-token prompts.
2. **Reuse also raises capacity.** At 2 req/s, distinct prefixes overloaded both A10s; reused prefixes needed a fraction of the prefill and stayed under 250 ms p95 on hits.
3. **SGLang's cache-hit TTFT has a higher floor on this setup:** 170 ms against vLLM's 73 ms, with both clients agreeing. Its distinct-prefix TTFT is only 74 ms higher (607 vs 533), so the gap is not prefill speed. Something adds about 100 ms on SGLang's hit path. On November 25, SGLang's low-load TTFT for short chat prompts was 69 ms from the same client location, so it is specific to this request shape: either long cached prefixes or the `/v1/completions` route. Not investigated.
4. **This does not reproduce SGLang's headline advantage.** The RadixAttention paper's gains come against engines without automatic prefix caching. vLLM 0.21 has it on, so both engines skip the shared prefill here. The comparison is between two prefix caches, not cache against no cache.

## Caveats

- One run per stage on one container per engine.
- Prefixes are random tokens. Real shared prompts (system prompts, documents) behave the same for caching but differ in text.
- Each client sent one warm-up request from its own prompts before measuring; it may have primed one prefix in the repeated stages.
- The distinct-prefix stages used 100 to 120 prefixes of about 2,100 tokens, far more than an A10's KV cache holds, so earlier prefixes were evicted. Eviction cannot turn a distinct prefix into a hit, so this does not affect those stages.
- No server-side metrics: Prometheus and Grafana were not running during this run. Queue depth and cache hit counters come only from the client results and the probe.
- About 21 minutes of A10 time per engine including startup, roughly 0.80 USD in total.

## Reproduce

See [Week 4: prefix cache comparison](../README.md#week-4-prefix-cache-comparison) for the commands. This run used stem `20261127-a10-prefix`.

## Raw data

[`raw/`](raw/) holds one report per stage: `raw/<engine>/vllm-bench/<stage>.json` (`vllm bench serve --save-detailed`, with per-request TTFTs) and `raw/<engine>/sglang-bench/<stage>.jsonl` (`sglang.bench_serving --output-details`), plus the probe output. The reports contain no endpoint hostnames.
