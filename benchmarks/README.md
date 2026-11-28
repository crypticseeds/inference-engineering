# Benchmarks

Reproducible workloads, raw outputs, and summarized results, one directory per run.

Every result must identify the model revision, engine version, hardware, input and output token lengths, sampling settings, concurrency or request rate, and whether requests were preempted. Check a run against the [pre-publish checklist](pre-publish-checklist.md) before quoting it elsewhere.

## Week 4: prefix cache comparison

[`prefix_modal.py`](prefix_modal.py) runs two clients from a Modal CPU container in us-east, next to the GPUs. Both send raw prompts with 2,048-token prefixes, 64-token suffixes and 32 output tokens to `/v1/completions`, at temperature 0 with `ignore_eos`.

| Client | Stage | Workload |
|---|---|---|
| vLLM 0.21.0 `vllm bench serve`, `prefix_repetition` | `unique-c1` | 100 distinct prefixes, one request at a time |
| | `repeated-c1` | 5 prefixes reused 20 times each, one at a time |
| | `unique-2rps` | 120 distinct prefixes at 2 req/s, up to 32 in flight |
| | `repeated-2rps` | 6 prefixes reused 20 times each at 2 req/s |
| SGLang 0.5.9 `sglang.bench_serving`, `generated-shared-prefix` (cross-check) | `gsp-unique-c1` | 100 groups of 1, one at a time |
| | `gsp-repeated-c1` | 5 groups of 20, one at a time |

Every stage has its own seed, so no stage reuses another's prefixes. Both datasets shuffle requests, so the repeated stages include the first, uncached use of each prefix. Both tools also send a warm-up request from their own prompts, which can prime one prefix. [vLLM benchmark CLI](https://docs.vllm.ai/en/v0.21.0/benchmarking/cli/) · [SGLang 0.5.9 benchmark source](https://github.com/sgl-project/sglang/blob/v0.5.9/python/sglang/bench_serving.py)

Start both engines with `modal serve` in separate terminals, then run one client per engine:

```bash
modal run benchmarks/prefix_modal.py --engine vllm --target https://<vllm-host> --stem <stem>
modal run benchmarks/prefix_modal.py --engine sglang --target https://<sglang-host> --stem <stem>
```

Before the benchmark, check that both engines report reused prompt tokens. The probe uses its own prompts, so it does not prime the measured prefixes.

```bash
python3 benchmarks/prefix_cache.py --vllm https://<vllm-host> --sglang https://<sglang-host>
```

## Runs

- [2026-10-18 Modal A10 dashboard validation](./2026-10-18-modal-a10-qwen3-8b/README.md) compares server-side and client-side latency.
- [2026-11-23 Modal A10 GuideLLM results](./2026-11-23-modal-a10-guidellm/README.md) records the sequential baseline and the 1 and 2 req/s stages.
- [2026-11-25 Modal A10 live comparison](./2026-11-25-modal-a10-live/README.md) repeats the six-rate sweep on vLLM and SGLang at once, from a Modal client, with a live Grafana dashboard.
- [2026-11-25 Modal H100 concurrency sweep](./2026-11-25-modal-h100-concurrency/README.md) pushes both engines from 1 to 1,000 concurrent requests with CUDA graphs and a 256-request batch.
- [2026-11-27 Modal A10 shared-prefix comparison](./2026-11-27-modal-a10-prefix/README.md) measures TTFT for distinct vs reused 2,048-token prefixes on both engines, one at a time and at 2 req/s.
- [2026-11-28 Modal L40S quantization](./2026-11-28-modal-l40s-quantization/README.md) compares Qwen3-8B BF16, FP8 and AWQ INT4 on GSM8K, WikiText perplexity, memory and a 1 to 250 concurrency sweep.
