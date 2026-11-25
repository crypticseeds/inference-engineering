# Qwen3-8B on Modal A10: latency and dashboard validation, 2026-10-18

Low-rate load against vLLM on one Modal A10, measured from both sides: vLLM's own `/metrics` (server) and [llm-slo-bench](https://github.com/crypticseeds/llm-slo-bench) plus curl (client). The goal was to validate the Grafana dashboard against independent client measurements and to establish decode latency against the hardware limit. It is not a capacity test: peak load was 2 req/s with at most 8 requests in flight, well under the 16-sequence batch limit.

## Setup

| Item | Value |
|---|---|
| Engine | vLLM 0.21.0 |
| Model | `Qwen/Qwen3-8B` at revision `b968826d9c46dd6066d109eabc6255188de91218`, BF16 (`--dtype auto`) |
| Hardware | One Modal A10 (24 GB), Modal us-east |
| Server flags | `--max-model-len 8192 --max-num-seqs 16 --gpu-memory-utilization 0.90 --enforce-eager` |
| App | [`deploy/modal/vllm_qwen3_8b.py`](../../deploy/modal/vllm_qwen3_8b.py) under `modal serve`, one container |
| Server metrics | Prometheus scraping `/metrics` every 5 s, Grafana dashboard from [`observability/`](../../observability/README.md) |
| Client | A Raspberry Pi over the public internet, HTTPS to Modal |
| Thinking mode | Disabled (`chat_template_kwargs` for curl, `/no_think` for llm-slo-bench) |

## Method

Two workloads, both with short prompts and up to 128 output tokens:

1. **Trickle** ([`trickle.md`](./trickle.md), [`trickle-client.txt`](./trickle-client.txt)): [`observability/trickle.sh`](../../observability/trickle.sh) with 2 closed-loop curl workers for 120 s, 1 s sleep between requests, a new TLS connection per request. Run twice; server counters read from `/metrics` after the first run, per-request client `time_total` saved for the second.
2. **Ramp** ([`ramp.json`](./ramp.json), [`ramp.html`](./ramp.html), [`ramp.jsonl`](./ramp.jsonl)): llm-slo-bench `ramp` with [`observability/slo-bench-modal.yaml`](../../observability/slo-bench-modal.yaml), streaming, open-loop 0.5 -> 1 -> 2 req/s for 2 minutes each, `max_in_flight: 8`. 300 requests, 20 prompt tokens each.

Client-side TTFT (llm-slo-bench) includes the full network path to Modal. Server-side TTFT is measured by vLLM from request arrival at the API server to the first output token, so it excludes the network.

## Results

### Ramp, client side (llm-slo-bench, HDR histograms)

300/300 succeeded; 0 dropped, timed out or errored.

| Metric | Mean | p50 | p95 | p99 |
|---|---|---|---|---|
| Time to first byte | 104.3 ms | 98.9 ms | 126.8 ms | 155.1 ms |
| Time to first token | 195.3 ms | 194.0 ms | 225.5 ms | 247.8 ms |
| Inter-chunk latency | 36.6 ms | 36.8 ms | 39.9 ms | 49.1 ms |
| Request duration | 2.80 s | 2.79 s | 2.97 s | 3.09 s |
| Output tokens/s per request | 27.7 | 27.6 | 29.3 | 29.5 |

21,664 completion tokens in total (about 72 per request). Both sanity gates in the config passed (p99 TTFT <= 2,000 ms, error rate <= 1%).

### Ramp, server side (vLLM `/metrics`)

| Metric | Value |
|---|---|
| Mean TTFT | 94.5 ms |

TODO(owner): record how the 94.5 ms was read (Grafana panel or a `/metrics` sum/count snapshot) and archive that snapshot here; no raw server-side export for the ramp is saved in this directory.

### Trickle

| Metric | Value | Source |
|---|---|---|
| Requests | 61/61 and 60/60 HTTP 200 | curl |
| Mean TTFT, run 1 | 82 ms (cold-start request excluded) | `/metrics` |
| Mean inter-token latency, run 1 | 36.6 ms (152.06 s / 4,154 samples) | `/metrics` |
| Mean output length, run 1 | about 68 tokens, all `stop` | `/metrics` |
| Client `time_total`, run 2 | min 2.58 s, p50 2.98 s, p95 3.42 s, max 4.55 s | curl |
| Mean queue time, run 2 | 11.5 µs | `/metrics` |

## Findings

1. **About half of client TTFT is outside the server.** Server-side mean TTFT was 94.5 ms; the client saw 195 ms. The roughly 100 ms difference is outside vLLM's view (network, TLS and Modal's proxy between the client and the container), so server-side dashboards alone understate user-facing TTFT by about 2x for this client.
2. **Decode runs at about 73% of the A10 bandwidth limit.** Each decode step reads all weights once: 16 GB / 600 GB/s = 27 ms minimum per token. Observed inter-token latency was 36.6 ms on the server (trickle) and 36.6 ms mean inter-chunk latency on the client (ramp), so 27 / 36.6 = about 73%, ignoring KV cache reads. Per-request decode speed of 27.7 tokens/s is consistent with 1 / 36.6 ms = 27.3 tokens/s.
3. **Server-side queue-time percentiles were an artifact.** vLLM's smallest `request_queue_time_seconds` bucket is 0.3 s, so `histogram_quantile` interpolates inside it. On an idle server the dashboard showed p50 150 ms and p99 297 ms while the real mean was 7.8 µs. The panel now plots `rate(_sum) / rate(_count)` (mean only) and showed 11.5 µs for trickle run 2.
4. **Three dashboard panels were fixed.** Queue Time and Prefill/Decode Time plotted `rate(_sum)` without dividing by `rate(_count)`, which has no per-request meaning; both now show per-request means. Max Generation Token in Sequence Group was replaced by a Request Throughput panel. Definitions are in [`observability/README.md`](../../observability/README.md#panel-definitions).
5. **Server-side tail percentiles are estimates.** vLLM's TTFT buckets jump from 0.1 s to 0.25 s, so any server-side p99 in that range is interpolated. Use the client HDR histograms for tails and the server means for attribution.

## Limitations

- Eager mode (`--enforce-eager`): CUDA graph mode was not measured.
- Low load only: at most 2 req/s and 8 in flight against a 16-sequence limit. No saturation, queueing or preemption behavior was exercised.
- One client location over the public internet; network figures do not transfer to other clients.
- Single run of each workload.

The Modal host is redacted to `MODAL_HOST` in `ramp.html`.
