# Observability

Local Prometheus and Grafana for vLLM and SGLang on Modal ([deploy/modal](../deploy/modal/README.md)).

Based on vLLM's [prometheus_grafana example](https://github.com/vllm-project/vllm/tree/79e205e8cdf7864b8b9f2d23c908b887f2f5badb/examples/observability/prometheus_grafana) at `79e205e`, with these changes:

- Prometheus scrapes the Modal endpoint over HTTPS instead of `host.docker.internal:8000`.
- Images pinned (`prom/prometheus:v3.5.0`, `grafana/grafana:12.1.0`); Prometheus data kept in a volume.
- Grafana on host port **3001**. Datasource and dashboard are provisioned, so there is no manual import.
- Dashboard: `model_name` defaults to `qwen3-8b`; three panels fixed or replaced (see the table).

## Files

| File | Purpose |
|---|---|
| `docker-compose.yaml` | Prometheus + Grafana |
| `prometheus.yaml` | Scrape job for the Modal host (replace `MODAL_HOST`) |
| `grafana/provisioning/` | Datasource and dashboard provider |
| `grafana.json` | vLLM server dashboard |
| `sglang-grafana.json` | SGLang server dashboard backed by live Prometheus scrapes |
| `trickle.sh` | Light curl load so panels move. Not a benchmark |
| `slo-bench-modal.yaml` | [llm-slo-bench](https://github.com/crypticseeds/llm-slo-bench) `ramp` config, 0.5 -> 1 -> 2 req/s |

## Run

```bash
# 1. GPU up (keep this terminal open; Ctrl-C stops the app)
modal serve deploy/modal/vllm_qwen3_8b.py      # note the printed https://... host

# 2. Point Prometheus and the ramp config at it
sed -i 's/MODAL_HOST/<host>/' observability/prometheus.yaml observability/slo-bench-modal.yaml
curl -s https://<host>/metrics | grep -E '^vllm:(num_requests_running|kv_cache_usage_perc)'

# 3. Stack up, check the target is UP at http://localhost:9090/targets, dashboard at http://localhost:3001
cd observability && docker compose up -d

# 4. Load, writing results to a dated run directory
OUT=../benchmarks/$(date +%F)-modal-a10-qwen3-8b
mkdir -p "$OUT"
BASE_URL=https://<host> ./trickle.sh 2 120 | tee "$OUT/trickle-client.txt"
SLO_BENCH_KEY=unused ~/REPOS/llm-slo-bench/bin/llm-slo-bench ramp --config slo-bench-modal.yaml \
  --out "$OUT/ramp.json" --html "$OUT/ramp.html" --raw-jsonl "$OUT/ramp.jsonl"

# 5. Shut down: Ctrl-C modal serve, then
docker compose down
cd .. && git checkout -- observability/prometheus.yaml observability/slo-bench-modal.yaml  # never commit the real host
```

Results from the 2026-10-18 run are in [`benchmarks/2026-10-18-modal-a10-qwen3-8b/`](../benchmarks/2026-10-18-modal-a10-qwen3-8b/README.md). Before committing a new run, replace the real host with `MODAL_HOST` in any file that embeds it (the HTML report does).

Prometheus scrapes every 5 s, which counts as traffic and keeps the A10 from scaling to zero. Always run `docker compose down` when finished.

## SGLang live smoke test

From the repo root, run `modal serve deploy/modal/sglang_qwen3_8b.py` in a separate terminal. Wait until `/health` returns HTTP 200. The launcher enables `/metrics`.

From `observability/`, enter the private hostname printed by Modal without `https://` or a trailing slash. This creates a scrape config outside the repo and disables the unused vLLM placeholder. If the current config already has a working vLLM target, the substitution preserves it.

```bash
read -r SGLANG_MODAL_HOST
export PROMETHEUS_CONFIG="$(mktemp -d "${TMPDIR:-/tmp}/sglang-prometheus.XXXXXX")/prometheus.yaml"
(
  umask 022
  sed -e 's/\["MODAL_HOST:443"\]/[]/' \
      -e "s|targets: \[\] # SGLang target.*|targets: [\"${SGLANG_MODAL_HOST}:443\"]|" \
      prometheus.yaml > "$PROMETHEUS_CONFIG"
)
docker compose up -d
```

Wait for `sglang` to report **UP** in [Prometheus Targets](http://127.0.0.1:9090/targets), then open [SGLang Live Metrics](http://127.0.0.1:3001/d/sglang-live). Use the normal local Grafana login.

```bash
BASE_URL="https://${SGLANG_MODAL_HOST}" bash ./trickle.sh 2 30
```

Check for HTTP 200 responses, positive request or token throughput, and latency samples. The queue can stay at zero under light traffic, and five-second scrapes can miss brief scheduler changes. These panels show server telemetry. GuideLLM supplies client-side latency, offered load, and request outcomes.

Stop `modal serve` with Ctrl-C once the smoke check is complete. The Prometheus volume retains the captured interval for screenshots.

## vLLM panel definitions

Every panel is server-side: vLLM measures it inside the API server process. TTFT and E2E start at frontend arrival (tokenization start), so they include queueing but not the network to the client. Client-side numbers (llm-slo-bench) include network and TLS to Modal and will be higher.

| Panel | Query (per `$model_name`) | Definition | Stage |
|---|---|---|---|
| Time To First Token Latency | p50/p90/p95/p99 + mean of `vllm:time_to_first_token_seconds` | Arrival to first output token. Includes tokenization, queue, prefill | Prefill + queue |
| Inter Token Latency | quantiles + mean of `vllm:inter_token_latency_seconds` | Gap between consecutive streamed outputs, one sample per output | Decode |
| Time Per Output Token (TPOT) | quantiles + mean of `vllm:request_time_per_output_token_seconds` | Per request: `(e2e - TTFT) / (output_tokens - 1)`. One-token requests recorded as 0 | Decode |
| E2E Request Latency | quantiles + mean of `vllm:e2e_request_latency_seconds` | Arrival to final token | Whole request |
| Queue Time per Request (mean) | `rate(_sum) / rate(_count)` of `vllm:request_queue_time_seconds` | Arrival in engine queue to first scheduled. **Fixed:** upstream plotted `rate(_sum)` alone (queue-seconds per second, no per-request meaning). Mean only: the smallest bucket is 0.3 s, so below that `histogram_quantile` just interpolates (an idle server showed p50 150 ms, p99 297 ms while the real mean was 7.8 µs) | Scheduler |
| Prefill and Decode Time per Request (mean) | `rate(_sum) / rate(_count)` of `vllm:request_prefill_time_seconds`, `vllm:request_decode_time_seconds` | Scheduled to first token; first to last token. **Fixed:** same `rate(_sum)`-only bug | Prefill / decode |
| Token Throughput | `rate(vllm:prompt_tokens_total)`, `rate(vllm:generation_tokens_total)` | Prompt tokens/s (prefill work) and output tokens/s (decode work), all users | System |
| Request Throughput | `sum(rate(vllm:request_success_total))` | Finished requests/s, all finish reasons. **Added:** upstream had no requests/s panel. Replaces "Max Generation Token in Sequence Group", which only differs from generation length under parallel sampling (`n>1`) and whose `rate(_sum)` had no definition | System |
| Scheduler State | `vllm:num_requests_running`, `vllm:num_requests_waiting` | Requests in the batch now; requests queued. Waiting > 0 sustained = saturated | Scheduler |
| Cache Utilization | `vllm:kv_cache_usage_perc` | Fraction of KV cache blocks in use (0-1). Near 1 = preemptions | Memory |
| Request Prompt Length | heatmap of `vllm:request_prompt_tokens` | Input length distribution | Workload |
| Request Generation Length | heatmap of `vllm:request_generation_tokens` | Output length distribution | Workload |
| Finish Reason | `increase(vllm:request_success_total)` by `finished_reason` | `stop` (EOS), `length` (hit max tokens), `abort` | Workload |

**Percentiles are bucket-limited.** Every quantile here comes from `histogram_quantile`, which interpolates linearly inside a bucket. vLLM's TTFT buckets jump from 0.1 s to 0.25 s, so a server-side TTFT p99 anywhere in that range is an estimate, not a measurement. For exact tails, use client-side HDR histograms (llm-slo-bench) and the server-side means.

Definitions follow the [vLLM metrics design doc](https://docs.vllm.ai/en/latest/design/metrics/).
