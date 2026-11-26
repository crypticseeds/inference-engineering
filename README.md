# Inference Engineering

A reproducible LLM serving stack: vLLM serving `Qwen/Qwen3-8B` behind an OpenAI-compatible API on rented GPUs, with Prometheus and Grafana for server-side metrics and client-side load testing to check them. Every result in this repository records the engine version, model revision, hardware and launch flags it was measured with.

## Architecture

```text
            OpenAI-compatible requests                 scrape every 5 s
 client  -------------------------------->  Modal A10  <-----------------  Prometheus  ---->  Grafana
 (curl,     /v1/chat/completions            vLLM 0.21.0      /metrics       (Docker          (dashboard,
  llm-slo-bench)                            Qwen3-8B BF16                    Compose)          port 3001)
```

- **Serving:** vLLM on one Modal A10, defined in [`deploy/modal/vllm_qwen3_8b.py`](./deploy/modal/vllm_qwen3_8b.py). The same container exposes the OpenAI API under `/v1` and Prometheus metrics at `/metrics`.
- **Metrics:** Prometheus scrapes `/metrics` over HTTPS; Grafana renders a dashboard adapted from vLLM's upstream example, with three panels corrected. See [`observability/`](./observability/README.md).
- **Load:** [`observability/trickle.sh`](./observability/trickle.sh) for closed-loop curl traffic and [llm-slo-bench](https://github.com/crypticseeds/llm-slo-bench) for open-loop rate ramps with client-side HDR histograms.

## Verified platforms

| Platform | How | Runbook |
|---|---|---|
| Modal | Serverless A10 (24 GB), vLLM 0.21.0, scale to zero after 120 s idle | [`deploy/modal/`](./deploy/modal/README.md) |
| Modal | Serverless A10 (24 GB), SGLang 0.5.9, dev endpoint checked from Bruno, Prometheus metrics enabled | [SGLang runbook](./variants/sglang/README.md) |
| RunPod | pip: vLLM installed with `uv` on a GPU pod, then `vllm serve` | [`deploy/runpod/`](./deploy/runpod/README.md#method-a-pip-on-a-pod) |
| RunPod | Image: official `vllm/vllm-openai` image as the pod container | [`deploy/runpod/`](./deploy/runpod/README.md#method-b-vllmvllm-openai-as-the-pod-image) |

## Results

Qwen3-8B BF16 on one Modal A10, vLLM 0.21.0 with `--enforce-eager`, low load (up to 2 req/s). Full setup, method and raw data: [`benchmarks/2026-10-18-modal-a10-qwen3-8b/`](./benchmarks/2026-10-18-modal-a10-qwen3-8b/README.md).

| Result | Value | What it shows |
|---|---|---|
| TTFT, server vs client (mean) | 94.5 ms (vLLM `/metrics`) vs 195 ms (llm-slo-bench) | About half of user-facing TTFT is network and proxy time that server metrics cannot see |
| Inter-token latency (mean) | 36.6 ms | About 73% of the A10 memory-bandwidth limit (16 GB of weights / 600 GB/s = 27 ms per token) |
| Queue time on an idle server | Dashboard p99 297 ms vs real mean 7.8 µs | Quantiles from vLLM's 0.3 s minimum bucket are interpolation artifacts; the panel now shows the mean |
| Dashboard panels fixed | 3 | Queue Time and Prefill/Decode Time divided by request count; Request Throughput replaces a panel with no definition |

## Repository layout

```text
inference-engineering/
├── deploy/
│   ├── modal/        # Modal app and runbook (A10, vLLM 0.21.0, Qwen3-8B)
│   ├── runpod/       # RunPod runbook, troubleshooting, and the original step-by-step guides in reference/
│   ├── docker/       # (planned) pinned container deployment
│   └── kubernetes/   # (planned) Kubernetes deployment with queue-depth autoscaling
├── observability/    # Prometheus + Grafana compose stack, dashboard, panel definitions, load scripts
├── benchmarks/       # One directory per run: setup, method, results, raw client and server data
├── variants/
│   ├── sglang/       # Verified SGLang deployment and API runbook
│   ├── fp8/          # (planned) FP8 quantization
│   └── awq/          # (planned) AWQ quantization
├── service/          # (planned) shared service code outside the serving engine, such as request middleware
└── router/           # (planned) cost- and latency-aware routing
```

Directories marked (planned) hold only a README describing their scope; see [Planned](#planned).

## Measurement conventions

1. Keep model, workload and hardware fixed when comparing engines or configurations.
2. Pin versions and save the exact launch command with every result.
3. Record input and output token lengths.
4. Report p50, p95 and p99, and keep TTFT separate from per-token decode latency.
5. Prefer client HDR histograms for tail latency; server-side `histogram_quantile` values are bucket-limited.
6. Do not publish measurements taken while the server is preempting requests.
7. Keep credentials and endpoint hostnames out of the repository; use placeholders such as `MODAL_HOST`.

## Planned

- Shared-prefix TTFT comparison between SGLang and vLLM on the same model, hardware and workload.
- FP8 and AWQ quantization, with a quality check alongside memory and latency.
- Kubernetes deployment with autoscaling on request queue depth rather than CPU.
- Cost- and latency-aware routing across deployments.

## Roadmap

This work follows the [Time to First Token roadmap](https://github.com/crypticseeds/time-to-first-token), a personal learning fork of the original [patchy631/time-to-first-token](https://github.com/patchy631/time-to-first-token) roadmap created by [patchy631](https://github.com/patchy631). The roadmap repository holds the curriculum, progress tracker and revision notes. This repository holds the service, deployment configuration, observability and benchmarks.

## Related tools

- [sre-inference-gateway](https://github.com/crypticseeds/sre-inference-gateway): multi-provider inference gateway with SRE patterns; its measurement runs (including `2026-10-18-vllm-modal`) are in [`docs/measurements`](https://github.com/crypticseeds/sre-inference-gateway/tree/main/docs/measurements).
- [llm-slo-bench](https://github.com/crypticseeds/llm-slo-bench): Go load generator and SLO gate for OpenAI-compatible streaming endpoints, used for the ramp results here.

## License

[MIT](./LICENSE)
