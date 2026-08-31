# Inference Engineering

An implementation repository for learning how to operate and optimize one reproducible Large Language Model (LLM) inference service.

The work follows the [Time to First Token roadmap](https://github.com/crypticseeds/time-to-first-token), a personal learning fork of the original [patchy631/time-to-first-token](https://github.com/patchy631/time-to-first-token) roadmap created by [patchy631](https://github.com/patchy631). The roadmap repository holds the curriculum, progress tracker, and revision notes. This repository holds the service, deployment configuration, observability, benchmarks, optimization variants, and final benchmark writeup.

## Goal

Build one OpenAI-compatible inference service and improve it in place:

- Serve a model with vLLM on rented Graphics Processing Unit (GPU) hardware
- Compare vLLM with SGLang using the same model and workload
- Measure Time to First Token (TTFT), Time Per Output Token (TPOT), throughput, queue depth, and cost
- Benchmark quantization, speculative decoding, and Key-Value (KV) cache strategies
- Deploy the service on Kubernetes and autoscale from queue depth
- Add cost-aware routing and per-request token budgets
- Publish exact commands, pinned versions, and reproducible results

## Current Direction

- **Primary engine:** vLLM
- **Candidate baseline model:** `Qwen/Qwen3-8B`
- **Current smoke-test environment:** Modal with one A10 GPU
- **Planned persistent environment:** RunPod with a 24-gigabyte GPU
- **Initial context target:** 8,192 tokens

These choices are provisional until the first deployment is validated. Record the exact engine version, model revision, GPU, launch command, and context length with every result.

## Repository Structure

```text
inference-engineering/
├── service/          # API-facing service code and shared runtime configuration
├── deploy/
│   ├── runpod/       # RunPod setup and launch instructions
│   ├── modal/        # Serverless development and smoke testing
│   ├── docker/       # Container-based deployment
│   └── kubernetes/   # Helm, manifests, and autoscaling configuration
├── observability/    # Prometheus, Grafana, dashboards, and metric definitions
├── benchmarks/       # Workloads, sweep scripts, raw results, and reports
├── variants/
│   ├── sglang/       # SGLang deployment using the baseline model
│   ├── fp8/          # FP8 configuration and measurements
│   └── awq/          # Activation-aware Weight Quantization configuration and measurements
├── router/           # Cost, latency, and quality routing with token budgets
└── old/              # Archived experiments retained for reference
```

Directories are populated only when their roadmap stage begins. Avoid adding disconnected demos: each change should extend or measure the same service.

## Working Rules

1. Keep the model, workload, and hardware fixed when comparing engines or configurations.
2. Pin versions and save exact launch commands before recording results.
3. Record exact input and output token lengths.
4. Report p50, p95, and p99 latency, keeping TTFT separate from TPOT.
5. Sweep request rate or concurrency rather than reporting one arbitrary load point.
6. Do not publish measurements taken while the server is preempting requests.
7. Keep credentials out of the repository. Runtime secrets must be injected through the environment.

## Roadmap Relationship

- **Learning plan and progress:** [crypticseeds/time-to-first-token](https://github.com/crypticseeds/time-to-first-token)
- **Implementation and benchmark:** this repository
- **Benchmark workspace:** [`benchmarks/`](./benchmarks/README.md)
- **Final result:** the roadmap will link back to the published benchmark in this repository

## Status

The repository is being reorganized around the 10-week roadmap. Earlier standalone vLLM pip and Docker notes are preserved in [`old/`](./old/README.md) and are not the current source of truth.

## License

[MIT](./LICENSE)
