# Plan: live benchmark observability, launcher cleanup and H100 sweep

Branch: `feat/live-benchmark-observability`, stacked on `docs/professionalize-repo` (PR #2).

## Goals

1. Watch vLLM and SGLang server metrics live in local Grafana while a benchmark runs, side by side.
2. Make the two Modal launchers consistent: same structure, shutdown, smoke test, constants and comments.
3. Run the benchmark client in a Modal CPU container next to the GPU, so the Pi only collects results.
4. Repeat the A10 six-rate GuideLLM sweep with both engines live.
5. Prepare and run one H100 GuideLLM concurrency sweep up to 1,000 concurrent requests within budget.

## Decisions to confirm

| # | Decision | Recommendation |
|---|---|---|
| D1 | Shared launcher code | Keep each launcher self-contained and runnable alone, with identical structure. A shared module needs Modal's `add_local_python_source` and couples the files. |
| D2 | GPU selection | One launcher per engine with a `GPU` profile (`A10` or `H100`) chosen by an environment variable at `modal serve` time. Each profile pins its own engine limits. |
| D3 | H100 engine limits | Remove `--enforce-eager` / `--disable-cuda-graph` on H100 (CUDA graphs on, as in production). Raise the running-request limit to 256 on both engines. Requests past 256 queue, which the dashboard shows. |
| D4 | Concurrent A10 run | Start both engines on separate A10s at once and drive them with two clients in parallel. Doubles GPU cost for the sweep window but gives the side-by-side live view. |
| D5 | Client tool | GuideLLM 0.8.1 for both runs. Check whether NVIDIA has superseded GenAI-Perf with AIPerf before adding either. |

## Tasks

### T1. Launcher consistency

- Same layout in both files: constants, image, volumes, `start_server`, `stop_server`, `Server` class, `smoke_test`.
- Shutdown terminates, waits, then kills on timeout in both.
- Smoke test runs in-container against loopback in both, and checks health, model list and a streamed chat reply.
- Both pass `bfloat16` explicitly and state that prefix caching is on and how cached tokens are reported.

Acceptance: `python -m py_compile` passes on both files. A diff of the two launchers shows only engine-specific flags and names. `modal run` smoke test passes on A10 for both engines.

### T2. Combined live dashboard

- One Grafana dashboard with vLLM and SGLang panels in matching rows: server p95 TTFT, request throughput, output tokens/s, running requests, queued requests, KV cache usage.
- Prometheus scrapes both targets from a private config outside the repo.

Acceptance: Grafana provisions the dashboard without errors. Both targets report UP. Every panel shows data during the smoke traffic.

### T3. Modal-hosted GuideLLM client

- A Modal app that runs a GuideLLM rate or concurrency sweep against a given endpoint and writes JSON reports to a Modal Volume. Pinned to the GPU's region.
- Reports are copied to the Pi with `modal volume get` and redacted before commit.

Acceptance: a one-stage sweep at 1 req/s against an A10 completes with zero errors and a report on the Pi.

### T4. A10 live sweep

- Same workload and six rates (1, 2, 4, 8, 16, 32 req/s) as the November 24 runs, both engines at once.
- Screenshots of the live dashboard at each stage; results written up next to the earlier runs.

Acceptance: six reports per engine, zero request errors outside the expected incomplete-at-cutoff counts, and screenshots saved in the run directory.

### T5. H100 concurrency sweep

- Dry run on H100 with one low concurrency point to confirm config, client and scrape.
- Sweep: 1, 10, 50, 100, 250, 500, 1,000 concurrent, 60 s per point plus warm-up, one engine at a time.
- Budget guard: stop Prometheus scraping and the apps right after the sweep.

Acceptance: total Modal spend for T5 under 8 USD, seven reports per engine, live screenshots.

## Out of scope

- Grafana Cloud export.
- Live client-side metrics (GuideLLM has no metrics exporter).
- The shared-prefix TTFT experiment (Week 4 buffer).
