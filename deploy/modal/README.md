# Modal: Qwen3-8B with vLLM on one A10

Runbook for serving `Qwen/Qwen3-8B` behind vLLM's OpenAI-compatible API on one Modal A10 GPU (24 GB), defined in [`vllm_qwen3_8b.py`](./vllm_qwen3_8b.py).

The app exposes:

```text
GET  /health
GET  /v1/models
POST /v1/chat/completions
GET  /metrics            # Prometheus metrics, scraped by observability/
```

> **Warning: the endpoint is unauthenticated.** The app sets `unauthenticated=True` and vLLM runs without `--api-key`, so anyone with the URL can send requests and spend your GPU budget. Use it only while you are actively running `modal run` or `modal serve`, do not share the URL, and never commit it (use the `MODAL_HOST` placeholder). Do not run `modal deploy` on this file. A persistent endpoint needs a separate authenticated configuration.

## Configuration

| Setting | Value | Reason |
|---|---|---|
| Model | `Qwen/Qwen3-8B` | 8B dense model that fits one 24 GB GPU in BF16 |
| Model revision | `b968826d9c46dd6066d109eabc6255188de91218` | Pinned so upstream model changes cannot alter results |
| vLLM | `0.21.0` | Pinned; matches Modal's official vLLM example at the time of writing |
| Base image | `nvidia/cuda:12.9.0-devel-ubuntu22.04`, Python 3.12 | CUDA userspace for the vLLM wheel |
| GPU profile | `GPU_PROFILE=A10` (default) or `H100` | Selects the GPU and the engine limits below |
| Precision | BF16 (`--dtype bfloat16`) | Qwen3-8B's native precision; weights take about 16 GB |
| Maximum model length | 8,192 tokens | Leaves GPU memory for KV cache and runtime overhead |
| Maximum sequences | A10: 16, H100: 256 | A10 keeps the conservative limit of the November 2026 sweeps; H100 admits more running requests for the concurrency sweep |
| CUDA graphs | A10: off (`--enforce-eager`), H100: on | A10 keeps eager mode for faster startup, lower memory pressure and comparability with earlier runs, so its results are eager-mode measurements. H100 uses CUDA graphs, as in production |
| GPU memory utilization | 90% | Reserves most GPU memory while keeping headroom |
| Prefix caching | On, with cached tokens reported in `usage.prompt_tokens_details` | Matches the SGLang launcher, where prefix caching is the default |
| Maximum containers | 1 | Prevents accidental scale-out |
| Minimum containers | 0 | No warm container is billed while idle |
| Idle scale-down | 120 seconds | Stops idle GPU billing quickly |
| Startup timeout | 15 minutes | Covers first-run image build, weight download and KV cache initialization |
| Volumes | `inference-engineering-huggingface-cache`, `inference-engineering-vllm-cache` | Weights and vLLM compile cache persist between runs |

## Cost controls

Before the first run:

1. Open Modal **Settings -> Usage & Billing** and set a workspace usage budget.
2. Confirm the remaining credits and the budget in the dashboard.

At runtime the app is capped at one GPU container and scales to zero after 120 idle seconds. Anything that sends requests keeps it awake, including the Prometheus scrape in [`observability/`](../../observability/README.md) (every 5 s). Stop the scraper when you stop the app.

## Prerequisites

From the repository root:

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install "modal>=1.3,<2"
modal setup
modal --version
modal token info
```

The model is public, so no Hugging Face token is needed.

## Deploy and verify

```bash
modal run deploy/modal/vllm_qwen3_8b.py
```

This starts vLLM inside one GPU container and checks it over loopback:

1. Polls `/health` until it returns 200, and fails early if vLLM exits.
2. Calls `/v1/models` and asserts `qwen3-8b` is served.
3. Sends one streaming chat request (`max_tokens: 64`, `temperature: 0`, thinking disabled via `chat_template_kwargs`) and asserts a non-empty reply that ends with `data: [DONE]`.

The [SGLang launcher](./sglang_qwen3_8b.py) has the same layout and runs the same checks. Prefix `GPU_PROFILE=H100` to either command to check the H100 profile. The command exits non-zero if any check fails. Modal stops the app when the command finishes; the weight and compile caches remain in the Volumes.

The first run is slow: it builds the CUDA and vLLM image, downloads about 16 GB of weights, then loads the model and initializes the KV cache. Later runs reuse the image layers and Volumes.

## Serve for load tests and dashboards

`modal run` exits after its checks. To keep the endpoint up for load generation or Grafana:

```bash
modal serve deploy/modal/vllm_qwen3_8b.py
```

Keep the terminal open and note the printed `https://...` host. Then check it from another terminal:

```bash
export MODAL_HOST=<host printed by modal serve>
curl -sf "https://$MODAL_HOST/health" && echo healthy
curl -s "https://$MODAL_HOST/v1/models"
curl -s "https://$MODAL_HOST/metrics" | grep -E '^vllm:(num_requests_running|kv_cache_usage_perc)'
```

Point Prometheus and the load tools at that host by following [`observability/README.md`](../../observability/README.md). Results from a run against this app are in [`benchmarks/2026-10-18-modal-a10-qwen3-8b/`](../../benchmarks/2026-10-18-modal-a10-qwen3-8b/README.md).

## Teardown

1. Stop load generators and the observability stack (`docker compose down` in `observability/`), so nothing keeps the container awake.
2. Press Ctrl-C in the `modal serve` terminal.
3. Confirm nothing is still running:

   ```bash
   modal app list
   modal app stop inference-engineering-qwen3-8b   # only if it is still listed as running
   ```

4. Restore placeholder files if you substituted the host: `git checkout -- observability/prometheus.yaml observability/slo-bench-modal.yaml`.

The Volumes are kept on purpose so the next start skips the weight download. List them with `modal volume list`.

## Troubleshooting

### Image build or model download takes several minutes

Expected on the first run. Modal caches image layers after a successful build, and the Hugging Face cache at `/root/.cache/huggingface` lives on a Modal Volume.

### CUDA out of memory

Lower context and batch capacity in `vllm_qwen3_8b.py`, one setting at a time:

```text
--max-model-len 4096
--max-num-seqs 8
--gpu-memory-utilization 0.85
```

### Health check times out

Read the streamed Modal logs above the error. Common causes are image build failure, model download failure, vLLM startup failure, or insufficient GPU memory.

### HTTP 503

The server is still starting or has scaled to zero. Under `modal serve`, wait and retry.
