# Modal: Qwen3-8B with vLLM

This runbook deploys `Qwen/Qwen3-8B` behind vLLM's OpenAI-compatible Application Programming Interface (API) on one Modal A10 Graphics Processing Unit (GPU).

Start with an ephemeral `modal run`. Do not create a persistent deployment until the health and chat tests pass.

## What a Successful Test Will Verify

- Modal can build the vLLM environment.
- Qwen3-8B fits on one 24-gigabyte A10 at an 8,192-token context limit.
- vLLM exposes `/health`, `/v1/models`, and `/v1/chat/completions`.
- The endpoint accepts the OpenAI chat request format.

It does not produce a performance baseline. The smoke test uses eager execution to reduce startup complexity and memory pressure.

## Cost Controls

Before running the app:

1. Open Modal **Settings -> Usage & Billing**.
2. Set a Workspace usage budget you are comfortable with.
3. Confirm the Starter credits and current budget in the dashboard.

The app requests one A10, allows only one container, keeps no warm container, and scales to zero after 120 idle seconds.

## Prerequisites

From the repository root:

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install "modal>=1.3,<2"
modal setup
```

If `uv` is not installed, follow the installation instructions at
[docs.astral.sh/uv](https://docs.astral.sh/uv/getting-started/installation/),
then rerun the commands above.

Verify installation and authentication:

```bash
modal --version
modal token info
```

The model is public, so this smoke test does not require a Hugging Face token.

## Run the Ephemeral Test

```bash
modal run deploy/modal/vllm_qwen3_8b.py
```

The first run has three slow stages:

1. Build the CUDA and vLLM image.
2. Download approximately 16 gigabytes of model weights.
3. Load the model and initialize its Key-Value (KV) cache.

The command waits for `/health`, lists the served models, then sends one chat request. A successful run ends with a response from `qwen3-8b`.

The app is ephemeral: Modal stops it after the local test finishes. The model and vLLM caches remain in Modal Volumes for faster later starts.

## Configuration Choices

| Setting | Value | Reason |
|---|---|---|
| Model | `Qwen/Qwen3-8B` | Baseline model selected for the roadmap |
| Model revision | Pinned commit | Prevent upstream model changes from altering results |
| vLLM | `0.21.0` | Pinned version from Modal's current official example |
| GPU | A10, 24 GB | Low-cost Modal GPU that can hold the BF16 model |
| Maximum model length | 8,192 tokens | Leaves GPU memory for KV cache and runtime overhead |
| Maximum sequences | 16 | Conservative smoke-test concurrency |
| GPU memory utilization | 90% | Reserves most GPU memory while retaining headroom |
| Eager execution | Enabled | Faster, simpler first validation; not the final benchmark mode |
| Maximum containers | 1 | Prevents accidental scale-out during testing |
| Idle scale-down | 120 seconds | Stops idle GPU billing quickly |

## Expected Checks

The script checks these routes:

```text
GET  /health
GET  /v1/models
POST /v1/chat/completions
```

The chat request disables Qwen's thinking mode so output length is predictable during this smoke test.

## Common Problems

### Image build takes several minutes

Expected on the first run. Modal caches image layers after a successful build.

### Model download takes several minutes

Expected on the first run. The Hugging Face cache is mounted at `/root/.cache/huggingface` using a Modal Volume.

### CUDA out of memory

Lower the context and request capacity in `vllm_qwen3_8b.py`:

```text
--max-model-len 4096
--max-num-seqs 8
--gpu-memory-utilization 0.85
```

Change one setting at a time and keep the working command.

### Health check times out

Inspect the streamed Modal logs above the error. Common causes are image build failure, model download failure, vLLM startup failure, or insufficient GPU memory.

### HTTP 503

The server is still starting or has scaled to zero. The test retries `503 Service Unavailable` responses until the startup deadline.

## After the Smoke Test

Once the temporary run succeeds:

1. Record the vLLM version, model revision, GPU, startup time, and launch command.
2. Remove `--enforce-eager` before collecting performance measurements.
3. Repeat the health and chat tests after changing the execution mode.
4. Do not deploy this file persistently: it intentionally uses an unauthenticated endpoint for a temporary `modal run` smoke test.
5. Create a separate authenticated deployment configuration if a persistent endpoint is needed later.
