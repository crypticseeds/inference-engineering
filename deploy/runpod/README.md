# RunPod: vLLM on a GPU pod

Runbook for serving vLLM's OpenAI-compatible API on a RunPod GPU pod. Two methods were used:

- **Method A, pip:** install vLLM with `uv` into a virtual environment on a general-purpose pod and run `vllm serve`.
- **Method B, container image:** launch the official `vllm/vllm-openai` image as the pod's own container.

Both expose the same API on port 8000. Problems and fixes are in [`troubleshooting.md`](./troubleshooting.md). The original step-by-step guides these steps come from are in [`reference/`](./reference/).

## Recorded environment

| Item | Method A (pip) | Method B (image) |
|---|---|---|
| vLLM | TODO(owner): record (`vllm --version`; a CUDA 12 wheel from the v0.20.2 release was installed) | TODO(owner): record the image tag actually run (`v0.21.0` is the recommended pin) |
| GPU | TODO(owner): record | TODO(owner): record |
| Driver / CUDA | TODO(owner): record (`nvidia-smi \| head -4`) | TODO(owner): record |
| Model and revision | TODO(owner): record | TODO(owner): record |
| Cloud tier | TODO(owner): record (Secure or Community) | TODO(owner): record |

## Choose a method: GPU and CUDA compatibility

The pod's NVIDIA driver is set by the RunPod host and cannot be changed from inside the pod. Recent `vllm/vllm-openai` images need CUDA 12.9 or newer, so the driver decides which method works.

Observed on RunPod (driver versions vary between Secure and Community Cloud and change over time):

| GPU | VRAM | Driver | CUDA | Method B (image) | Method A (pip) wheel |
|---|---|---|---|---|---|
| NVIDIA L40S | 48 GB | 580.126.09 | 13.0 | Yes | Default (CUDA 13) |
| NVIDIA RTX PRO 4500 | | 580.126.09 | 13.0 | Yes | Default (CUDA 13) |
| NVIDIA RTX PRO 4000 | | 580.159.04 | 13.0 | Yes | Default (CUDA 13) |
| NVIDIA A40 | 48 GB | 570.211.01 | 12.8 | No | CUDA 12 wheel |
| NVIDIA L4 | | 570.195.03 | 12.8 | No | CUDA 12 wheel |
| NVIDIA GeForce RTX 5090 | | 570.195.03 | 12.8 | No | CUDA 12 wheel |
| NVIDIA GeForce RTX 4090 | 24 GB | 565.57.01 | 12.7 | No | CUDA 12 wheel |

Rule: on a CUDA 13.0 GPU either method works. On a CUDA 12.x GPU use Method A with a CUDA 12 wheel.

## Why not `docker run` inside a pod

A RunPod pod is already a Docker container. Running `docker run` inside it needs a Docker daemon in the container, access to `/var/run/docker.sock`, and privileged mode or extra kernel capabilities. RunPod provides none of these, so attempts fail with one of:

```text
Cannot connect to the Docker daemon at unix:///var/run/docker.sock. Is the docker daemon running?
docker: Error response from daemon: authorization denied
```

or `docker` is not installed at all. To run vLLM in a container on RunPod, give RunPod the image and let it run the container: that is Method B.

## Method A: pip on a pod

### 1. Create the pod

- **Pods -> + Deploy**, pick the GPU.
- Container image: `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` (CUDA 12.8.1). Its bundled PyTorch build is not compatible with its CUDA version, so install into a fresh virtual environment, not the system Python.
- Container disk: at least 30 GB. Volume disk: 50 GB mounted at `/workspace` (persists across restarts).
- Expose HTTP port `8000`.
- Connect with the SSH command from **Connect**.

### 2. Check the driver before installing

```bash
nvidia-smi | head -4      # driver and CUDA version
uname -m                  # architecture, for picking a wheel
```

### 3. Environment and model cache

```bash
cd /workspace
pip install uv
uv venv --python 3.12 --seed /workspace/vllm-env
source /workspace/vllm-env/bin/activate
export HF_HOME=/workspace/hf_cache   # weights survive pod restarts
mkdir -p "$HF_HOME"
```

Set `HF_TOKEN` only for gated models, as a RunPod pod environment variable rather than in a shell file.

### 4. Install vLLM

Since vLLM v0.21.0 the default PyPI wheel targets CUDA 13. Install the build that matches the driver:

| Pod CUDA | Install |
|---|---|
| 13.x | `uv pip install vllm --torch-backend=auto` |
| 12.x | A CUDA 12 wheel from a vLLM release page ([v0.20.2](https://github.com/vllm-project/vllm/releases/tag/v0.20.2) provides `cu129` wheels; pick the filename for your architecture): `uv pip install <WHEEL_URL>` |

Verify:

```bash
python -c "import vllm; print(vllm.__version__)"
vllm --version
```

Pin the working version afterwards: `pip freeze > /workspace/requirements-vllm.txt`.

### 5. Launch

Run inside `tmux new -s vllm` (detach with Ctrl-B D) so the server survives SSH disconnects. Add `--api-key "$VLLM_API_KEY"` whenever the port is exposed (see [Access](#access)).

24 GB GPU (for example RTX 4090):

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --host 0.0.0.0 --port 8000 \
  --dtype auto \
  --gpu-memory-utilization 0.85 \
  --max-model-len 4096 \
  --max-num-seqs 128 \
  --served-model-name qwen2.5-7b
```

48 GB GPU (for example A40 or L40S):

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --host 0.0.0.0 --port 8000 \
  --dtype auto \
  --gpu-memory-utilization 0.90 \
  --max-model-len 8192 \
  --max-num-seqs 256 \
  --served-model-name qwen2.5-7b \
  --enable-prefix-caching
```

The server is ready when the log shows `Uvicorn running on http://0.0.0.0:8000`. The first start includes the weight download.

## Method B: `vllm/vllm-openai` as the pod image

1. **Pods -> + New Pod -> Custom.**
2. Container image: `vllm/vllm-openai:v0.21.0`. Pin a tag; `latest` changes between pulls.
3. Environment: `HF_TOKEN` (and `HUGGING_FACE_HUB_TOKEN`) only for gated models.
4. Container start command: model and flags only. The image entrypoint is already `vllm serve`, so do not repeat it.
5. Expose HTTP port `8000`.
6. GPU: a CUDA 13.0 GPU from the table above (L40S 48 GB, RTX PRO 4500 or RTX PRO 4000).
7. Storage: container disk at least 30 GB; a Network Volume mounted at `/root/.cache/huggingface` so weights are not downloaded on every restart.
8. Deploy. vLLM starts on container boot.

Start command for `Qwen/Qwen3-8B` (about 18 GB of VRAM) on a 48 GB GPU:

```text
Qwen/Qwen3-8B \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype auto \
  --enforce-eager \
  --gpu-memory-utilization 0.90 \
  --max-model-len 8128 \
  --max-num-seqs 256
```

TODO(owner): confirm `--max-model-len 8128` (the value in `reference/02-vllm-docker/runpod.md`) versus 8192.

Start command for `mistralai/Mistral-7B-Instruct-v0.2` (about 16 GB of VRAM):

```text
mistralai/Mistral-7B-Instruct-v0.2 \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype auto \
  --gpu-memory-utilization 0.90 \
  --max-model-len 8192
```

Without `--served-model-name`, the API model id is the Hugging Face path.

## Access

The pod URL is `https://<pod-id>-8000.proxy.runpod.net` (**Connect -> HTTP Service**). It is reachable from the internet, and vLLM accepts any request unless started with `--api-key`. Alternatively tunnel instead of exposing the port:

```bash
ssh root@<ip> -p <ssh-port> -L 8000:localhost:8000 -N
```

## Verify

```bash
BASE=https://<pod-id>-8000.proxy.runpod.net   # or http://localhost:8000 through the tunnel
AUTH="Authorization: Bearer $VLLM_API_KEY"     # omit if started without --api-key

curl -sf -o /dev/null -w '%{http_code}\n' -H "$AUTH" "$BASE/health"   # 200; 502 from the proxy while loading
curl -s -H "$AUTH" "$BASE/v1/models"                                # use the returned id as "model"
curl -s -H "$AUTH" "$BASE/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"model": "<id from /v1/models>", "messages": [{"role": "user", "content": "What is PagedAttention?"}], "max_tokens": 128}'
```

On the pod (Method A, or a web terminal for Method B), watch GPU memory and utilization while requests run:

```bash
watch -n 2 "nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader"
```

Expect memory to rise after model load and stay stable, and utilization to rise toward 80-100% during generation.

## Teardown

Stop the pod when finished; a running pod bills for the GPU whether or not it serves requests. The `/workspace` volume or Network Volume keeps the weights for the next start.
