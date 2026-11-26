# vLLM on RunPod — Installation & Testing Guide

This guide walks you through installing vLLM on a RunPod GPU pod and verifying it works, end to end. It covers the known PyTorch/CUDA version mismatch issue in the default RunPod image and shows how to test using both curl and Postman.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Create & Connect to Your RunPod](#2-create--connect-to-your-runpod)
3. [Verify Your GPU](#3-verify-your-gpu)
4. [Critical: Fix the PyTorch/CUDA Version Mismatch](#4-critical-fix-the-pytorchcuda-version-mismatch)
5. [Set Up Your Environment](#5-set-up-your-environment)
6. [Install vLLM](#6-install-vllm)
7. [Start the vLLM Server](#7-start-the-vllm-server)
8. [Testing with curl](#8-testing-with-curl)
9. [Testing with Postman](#9-testing-with-postman)
10. [Monitor GPU During Inference](#10-monitor-gpu-during-inference)

---

## 1. Prerequisites

- A [RunPod](https://runpod.io) account with billing set up
- An SSH public key added to RunPod (Settings → SSH Public Keys)
- A [Hugging Face](https://huggingface.co) account and access token (for model downloads)

---

## 2. Create & Connect to Your RunPod

### Create the pod

1. Log in to RunPod → **Pods** → **+ Deploy**
2. Select a GPU — RTX 4090 (24 GB) is recommended for 7B models
3. Under **Container Image**, use:
   ```
   runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404
   ```
   > **Important:** This image has the correct CUDA version (12.8.1) but ships with a PyTorch build that is incompatible with it. You must update PyTorch before installing vLLM — see [Section 4](#4-critical-fix-the-pytorchcuda-version-mismatch).
4. Set **Container Disk** to at least **30 GB**
5. Set **Volume Disk** to **50 GB**, mounted at `/workspace` (persists across restarts)
6. Under **Expose HTTP Ports**, add `8000`
7. Click **Deploy**

### Connect via SSH

Once the pod is Running:

1. Click **Connect** on your pod
2. Copy the SSH command (e.g. `ssh root@<ip> -p <port>`)
3. Run it from your terminal:

```bash
ssh root@<ip> -p <port>
```

---

## 3. Verify Your GPU

Once connected, confirm the GPU is visible:

```bash
nvidia-smi
```

Expected output:

```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 550.90.07    Driver Version: 550.90.07    CUDA Version: 12.4    |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  NVIDIA RTX 4090     Off  | 00000000:01:00.0 Off |                  Off |
| 30%   35C    P8    20W / 450W |    500MiB / 24576MiB |      0%      Default |
+-----------------------------------------------------------------------------+
```

> **[IMAGE PLACEHOLDER: nvidia-smi output showing GPU name, CUDA version, memory usage, and GPU utilization]**

Key things to check:
- **CUDA Version** — must be 12.x for this guide
- **Memory-Usage** — should be near zero (idle)
- **GPU-Util** — should be 0% when idle

---

## 4. Critical: Fix the PyTorch/CUDA Version Mismatch

> **Read this before installing anything.**

The RunPod image `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` ships with a PyTorch build that is **not compatible** with its CUDA version. If you use that image (or encounter this on any pod), you will get errors like:

```
ImportError: libcudart.so.13: cannot open shared object file
```

This happens because vLLM v0.21.0+ defaults to a CUDA 13 wheel, but the system only has CUDA 12.

### Check what you have

```bash
nvidia-smi | head -4      # check driver and CUDA version
nvcc --version            # check toolkit version (if available)
python -c "import torch; print(torch.version.cuda)"  # check PyTorch's CUDA
```

> **[IMAGE PLACEHOLDER: terminal showing nvidia-smi CUDA version vs torch.version.cuda mismatch]**

### Fix: Update PyTorch to match your CUDA version

```bash
# Activate your venv first (see Section 5)
source /workspace/vllm-env/bin/activate

# Uninstall the incompatible PyTorch
pip uninstall torch torchvision torchaudio -y

# Install PyTorch built for CUDA 12.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Verify PyTorch now sees the GPU
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
# Expected: True 12.8
```

### Fix: Install the correct vLLM wheel (CUDA 12)

If `uv pip install vllm` pulled a CUDA 13 wheel, wipe the environment and reinstall:

```bash
deactivate
rm -rf /workspace/vllm-env

uv venv --python 3.12 --seed /workspace/vllm-env
source /workspace/vllm-env/bin/activate

# For x86_64 with CUDA 12 — get the wheel URL from:
# https://github.com/vllm-project/vllm/releases/tag/v0.20.2
# Look for a filename containing "cu129" for your architecture
uv pip install <WHEEL_URL>
```

### Rule of thumb

| System CUDA | vLLM install flag |
|---|---|
| CUDA 12.x | `--torch-backend=cu129` or use cu12 wheel directly |
| CUDA 13.x | `--torch-backend=auto` |

Always run `nvidia-smi` before installing vLLM on any new machine.

---

## 5. Set Up Your Environment

### Create a virtual environment

```bash
cd /workspace

# Install uv (fast package manager recommended by vLLM)
pip install uv

# Create a virtual environment
uv venv --python 3.12 --seed vllm-env

# Activate it
source /workspace/vllm-env/bin/activate

# Verify
python --version   # Should show Python 3.12.x
```

### Set your Hugging Face token

```bash
export HF_TOKEN="hf_your_token_here"
echo 'export HF_TOKEN="hf_your_token_here"' >> ~/.bashrc
```

### Point the model cache to your persistent volume

```bash
export HF_HOME="/workspace/hf_cache"
mkdir -p /workspace/hf_cache
echo 'export HF_HOME="/workspace/hf_cache"' >> ~/.bashrc
```

This prevents re-downloading models (7–14 GB each) every time your pod restarts.

---

## 6. Install vLLM

With your virtual environment active:

```bash
uv pip install vllm --torch-backend=auto
```

> If this fails with a `libcudart.so.13` error, follow [Section 4](#4-critical-fix-the-pytorchcuda-version-mismatch).

### Verify the installation

```bash
python -c "import vllm; print(vllm.__version__)"
vllm --version
```

Both should print the same version number.

---

## 7. Start the vLLM Server

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype auto \
  --gpu-memory-utilization 0.90 \
  --max-model-len 8192 \
  --served-model-name qwen2.5-7b
```

The server is ready when you see a line like:

```
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

This takes 2–5 minutes on first run (model download + load). Subsequent starts are faster.

> **[IMAGE PLACEHOLDER: terminal showing vLLM startup logs with model loading progress]**

### Keep the server running after SSH disconnect

```bash
# Option 1: tmux (recommended)
tmux new -s vllm
# run the serve command inside tmux
# detach with Ctrl+B then D

# Option 2: nohup
nohup vllm serve Qwen/Qwen2.5-7B-Instruct \
  --host 0.0.0.0 --port 8000 --dtype auto \
  --gpu-memory-utilization 0.90 --max-model-len 8192 \
  --served-model-name qwen2.5-7b > /workspace/vllm.log 2>&1 &
```

---

## 8. Testing with curl

Your base URL is the RunPod proxy URL:

```
https://<pod-id>-8000.proxy.runpod.net
```

Find it in the RunPod dashboard → your pod → **Connect** → HTTP Service.

### Health check

```bash
curl https://<pod-id>-8000.proxy.runpod.net/health
```

Returns `200 OK` when vLLM is ready. If you get `502`, the server is still loading — wait 30 seconds and retry.

### List loaded models

```bash
curl https://<pod-id>-8000.proxy.runpod.net/v1/models
```

Expected response:

```json
{
  "object": "list",
  "data": [
    {
      "id": "qwen2.5-7b",
      "object": "model",
      "created": 1234567890,
      "owned_by": "vllm"
    }
  ]
}
```

Use the `id` value as your `model` field in all subsequent requests.

### Chat completions

```bash
curl https://<pod-id>-8000.proxy.runpod.net/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-7b",
    "messages": [
      { "role": "user", "content": "What is the capital of France?" }
    ],
    "max_tokens": 128,
    "temperature": 0.7
  }'
```

Expected response:

```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "model": "qwen2.5-7b",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "The capital of France is Paris."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 15,
    "completion_tokens": 9,
    "total_tokens": 24
  }
}
```

### Text completions (non-chat)

```bash
curl https://<pod-id>-8000.proxy.runpod.net/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-7b",
    "prompt": "The capital of France is",
    "max_tokens": 64
  }'
```

### Streaming response

```bash
curl https://<pod-id>-8000.proxy.runpod.net/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-7b",
    "messages": [{ "role": "user", "content": "Count from 1 to 5." }],
    "max_tokens": 64,
    "stream": true
  }'
```

Tokens arrive as `data: {...}` lines as they are generated, ending with `data: [DONE]`.

---

## 9. Testing with Postman

### Set up a collection

1. In Postman, click **New Collection** → name it **vLLM RunPod**
2. Go to **Variables** tab and add:

   | Variable | Initial Value |
   |---|---|
   | `base_url` | `https://<pod-id>-8000.proxy.runpod.net` |
   | `model` | `qwen2.5-7b` |

3. Save the collection. Use `{{base_url}}` and `{{model}}` in all requests — update only the variable when you switch pods.

> **[IMAGE PLACEHOLDER: Postman collection variables panel with base_url and model set]**

### Request 1 — Health check

- **Method**: `GET`
- **URL**: `{{base_url}}/health`
- Click **Send** — expect `200 OK`

### Request 2 — List models

- **Method**: `GET`
- **URL**: `{{base_url}}/v1/models`
- Click **Send** — confirm your model name appears in the response

> **[IMAGE PLACEHOLDER: Postman showing GET /v1/models response with model id]**

### Request 3 — Chat completions

- **Method**: `POST`
- **URL**: `{{base_url}}/v1/chat/completions`
- **Headers**: `Content-Type: application/json`
- **Body** (raw JSON):

```json
{
  "model": "{{model}}",
  "messages": [
    { "role": "user", "content": "Hello, what can you do?" }
  ],
  "max_tokens": 256,
  "temperature": 0.7
}
```

> **[IMAGE PLACEHOLDER: Postman showing POST /v1/chat/completions with JSON body and 200 response]**

### Request 4 — Text completions

- **Method**: `POST`
- **URL**: `{{base_url}}/v1/completions`
- **Body** (raw JSON):

```json
{
  "model": "{{model}}",
  "prompt": "The capital of France is",
  "max_tokens": 64
}
```

### Authentication (if you started vLLM with `--api-key`)

Add this header to all requests:

```
Authorization: Bearer <your-api-key>
```

---

## 10. Monitor GPU During Inference

While vLLM is serving requests, open a second SSH session and run:

```bash
watch -n 2 nvidia-smi
```

This refreshes every 2 seconds. During an active inference request you should see:

- **GPU-Util** spike to 80–100%
- **Memory-Usage** increase as the KV cache fills

> **[IMAGE PLACEHOLDER: nvidia-smi showing high GPU utilization during inference with memory usage near the configured limit]**

For a compact view (memory and utilization only):

```bash
watch -n 2 "nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv"
```

> **[IMAGE PLACEHOLDER: compact nvidia-smi CSV output showing memory used vs total and GPU % utilization]**

For CPU and system metrics:

```bash
htop
```

> **[IMAGE PLACEHOLDER: htop output showing CPU core utilization and RAM usage while vLLM is running]**

### What to watch for

| Metric | Healthy | Warning |
|---|---|---|
| GPU-Util at idle | 0% | >5% (something is leaking) |
| GPU-Util during inference | 80–100% | <50% (throughput bottleneck) |
| Memory-Usage | Stable after model load | Growing continuously (memory leak) |
| CPU usage | Low (10–30%) | Sustained 100% (CPU bottleneck) |

---

## Quick Reference

| Task | Command |
|---|---|
| Check CUDA version | `nvidia-smi \| head -4` |
| Check PyTorch CUDA | `python -c "import torch; print(torch.version.cuda)"` |
| Activate venv | `source /workspace/vllm-env/bin/activate` |
| Start vLLM | `vllm serve Qwen/Qwen2.5-7B-Instruct --host 0.0.0.0 --port 8000 --dtype auto --gpu-memory-utilization 0.90 --max-model-len 8192 --served-model-name qwen2.5-7b` |
| Health check | `curl https://<pod-id>-8000.proxy.runpod.net/health` |
| List models | `curl https://<pod-id>-8000.proxy.runpod.net/v1/models` |
| Watch GPU | `watch -n 2 nvidia-smi` |

---

## Common Issues

| Symptom | Likely Cause | Fix |
|---|---|---|
| `ImportError: libcudart.so.13` | vLLM CUDA 13 wheel on CUDA 12 system | See [Section 4](#4-critical-fix-the-pytorchcuda-version-mismatch) |
| `502 Bad Gateway` from proxy | vLLM still loading | Wait and retry `/health` |
| `404 model not found` | Wrong model name in request | Check `/v1/models` for the correct id |
| `CUDA out of memory` | `--gpu-memory-utilization` too high | Lower to `0.85` and restart |
| Port not accessible | Port 8000 not exposed on pod | Add port 8000 in RunPod dashboard |

For detailed error explanations, see [troubleshooting.md](./troubleshooting.md).
