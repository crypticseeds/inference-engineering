# Phase 2: Running vLLM via Docker on RunPod — The Definitive Guide

> **Series:** MLOps for Inference — Phase 2 of 8
> **Builds on:** Phase 1 (vLLM via pip on RunPod)
> **Goal:** Understand how Docker really works on RunPod, why Docker-in-Docker fails, and the correct way to run vLLM in a container on RunPod.

---

## The Core Problem — Why Docker-in-Docker Fails on RunPod

This trips up everyone. Here's what's actually happening.

**When you rent a RunPod pod, you are already inside a Docker container.**

RunPod's entire platform is Docker. Every pod you launch — whether you chose the PyTorch template, a Jupyter template, or anything else — is a Docker container running on RunPod's host machines. The architecture looks like this:

```
RunPod Host Machine (bare metal)
└── Docker daemon (running on the host)
    └── Your Pod (a Docker container)
        └── You SSH in here ← this is where you land
```

When you SSH into your pod and try to run `docker run ...`, you're trying to start a Docker container **inside** a Docker container. This requires:

1. The Docker daemon to be running inside your container
2. `/var/run/docker.sock` to exist and be accessible
3. Privileged mode or specific kernel capabilities

RunPod does **not** give you any of these by default because:

- Running a full Docker daemon inside a container requires `--privileged` mode
- `--privileged` is a major security risk on shared GPU infrastructure
- RunPod doesn't expose the host Docker socket to your containers

**The errors you'll see** are typically one of:

```
Cannot connect to the Docker daemon at unix:///var/run/docker.sock.
Is the docker daemon running?
```

```
docker: Error response from daemon: authorization denied
```

Or simply nothing — because `docker` isn't even installed inside the pod.

### The right mental model

Even though you can't run `docker run` inside a RunPod pod, Docker is still central to Phase 2. The shift in thinking you need is:

> On RunPod, **you are not the Docker runtime — RunPod is.** Your job is to tell RunPod which image to launch. RunPod handles the `docker run`.

This is exactly how production systems work. In Kubernetes, ECS, or any container scheduler, you never call `docker run` yourself — the scheduler does it on your behalf. RunPod is the same pattern, without the orchestration overhead.

---

## GPU Compatibility — Read This First

The latest vLLM Docker images require **CUDA 12.9 or higher**. Not all RunPod GPUs meet this requirement.

> For the compatibility table and detailed explanation, see [troubleshooting.md](./troubleshooting.md).

**Quick reference:**

| CUDA Version | vLLM Docker | Recommended GPUs |
|---|---|---|
| 13.0 | Yes | L40S, RTX PRO 4500, RTX PRO 4000 |
| 12.8 | No | A40, L4, RTX 5090 |
| 12.7 | No | RTX 4090 |

**Always select a CUDA 13.0 GPU when deploying the vLLM Docker image on RunPod.** If you can only access a CUDA 12.x GPU, use the pip/uv installation method from Phase 1 instead.

---

## Deploying vLLM — Launch the Official Docker Image as a RunPod Pod

The correct way to run vLLM via Docker on RunPod is to tell RunPod to launch the `vllm/vllm-openai` image as the pod itself — not to spin up a generic pod and try to run Docker inside it.

**How it works:**

```
RunPod Host
└── Docker daemon
    └── vllm/vllm-openai:latest  ← This IS your pod
        └── vLLM server starts automatically on container boot
```

**Step 1 — Go to RunPod → Pods → + New Pod**

**Step 2 — Click "Custom" instead of choosing a template**

**Step 3 — Enter the container image:**

```
vllm/vllm-openai:latest
```

Pinning a specific version is strongly recommended for reproducibility:

```
vllm/vllm-openai:v0.21.0
```

**Step 4 — Set environment variables:**

```
HF_TOKEN=hf_yourtoken
HUGGING_FACE_HUB_TOKEN=hf_yourtoken
```

**Step 5 — Set the container start command** (choose based on your GPU VRAM):

> **Note:** Do not include `vllm serve` — the `vllm/vllm-openai` image already sets that as its entrypoint. You only need to pass the model name and flags.

*Mistral-7B-Instruct — requires ~16 GB VRAM:*

```
mistralai/Mistral-7B-Instruct-v0.2 \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype auto \
  --gpu-memory-utilization 0.90 \
  --max-model-len 8192
```

*Qwen3-8B — requires ~18 GB VRAM:*

```
Qwen/Qwen3-8B \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype auto \
  --enforce-eager \
  --gpu-memory-utilization 0.90 \
  --max-model-len 8128 \
  --max-num-seqs 256
```

**Step 6 — Expose port `8000` as HTTP**

**Step 7 — Set GPU and storage:**

- GPU: L40S (48 GB) — recommended. RTX PRO 4500 or RTX PRO 4000 also work. All are CUDA 13.0.
- Container Disk: 30 GB minimum (for image layers and model cache)
- Network Volume: mount at `/root/.cache/huggingface` so downloaded models persist across pod restarts — without this, the model re-downloads on every restart

**Step 8 — Deploy the pod.**

The container boots with vLLM pre-installed. On first boot the model downloads to the HF cache; subsequent boots use the cached version if a Network Volume is attached. The server starts automatically and is ready to serve requests within a few minutes.

**Why this counts as "running vLLM via Docker":**

- `vllm/vllm-openai` is an official Docker image published by the vLLM team
- RunPod is running it via Docker — the isolation and reproducibility guarantees are fully in effect
- You're not calling `docker run` — RunPod is, just as Kubernetes or ECS would in production

---

## Testing the Endpoint

Once the pod is running, get your exposed endpoint URL from the RunPod console. It follows the format `https://PODID-8000.proxy.runpod.net`.

### Testing with curl

**Health check:**

```bash
curl https://PODID-8000.proxy.runpod.net/health
```

Expected response: `{"status":"ok"}`

**List loaded models:**

```bash
curl https://PODID-8000.proxy.runpod.net/v1/models
```

**Chat completion:**

```bash
curl https://PODID-8000.proxy.runpod.net/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mistralai/Mistral-7B-Instruct-v0.2",
    "messages": [{"role": "user", "content": "What is PagedAttention?"}],
    "max_tokens": 200
  }'
```

### Testing with Postman

1. Open your Postman collection from Phase 1
2. Go to the collection's **Variables** tab
3. Update `base_url` to your RunPod proxy URL: `https://PODID-8000.proxy.runpod.net`
4. All saved requests (health check, `/v1/models`, `/v1/chat/completions`) now route to the Docker deployment without any other changes needed

---

## Key Takeaways

- **RunPod pods are Docker containers.** You cannot run Docker inside them — RunPod runs Docker for you.
- **Specify the image at pod creation time.** This is how you deploy a Docker image on RunPod.
- **CUDA 13.0 GPUs are required** for the latest vLLM Docker images. Use L40S, RTX PRO 4500, or RTX PRO 4000.
- **Network Volumes prevent expensive re-downloads.** Always mount one at `/root/.cache/huggingface`.
- **This is the production pattern.** Kubernetes, ECS, and every other container scheduler work the same way — the scheduler calls `docker run`, not you.

---

## Next: Phase 3

Phase 3 introduces benchmarking. Now that you have vLLM running both via pip (Phase 1) and Docker (Phase 2), you'll run throughput and latency benchmarks against both deployments and understand what tokens/sec, TTFT (time to first token), and ITL (inter-token latency) actually mean — and how to improve them.
