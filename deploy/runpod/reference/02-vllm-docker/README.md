# Phase 2: Running vLLM via Docker on RunPod — A Complete Guide

> **Series:** MLOps for Inference — Phase 2 of 8  
> **Builds on:** Phase 1 (vLLM via pip). All the flags, concepts, and API knowledge carry forward unchanged.  
> **Goal:** Run the exact same vLLM server inside a Docker container. Understand *why* Docker exists, what it actually does differently from pip, and what every flag in the `docker run` command means. Then graduate to `docker-compose`.

---

## Table of Contents

1. [Why Docker? What problem does it actually solve?](#1-why-docker-what-problem-does-it-actually-solve)
2. [Docker concepts you must understand first](#2-docker-concepts-you-must-understand-first)
3. [What is the vllm/vllm-openai image?](#3-what-is-the-vllmvllm-openai-image)
4. [Setting up the RunPod pod for Docker](#4-setting-up-the-runpod-pod-for-docker)
5. [Your first docker run command — explained completely](#5-your-first-docker-run-command--explained-completely)
6. [Volume mounts deep dive — bind mounts vs named volumes](#6-volume-mounts-deep-dive--bind-mounts-vs-named-volumes)
7. [Shared memory — --ipc=host vs --shm-size](#7-shared-memory----ipchost-vs---shm-size)
8. [GPU passthrough — how Docker sees your GPU](#8-gpu-passthrough--how-docker-sees-your-gpu)
9. [Environment variables in Docker](#9-environment-variables-in-docker)
10. [Useful docker commands for day-to-day work](#10-useful-docker-commands-for-day-to-day-work)
11. [Graduating to docker-compose](#11-graduating-to-docker-compose)
12. [Writing a custom Dockerfile on top of vllm-openai](#12-writing-a-custom-dockerfile-on-top-of-vllm-openai)
13. [Pinning versions — never use :latest in production](#13-pinning-versions--never-use-latest-in-production)
14. [Common Docker + vLLM errors and fixes](#14-common-docker--vllm-errors-and-fixes)
15. [pip vs Docker — a direct comparison](#15-pip-vs-docker--a-direct-comparison)
16. [What you learned — exit checklist](#16-what-you-learned--exit-checklist)

---

## 1. Why Docker? What problem does it actually solve?

In Phase 1 you installed vLLM directly onto a RunPod pod with pip. It worked — but it had problems you may have already felt:

- **It's fragile.** The environment on that specific pod is now unique. If you spin up a new pod, you have to repeat every install step. If a package upgrades automatically, something may break.
- **It's hard to share.** If a colleague wants to reproduce your setup, they need your exact Python version, your exact CUDA version, your exact pip packages, and your exact environment variables. Good luck.
- **It's tied to the machine.** Move to a different host, different OS, different cloud provider — everything needs redoing.

Docker solves all three problems with one idea: **package the entire environment — code, dependencies, OS libraries, configuration — into a single portable unit called an image.**

You ship the image, not the instructions. Anyone who pulls that image gets an identical environment regardless of what machine they're running on.

### The container mental model

Think of it this way:

```
Without Docker:              With Docker:

Your vLLM code               Your vLLM code
    ↕ depends on                 ↕ packaged together in
PyTorch                      PyTorch
    ↕ depends on             CUDA libraries
CUDA libraries               Python 3.12
    ↕ depends on             Ubuntu 22.04 base
OS libraries                 ─────────────────
    ↕ depends on             Container Image (one file)
Host OS
```

The container runs on top of the host's Linux kernel (it shares the kernel — this is not a full VM), but everything else is isolated inside the image. The host OS doesn't need Python, PyTorch, or CUDA toolkit installed. Only the NVIDIA driver needs to be on the host.

### Why this matters for MLOps

Every production inference system you'll encounter runs containers — not bare pip installs. Kubernetes (Phase 5), KServe (Phase 6), and every managed ML platform orchestrate containers, not Python environments. Phase 2 is where you learn the foundations that everything else builds on.

---

## 2. Docker concepts you must understand first

### Image vs Container

An **image** is the blueprint — a read-only, layered snapshot of a filesystem. Think of it like a class definition in code.

A **container** is a running instance of an image — it has the image's filesystem plus a thin writable layer on top for any changes made during its lifetime. Think of it like an object instantiated from a class.

```
Image: vllm/vllm-openai:latest   (blueprint, stored on disk)
    ↓  docker run
Container: vllm-server            (running instance, has its own PID, network, filesystem)
```

You can run many containers from the same image. When a container stops, by default its writable layer is discarded — that's why volume mounts exist.

### Layers

Docker images are built in layers. Each instruction in a Dockerfile adds a layer. Layers are cached and shared between images. This is why pulling `vllm/vllm-openai` is fast after the first time — Docker only downloads layers that have changed.

```
vllm/vllm-openai:latest
├── Layer 1: Ubuntu 22.04 base        (shared with many other images)
├── Layer 2: CUDA 12.4 libraries      (shared with other CUDA images)
├── Layer 3: Python 3.12 + pip        (shared with Python images)
├── Layer 4: PyTorch 2.5              (shared with PyTorch images)
└── Layer 5: vLLM + dependencies      (unique to this image)
```

### Registry

Docker Hub is the default registry — a place to store and share images. `docker pull vllm/vllm-openai:latest` fetches from Docker Hub. In production you'd use a private registry (ECR, GCR, Harbor). For now, Docker Hub is all you need.

### The ENTRYPOINT and CMD

Every Docker image has an ENTRYPOINT — the command that runs when the container starts. For `vllm/vllm-openai`, the entrypoint is the `vllm serve` command itself. Anything you pass after the image name in `docker run` becomes arguments to that entrypoint — which is why you can pass your vLLM flags directly after the image name.

```bash
docker run vllm/vllm-openai:latest --model Qwen/Qwen2.5-7B-Instruct
#                                   ↑ these are passed to: vllm serve
```

---

## 3. What is the vllm/vllm-openai image?

The official vLLM Docker image is published to Docker Hub at `vllm/vllm-openai`. It is built and maintained by the vLLM project team and released alongside every vLLM version.

### What's inside it

- Ubuntu 22.04 base
- CUDA 12.x toolkit and libraries
- Python 3.12
- PyTorch (matching the CUDA version)
- vLLM and all its dependencies (FlashAttention, xFormers, etc.)
- The `vllm serve` entrypoint

### What's NOT inside it

- Model weights (downloaded at runtime or mounted as a volume)
- Your HuggingFace token
- Optional audio/video dependencies (excluded for licensing reasons — you add these in a custom Dockerfile if needed)

### Image size

The image is approximately 9–12 GB. This is large because it contains full CUDA libraries and PyTorch. Pull it once and Docker caches it locally. On RunPod, this pull happens on the pod's network, which is fast.

### Image tags

Tags follow the pattern `vllm/vllm-openai:<version>`:

```bash
vllm/vllm-openai:latest          # Latest release (changes over time — not for production)
vllm/vllm-openai:v0.8.5         # Specific pinned version (use this)
vllm/vllm-openai:v0.8.5-cu124   # Version + specific CUDA version
```

> **Rule:** Never use `:latest` once you have a working configuration. We cover this in section 13.

---

## 4. Setting up the RunPod pod for Docker

For Phase 2 you need a pod that has Docker installed and a GPU available. You have two options on RunPod:

### Option A — Use a RunPod base image that has Docker pre-installed (Recommended)

In the RunPod pod creation UI:
- **Container Image:** `runpod/base:0.6.2-cuda12.4.1` or `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`
- Docker is pre-installed on RunPod GPU pods by default

Verify after SSH:
```bash
docker --version
# Docker version 27.x.x, build ...

docker run --rm hello-world
# Should print: "Hello from Docker!"
```

### Option B — Start from the vllm/vllm-openai image directly as your RunPod template

RunPod also lets you specify `vllm/vllm-openai:latest` as the pod's container image directly. In this mode RunPod launches the image for you — but you lose the ability to experiment with `docker run` yourself. Use Option A for learning; Option B is for when you already know what you're doing.

### Persistent volume

Same as Phase 1 — mount a volume at `/workspace` with at least 50 GB. Your model cache goes there so you don't re-download across pod restarts.

```
/workspace/
├── hf_cache/        ← model weights (HF_HOME)
├── docker-runs/     ← your shell scripts
└── compose/         ← your docker-compose files
```

### Verify the NVIDIA Container Toolkit is present

This is the bridge between Docker and your GPU. On RunPod it's always pre-installed, but verify:

```bash
nvidia-smi                           # Host can see the GPU

docker run --rm --gpus all \
  nvidia/cuda:12.4.1-base-ubuntu22.04 \
  nvidia-smi                         # Container can also see the GPU
```

If the second command works, you're ready. If it fails with "could not select device driver", the NVIDIA Container Toolkit is not configured — on RunPod this should never happen, but on a bare DigitalOcean droplet you'd need to install it:

```bash
# On a bare Ubuntu server (not needed on RunPod):
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor \
  -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt update && sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

---

## 5. Your first docker run command — explained completely

Here is the complete command. Run it and it will pull the image, start the container, and serve vLLM — identical to Phase 1 in terms of the API, but now running inside a container.

```bash
docker run \
  --name vllm-server \
  --runtime nvidia \
  --gpus all \
  --ipc=host \
  -p 8000:8000 \
  -v /workspace/hf_cache:/root/.cache/huggingface \
  -e HF_TOKEN=$HF_TOKEN \
  -e HF_HOME=/root/.cache/huggingface \
  --restart unless-stopped \
  vllm/vllm-openai:latest \
    --model Qwen/Qwen2.5-7B-Instruct \
    --host 0.0.0.0 \
    --port 8000 \
    --dtype auto \
    --gpu-memory-utilization 0.90 \
    --max-model-len 4096 \
    --max-num-seqs 256 \
    --served-model-name qwen2.5-7b
```

### Every flag explained

**`--name vllm-server`**  
Gives the container a human-readable name. Without this, Docker assigns a random name like `epic_goldstine`. With a name you can reference it in other commands: `docker logs vllm-server`, `docker stop vllm-server`, `docker exec vllm-server nvidia-smi`.

**`--runtime nvidia`**  
Tells Docker to use the NVIDIA container runtime instead of the default `runc`. This is what enables GPU passthrough. On newer Docker versions with the NVIDIA Container Toolkit configured, `--gpus all` alone is sufficient and `--runtime nvidia` is implicit — but being explicit never hurts.

**`--gpus all`**  
Passes all GPUs on the host into the container. Inside the container, `nvidia-smi` will show every GPU. If you want to restrict to specific GPUs: `--gpus '"device=0"'` for GPU 0 only, or `--gpus '"device=0,1"'` for GPUs 0 and 1. This matters when you run multiple containers on a multi-GPU host and want to partition the GPUs between them.

**`--ipc=host`**  
Shares the host's IPC (Inter-Process Communication) namespace with the container. This is required because PyTorch uses POSIX shared memory (`/dev/shm`) to transfer tensors between processes — especially for tensor parallelism (multi-GPU inference). Without this, the container gets its own isolated `/dev/shm` with a default size of only 64 MB, which is far too small. We go deep on this in section 7.

**`-p 8000:8000`**  
Port mapping: `HOST_PORT:CONTAINER_PORT`. The container's port 8000 is mapped to the host's port 8000. Traffic hitting port 8000 on the host is forwarded into the container. If port 8000 on the host is already taken, change the left side: `-p 8001:8000` maps container port 8000 to host port 8001. You always query the host port from outside.

**`-v /workspace/hf_cache:/root/.cache/huggingface`**  
Mounts your persistent volume's cache directory into the container at the path where HuggingFace Hub downloads models inside the container. This is a **bind mount** — the host directory and the container directory point to the same filesystem location. When vLLM downloads a model inside the container, it lands on your persistent volume and survives container restarts. We cover this in depth in section 6.

**`-e HF_TOKEN=$HF_TOKEN`**  
Injects the environment variable `HF_TOKEN` into the container from your current shell. The `$HF_TOKEN` is evaluated by your shell — so it must be set in your host session: `export HF_TOKEN="hf_xxx"`. Inside the container, vLLM uses this to authenticate downloads of gated models.

**`-e HF_HOME=/root/.cache/huggingface`**  
Tells the HuggingFace library inside the container where to store and look for models. Must match the container-side path in your volume mount.

**`--restart unless-stopped`**  
If the container crashes or the host reboots, Docker automatically restarts it — unless you explicitly stopped it with `docker stop`. Options are: `no` (default, never restart), `always` (restart even after explicit stop), `on-failure` (restart on non-zero exit code), `unless-stopped` (restart always except after explicit stop). For a server process, `unless-stopped` is the right choice.

**`vllm/vllm-openai:latest`**  
The image to run. Everything before this line is Docker configuration. Everything after this line is passed to the image's ENTRYPOINT (`vllm serve`) as arguments.

**`--model` through `--served-model-name`**  
These are your vLLM flags — identical to Phase 1. They are passed directly to `vllm serve` inside the container. All flags from Phase 1's section 9 work here.

### Running it detached (background)

The command above runs in the foreground — your terminal is attached to the container's stdout. To run it in the background, add `-d`:

```bash
docker run -d \
  --name vllm-server \
  ...
```

The container starts in the background and you get your terminal back immediately. Check its output with `docker logs vllm-server -f`.

---

## 6. Volume mounts deep dive — bind mounts vs named volumes

This is one of the most important concepts in Docker for MLOps. Model weights are large and expensive to re-download. You need them to survive container restarts.

Docker provides two ways to persist data:

### Bind mounts

A bind mount maps a specific path on the host to a path in the container. The host controls the data.

```bash
-v /workspace/hf_cache:/root/.cache/huggingface
#  ↑ host path         ↑ container path
```

- The host directory must exist (or Docker creates an empty one)
- If the host directory is empty, it **overwrites** the container path with the empty directory — this can delete files the image had there
- Changes from either side are immediately visible to the other
- You can browse, backup, and manage the files directly from the host: `ls /workspace/hf_cache/`
- **Best for:** model weight caches, config files you want to edit on the host, log directories

```bash
# Create the directory first — always
mkdir -p /workspace/hf_cache

# Then mount it
-v /workspace/hf_cache:/root/.cache/huggingface
```

### Named volumes

A named volume is managed by Docker. Docker stores the data in `/var/lib/docker/volumes/<name>/_data` on the host, but you reference it by name.

```bash
# In docker run:
-v hf-model-cache:/root/.cache/huggingface

# In docker-compose.yml:
volumes:
  hf-model-cache:
```

- If the volume doesn't exist, Docker creates it and **copies the container's existing files** into it (unlike bind mount which overwrites)
- You cannot easily browse the files from the host (they're in `/var/lib/docker/volumes/`)
- Docker manages lifecycle — volumes persist until you explicitly delete them
- More portable — works identically on any host without worrying about path structure
- **Best for:** production persistent data, database files

### Which to use for vLLM?

**Use a bind mount for the HuggingFace cache.** Here's why:

1. You want to see and manage the model files directly — `ls /workspace/hf_cache/hub/` lets you see which models are downloaded
2. Your RunPod volume is already at `/workspace` — a bind mount lets you use that storage directly
3. You might want to pre-download models, inspect them, or delete specific ones without going through Docker commands

```bash
# Pre-download a model to the cache before even starting the container
export HF_HOME=/workspace/hf_cache
huggingface-cli download Qwen/Qwen2.5-7B-Instruct

# Now when Docker starts, the model is already there — no download wait
docker run ... -v /workspace/hf_cache:/root/.cache/huggingface ...
```

### The path translation rule

The left side of `-v` is always the **host path**. The right side is always the **container path**. They can be completely different strings — they just point to the same underlying storage.

```
Host filesystem:                   Container filesystem:
/workspace/hf_cache/               /root/.cache/huggingface/
├── hub/                     ↔     ├── hub/
│   └── models--Qwen.../           │   └── models--Qwen.../
└── token                          └── token
```

---

## 7. Shared memory — --ipc=host vs --shm-size

This is the flag that trips up almost every beginner doing multi-GPU Docker deployments. Understanding it properly saves hours of debugging cryptic NCCL errors.

### What is shared memory?

Shared memory (`/dev/shm`) is a section of RAM that multiple processes can access simultaneously without going through the kernel's normal file I/O. It's orders of magnitude faster than writing to disk or passing data through sockets. PyTorch uses it heavily for:

- Transferring tensor data between worker processes (DataLoader workers)
- NCCL (NVIDIA Collective Communications Library) communication between GPU processes during tensor parallelism
- Passing KV cache blocks between vLLM's scheduler and worker processes

### The Docker default is dangerously small

By default, every Docker container gets `/dev/shm` with **64 MB** of shared memory. For a single-GPU vLLM deployment with small models, this sometimes works. For anything with tensor parallelism, multiple workers, or high concurrency, it fails with errors like:

```
RuntimeError: DataLoader worker (pid 1234) is killed by signal: Bus error
OSError: [Errno 28] No space left on device
ncclSystemError: System call failed. Last error: No space left on device
```

None of these messages say "shared memory" — they look like completely different errors. That's why this trips people up.

### Option 1: --ipc=host (Recommended)

```bash
--ipc=host
```

This removes the isolation entirely — the container uses the **host's** `/dev/shm` directly. The host's `/dev/shm` is typically sized at half of system RAM (e.g., on a 256 GB RAM host, `/dev/shm` is 128 GB). You never run out.

**Tradeoff:** Any process on the host can access the shared memory objects created by the container. On a dedicated inference pod this is not a concern. On a multi-tenant shared host, it is.

### Option 2: --shm-size (More controlled)

```bash
--shm-size 16g      # for single GPU
--shm-size 32g      # for multi-GPU / tensor parallel
```

This keeps isolation but gives the container a larger `/dev/shm`. Use this when you can't use `--ipc=host` for security reasons (e.g., shared hosts, enterprise environments).

**Sizing guide:**

| Setup | Minimum shm-size |
|---|---|
| Single GPU, 7B model | 8g |
| Single GPU, high concurrency | 16g |
| 2–4 GPU tensor parallelism | 16g |
| 4–8 GPU tensor parallelism | 32g |

### Practical recommendation for RunPod

Use `--ipc=host`. You're on a dedicated pod — there are no other tenants sharing the host's IPC namespace. It's simpler, always sufficient, and matches what the vLLM documentation recommends.

---

## 8. GPU passthrough — how Docker sees your GPU

Docker containers are isolated from the host by default — they cannot see any hardware except what you explicitly pass in. GPUs are passed in through the NVIDIA Container Toolkit.

### How it works

The NVIDIA Container Toolkit installs a custom OCI runtime hook. When Docker starts a container with `--gpus`, this hook:

1. Detects which GPU(s) you requested
2. Mounts the necessary NVIDIA device files (`/dev/nvidia0`, `/dev/nvidiactl`, etc.) into the container
3. Mounts the NVIDIA driver libraries into the container
4. Sets `CUDA_VISIBLE_DEVICES` appropriately

The container never needs the CUDA toolkit installed — only the NVIDIA driver on the host. The image (`vllm/vllm-openai`) includes the CUDA runtime libraries, but the driver itself is provided by the host through this passthrough mechanism.

### GPU selection flags

```bash
--gpus all                           # All GPUs
--gpus '"device=0"'                  # GPU 0 only (note the quotes)
--gpus '"device=0,1"'                # GPUs 0 and 1
--gpus '"device=GPU-<uuid>"'         # By UUID from nvidia-smi -L
```

### Verify GPU visibility inside the container

```bash
# Run nvidia-smi inside your running container
docker exec vllm-server nvidia-smi

# Run it inline without starting a server
docker run --rm --gpus all vllm/vllm-openai:latest nvidia-smi
```

If this shows your GPU, Docker can see it. If it shows `No devices were found`, the NVIDIA Container Toolkit isn't configured.

### CUDA_VISIBLE_DEVICES inside containers

When you set `--gpus '"device=0,1"'`, Docker sets `CUDA_VISIBLE_DEVICES=0,1` inside the container automatically. If you also pass `-e CUDA_VISIBLE_DEVICES=0` explicitly, you can accidentally override this and restrict GPU access further. Don't set `CUDA_VISIBLE_DEVICES` manually unless you have a specific reason.

---

## 9. Environment variables in Docker

Environment variables are the clean way to inject secrets, configuration, and runtime settings into containers. In vLLM deployments, you'll use them constantly.

### Three ways to pass environment variables

**1. Inline with -e (for individual variables):**
```bash
docker run -e HF_TOKEN=hf_xxx -e HF_HOME=/root/.cache/huggingface ...
```

**2. Reference from your host shell with -e (no value = inherit):**
```bash
export HF_TOKEN="hf_xxx"
docker run -e HF_TOKEN ...          # No = sign — inherits from host shell
```

**3. From a .env file with --env-file (best practice):**
```bash
# Create /workspace/.env
cat > /workspace/.env << 'EOF'
HF_TOKEN=hf_your_token_here
HF_HOME=/root/.cache/huggingface
HF_HUB_ENABLE_HF_TRANSFER=1
VLLM_LOGGING_LEVEL=INFO
EOF

docker run --env-file /workspace/.env ...
```

The `.env` file is the cleanest approach — one place for all configuration, easy to audit, never accidentally echoed in shell history. Add it to `.gitignore` so your token never gets committed.

### Important vLLM environment variables

These go inside the container as `-e` flags:

| Variable | Description |
|---|---|
| `HF_TOKEN` | HuggingFace authentication token |
| `HF_HOME` | HuggingFace cache directory inside container |
| `HF_HUB_ENABLE_HF_TRANSFER` | Set to `1` for faster model downloads (uses Rust) |
| `VLLM_LOGGING_LEVEL` | `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `CUDA_VISIBLE_DEVICES` | Override GPU visibility (usually leave this to Docker) |
| `NCCL_DEBUG` | Set to `INFO` or `TRACE` for debugging multi-GPU NCCL issues |
| `VLLM_WORKER_MULTIPROC_METHOD` | `fork` or `spawn` for multiprocessing (leave as default) |

---

## 10. Useful docker commands for day-to-day work

### Container lifecycle

```bash
# Start a new container
docker run --name vllm-server ...

# Stop a running container (graceful — sends SIGTERM)
docker stop vllm-server

# Start an existing stopped container
docker start vllm-server

# Restart
docker restart vllm-server

# Remove a stopped container
docker rm vllm-server

# Stop and remove in one command
docker rm -f vllm-server

# List running containers
docker ps

# List all containers (including stopped)
docker ps -a
```

### Logs

```bash
# Stream live logs (like tail -f)
docker logs vllm-server -f

# Last 100 lines
docker logs vllm-server --tail 100

# With timestamps
docker logs vllm-server -f --timestamps
```

> **Key difference from Phase 1:** In Phase 1, logs printed directly to your terminal. In Docker, logs are captured by the Docker daemon and stored separately. `docker logs` retrieves them. This means even if your SSH session disconnects, the logs are preserved.

### Execute commands inside a running container

```bash
# Open an interactive shell inside the running vLLM container
docker exec -it vllm-server /bin/bash

# Run a single command inside the container
docker exec vllm-server nvidia-smi
docker exec vllm-server df -h              # Check disk space inside container
docker exec vllm-server env | grep HF     # Verify env vars are set

# Check Python packages installed in the container
docker exec vllm-server pip list | grep vllm
```

### Inspect a container

```bash
# Full container details (JSON)
docker inspect vllm-server

# Just the IP address
docker inspect vllm-server --format '{{.NetworkSettings.IPAddress}}'

# Check volume mounts
docker inspect vllm-server --format '{{json .Mounts}}' | python3 -m json.tool
```

### Image management

```bash
# List downloaded images
docker images

# Pull an image without running it
docker pull vllm/vllm-openai:latest

# Remove an image (must stop containers using it first)
docker rmi vllm/vllm-openai:latest

# Show image layers and history
docker history vllm/vllm-openai:latest

# Clean up unused images, containers, networks (be careful)
docker system prune

# Nuclear option — delete everything including volumes
docker system prune -a --volumes
```

### Resource usage

```bash
# Live CPU, RAM, GPU usage per container
docker stats

# Stats for a specific container
docker stats vllm-server
```

---

## 11. Graduating to docker-compose

`docker run` with all its flags gets unwieldy fast. A command with 15 flags spread across multiple lines is hard to read, hard to version control, and easy to get wrong.

`docker-compose` solves this by letting you define everything in a YAML file. Run `docker compose up` and it handles the rest.

### Installing docker-compose (if not present)

On RunPod pods, Compose V2 (integrated into Docker as `docker compose`) is usually available:

```bash
docker compose version
# Docker Compose version v2.x.x
```

If not:
```bash
sudo apt install docker-compose-plugin
```

### The docker-compose.yml for vLLM

Save this as `/workspace/compose/docker-compose.yml`:

```yaml
services:
  vllm:
    image: vllm/vllm-openai:latest    # Pin this in production — see section 13
    container_name: vllm-server
    
    # GPU passthrough
    runtime: nvidia
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    
    # Shared memory — required for PyTorch multiprocessing
    ipc: host

    # Port mapping
    ports:
      - "8000:8000"
    
    # Volume mounts
    volumes:
      - /workspace/hf_cache:/root/.cache/huggingface
    
    # Environment variables
    environment:
      - HF_TOKEN=${HF_TOKEN}              # Read from host shell or .env file
      - HF_HOME=/root/.cache/huggingface
      - HF_HUB_ENABLE_HF_TRANSFER=1
      - VLLM_LOGGING_LEVEL=INFO
    
    # Restart policy
    restart: unless-stopped
    
    # Health check — Docker will monitor this and restart if it fails
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 5
      start_period: 120s    # Give vLLM time to load the model before checking
    
    # vLLM serve arguments — passed to the ENTRYPOINT
    command: >
      --model Qwen/Qwen2.5-7B-Instruct
      --host 0.0.0.0
      --port 8000
      --dtype auto
      --gpu-memory-utilization 0.90
      --max-model-len 8192
      --max-num-seqs 256
      --served-model-name qwen2.5-7b
      --enable-prefix-caching
```

### Running it

```bash
cd /workspace/compose

# Start (pulls image if not present, starts container)
docker compose up -d

# View logs
docker compose logs -f

# Stop
docker compose down

# Restart
docker compose restart

# Rebuild and restart (if you change the compose file)
docker compose up -d --force-recreate
```

### The .env file for secrets

Create `/workspace/compose/.env` alongside your `docker-compose.yml`:

```bash
cat > /workspace/compose/.env << 'EOF'
HF_TOKEN=hf_your_actual_token_here
EOF

chmod 600 /workspace/compose/.env    # Only owner can read
```

Docker Compose automatically reads `.env` from the same directory as `docker-compose.yml`. The `${HF_TOKEN}` in your compose file is substituted from this file. Never commit `.env` to git.

### Why the healthcheck matters

The `healthcheck` block tells Docker to probe `GET /health` every 30 seconds. If it fails 5 times in a row, Docker marks the container as `unhealthy`. With `restart: unless-stopped`, Docker can automatically restart unhealthy containers. This is the simplest form of self-healing — more on this in Phase 5 (Kubernetes), but it's good practice to understand it here.

The `start_period: 120s` is critical for vLLM — model loading takes 60–180 seconds. Without a start period, Docker starts health-checking immediately, fails repeatedly during model load, and potentially restarts the container before it's even finished starting. `start_period` tells Docker to wait before counting failures.

---

## 12. Writing a custom Dockerfile on top of vllm-openai

Sometimes you need to add things to the official image — additional Python packages, custom model pre-loading, or configuration files. The right way to do this is to build on top of the official image rather than modifying it.

### Example: Adding audio processing support

The official image excludes audio dependencies for licensing reasons. If you need them:

```dockerfile
# Save as: /workspace/compose/Dockerfile
FROM vllm/vllm-openai:latest

# Install audio dependencies
# Pin the vLLM version to match the base image
RUN uv pip install --system "vllm[audio]==0.8.5"

# Optional: pre-download a model into the image
# (makes startup faster but image becomes huge — tradeoff)
# ARG HF_TOKEN
# RUN huggingface-cli download Qwen/Qwen2.5-7B-Instruct
```

```yaml
# In docker-compose.yml, change:
# image: vllm/vllm-openai:latest
# To:
build:
  context: .
  dockerfile: Dockerfile
```

```bash
# Build your custom image
docker compose build

# Run it
docker compose up -d
```

### Example: Pre-downloading the model at build time

This bakes the model into the image. The image becomes ~25 GB, but startup is instant (no download). Useful for production environments where startup time matters and pulling 14 GB at runtime is unacceptable.

```dockerfile
FROM vllm/vllm-openai:latest

# Build argument for the token (passed at build time, not stored in image)
ARG HF_TOKEN

# Pre-download the model weights
RUN --mount=type=secret,id=hf_token \
    HF_TOKEN=$(cat /run/secrets/hf_token) \
    huggingface-cli download Qwen/Qwen2.5-7B-Instruct \
    --local-dir /models/qwen2.5-7b
```

```bash
# Build with the secret — token is not baked into the image layer
docker buildx build \
  --secret id=hf_token,env=HF_TOKEN \
  -t my-vllm:latest .
```

### The golden rule of custom Dockerfiles

**Always build FROM a pinned version, never from `:latest`.**

```dockerfile
# Bad — this changes every vLLM release, breaking your build unpredictably
FROM vllm/vllm-openai:latest

# Good — this is stable forever
FROM vllm/vllm-openai:v0.8.5
```

---

## 13. Pinning versions — never use :latest in production

`:latest` is a lie. It means "whatever the maintainer decided to tag as latest at the time you pulled it." Two pulls of `:latest` one week apart can give you completely different images with different vLLM versions, different PyTorch, different behaviour.

### Find the current stable version

```bash
# Check what version is inside a running container
docker exec vllm-server python -c "import vllm; print(vllm.__version__)"

# Or check Docker Hub for available tags
# https://hub.docker.com/r/vllm/vllm-openai/tags
```

### Pin it in your compose file

```yaml
# Instead of:
image: vllm/vllm-openai:latest

# Use:
image: vllm/vllm-openai:v0.8.5
```

### Upgrade deliberately

When you want to upgrade vLLM:
1. Check the [vLLM release notes](https://github.com/vllm-project/vllm/releases) for breaking changes
2. Test the new version with your model and workload
3. Update the version tag in your compose file
4. Re-deploy with `docker compose up -d --force-recreate`

This is the discipline that prevents "it was working yesterday, nothing changed" incidents.

---

## 14. Common Docker + vLLM errors and fixes

### Container exits immediately with no useful output

```bash
docker ps -a    # Container shows "Exited (1)"
docker logs vllm-server    # Check what happened
```

Most common causes:
- Missing `--gpus all` — container can't see GPU, vLLM fails to initialise CUDA
- Wrong vLLM flags passed — typo in `--model` name or unknown flag
- OOM — same causes as Phase 1, same fixes

---

### Shared memory errors

```
RuntimeError: DataLoader worker is killed by signal: Bus error
ncclSystemError: No space left on device
```

**Fix:** Add `--ipc=host` or increase `--shm-size 16g`. These errors don't obviously say "shared memory" — you have to know what they mean.

---

### GPU not visible inside container

```
RuntimeError: No CUDA GPUs are available
```

**Fix:** Verify `--gpus all` is in your `docker run` command. Verify the NVIDIA Container Toolkit is installed. Run `docker exec vllm-server nvidia-smi` to check GPU visibility.

---

### Model download fails inside container

```
requests.exceptions.HTTPError: 401 Client Error: Unauthorized
```

**Fix:** Your `HF_TOKEN` environment variable isn't reaching the container. Check:
```bash
docker exec vllm-server env | grep HF
# Must show: HF_TOKEN=hf_xxx
```

If it's missing, you likely forgot `-e HF_TOKEN=$HF_TOKEN` in your run command, or `$HF_TOKEN` wasn't set in your host shell when you ran the command.

---

### Port already in use

```
Error starting userland proxy: listen tcp 0.0.0.0:8000: bind: address already in use
```

**Fix:**
```bash
# Find what's using the port (on the host, not inside the container)
lsof -i :8000
# Or:
docker ps    # Check if another container is already using it

# Use a different host port
-p 8001:8000    # Host 8001 → container 8000
```

---

### Container starts but health check fails

```
vllm-server is unhealthy
```

Check if the model is still loading (startup takes 60–180 seconds):
```bash
docker logs vllm-server -f    # Watch for "Application startup complete"
```

If it's still loading, your `start_period` in the healthcheck is too short. Increase it:
```yaml
healthcheck:
  start_period: 300s    # 5 minutes for large models
```

---

### Volume data not persisting after container restart

This happens when you use an **anonymous volume** instead of a bind mount or named volume.

```bash
# Wrong — creates an anonymous volume that's deleted with the container
docker run -v /root/.cache/huggingface ...

# Right — bind mount to a persistent host path
docker run -v /workspace/hf_cache:/root/.cache/huggingface ...
```

Anonymous volumes (no host path, just a container path) are deleted by `docker rm`. Always use a host path or named volume for anything you want to keep.

---

## 15. pip vs Docker — a direct comparison

Now that you've done both, here's the honest comparison:

| Aspect | pip (Phase 1) | Docker (Phase 2) |
|---|---|---|
| Setup time | 5–10 min install | 5 min pull (then instant) |
| Reproducibility | Poor — depends on host state | Excellent — image is immutable |
| Portability | Host-specific | Runs anywhere with Docker + GPU |
| Debuggability | Everything visible on host | Need `docker exec` to get inside |
| Isolation | None — shares host Python | Full — own filesystem and processes |
| Dependency conflicts | Possible with other pip packages | Impossible — fully isolated |
| GPU access | Direct | Via NVIDIA Container Toolkit |
| Log management | Terminal / nohup file | `docker logs` — managed by daemon |
| Auto-restart | Manual / systemd | `--restart unless-stopped` |
| vLLM flags | Identical | Identical |
| API surface | Identical | Identical |

**The flags, the API, the metrics, and the model behaviour are 100% identical.** The only difference is the wrapper. This is the key insight of Phase 2 — Docker doesn't change what vLLM does, it changes how it's packaged and operated.

---

## 16. What you learned — exit checklist

Before moving to Phase 3 (Benchmarking), verify you can answer these:

**Docker concepts:**
- [ ] What is the difference between a Docker image and a container?
- [ ] What does `--gpus all` do and what is it actually passing into the container?
- [ ] Why does `--ipc=host` exist, and what happens without it during tensor parallel inference?
- [ ] What is the difference between a bind mount and a named volume? Which should you use for model weights and why?
- [ ] What does `-p 8000:8000` mean? If port 8000 on the host is busy, how do you fix it without changing vLLM's listening port?
- [ ] What is the ENTRYPOINT of the vllm/vllm-openai image and how do your `--model` flags get passed to it?
- [ ] Why should you never use `:latest` in production?

**Practical skills:**
- [ ] Run vLLM in a Docker container on RunPod
- [ ] Check container logs with `docker logs`
- [ ] Open a shell inside the running container with `docker exec -it`
- [ ] Run vLLM via docker-compose with a proper `.env` file for secrets
- [ ] Write a minimal custom Dockerfile that extends `vllm/vllm-openai`
- [ ] Fix a shared memory error
- [ ] Verify GPU visibility inside the container

**Key mental model:**
- [ ] Explain in one sentence why the vLLM API from Phase 1 and Phase 2 are identical even though the deployment method is completely different.

---

## Quick reference card

```bash
# ── Environment setup ──────────────────────────────────────────────────────
export HF_TOKEN="hf_xxx"
mkdir -p /workspace/hf_cache /workspace/compose

# ── docker run (single command) ────────────────────────────────────────────
docker run -d \
  --name vllm-server \
  --runtime nvidia \
  --gpus all \
  --ipc=host \
  -p 8000:8000 \
  -v /workspace/hf_cache:/root/.cache/huggingface \
  -e HF_TOKEN=$HF_TOKEN \
  -e HF_HOME=/root/.cache/huggingface \
  --restart unless-stopped \
  vllm/vllm-openai:latest \
    --model Qwen/Qwen2.5-7B-Instruct \
    --host 0.0.0.0 --port 8000 \
    --dtype auto \
    --gpu-memory-utilization 0.90 \
    --max-model-len 8192 \
    --served-model-name qwen2.5-7b

# ── Day-to-day commands ────────────────────────────────────────────────────
docker logs vllm-server -f             # Live logs
docker exec -it vllm-server bash       # Shell inside container
docker exec vllm-server nvidia-smi     # Check GPU inside container
docker stats vllm-server               # Resource usage
docker stop vllm-server                # Graceful stop
docker rm -f vllm-server               # Force remove

# ── docker-compose ─────────────────────────────────────────────────────────
cd /workspace/compose
docker compose up -d                   # Start
docker compose logs -f                 # Logs
docker compose down                    # Stop and remove
docker compose restart                 # Restart

# ── Health check ───────────────────────────────────────────────────────────
curl http://localhost:8000/health
curl http://localhost:8000/v1/models
```

---

## What's next — Phase 3: Benchmarking

In Phase 3, you'll finally measure what you've built. Both your pip deployment and your Docker deployment serve the same API — so you can benchmark them head to head.

You'll learn:
- vLLM's built-in benchmark scripts (`benchmark_serving.py`, `benchmark_throughput.py`)
- How to measure TTFT, ITL, throughput, and concurrency properly
- How to read the output and understand what the numbers mean
- How to stress-test your server with Locust
- How to watch GPU metrics with `nvidia-smi` and `dcgm-exporter` during load
- How to build a baseline you'll use to evaluate every tool you add in later phases

Everything you measure in Phase 3 becomes your reference point for all of Phases 4–8.

---

*Last updated: May 2026. Based on official vLLM Docker documentation at docs.vllm.ai/en/stable/deployment/docker.*
