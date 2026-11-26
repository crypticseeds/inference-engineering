# inference-engineering

# Phase 1: Running vLLM on RunPod via pip — A Complete Beginner's Guide

> **Series:** MLOps for Inference — Phase 1 of 8  
> **Goal:** Deploy your first LLM inference server from scratch, understand every moving part, and build the mental model you'll need for every phase that follows.  
> **You will need:** A RunPod account, a Hugging Face account (free), ~$2–5 in RunPod credits.

---

## Table of Contents

1. [What is vLLM and why does it matter?](#1-what-is-vllm-and-why-does-it-matter)
2. [Core concepts before you touch the keyboard](#2-core-concepts-before-you-touch-the-keyboard)
3. [Choosing your GPU on RunPod](#3-choosing-your-gpu-on-runpod)
4. [Spinning up a pod and connecting via SSH](#4-spinning-up-a-pod-and-connecting-via-ssh)
5. [Setting up your environment](#5-setting-up-your-environment)
6. [Installing vLLM](#6-installing-vllm)
7. [Choosing a model](#7-choosing-a-model)
8. [Your first vLLM serve command — explained line by line](#8-your-first-vllm-serve-command--explained-line-by-line)
9. [The essential flags reference](#9-the-essential-flags-reference)
10. [Querying the server — curl and Python](#10-querying-the-server--curl-and-python)
11. [Reading the startup logs — what everything means](#11-reading-the-startup-logs--what-everything-means)
12. [Watching GPU memory in real time](#12-watching-gpu-memory-in-real-time)
13. [Common errors and how to fix them](#13-common-errors-and-how-to-fix-them)
14. [Best practices for pip deployments](#14-best-practices-for-pip-deployments)
15. [What you learned — exit checklist](#15-what-you-learned--exit-checklist)

---

## 1. What is vLLM and why does it matter?

vLLM (Virtual Large Language Model server) is an open-source inference engine built at UC Berkeley's Sky Computing Lab. It solves a very specific problem: running LLMs on GPUs fast and efficiently.

### The problem it solves

When you run a language model, it processes text using an internal mechanism called the **KV cache** (Key-Value cache). Every token in the conversation has to be stored so the model can "remember" what came before. The naive approach to managing this memory is wasteful — it reserves large chunks of GPU RAM upfront and holds onto them even when most of that space sits empty.

vLLM introduced two breakthrough ideas:

**PagedAttention** — Instead of allocating one giant contiguous block of GPU memory per request, vLLM manages the KV cache in small, fixed-size pages (blocks), exactly like how an operating system manages virtual memory for programs. Pages are allocated on demand and freed immediately when done. This eliminates GPU memory fragmentation almost entirely.

**Continuous batching** — Traditional servers process one batch at a time: start 10 requests, wait for all 10 to finish, then start the next 10. vLLM processes a _stream_ of requests. As soon as one request finishes generating a token, that slot is immediately handed to the next waiting request. The GPU is never idle waiting for slow requests to complete. This is why vLLM achieves 24x higher throughput than a naive HuggingFace Transformers deployment.

### Why you should care

Every production LLM serving system you will encounter — whether it's Triton, KServe, Ray Serve, or a managed cloud service — either uses vLLM internally or was designed to compete with it. Understanding vLLM at the raw pip level means you understand the engine, not just the interface.

---

## 2. Core concepts before you touch the keyboard

Do not skip this section. These terms will appear constantly throughout your vLLM journey.

### GPU VRAM

GPU RAM (VRAM) is where the model lives. Unlike CPU RAM, it cannot be swapped to disk during inference without catastrophic slowdowns. You need enough VRAM for:

1. **Model weights** — the actual parameters of the model
2. **KV cache** — temporary storage for attention computations per active request
3. **Activations** — intermediate values computed during a forward pass
4. **Framework overhead** — PyTorch, CUDA kernels, etc.

A rough rule of thumb for model weight memory:

|Model size|FP32 (32-bit)|FP16/BF16 (16-bit)|INT8 (8-bit)|INT4 (4-bit)|
|---|---|---|---|---|
|7B params|~28 GB|~14 GB|~7 GB|~3.5 GB|
|13B params|~52 GB|~26 GB|~13 GB|~6.5 GB|
|70B params|~280 GB|~140 GB|~70 GB|~35 GB|

You need VRAM _beyond_ these numbers for KV cache headroom. This is why a 7B model in FP16 on a 24 GB GPU can still OOM (Out of Memory) — the weights alone use 14 GB, leaving only 10 GB for KV cache and overhead.

### Dtype (Data Type)

The numeric precision used to store the model's weights and perform computations.

- **FP32 (float32)** — Full 32-bit precision. Rarely used for inference; 2x the memory of FP16 for no meaningful quality gain.
- **FP16 (float16)** — 16-bit. The workhorse for older GPUs (RTX 3090, A100 40GB). Slightly less stable numerically.
- **BF16 (bfloat16)** — 16-bit with a wider exponent range. More numerically stable than FP16. Preferred for A100, H100, and newer GPUs that have native BF16 tensor cores.
- **INT8 / INT4** — Quantized formats. Dramatically reduce memory usage at a small quality cost. Usually loaded via quantized model checkpoints (AWQ, GPTQ, GGUF).

**Practical rule:** Use `--dtype bfloat16` on A100/H100. Use `--dtype float16` on RTX 3090/4090. Use `--dtype auto` and let vLLM decide.

### Tensor Parallelism

If a model is too large to fit on one GPU, you can split it across multiple GPUs. Tensor parallelism splits the model's weight matrices across GPUs so each GPU holds a slice. The GPUs communicate via NVLink (fast, on the same machine) or PCIe. This is controlled by `--tensor-parallel-size N`, where N is the number of GPUs to spread across.

### Context Length / Max Model Length

The maximum number of tokens (words/subwords) a model can process at once — prompt + response combined. A model with a 128K context window can theoretically process enormous documents. But a longer context = more KV cache = more VRAM. You can always cap it lower with `--max-model-len` to save memory.

### OpenAI-compatible API

vLLM exposes the exact same HTTP API that OpenAI's ChatGPT API uses. This means any application written to talk to OpenAI can talk to vLLM with one line changed: the base URL. The two main endpoints are:

- `POST /v1/completions` — text-in, text-out (older style)
- `POST /v1/chat/completions` — structured conversation (system/user/assistant messages)

### Tokens

LLMs don't read words — they read tokens. A token is roughly 0.75 words in English. "Hello world" is 2–3 tokens. A 4000-token context window can hold about 3000 words. Keep this in mind when sizing `--max-model-len`.

---

## 3. Choosing your GPU on RunPod

RunPod offers two pod tiers:

- **Secure Cloud** — dedicated hardware, more consistent performance, slightly more expensive. Use this.
- **Community Cloud** — shared infrastructure, cheaper, less predictable. Fine for experiments.

### GPU selection by model size

For Phase 1, you'll work with 7B parameter models. Here's your decision matrix:

|GPU|VRAM|7B FP16 fit?|Notes|
|---|---|---|---|
|**RTX 4090**|24 GB|Tight — use `--max-model-len 4096`|Good learning GPU, cheap|
|**A40**|48 GB|Comfortable|Great balance of cost and space|
|**L40S**|48 GB|Comfortable|Faster than A40, similar price|
|**A100 40GB**|40 GB|Comfortable|Production-grade, more expensive|
|**A100 80GB**|80 GB|Very comfortable|Run 13B or quantized 70B|
|**H100**|80 GB|Overkill for 7B|Save this for Phase 4+|

**Recommendation for this phase:** Start with an **RTX 4090** (cheapest per hour, teaches you memory constraints) or an **A40/L40S** (comfortable headroom, fewer OOM surprises). Don't use an H100 yet — you'll spend money without learning more.

---

## 4. Spinning up a pod and connecting via SSH

### Step 1: Create the pod

1. Log into [runpod.io](https://runpod.io) → **Pods** → **+ Deploy**
2. Select **GPU** type (RTX 4090 or A40)
3. Under **Container Image**, use: `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`
    - This image includes: CUDA 12.8, Python 3.11+, PyTorch, and all NVIDIA drivers pre-installed. You do not need to install drivers yourself.
4. Set **Container Disk** to at least **30 GB** (model weights take space)
5. Set **Volume Disk** to **50 GB** — mount it at `/workspace`. This persists across pod restarts.
6. Under **Expose Ports**, add: `8000` (this is where vLLM will listen)
7. Click **Deploy**

### Step 2: Get your SSH command

Once the pod is Running:

1. Click **Connect** on your pod
2. Copy the **SSH** command — it looks like: `ssh root@<ip> -p <port> -i ~/.ssh/id_rsa`

### Step 3: Add your SSH key (if you haven't)

```bash
# On your local machine — generate a key if you don't have one
ssh-keygen -t ed25519 -C "your_email@example.com"

# Copy the public key
cat ~/.ssh/id_ed25519.pub
```

Paste this into RunPod: **Settings** → **SSH Public Keys** → Add.

### Step 4: Connect

```bash
ssh root@<ip> -p <port>
```

You're in. You'll see a shell prompt like `root@<container_id>:~#`.

---

## 5. Setting up your environment

### Verify your GPU is visible

```bash
nvidia-smi
```

You should see output like:

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

Key things to note:

- **Memory-Usage**: `500MiB / 24576MiB` means 500 MB used, 24 GB total
- **GPU-Util**: Should be near 0% when idle, spikes to 90-100% during inference
- **Driver Version / CUDA Version**: Must be CUDA 12.x for modern vLLM

### Set up a Python virtual environment

Even though the RunPod image has Python pre-installed, always use a virtual environment to isolate your vLLM installation from the system Python.

```bash
# Move to persistent volume so your environment survives pod restarts
cd /workspace

# Install uv — the fast Python package manager recommended by vLLM docs
pip install uv

# Create a virtual environment with Python 3.12
uv venv --python 3.12 --seed vllm-env

# Activate it
source /workspace/vllm-env/bin/activate

# Verify
python --version
# Should show: Python 3.12.x
```

> **Why uv instead of pip?** uv is 10–100x faster than pip for resolving and installing packages. For vLLM which has many heavy dependencies (PyTorch, CUDA extensions), this matters. vLLM's own documentation recommends uv.

### Set your Hugging Face token

Many models (especially Llama 3) require accepting a license agreement on Hugging Face before downloading. You authenticate via a token.

1. Go to [huggingface.co](https://huggingface.co) → Settings → Access Tokens → New token (read permission is enough)
2. Set it in your shell:

```bash
export HF_TOKEN="hf_your_token_here"

# Also add it to your shell config so it persists
echo 'export HF_TOKEN="hf_your_token_here"' >> ~/.bashrc
```

### Set the model cache to your persistent volume

By default, Hugging Face downloads models to `~/.cache/huggingface`. On RunPod, the home directory is on the container disk, which is wiped when the pod stops. Point the cache to your persistent volume instead:

```bash
export HF_HOME="/workspace/hf_cache"
mkdir -p /workspace/hf_cache

# Add to bashrc for persistence
echo 'export HF_HOME="/workspace/hf_cache"' >> ~/.bashrc
```

> **Why this matters:** A 7B model is roughly 14 GB. Without this, every time your pod restarts (or you try a different pod), you re-download 14 GB. With the volume, you download once and reuse forever.

---

## 6. Installing vLLM

With your virtual environment active:

```bash
uv pip install vllm --torch-backend=auto
```

The `--torch-backend=auto` flag tells uv to detect your CUDA version and install the matching version of PyTorch automatically. This takes 3–10 minutes depending on connection speed — vLLM pulls PyTorch and many CUDA-compiled extensions.

### Verify the installation

```bash
python -c "import vllm; print(vllm.__version__)"
# Should print: 0.x.x (whatever the latest stable is)

vllm --version
# Should print the same
```

---

## 7. Choosing a model

For Phase 1, use a small, well-tested model with no license hurdles. Here are the best options ranked by learning value:

### Option A — Qwen2.5-7B-Instruct (Recommended for beginners)

```
Qwen/Qwen2.5-7B-Instruct
```

- No license agreement required
- 7B parameters, runs on any 16GB+ GPU
- Strong instruction-following, good for testing chat completions
- Fits comfortably on RTX 4090 in FP16

### Option B — Mistral-7B-Instruct-v0.3

```
mistralai/Mistral-7B-Instruct-v0.3
```

- No gated access required
- Very popular, tons of community docs
- Excellent baseline for benchmarking

### Option C — Meta-Llama-3.1-8B-Instruct (Requires HF license acceptance)

```
meta-llama/Meta-Llama-3.1-8B-Instruct
```

- Must accept license at huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct
- The industry standard reference model
- Use once you have HF token set up

### Option D — Phi-3-mini-4k-instruct (Good for tight VRAM)

```
microsoft/Phi-3-mini-4k-instruct
```

- Only 3.8B parameters — half the size
- Runs on 8GB GPUs with room to spare
- Good if you're on a tight budget or want to test flags without waiting for big downloads

> **Start with Qwen2.5-7B-Instruct.** No token required, no license gate, downloads quickly, and behaves predictably.

---

## 8. Your first vLLM serve command — explained line by line

Here is the complete command to start your first vLLM server:

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype auto \
  --gpu-memory-utilization 0.90 \
  --max-model-len 8192 \
  --max-num-seqs 256 \
  --served-model-name qwen2.5-7b
```

### Line-by-line explanation

**`vllm serve Qwen/Qwen2.5-7B-Instruct`**  
This is the model identifier — the Hugging Face repo path. vLLM will download it from HuggingFace Hub on first run (to `$HF_HOME`), then load from disk on subsequent runs. You can also pass a local path to a downloaded model folder.

**`--host 0.0.0.0`**  
Binds the server to all network interfaces. By default, vLLM only listens on `127.0.0.1` (localhost), which means only processes on the same machine can reach it. Setting `0.0.0.0` makes it accessible from outside the pod — required for hitting it from your laptop or other services.

**`--port 8000`**  
The HTTP port the server listens on. This must match the port you exposed in RunPod's pod settings. If you're running multiple models on one machine, use different ports (8000, 8001, etc.).

**`--dtype auto`**  
Tells vLLM to automatically pick the best data type for your GPU. On A100/H100, it picks BF16. On RTX GPUs, it picks FP16. Always start with `auto` — only override if you have a specific reason.

**`--gpu-memory-utilization 0.90`**  
vLLM pre-allocates a fraction of your GPU's total VRAM for the KV cache. Here, `0.90` means 90%. The remaining 10% is kept as a safety buffer for PyTorch framework overhead, CUDA kernels, and other system use.

Why not 1.0? Because if you use 100% and something unexpected needs a small amount of memory, the process crashes with OOM. 0.90 is the recommended default. Go lower (0.85, 0.80) if you see OOM errors at startup.

**`--max-model-len 8192`**  
Caps the maximum context length at 8192 tokens (prompt + response combined). The Qwen2.5-7B model supports up to 128K tokens natively, but allocating KV cache for 128K tokens on every startup wastes enormous VRAM. Always set this to the maximum you actually need. 8192 tokens ≈ ~6000 words, which is enough for most tasks.

**`--max-num-seqs 256`**  
The maximum number of requests vLLM will process concurrently. Setting this too high can cause OOM if all 256 sequences are long. Setting it too low wastes throughput. 256 is a sensible default; reduce to 64–128 if you see memory pressure.

**`--served-model-name qwen2.5-7b`**  
Sets the model name that appears in API responses and that clients pass in requests. By default, this is the full HuggingFace path (`Qwen/Qwen2.5-7B-Instruct`). Setting a clean alias means your API clients don't need to know the HF path — they just use `qwen2.5-7b`. This is important for compatibility and for when you want to swap models without changing client code.

---

## 9. The essential flags reference

Below are all the flags you need to understand for Phase 1, grouped by category.

### Model loading flags

|Flag|Type|Default|Description|
|---|---|---|---|
|`--model` (positional)|string|required|HuggingFace model ID or local path|
|`--dtype`|string|`auto`|`auto`, `float16`, `bfloat16`, `float32`|
|`--quantization`|string|`None`|`awq`, `gptq`, `gguf`, `fp8`, `bitsandbytes` — for quantized models|
|`--revision`|string|`None`|Specific HF model revision/commit hash|
|`--tokenizer`|string|same as model|Use a different tokenizer than the model|
|`--trust-remote-code`|flag|off|Allow running custom model code from HF hub. Required for some models like Falcon.|
|`--load-format`|string|`auto`|`auto`, `pt` (PyTorch), `safetensors`, `npcache`, `dummy`|

> **On `--trust-remote-code`:** Only enable this for models you trust. It runs Python code from the model repo on your machine.

### Memory and context flags

|Flag|Type|Default|Description|
|---|---|---|---|
|`--gpu-memory-utilization`|float|`0.90`|Fraction of GPU VRAM to use (0.0–1.0)|
|`--max-model-len`|int|model's native max|Maximum tokens per request (prompt + output)|
|`--max-num-seqs`|int|`256`|Max concurrent sequences in the engine|
|`--max-num-batched-tokens`|int|auto|Max tokens per forward pass (increases throughput at cost of latency)|
|`--swap-space`|int|`4`|CPU RAM (GB) used for swapping KV blocks. Increase for high-concurrency workloads.|
|`--kv-cache-dtype`|string|`auto`|`auto`, `fp8` — FP8 KV cache reduces memory by ~50% on supported GPUs|

### Parallelism flags

|Flag|Type|Default|Description|
|---|---|---|---|
|`--tensor-parallel-size`|int|`1`|Number of GPUs to shard the model across. Set to number of GPUs for multi-GPU pods.|
|`--pipeline-parallel-size`|int|`1`|Distribute layers across GPUs (for very large models). Rarely needed for 7B.|

### Server flags

|Flag|Type|Default|Description|
|---|---|---|---|
|`--host`|string|`127.0.0.1`|Bind address. Use `0.0.0.0` for external access.|
|`--port`|int|`8000`|HTTP port|
|`--served-model-name`|string|model path|Alias returned in API responses|
|`--api-key`|string|`None`|Simple Bearer token auth. Clients must send `Authorization: Bearer <key>`|
|`--max-log-len`|int|`None`|Truncate prompt logging. Useful for privacy in production.|
|`--disable-log-requests`|flag|off|Suppress per-request logging. Use in production for lower overhead.|
|`--enable-prefix-caching`|flag|off|Cache KV states for repeated prompt prefixes. Speeds up chatbot workloads significantly.|

### Sampling / generation flags

|Flag|Type|Default|Description|
|---|---|---|---|
|`--max-logprobs`|int|`20`|Max number of log probabilities to return when `logprobs` is requested|
|`--rope-scaling`|json|`None`|RoPE scaling config for extending context beyond trained length|

---

## 10. Querying the server — curl and Python

Once the server starts (look for `INFO: Application startup complete` in the logs), you can query it.

### Finding your RunPod URL

In the RunPod UI:

1. Click on your running pod
2. Click **Connect** → **HTTP Service**
3. Your URL will look like: `https://<pod-id>-8000.proxy.runpod.net`

Or if using the direct IP via SSH tunneling:

```bash
# On your local machine, forward pod port 8000 to localhost:8000
ssh root@<ip> -p <ssh_port> -L 8000:localhost:8000 -N &
```

Then use `http://localhost:8000` as your base URL.

### Test 1: Health check

```bash
curl http://localhost:8000/health
# Returns: OK
```

### Test 2: List available models

```bash
curl http://localhost:8000/v1/models | python3 -m json.tool
```

You should see your model name in the response:

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

### Test 3: Chat completions (the main API)

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-7b",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "What is PagedAttention in vLLM?"}
    ],
    "max_tokens": 300,
    "temperature": 0.7
  }'
```

### Test 4: Completions API (text-in, text-out)

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-7b",
    "prompt": "The capital of France is",
    "max_tokens": 50,
    "temperature": 0
  }'
```

### Test 5: Streaming response

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-7b",
    "messages": [{"role": "user", "content": "Count from 1 to 20 slowly"}],
    "max_tokens": 200,
    "stream": true
  }'
```

With `stream: true`, tokens are returned as Server-Sent Events (SSE) — you'll see them appear token by token. This is how ChatGPT's typing effect works.

### Python client

```python
# pip install openai
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed",  # required by openai client but ignored by vLLM unless you set --api-key
)

# Chat completion
response = client.chat.completions.create(
    model="qwen2.5-7b",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain KV cache in 3 sentences."}
    ],
    max_tokens=200,
    temperature=0.7,
)

print(response.choices[0].message.content)

# Streaming
stream = client.chat.completions.create(
    model="qwen2.5-7b",
    messages=[{"role": "user", "content": "Write a haiku about GPU memory"}],
    max_tokens=100,
    stream=True,
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

### Understanding the response object

```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1716768000,
  "model": "qwen2.5-7b",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "PagedAttention is..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 32,
    "completion_tokens": 47,
    "total_tokens": 79
  }
}
```

- **`finish_reason`**: `stop` = model finished naturally. `length` = hit `max_tokens` limit. `content_filter` = flagged content.
- **`usage`**: Token counts for this request. Critical for cost tracking in production.
- **`prompt_tokens`**: How many tokens your system prompt + user message consumed. Long system prompts are expensive.

---

## 11. Reading the startup logs — what everything means

When you run `vllm serve`, a torrent of log lines appears. Here's what to look for:

```
INFO 05-16 10:00:01 config.py:849] This model supports multiple tasks: {'embed', 'classify', 'reward', 'generate', 'score'}.
```

→ The model's capability. You care about `generate` for text generation.

```
INFO 05-16 10:00:05 model_runner.py:1065] Loading model weights took 13.42 GB and 8.231 seconds
```

→ How much VRAM the model weights consumed. Verify this matches your expectations (Qwen2.5-7B in FP16 should be ~14 GB).

```
INFO 05-16 10:00:12 kv_cache_utils.py:634] GPU KV cache size: 187,392 tokens
```

→ **This is one of the most important lines.** It tells you the total KV cache capacity — how many tokens worth of context can be simultaneously active across all requests. A higher number means you can serve more concurrent long conversations.

```
INFO 05-16 10:00:12 kv_cache_utils.py:638] Maximum concurrency for 8,192 tokens per request: 22.9x
```

→ With your `--max-model-len` of 8192, vLLM estimates it can handle ~22 concurrent full-length requests. This is your theoretical concurrency ceiling.

```
INFO 05-16 10:00:13 scheduler.py:65] Scheduler configuration: chunked-prefill is enabled
```

→ Chunked prefill is on. This means long prompts are processed in chunks rather than all at once, improving latency for the first token. It's enabled by default in recent vLLM versions.

```
INFO 05-16 10:00:15 api_server.py:239] Starting vLLM API server on http://0.0.0.0:8000
INFO 05-16 10:00:15 launcher.py:29] Available routes are:
GET /health
GET /v1/models
POST /v1/chat/completions
POST /v1/completions
POST /v1/embeddings
```

→ Server is up and ready. Routes are listed. You can now send requests.

During inference, you'll see lines like:

```
INFO 05-16 10:01:45 engine.py:390] Avg prompt throughput: 1893.2 tokens/s, Avg generation throughput: 48.7 tokens/s, Running: 2 reqs, Waiting: 0 reqs, GPU KV cache usage: 2.1%, Prefix cache hit rate: 0.0%
```

- **Prompt throughput**: How fast the model processes input tokens (prefill phase)
- **Generation throughput**: How fast the model generates output tokens (decode phase)
- **Running / Waiting**: Active and queued requests
- **GPU KV cache usage**: What % of your KV cache is in use. If this consistently hits 90%+, you're at capacity.
- **Prefix cache hit rate**: If `--enable-prefix-caching` is on, this shows what % of prompts are hitting the cache.

---

## 12. Watching GPU memory in real time

Open a second SSH session to your RunPod pod. Run these alongside your vLLM server:

### Basic watch (refreshes every 2 seconds)

```bash
watch -n 2 nvidia-smi
```

### Compact format — just memory and utilization

```bash
watch -n 1 "nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader"
```

Output:

```
NVIDIA RTX 4090, 15823 MiB, 24576 MiB, 87 %
```

### What to watch for

|Metric|Idle|During inference|
|---|---|---|
|Memory used|~14 GB (weights)|Increases with each request, stabilises|
|GPU utilization|0–2%|Spikes to 80–100% per batch|
|Memory total|Fixed (e.g. 24576 MiB)|Fixed|

If GPU utilization is consistently low during inference, your batch size is too small — vLLM isn't saturating the GPU. This matters for throughput but not for a single-user test.

### Log memory usage from Python

```python
import subprocess

result = subprocess.run(
    ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
    capture_output=True, text=True
)
used, total = result.stdout.strip().split(", ")
print(f"VRAM: {used} / {total} MB  ({100*int(used)/int(total):.1f}%)")
```

---

## 13. Common errors and how to fix them

### Out of Memory (OOM) at startup

```
torch.cuda.OutOfMemoryError: CUDA out of memory.
```

**Cause:** `--gpu-memory-utilization` is too high, or `--max-model-len` is too large for available VRAM.

**Fix options (try in order):**

1. Lower `--gpu-memory-utilization 0.85` → `0.80`
2. Lower `--max-model-len 8192` → `4096`
3. Use a quantized model (add `--quantization awq` with an AWQ model)
4. Upgrade to a larger GPU

---

### CUDA out of memory during inference (preemption warning)

```
WARNING scheduler.py:1057 Sequence group X is preempted by PreemptionMode.RECOMPUTE
```

**Cause:** You have more concurrent requests than your KV cache can hold. vLLM has to pause some requests and recompute.

**Fix options:**

1. Increase `--gpu-memory-utilization`
2. Lower `--max-model-len`
3. Lower `--max-num-seqs`
4. Use `--kv-cache-dtype fp8` (halves KV cache memory, minor quality impact)

---

### Model not found / 404

```
ValueError: Model meta-llama/Llama-3.1-8B-Instruct is not accessible
```

**Cause:** Either a typo in the model name, or it's a gated model and your HF_TOKEN isn't set, or you haven't accepted the license on HuggingFace.

**Fix:**

1. Check `echo $HF_TOKEN` — must be set
2. Visit the model page on HuggingFace and accept the access agreement
3. Verify the exact model ID (case sensitive)

---

### Port already in use

```
OSError: [Errno 98] Address already in use
```

**Cause:** Another vLLM process (or another app) is already using port 8000.

**Fix:**

```bash
# Find what's using port 8000
lsof -i :8000

# Kill it
kill -9 <PID>

# Or use a different port
vllm serve ... --port 8001
```

---

### Slow first request

Not an error — it's expected. The first request triggers JIT (Just-In-Time) CUDA kernel compilation. This can take 30–120 seconds. Subsequent requests are fast. If you need consistent first-request latency, use `--enforce-eager` to disable JIT compilation (slower peak throughput, predictable latency).

---

### Trust remote code error

```
ValueError: The model contains custom code...set trust_remote_code=True
```

**Fix:** Add `--trust-remote-code` to your serve command. Read the model's HuggingFace page first to confirm you trust it.

---

## 14. Best practices for pip deployments

These are things professionals do that beginners skip — learn them now.

### Keep vLLM in a virtualenv, always

Never install vLLM into the system Python. It has dozens of heavy dependencies (PyTorch, specific CUDA versions) that will conflict with other projects. Always activate your venv before starting the server:

```bash
source /workspace/vllm-env/bin/activate
vllm serve ...
```

### Pin your vLLM version for reproducibility

vLLM moves fast — major API changes and performance improvements ship weekly. If you get a working config, pin the version:

```bash
uv pip install vllm==0.8.5  # replace with current stable
pip freeze > /workspace/requirements-vllm.txt
```

This way you can recreate the exact environment later.

### Use nohup or tmux to keep the server running after SSH disconnect

If your SSH session disconnects, your vLLM process dies:

```bash
# Option 1: nohup (simple)
nohup vllm serve Qwen/Qwen2.5-7B-Instruct \
  --host 0.0.0.0 --port 8000 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 8192 \
  > /workspace/vllm.log 2>&1 &

# Check it's running
tail -f /workspace/vllm.log

# Option 2: tmux (better — allows re-attaching)
tmux new -s vllm
vllm serve ...
# Detach with Ctrl+B, then D
# Re-attach with:
tmux attach -t vllm
```

### Never use `--gpu-memory-utilization 1.0`

Always keep at least 5–10% headroom. At 100%, any slight overhead from CUDA kernels or a large unexpected batch will crash your process.

### Set `--api-key` even for personal use

Get into the habit. Add a simple key so you don't accidentally expose an open endpoint:

```bash
vllm serve ... --api-key "my-secret-key-123"
```

Then in curl:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer my-secret-key-123" \
  ...
```

### Log to a file with timestamps

Always save your logs. You'll thank yourself when debugging:

```bash
vllm serve ... 2>&1 | tee -a /workspace/vllm_$(date +%Y%m%d_%H%M%S).log
```

### Always set `HF_HOME` to your persistent volume

We covered this in section 5, but it bears repeating: a 7B model download is ~14 GB and takes 5–10 minutes on RunPod's network. Without a persistent cache, you pay this cost every single time you restart. With a mounted volume at `/workspace/hf_cache`, you pay it once.

### Start small with `--max-model-len`

Set `--max-model-len` to what you actually need, not to the model's theoretical maximum. A Qwen2.5-7B model can do 128K context, but you almost never need that. Starting at 8192 reserves far less KV cache, leaving more headroom for concurrent requests.

### Use `--enable-prefix-caching` for chat applications

If you're running a chatbot or an agent where many requests share the same long system prompt, prefix caching is a free 30–70% latency improvement. It caches the KV state of the repeated prefix so the model doesn't reprocess it:

```bash
vllm serve ... --enable-prefix-caching
```

Watch the `Prefix cache hit rate` in the logs climb as requests come in.

### Understand the GPU during inference

Keep `nvidia-smi` running in a second terminal. When a request comes in:

- GPU utilization should jump to 70–100%
- Memory used should increase slightly (KV cache for that request)
- When the request finishes, memory should decrease back to baseline

If utilization never climbs high, you're sending requests too slowly to saturate the GPU. This is fine for single-user testing — matters for benchmarking in Phase 3.

---

## 15. What you learned — exit checklist

Before moving to Phase 1b (Docker) or Phase 2 (Benchmarking), verify you can answer all of these from memory:

**Concepts:**

- [ ] What does PagedAttention solve, and how does it differ from naive KV cache allocation?
- [ ] What is continuous batching and why does it increase throughput?
- [ ] What is the difference between FP16 and BF16, and when do you choose each?
- [ ] What does `--gpu-memory-utilization 0.90` actually allocate? (Answer: 90% of VRAM pre-allocated for KV cache blocks)
- [ ] Why would you lower `--max-model-len` from 128K to 8K even if the model supports 128K?
- [ ] What does `--tensor-parallel-size` do, and when would you use it?
- [ ] What is the difference between prompt tokens and completion tokens in the usage object?

**Practical skills:**

- [ ] Start a vLLM server on RunPod via pip
- [ ] Query the `/v1/chat/completions` endpoint with curl
- [ ] Query it with the Python openai library
- [ ] Read the startup logs and find the KV cache size line
- [ ] Watch GPU memory with `nvidia-smi` while requests run
- [ ] Keep the server running after SSH disconnect with `nohup` or `tmux`
- [ ] Fix an OOM error by adjusting flags

**Things you've seen in the logs:**

- [ ] `GPU KV cache size: X tokens` — what does this tell you?
- [ ] `Maximum concurrency for Y tokens per request: Z` — what does this mean?
- [ ] `Avg prompt throughput: X tokens/s, Avg generation throughput: Y tokens/s` — what's the difference between these two numbers?

---

## Quick reference card

```bash
# Environment setup (run once per pod)
cd /workspace
pip install uv
uv venv --python 3.12 --seed vllm-env
source /workspace/vllm-env/bin/activate
export HF_TOKEN="hf_xxx"
export HF_HOME="/workspace/hf_cache"
uv pip install vllm --torch-backend=auto

# Start server (RTX 4090 — 24GB)
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --host 0.0.0.0 --port 8000 \
  --dtype auto \
  --gpu-memory-utilization 0.85 \
  --max-model-len 4096 \
  --max-num-seqs 128 \
  --served-model-name qwen2.5-7b

# Start server (A40/L40S — 48GB)
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --host 0.0.0.0 --port 8000 \
  --dtype auto \
  --gpu-memory-utilization 0.90 \
  --max-model-len 8192 \
  --max-num-seqs 256 \
  --served-model-name qwen2.5-7b \
  --enable-prefix-caching

# Health check
curl http://localhost:8000/health

# Chat completion
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen2.5-7b","messages":[{"role":"user","content":"Hello"}],"max_tokens":100}'

# Watch GPU
watch -n 2 "nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader"
```

---

## What's next — Phase 1b: Docker

In Phase 1b, you'll run the exact same model and flags but inside a Docker container. The key things you'll learn:

- How `--gpus all` passes the GPU into the container
- Volume mounts for the model cache (so you don't re-download)
- Port mapping (`-p 8000:8000`)
- Environment variable injection (`-e HF_TOKEN=...`)
- The difference between the container's filesystem and the host's
- Writing a `docker-compose.yml` for a repeatable deployment

Everything in this guide (the flags, the API, the metrics) carries forward unchanged. The only difference is the wrapper around it.

---

_Last updated: May 2026. Based on vLLM stable documentation at docs.vllm.ai._
