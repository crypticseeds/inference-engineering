"""vLLM serving Qwen3-8B on one Modal GPU.

GPU_PROFILE selects the GPU and engine limits (default A10). MODEL_VARIANT
selects the checkpoint (default bf16); every variant is served as qwen3-8b.

    modal run deploy/modal/vllm_qwen3_8b.py                      # smoke test
    modal serve deploy/modal/vllm_qwen3_8b.py                    # dev endpoint
    GPU_PROFILE=H100 modal serve deploy/modal/vllm_qwen3_8b.py   # H100 profile
    GPU_PROFILE=L40S MODEL_VARIANT=awq modal serve deploy/modal/vllm_qwen3_8b.py

The endpoint is unauthenticated. Never `modal deploy` this file.
"""

import json
import os
import subprocess
import time
import urllib.error
import urllib.request

import modal

# Qwen's own checkpoints. fp8 is block-wise FP8 weights and activations; awq is
# 4-bit AWQ weights with 16-bit activations.
VARIANTS = {
    "bf16": ("Qwen/Qwen3-8B", "b968826d9c46dd6066d109eabc6255188de91218"),
    "fp8": ("Qwen/Qwen3-8B-FP8", "220b46e3b2180893580a4454f21f22d3ebb187d3"),
    "awq": ("Qwen/Qwen3-8B-AWQ", "4da05a8edb55c6046cce958586c33b61da07bb79"),
}
MODEL_VARIANT = os.environ.get("MODEL_VARIANT", "bf16")
if MODEL_VARIANT not in VARIANTS:
    raise ValueError(f"MODEL_VARIANT must be one of {sorted(VARIANTS)}, got {MODEL_VARIANT!r}")
MODEL_NAME, MODEL_REVISION = VARIANTS[MODEL_VARIANT]
SERVED_MODEL_NAME = "qwen3-8b"
CONTEXT_LENGTH = 8192
PORT = 8000
LOCAL_URL = f"http://127.0.0.1:{PORT}"
STARTUP_TIMEOUT_SECONDS = 15 * 60
VLLM_VERSION = "0.21.0"

# A10 keeps the eager, 16-request configuration of the November 2026 sweeps.
# H100 and L40S enable CUDA graphs and admit up to 256 running requests.
# L40S (Ada) is the cheapest profile with native FP8 compute, used for Week 6.
PROFILES = {
    "A10": {"max_running_requests": 16, "cuda_graphs": False},
    "H100": {"max_running_requests": 256, "cuda_graphs": True},
    "L40S": {"max_running_requests": 256, "cuda_graphs": True},
}
GPU_PROFILE = os.environ.get("GPU_PROFILE", "A10")
if GPU_PROFILE not in PROFILES:
    raise ValueError(f"GPU_PROFILE must be one of {sorted(PROFILES)}, got {GPU_PROFILE!r}")
PROFILE = PROFILES[GPU_PROFILE]

image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install(f"vllm=={VLLM_VERSION}")
    .env(
        {
            "GPU_PROFILE": GPU_PROFILE,
            "MODEL_VARIANT": MODEL_VARIANT,
            "HF_XET_HIGH_PERFORMANCE": "1",
            "VLLM_LOG_STATS_INTERVAL": "1",
        }
    )
)
volumes = {
    "/root/.cache/huggingface": modal.Volume.from_name(
        "inference-engineering-huggingface-cache", create_if_missing=True
    ),
    "/root/.cache/vllm": modal.Volume.from_name(
        "inference-engineering-vllm-cache", create_if_missing=True
    ),
}
# One app per variant, so the variants can be served side by side.
app = modal.App(
    "inference-engineering-qwen3-8b" + ("" if MODEL_VARIANT == "bf16" else f"-{MODEL_VARIANT}")
)


def start_server(host: str) -> subprocess.Popen:
    # Prefix caching is on in both engines, and both report cached prompt
    # tokens in the response usage (prompt_tokens_details.cached_tokens).
    command = [
        "vllm", "serve", MODEL_NAME,
        "--revision", MODEL_REVISION,
        "--served-model-name", SERVED_MODEL_NAME,
        "--host", host,
        "--port", str(PORT),
        "--dtype", "bfloat16",
        "--max-model-len", str(CONTEXT_LENGTH),
        "--max-num-seqs", str(PROFILE["max_running_requests"]),
        "--gpu-memory-utilization", "0.90",
        "--enable-prefix-caching",
        "--enable-prompt-tokens-details",
    ]
    if not PROFILE["cuda_graphs"]:
        command.append("--enforce-eager")
    print("Starting:", " ".join(command), flush=True)
    return subprocess.Popen(command)


def stop_server(server: subprocess.Popen) -> None:
    if server.poll() is None:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait()


@app.server(
    image=image,
    gpu=GPU_PROFILE,
    max_containers=1,
    min_containers=0,
    scaledown_window=120,
    startup_timeout=STARTUP_TIMEOUT_SECONDS,
    volumes=volumes,
    port=PORT,
    unauthenticated=True,
    # Keep the GPU in the benchmark clients' region. Modal does not allow
    # pinning the cloud here, so the startup log records which one it chose.
    compute_region="us-east",
)
class Server:
    @modal.enter()
    def start(self):
        print(
            "GPU placement:",
            os.environ.get("MODAL_CLOUD_PROVIDER", "unknown"),
            os.environ.get("MODAL_REGION", "unknown"),
            flush=True,
        )
        self.process = start_server("0.0.0.0")

    @modal.exit()
    def stop(self):
        stop_server(self.process)


@app.function(image=image, gpu=GPU_PROFILE, max_containers=1, timeout=STARTUP_TIMEOUT_SECONDS, volumes=volumes)
def smoke_test():
    server = start_server("127.0.0.1")
    try:
        wait_until_healthy(server)
        check_model_list()
        check_streaming_chat()
    finally:
        stop_server(server)


@app.local_entrypoint()
def main():
    smoke_test.remote()


def wait_until_healthy(server: subprocess.Popen) -> None:
    deadline = time.monotonic() + STARTUP_TIMEOUT_SECONDS - 60
    while time.monotonic() < deadline:
        if server.poll() is not None:
            raise RuntimeError(f"vLLM exited during startup ({server.returncode})")
        try:
            with urllib.request.urlopen(f"{LOCAL_URL}/health", timeout=5):
                print("Health check passed", flush=True)
                return
        except (urllib.error.URLError, ConnectionError):
            time.sleep(2)
    raise TimeoutError("vLLM did not become healthy before the startup timeout")


def check_model_list() -> None:
    with urllib.request.urlopen(f"{LOCAL_URL}/v1/models", timeout=10) as response:
        model_ids = {model["id"] for model in json.load(response)["data"]}
    if SERVED_MODEL_NAME not in model_ids:
        raise RuntimeError(f"Expected {SERVED_MODEL_NAME}, got {sorted(model_ids)}")
    print(f"Model discovery passed: {SERVED_MODEL_NAME}", flush=True)


def check_streaming_chat() -> None:
    request = urllib.request.Request(
        f"{LOCAL_URL}/v1/chat/completions",
        data=json.dumps(
            {
                "model": SERVED_MODEL_NAME,
                "messages": [{"role": "user", "content": "Say hello in one sentence."}],
                "max_tokens": 64,
                "temperature": 0,
                "chat_template_kwargs": {"enable_thinking": False},
                "stream": True,
            }
        ).encode(),
        headers={"Content-Type": "application/json"},
    )
    text = []
    done = False
    with urllib.request.urlopen(request, timeout=60) as response:
        if response.headers.get_content_type() != "text/event-stream":
            raise RuntimeError("Expected a server-sent event stream")
        for line in response:
            if not line.startswith(b"data: "):
                continue
            data = line.removeprefix(b"data: ").strip()
            if data == b"[DONE]":
                done = True
                break
            for choice in json.loads(data)["choices"]:
                text.append(choice["delta"].get("content") or "")
    if not done or not "".join(text).strip():
        raise RuntimeError("vLLM did not stream a complete, non-empty reply")
    print("Streaming chat passed:", "".join(text), flush=True)
