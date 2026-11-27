"""Shared-prefix TTFT benchmark from a Modal CPU container.

    modal run benchmarks/prefix_modal.py --engine vllm --target https://<host> --stem <stem>
    modal run benchmarks/prefix_modal.py --engine sglang --target https://<host> --stem <stem>

Two clients, each run in the image of the engine that ships it:

- vLLM 0.21.0 `vllm bench serve`, prefix_repetition dataset (primary)
- SGLang 0.5.9 `sglang.bench_serving`, generated-shared-prefix dataset (cross-check)

Both send raw prompts to /v1/completions. Every stage uses its own seed, so no
stage reuses another stage's prefixes. Reports go to the
inference-engineering-benchmark-results Volume under <stem>/:

    modal volume get inference-engineering-benchmark-results <stem> <local-dir>

Reports record the target URL. Replace it with MODAL_HOST before committing.
"""

import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

import modal

MODEL_NAME = "Qwen/Qwen3-8B"
MODEL_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
SERVED_MODEL_NAME = "qwen3-8b"
RESULTS_DIR = "/results"
READY_TIMEOUT_SECONDS = 15 * 60
# The engines downloaded this revision into the shared cache Volume. Loading the
# tokenizer from it pins the revision for both clients.
TOKENIZER = (
    "/root/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/" + MODEL_REVISION
)

PREFIX_LEN = 2048
SUFFIX_LEN = 64
OUTPUT_LEN = 32

# (report name, prefixes, requests, seed, request rate, max concurrency).
# One at a time: 100 distinct prefixes vs 5 prefixes reused 20 times each.
# Loaded: the same two workloads at a fixed 2 req/s, about the rate at which
# distinct 2,112-token prompts should approach the A10's prefill capacity.
VLLM_STAGES = [
    ("unique-c1", 100, 100, 42, "inf", 1),
    ("repeated-c1", 5, 100, 43, "inf", 1),
    ("unique-2rps", 120, 120, 44, "2", 32),
    ("repeated-2rps", 6, 120, 45, "2", 32),
]
# (report name, groups, prompts per group, seed), one request at a time.
SGLANG_STAGES = [
    ("gsp-unique-c1", 100, 1, 46),
    ("gsp-repeated-c1", 5, 20, 47),
]

vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install("vllm==0.21.0")
)
sglang_image = modal.Image.from_registry("lmsysorg/sglang:v0.5.9").entrypoint([])
results = modal.Volume.from_name("inference-engineering-benchmark-results", create_if_missing=True)
hf_cache = modal.Volume.from_name("inference-engineering-huggingface-cache", create_if_missing=True)
app = modal.App("inference-engineering-prefix-bench")

# Same region as the GPU dev endpoints and the GuideLLM client.
function_options = {
    "region": "us-east",
    "cpu": 4,
    "memory": 8192,
    "timeout": 60 * 60,
    "volumes": {RESULTS_DIR: results, "/root/.cache/huggingface": hf_cache},
}


@app.function(image=vllm_image, **function_options)
def run_vllm_bench(target: str, out_dir: str) -> list[str]:
    wait_until_healthy(target)
    written = []
    for name, prefixes, requests, seed, rate, concurrency in VLLM_STAGES:
        command = [
            "vllm", "bench", "serve",
            "--backend", "openai", "--endpoint", "/v1/completions", "--base-url", target,
            "--model", MODEL_NAME, "--tokenizer", TOKENIZER,
            "--served-model-name", SERVED_MODEL_NAME,
            "--dataset-name", "prefix_repetition", "--num-prompts", str(requests),
            "--prefix-repetition-prefix-len", str(PREFIX_LEN),
            "--prefix-repetition-suffix-len", str(SUFFIX_LEN),
            "--prefix-repetition-output-len", str(OUTPUT_LEN),
            "--prefix-repetition-num-prefixes", str(prefixes),
            "--seed", str(seed), "--temperature", "0", "--ignore-eos",
            "--request-rate", rate, "--burstiness", "10000", "--max-concurrency", str(concurrency),
            "--metric-percentiles", "50,95,99",
            "--save-result", "--save-detailed",
            "--result-dir", f"{RESULTS_DIR}/{out_dir}", "--result-filename", f"{name}.json",
        ]
        written.append(run_stage(name, command, out_dir))
    return written


@app.function(image=sglang_image, **function_options)
def run_sglang_bench(engine: str, target: str, out_dir: str) -> list[str]:
    wait_until_healthy(target)
    # Both choices post to /v1/completions with the same request handler.
    backend = "sglang-oai" if engine == "sglang" else "vllm"
    written = []
    for name, groups, per_group, seed in SGLANG_STAGES:
        command = [
            "python3", "-m", "sglang.bench_serving",
            "--backend", backend, "--base-url", target,
            "--model", MODEL_NAME, "--tokenizer", TOKENIZER,
            "--served-model-name", SERVED_MODEL_NAME,
            "--dataset-name", "generated-shared-prefix",
            "--gsp-num-groups", str(groups), "--gsp-prompts-per-group", str(per_group),
            "--gsp-system-prompt-len", str(PREFIX_LEN), "--gsp-question-len", str(SUFFIX_LEN),
            "--gsp-output-len", str(OUTPUT_LEN), "--gsp-range-ratio", "1.0",
            "--num-prompts", str(groups * per_group), "--seed", str(seed),
            "--max-concurrency", "1",
            "--output-file", f"{RESULTS_DIR}/{out_dir}/{name}.jsonl", "--output-details",
        ]
        written.append(run_stage(name, command, out_dir))
    return written


def run_stage(name: str, command: list[str], out_dir: str) -> str:
    Path(RESULTS_DIR, out_dir).mkdir(parents=True, exist_ok=True)
    print(f"== {name}", flush=True)
    subprocess.run(command, check=True)
    results.commit()
    return f"{out_dir}/{name}"


def wait_until_healthy(target: str) -> None:
    # Keep GPU cold starts out of the measured stages.
    deadline = time.monotonic() + READY_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{target}/health", timeout=30):
                print("Endpoint healthy", flush=True)
                return
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            time.sleep(5)
    raise TimeoutError(f"{target}/health did not return 200 within {READY_TIMEOUT_SECONDS} s")


@app.local_entrypoint()
def main(engine: str, target: str, stem: str):
    if engine not in ("vllm", "sglang"):
        raise SystemExit("--engine must be vllm or sglang")
    target = target.rstrip("/")
    out_dir = f"{stem}/{engine}"
    # vLLM's client first, then SGLang's, against the same warm endpoint.
    for path in run_vllm_bench.remote(target, f"{out_dir}/vllm-bench"):
        print("wrote", path)
    for path in run_sglang_bench.remote(engine, target, f"{out_dir}/sglang-bench"):
        print("wrote", path)
