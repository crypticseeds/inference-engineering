"""Run a GuideLLM 0.8.1 plan from a Modal CPU container instead of the Pi.

    modal run benchmarks/guidellm_modal.py --plan smoke --target https://<host> --stem <stem>

Each stage writes one JSON report to the inference-engineering-benchmark-results
Volume under <stem>/. Copy them out with:

    modal volume get inference-engineering-benchmark-results <stem> <local-dir>

Reports record the target URL. Replace it with MODAL_HOST before committing.
"""

import os
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

import modal

MODEL_NAME = "Qwen/Qwen3-8B"
MODEL_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
SERVED_MODEL_NAME = "qwen3-8b"
GUIDELLM_VERSION = "0.8.1"
RESULTS_DIR = "/results"
READY_TIMEOUT_SECONDS = 15 * 60

# Workload of the November 2026 sweeps: synthetic text, seed 42, 64 prompt and
# 16 output tokens, streamed /v1/chat/completions.
DATA = "kind=synthetic_text,prompt_tokens=64,output_tokens=16"

# Each stage: (report name, --profile value, --constraint values).
# a10-rate repeats the November sweeps: 4-request cap below 8 req/s, 64 above.
PLANS = {
    "smoke": {
        "request_timeout": 30,
        "stages": [("smoke-1rps", "kind=constant,rate=1,max_concurrency=4,warmup=5", ["kind=max_duration,seconds=20"])],
    },
    "a10-rate": {
        "request_timeout": 30,
        "stages": [("warmup", "kind=synchronous", ["kind=max_requests,count=3"])]
        + [
            (f"{rate}rps", f"kind=constant,rate={rate},max_concurrency={4 if rate < 8 else 64},warmup=5",
             ["kind=max_duration,seconds=60"])
            for rate in (1, 2, 4, 8, 16, 32)
        ],
    },
    # Short check of the H100 profile and client before the full sweep.
    "h100-dry": {
        "request_timeout": 120,
        "stages": [("warmup", "kind=synchronous", ["kind=max_requests,count=3"])]
        + [
            (f"c{streams}", f"kind=concurrent,streams={streams},warmup=5", ["kind=max_duration,seconds=30"])
            for streams in (10, 250)
        ],
    },
    # 16 CPUs so the client is less likely than the engine to limit throughput
    # at 1,000 concurrent streams.
    "h100-concurrency": {
        "request_timeout": 120,
        "cpu": 16,
        "stages": [("warmup", "kind=synchronous", ["kind=max_requests,count=3"])]
        + [
            (f"c{streams}", f"kind=concurrent,streams={streams},warmup=5", ["kind=max_duration,seconds=60"])
            for streams in (1, 10, 50, 100, 250, 500, 1000)
        ],
    },
    # Week 6: the same stages up to 250 concurrent, once per quantized variant.
    "quant-concurrency": {
        "request_timeout": 120,
        "stages": [("warmup", "kind=synchronous", ["kind=max_requests,count=3"])]
        + [
            (f"c{streams}", f"kind=concurrent,streams={streams},warmup=5", ["kind=max_duration,seconds=60"])
            for streams in (1, 10, 50, 100, 250)
        ],
    },
}

image = modal.Image.debian_slim(python_version="3.12").uv_pip_install(f"guidellm=={GUIDELLM_VERSION}")
results = modal.Volume.from_name("inference-engineering-benchmark-results", create_if_missing=True)
hf_cache = modal.Volume.from_name("inference-engineering-huggingface-cache", create_if_missing=True)
app = modal.App("inference-engineering-guidellm")


# Same region as the GPU dev endpoints (us-east in their hostnames), so client
# placement does not differ between engines or runs.
@app.function(
    image=image,
    region="us-east",
    cpu=8,
    memory=16384,
    timeout=45 * 60,
    volumes={RESULTS_DIR: results, "/root/.cache/huggingface": hf_cache},
)
def run_plan(plan: str, target: str, stem: str) -> list[str]:
    spec = PLANS[plan]
    print(
        "Client placement:",
        os.environ.get("MODAL_CLOUD_PROVIDER", "unknown"),
        os.environ.get("MODAL_REGION", "unknown"),
        flush=True,
    )
    wait_until_healthy(target)
    out_dir = Path(RESULTS_DIR) / stem
    out_dir.mkdir(parents=True, exist_ok=False)
    written = []
    for name, profile, constraints in spec["stages"]:
        path = out_dir / f"{name}.json"
        command = [
            "guidellm", "run",
            "--backend",
            f"kind=openai_http,target={target},model={SERVED_MODEL_NAME},verify=true,timeout={spec['request_timeout']}",
            "--tokenizer",
            f"kind=huggingface_auto,model={MODEL_NAME},load_kwargs.revision={MODEL_REVISION}",
            "--data", DATA,
            "--profile", profile,
            *[arg for constraint in constraints for arg in ("--constraint", constraint)],
            "--output", f"kind=json,path={path}",
            "--disable-console-interactive",
        ]
        print(f"== {name}: {profile}", flush=True)
        subprocess.run(command, check=True)
        results.commit()
        written.append(str(path.relative_to(RESULTS_DIR)))
    return written


def wait_until_healthy(target: str) -> None:
    # The GPU container may still be loading the model. GuideLLM exits on the
    # first failed /health check, so wait here instead.
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
def main(plan: str, target: str, stem: str):
    if plan not in PLANS:
        raise SystemExit(f"--plan must be one of {sorted(PLANS)}")
    cpu = PLANS[plan].get("cpu", 8)
    runner = run_plan.with_options(cpu=cpu, memory=cpu * 2048)
    for path in runner.remote(plan, target.rstrip("/"), stem):
        print("wrote", path)
