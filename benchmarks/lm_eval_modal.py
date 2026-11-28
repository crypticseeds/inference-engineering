"""Run the Week 6 quality proxy with lm-evaluation-harness from a Modal CPU container.

    modal run benchmarks/lm_eval_modal.py --target https://<host> --stem <stem>
    modal run benchmarks/lm_eval_modal.py --target https://<host> --stem <stem> --limit 5   # smoke

Tasks, against the served qwen3-8b through /v1/completions:

- gsm8k: grade-school maths, 5-shot, exact match over all 1,319 test questions
- wikitext: document-level perplexity on the WikiText-2 test set (rolling log-likelihood)

Results go to the inference-engineering-benchmark-results Volume under <stem>/:

    modal volume get inference-engineering-benchmark-results <stem> <local-dir>
"""

import subprocess
import time
import urllib.error
import urllib.request

import modal

MODEL_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
SERVED_MODEL_NAME = "qwen3-8b"
LM_EVAL_VERSION = "0.4.13"
RESULTS_DIR = "/results"
READY_TIMEOUT_SECONDS = 15 * 60
TASKS = "gsm8k,wikitext"
# The quantized checkpoints ship the same tokenizer as the BF16 model, so one
# pinned tokenizer serves every variant.
TOKENIZER = "/root/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/" + MODEL_REVISION

# The [api] extra omits transformers, which local-completions needs for its tokenizer.
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install(
    f"lm-eval[api]=={LM_EVAL_VERSION}", "transformers==5.22.2"
)
results = modal.Volume.from_name("inference-engineering-benchmark-results", create_if_missing=True)
hf_cache = modal.Volume.from_name("inference-engineering-huggingface-cache", create_if_missing=True)
app = modal.App("inference-engineering-lm-eval")


# Same region as the GPU dev endpoints and the other benchmark clients.
@app.function(
    image=image,
    region="us-east",
    cpu=4,
    memory=8192,
    timeout=90 * 60,
    volumes={RESULTS_DIR: results, "/root/.cache/huggingface": hf_cache},
)
def run_eval(target: str, stem: str, limit: int) -> str:
    wait_until_healthy(target)
    out_dir = f"{RESULTS_DIR}/{stem}"
    command = [
        "lm_eval",
        "--model", "local-completions",
        "--model_args",
        (
            f"model={SERVED_MODEL_NAME},base_url={target}/v1/completions,tokenizer={TOKENIZER},"
            "tokenizer_backend=huggingface,tokenized_requests=False,num_concurrent=32,max_retries=3"
        ),
        "--tasks", TASKS,
        "--batch_size", "1",
        "--output_path", out_dir,
        "--log_samples",
    ]
    if limit:
        command += ["--limit", str(limit)]
    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, check=True)
    results.commit()
    return stem


def wait_until_healthy(target: str) -> None:
    # Keep GPU cold starts out of the evaluation.
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
def main(target: str, stem: str, limit: int = 0):
    print("wrote", run_eval.remote(target.rstrip("/"), stem, limit))
