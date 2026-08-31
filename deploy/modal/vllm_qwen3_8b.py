import json
import subprocess
import time
import urllib.error
import urllib.request

import modal


MODEL_NAME = "Qwen/Qwen3-8B"
MODEL_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
SERVED_MODEL_NAME = "qwen3-8b"
VLLM_VERSION = "0.21.0"
VLLM_PORT = 8000
STARTUP_TIMEOUT_SECONDS = 15 * 60

vllm_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.9.0-devel-ubuntu22.04",
        add_python="3.12",
    )
    .entrypoint([])
    .uv_pip_install(f"vllm=={VLLM_VERSION}")
    .env(
        {
            "HF_XET_HIGH_PERFORMANCE": "1",
            "VLLM_LOG_STATS_INTERVAL": "1",
        }
    )
)

hf_cache = modal.Volume.from_name(
    "inference-engineering-huggingface-cache",
    create_if_missing=True,
)
vllm_cache = modal.Volume.from_name(
    "inference-engineering-vllm-cache",
    create_if_missing=True,
)

app = modal.App("inference-engineering-qwen3-8b")


@app.server(
    image=vllm_image,
    gpu="A10",
    max_containers=1,
    min_containers=0,
    scaledown_window=120,
    startup_timeout=STARTUP_TIMEOUT_SECONDS,
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/root/.cache/vllm": vllm_cache,
    },
    port=VLLM_PORT,
    unauthenticated=True,
)
class Server:
    @modal.enter()
    def start(self):
        command = [
            "vllm",
            "serve",
            MODEL_NAME,
            "--revision",
            MODEL_REVISION,
            "--served-model-name",
            SERVED_MODEL_NAME,
            "--host",
            "0.0.0.0",
            "--port",
            str(VLLM_PORT),
            "--dtype",
            "auto",
            "--gpu-memory-utilization",
            "0.90",
            "--max-model-len",
            "8192",
            "--max-num-seqs",
            "16",
            "--enforce-eager",
        ]
        print("Starting:", " ".join(command), flush=True)
        self.process = subprocess.Popen(command)

    @modal.exit()
    def stop(self):
        self.process.terminate()


def request_json(
    url: str,
    method: str = "GET",
    payload: dict | None = None,
    timeout: int = 60,
) -> dict | str:
    body = json.dumps(payload).encode() if payload is not None else None
    headers = {"Content-Type": "application/json"} if body is not None else {}
    request = urllib.request.Request(
        url,
        data=body,
        headers=headers,
        method=method,
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        content = response.read().decode()
        if response.headers.get_content_type() == "application/json":
            return json.loads(content)
        return content


@app.local_entrypoint()
def test():
    base_url = Server.get_url()
    deadline = time.monotonic() + STARTUP_TIMEOUT_SECONDS

    print(f"Waiting for vLLM at {base_url}")
    while time.monotonic() < deadline:
        try:
            request_json(f"{base_url}/health", timeout=30)
            break
        except urllib.error.HTTPError as error:
            if error.code != 503:
                raise
        except urllib.error.URLError:
            pass
        time.sleep(2)
    else:
        raise TimeoutError("vLLM did not become healthy before the startup timeout")

    print("Health check passed")

    models = request_json(f"{base_url}/v1/models")
    assert isinstance(models, dict)
    model_ids = {model["id"] for model in models.get("data", [])}
    assert SERVED_MODEL_NAME in model_ids, (
        f"Expected {SERVED_MODEL_NAME!r} in served models, got {sorted(model_ids)!r}"
    )
    print("Models:", json.dumps(models, indent=2))

    response = request_json(
        f"{base_url}/v1/chat/completions",
        method="POST",
        payload={
            "model": SERVED_MODEL_NAME,
            "messages": [
                {
                    "role": "user",
                    "content": "Explain PagedAttention in two short sentences.",
                }
            ],
            "max_tokens": 96,
            "temperature": 0.0,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        timeout=180,
    )
    assert isinstance(response, dict)
    assert response.get("model") == SERVED_MODEL_NAME
    choices = response.get("choices")
    assert isinstance(choices, list) and choices
    message = choices[0].get("message", {})
    assert message.get("role") == "assistant"
    assert message.get("content", "").strip()
    print("Chat response:", json.dumps(response, indent=2))
    print("Smoke test passed")
