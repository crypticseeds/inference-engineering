import json
import subprocess
import time
import urllib.error
import urllib.request

import modal


MODEL_NAME = "Qwen/Qwen3-8B"
MODEL_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
SERVED_MODEL_NAME = "qwen3-8b"
PORT = 30000
BASE_URL = f"http://127.0.0.1:{PORT}"

image = modal.Image.from_registry("lmsysorg/sglang:v0.5.9").entrypoint([])
hf_cache = modal.Volume.from_name(
    "inference-engineering-huggingface-cache",
    create_if_missing=True,
)
app = modal.App("inference-engineering-sglang-qwen3-8b")


def start_server(host: str) -> subprocess.Popen:
    # Match the model revision, precision, context cap, and eager mode of vLLM.
    command = [
        "python3", "-m", "sglang.launch_server",
        "--model-path", MODEL_NAME,
        "--revision", MODEL_REVISION,
        "--served-model-name", SERVED_MODEL_NAME,
        "--host", host,
        "--port", str(PORT),
        "--dtype", "bfloat16",
        "--context-length", "8192",
        "--max-running-requests", "16",
        "--mem-fraction-static", "0.85",
        "--disable-cuda-graph",
        "--enable-metrics",
        "--enable-cache-report",
    ]
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
    gpu="A10",
    max_containers=1,
    min_containers=0,
    scaledown_window=120,
    startup_timeout=15 * 60,
    volumes={"/root/.cache/huggingface": hf_cache},
    port=PORT,
    unauthenticated=True,
)
class Server:
    @modal.enter()
    def start(self):
        self.process = start_server("0.0.0.0")

    @modal.exit()
    def stop(self):
        stop_server(self.process)


@app.function(
    image=image,
    gpu="A10",
    max_containers=1,
    timeout=15 * 60,
    volumes={"/root/.cache/huggingface": hf_cache},
)
def smoke_test():
    server = start_server("127.0.0.1")
    try:
        deadline = time.monotonic() + 12 * 60
        while time.monotonic() < deadline:
            if server.poll() is not None:
                raise RuntimeError(f"SGLang exited during startup ({server.returncode})")
            try:
                with urllib.request.urlopen(f"{BASE_URL}/health", timeout=5):
                    break
            except urllib.error.URLError:
                time.sleep(2)
        else:
            raise TimeoutError("SGLang did not become healthy")
        print("Health check passed", flush=True)

        with urllib.request.urlopen(f"{BASE_URL}/v1/models", timeout=10) as response:
            model_ids = {model["id"] for model in json.load(response)["data"]}
        if SERVED_MODEL_NAME not in model_ids:
            raise RuntimeError(f"Expected {SERVED_MODEL_NAME}, got {sorted(model_ids)}")
        print(f"Model discovery passed: {SERVED_MODEL_NAME}", flush=True)

        request = urllib.request.Request(
            f"{BASE_URL}/v1/chat/completions",
            data=json.dumps({
                "model": SERVED_MODEL_NAME,
                "messages": [{"role": "user", "content": "Say hello in one sentence."}],
                "max_tokens": 64,
                "temperature": 0,
                "chat_template_kwargs": {"enable_thinking": False},
                "stream": True,
            }).encode(),
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
            raise RuntimeError("SGLang did not stream a complete, non-empty reply")
        print("Streaming chat passed:", "".join(text), flush=True)
    finally:
        stop_server(server)
