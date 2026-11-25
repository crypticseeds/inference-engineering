"""Compare client TTFT for fresh and reused prefixes on warm serving engines."""

import argparse
import http.client
import json
import os
import statistics
import sys
import time
import uuid
from urllib.parse import urlsplit


MODEL = "qwen3-8b"
PAIRS = 10
CONTEXT = (
    "The warehouse catalog lists spare conveyor belts, optical sensors, "
    "steel brackets, shipping labels, circuit boards, and motor assemblies.\n"
) * 96


def sample(connection, headers, prefix, subject):
    body = json.dumps({
        "model": MODEL,
        "messages": [
            {"role": "system", "content": prefix},
            {"role": "user", "content": f"Reply with one short sentence about {subject}."},
        ],
        "temperature": 0,
        "max_tokens": 24,
        "chat_template_kwargs": {"enable_thinking": False},
        "stream": True,
        "stream_options": {"include_usage": True},
    })
    start = time.perf_counter()
    connection.request("POST", "/v1/chat/completions", body, headers)
    with connection.getresponse() as response:
        if response.status != 200:
            raise RuntimeError(f"HTTP {response.status} during chat request")
        ttft = None
        usage = None
        done = False
        for line in response:
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if data == b"[DONE]":
                done = True
                break
            chunk = json.loads(data)
            usage = chunk.get("usage") or usage
            if ttft is None and any(c.get("delta", {}).get("content") for c in chunk["choices"]):
                ttft = (time.perf_counter() - start) * 1000
        # Consume the rest of the HTTP body so the same connection can be reused.
        response.read()
    if not done or ttft is None or usage is None:
        raise RuntimeError("Expected a complete chat stream with content and token usage")
    return {
        "ttft_ms": round(ttft, 3),
        "prompt_tokens": usage["prompt_tokens"],
        "completion_tokens": usage["completion_tokens"],
        "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0),
    }


def compare(url, prefixes, proxy_headers):
    target = urlsplit(url)
    if target.scheme != "https" or not target.hostname or target.path not in ("", "/"):
        raise ValueError("Use the HTTPS base URL printed by Modal, without /v1")
    headers = {"Content-Type": "application/json", **proxy_headers}
    connection = http.client.HTTPSConnection(target.hostname, target.port, timeout=45)
    try:
        # Probe readiness separately so cold-start time cannot enter the samples.
        deadline = time.monotonic() + 15 * 60
        while True:
            connection.request("GET", "/health", headers=headers)
            with connection.getresponse() as response:
                status = response.status
                response.read()
            if status == 200:
                break
            if status != 503 or time.monotonic() >= deadline:
                raise RuntimeError(f"HTTP {status} while waiting for engine health")
            time.sleep(2)
        sample(connection, headers, "Benchmark warmup.", "rain")
        pairs = []
        for prefix in prefixes:
            miss = sample(connection, headers, prefix, "rain")
            hit = sample(connection, headers, prefix, "wind")
            if miss["cached_tokens"] > 0.1 * miss["prompt_tokens"] or not (
                hit["cached_tokens"] >= 0.75 * hit["prompt_tokens"]
            ):
                raise RuntimeError(f"Cache reuse could not be verified: {miss=}, {hit=}")
            pairs.append({"miss": miss, "hit": hit})
        return {
            "miss_p50_ms": statistics.median(pair["miss"]["ttft_ms"] for pair in pairs),
            "hit_p50_ms": statistics.median(pair["hit"]["ttft_ms"] for pair in pairs),
            "pairs": pairs,
        }
    finally:
        connection.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vllm", required=True, help="vLLM HTTPS base URL")
    parser.add_argument("--sglang", required=True, help="SGLang HTTPS base URL")
    args = parser.parse_args()
    token = os.getenv("MODAL_PROXY_BEARER")
    sglang_headers = {"Authorization": f"Bearer {token}"} if token else {}
    prefixes = [f"Catalog {uuid.uuid4().hex}\n{CONTEXT}" for _ in range(PAIRS)]
    try:
        print("Measuring vLLM…", file=sys.stderr)
        vllm = compare(args.vllm, prefixes, {})
        print("Measuring SGLang…", file=sys.stderr)
        sglang = compare(args.sglang, prefixes, sglang_headers)
    except (OSError, RuntimeError, ValueError, http.client.HTTPException) as error:
        parser.exit(1, f"Benchmark stopped: {error}\n")
    print(json.dumps({"model": MODEL, "vllm": vllm, "sglang": sglang}, indent=2))


if __name__ == "__main__":
    main()
