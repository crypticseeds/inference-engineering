# Benchmarks

Reproducible workloads, raw outputs, and summarized results, one directory per run.

Every result must identify the model revision, engine version, hardware, input and output token lengths, sampling settings, concurrency or request rate, and whether requests were preempted.

## Week 4: prefix cache comparison

Use vLLM 0.21.0's `vllm bench serve` as the benchmark client for both engines. Its `prefix_repetition` dataset exercises the shared OpenAI-compatible `/v1/completions` endpoint. Run 100 requests per condition, one at a time. Compare 100 distinct prefixes to 5 repeated prefixes, keeping the requested prefix, suffix, and output lengths fixed. [Benchmark CLI reference](https://docs.vllm.ai/en/v0.21.0/benchmarking/cli/)

From the repository root, run `modal serve deploy/modal/vllm_qwen3_8b.py` and `modal serve deploy/modal/sglang_qwen3_8b.py` in separate terminals. Both dev endpoints allow unauthenticated requests. Run the following from the repository root, replacing the placeholders with the HTTPS base URLs printed by Modal.

```bash
VLLM_BASE_URL='<VLLM_BASE_URL>'
SGLANG_BASE_URL='<SGLANG_BASE_URL>'

bench_prefix() {
  vllm bench serve \
    --backend openai --endpoint /v1/completions --base-url "$1" \
    --model Qwen/Qwen3-8B --served-model-name qwen3-8b \
    --dataset-name prefix_repetition --num-prompts 100 \
    --prefix-repetition-prefix-len 2048 --prefix-repetition-suffix-len 64 \
    --prefix-repetition-output-len 32 --prefix-repetition-num-prefixes "$2" \
    --seed "$3" --max-concurrency 1 --temperature 0 --ignore-eos \
    --ready-check-timeout-sec 900 --metric-percentiles 50,95,99 \
    --save-result --save-detailed --result-filename "$4"
}

OPENAI_API_KEY= bench_prefix "$VLLM_BASE_URL" 100 42 vllm-unique.json
OPENAI_API_KEY= bench_prefix "$VLLM_BASE_URL" 5 43 vllm-repeated.json
OPENAI_API_KEY= bench_prefix "$SGLANG_BASE_URL" 100 42 sglang-unique.json
OPENAI_API_KEY= bench_prefix "$SGLANG_BASE_URL" 5 43 sglang-repeated.json
```

Check that all 100 requests completed per run. Report p50, p95, and p99 client TTFT, the recorded token lengths, and any failed requests. Use the same seeds across engines and different seeds across conditions. The tool sends a readiness request using the first prompt, which can warm one prefix. The repeated-prefix condition also includes the first use of each unprimed prefix. Report both conditions as workloads with different levels of reuse.

The paired chat probe is available to check whether both engines actually report reused tokens. Run it before the benchmark to keep its prompts separate from the measured workload.

```bash
python3 benchmarks/prefix_cache.py --vllm "$VLLM_BASE_URL" --sglang "$SGLANG_BASE_URL"
```

SGLang 0.5.9 also ships `python3 -m sglang.bench_serving` with a `generated-shared-prefix` dataset. Its `sglang-oai` and `vllm` backends both use the same `/v1/completions` request handler. For a second-tool cross-check, use that client against both engines and report the workload and results together. [SGLang 0.5.9 benchmark source](https://github.com/sgl-project/sglang/blob/v0.5.9/python/sglang/bench_serving.py)

Press Ctrl-C in both serving terminals when finished. The live benchmark is pending.

## Runs

- [2026-10-18 Modal A10 dashboard validation](./2026-10-18-modal-a10-qwen3-8b/README.md) compares server-side and client-side latency.
- [2026-11-23 Modal A10 GuideLLM results](./2026-11-23-modal-a10-guidellm/README.md) records the sequential baseline and the 1 and 2 req/s stages.
