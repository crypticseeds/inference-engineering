# Week 6: quantization variants on an L40S

Roadmap deliverable: FP8 and INT4 variants benchmarked against the 16-bit baseline on throughput and a quality proxy. The quality table is the deliverable, not the speedup.

## Decisions

| Decision | Choice | Why |
|---|---|---|
| GPU | One Modal L40S (Ada, 48 GB) per variant, region pinned to us-east (`compute_region`; Modal does not allow pinning the cloud) | Cheapest GPU with native FP8 tensor cores. On the A10 (Ampere), FP8 checkpoints run weight-only, which would not show what FP8 compute buys |
| Variants | `Qwen/Qwen3-8B` (BF16), `Qwen/Qwen3-8B-FP8` (block-wise FP8), `Qwen/Qwen3-8B-AWQ` (INT4 weights), revisions pinned in the launcher | Same publisher for all three |
| Engine | vLLM 0.21.0, `GPU_PROFILE=L40S`: CUDA graphs, 256 running requests, 90% memory, 8,192 context, KV cache in BF16 for every variant | Only the weights change between variants |
| Throughput | GuideLLM `quant-concurrency` plan: 1, 10, 50, 100, 250 concurrent, 60 s each, 72-token prompts / 16 output tokens (Week 5 workload) | Comparable with Week 5 |
| Memory | Weight memory and KV cache capacity from the vLLM startup log | Where INT4 and FP8 should save the most |
| Quality | lm-evaluation-harness 0.4.13 `local-completions`: GSM8K 5-shot exact match (1,319 questions) and WikiText-2 document perplexity | Reasoning plus language modelling |

## Steps and checks

1. **Serve the three variants side by side.** Check: `/health` returns 200 on all three; the startup log names the expected quantization method.
2. **Quality smoke, `--limit 5`.** Check: lm-eval finishes on each variant and reports GSM8K exact match and WikiText perplexity; no request errors.
3. **Full quality run** on each variant. Check: 1,319 GSM8K samples and all WikiText documents scored.
4. **Throughput sweep** on each variant. Check: zero request errors; dispatch delay under 0.2 s.
5. **Write-up** in `benchmarks/<date>-modal-l40s-quantization/`, raw reports with hostnames removed. Check: the table has throughput, memory and quality per variant, with versions and commands.

Estimated GPU time about 25 to 35 minutes per variant, three L40S in parallel.
