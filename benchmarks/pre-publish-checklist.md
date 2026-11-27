# Pre-publish benchmark checklist

Check a run against every item before quoting its numbers outside this repo. A run that fails an item can still be published if the write-up states the limitation next to the result.

| # | Item | What passing looks like |
|---|---|---|
| 1 | Swept the load | Rate or concurrency stages from a low-load baseline to past saturation, not one load point |
| 2 | Reported exact input and output token lengths | Token counts as the server saw them (after the chat template), not only the configured values |
| 3 | Reported p50, p95 and p99 | All three for TTFT, inter-token latency and request latency, with sample counts where p99 is thin |
| 4 | Kept TTFT and inter-token latency separate | Separate columns; never one blended latency |
| 5 | Did not measure at the preemption point | Preemption counter and KV cache usage recorded; no stage ran with preemptions unless that was the point |
| 6 | Used realistic, not uniform, length distributions | A dataset or a distribution of prompt and output lengths, or the write-up says why fixed lengths were used |
| 7 | Accounted for tokenizer differences | Same tokenizer across compared engines, or results in requests/s for a defined workload |
| 8 | Pinned versions and published exact commands | Engine, model revision, client and image versions pinned; the exact launch and client commands in the repo |

Also confirm the client delivered the load (GuideLLM dispatch delay under 0.2 s, or the stage is labeled as saturated) and that committed files contain no live endpoint hostnames.

## Audit of the Week 5 runs and the Week 4 prefix run

Audited 2026-11-27. ✅ passes, ⚠️ partial, ❌ fails.

| # | [A10 rate sweep, Nov 23-24](2026-11-23-modal-a10-guidellm/README.md) | [A10 live, Nov 25](2026-11-25-modal-a10-live/README.md) | [H100 concurrency, Nov 25](2026-11-25-modal-h100-concurrency/README.md) | [A10 prefix, Nov 27](2026-11-27-modal-a10-prefix/README.md) |
|---|---|---|---|---|
| 1 Swept | ✅ 1 to 32 req/s | ✅ 1 to 32 req/s | ✅ 1 to 1,000 concurrent | ⚠️ one-at-a-time and 2 req/s only; capacity bracketed, not swept |
| 2 Token lengths | ✅ 72 in / 16 out after formatting | ✅ 72 / 16 | ✅ 72 / 16 | ✅ 2,112 / 32 (raw completions, no template) |
| 3 p50/p95/p99 | ✅ p99 reported; small samples noted | ⚠️ p99 omitted from the write-up; in raw reports | ⚠️ p99 omitted from the write-up; in raw reports with large samples | ✅ |
| 4 TTFT vs ITL | ✅ | ✅ | ✅ | ✅ |
| 5 Preemption | ✅ vLLM counter 0; SGLang retractions 0 | ⚠️ KV cache peaked near 5%; preemption count not stated in the write-up | ✅ 0 preemptions and retractions, KV under 6% | ⚠️ no server metrics this run; overloaded distinct stages are labeled as such |
| 6 Realistic lengths | ❌ fixed 64/16 | ❌ fixed | ❌ fixed | ❌ fixed, random-token prompts |
| 7 Tokenizer | ✅ same model and tokenizer on both engines | ✅ | ✅ | ✅ pinned tokenizer snapshot in both clients |
| 8 Versions and commands | ✅ | ✅ launchers plus `guidellm_modal.py --plan a10-rate` | ✅ `--plan h100-concurrency` | ✅ `prefix_modal.py` |

## Gaps

1. **Realistic length distributions (item 6) fail on every run.** All runs used fixed lengths on purpose, to isolate load from prompt size. That is right for comparing engines and finding the knee, but the absolute capacity numbers only hold for 72/16 or 2,112/32-token requests. Before publishing capacity claims, rerun the headline points with a dataset such as ShareGPT, or a length distribution, and say which numbers came from which.
2. **p99 is missing from two write-ups (item 3).** The A10 live and H100 READMEs report p50 and p95. The raw GuideLLM reports contain p99 for every stage (for example, vLLM at 1,000 concurrent: 7,210 ms p99 TTFT over 16,704 samples), so this is an edit to the tables, not a rerun.
3. **The prefix run brackets capacity instead of sweeping it (item 1).** 2 req/s already overloaded both engines on distinct prefixes. Finding each engine's distinct-prefix capacity would need rates between 0.6 and 2 req/s.
4. **The prefix run has no server-side metrics (item 5).** Prometheus was not running. The client results show overload clearly, but queue depth and cache hit counters were not recorded.
