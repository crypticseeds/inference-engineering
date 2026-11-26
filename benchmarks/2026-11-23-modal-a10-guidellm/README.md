# GuideLLM on a Modal A10, 2026-11-23

Week 5 began with a GuideLLM smoke test, a sequential baseline, and two fixed-rate runs against vLLM. At 2 req/s, completed throughput matched the offered rate. Median client time to first token (TTFT) rose from 247.8 ms to 300.3 ms and median inter-token latency (ITL) rose from 33.6 ms to 34.4 ms. The serving logs showed zero waiting requests at the sampled points during the 2 req/s stage.

## Setup

| Item | Value |
|---|---|
| Engine and GPU | vLLM 0.21.0 on one Modal NVIDIA A10 (24 GB), us-east |
| Model and tokenizer | `Qwen/Qwen3-8B` at `b968826d9c46dd6066d109eabc6255188de91218`, BF16, served as `qwen3-8b` |
| Server flags | `--max-model-len 8192 --max-num-seqs 16 --gpu-memory-utilization 0.90 --enforce-eager --enable-prefix-caching --enable-prompt-tokens-details` |
| Client | GuideLLM 0.8.1 on a Raspberry Pi, streaming `/v1/chat/completions` over HTTPS |
| Workload | Synthetic text, seed 42, 64 configured prompt tokens and 16 output tokens. GuideLLM reported 72 input tokens after chat formatting and 16 output tokens per completed request. |
| Sampling | The server loaded the model defaults of temperature 0.6, top-k 20, and top-p 0.95. The GuideLLM backend had no extra request arguments. |
| Fixed-rate stages | 30-second duration limit, 5-second requested warm-up, at most four client requests in flight |

## Results

| Metric | Sequential | 1 req/s | 2 req/s |
|---|---:|---:|---:|
| Measured duration | 6.6 s | 26.0 s | 25.5 s |
| Completed requests | 10 | 27 | 51 |
| Incomplete requests | 0 | 0 | 2 |
| Request errors | 0 | 0 | 0 |
| Completed requests/sec | 1.51 | 1.04 | 2.00 |
| Median requests in flight | 1 | 1 | 2 |
| Client TTFT p50 | 155.9 ms | 247.8 ms | 300.3 ms |
| Client TTFT p95 | 162.1 ms | 256.1 ms | 319.5 ms |
| Client TTFT p99 | 162.1 ms | 259.8 ms | 422.3 ms |
| Client ITL p50 | 33.6 ms | 33.6 ms | 34.4 ms |

The first smoke request completed and reported 865.8 ms TTFT. The server logged Triton JIT compilation during that request. The ten-request sequential run followed on the same container.

GuideLLM marked two requests incomplete at the 2 req/s measurement cutoff. Both were cancelled and had started within roughly half a second of the cutoff. In the 2 req/s stage, observed waiting requests were zero and sampled GPU KV-cache usage was about 0.3%. Server preemptions were not independently verified with a counter snapshot.

The constant-rate runs used separate Modal containers, each started before its measurement window. The TTFT difference may include container or network variation. GuideLLM could not estimate TTFT tail confidence intervals at these sample sizes. Treat the p99 values as descriptive and collect more samples for any capacity claim.

The original reports are in [`raw/`](raw/) as `week5-guidellm-baseline.json`, `week5-guidellm-1rps.json`, and `week5-guidellm-2rps.json`.

## Reproduce and extend the sweep

Start the Modal app from the repository root and copy its printed HTTPS base URL into `VLLM_BASE_URL`. Wait for `/health` to return HTTP 200 before running GuideLLM. [Modal launch file](../../deploy/modal/vllm_qwen3_8b.py) · [Modal runbook](../../deploy/modal/README.md#serve-for-load-tests-and-dashboards)

```bash
modal serve deploy/modal/vllm_qwen3_8b.py
```

Run this in a separate terminal with GuideLLM already cached by `uv`. Set `RATE` to `1` or `2`. Use a distinct report path for a repeat run.

```bash
VLLM_BASE_URL='<MODAL_BASE_URL>'
RATE=2
uvx --offline --python 3.12 --from guidellm==0.8.1 guidellm run \
  --backend "kind=openai_http,target=$VLLM_BASE_URL,model=qwen3-8b,verify=true,timeout=30" \
  --tokenizer 'kind=huggingface_auto,model=Qwen/Qwen3-8B,load_kwargs.revision=b968826d9c46dd6066d109eabc6255188de91218' \
  --data kind=synthetic_text,prompt_tokens=64,output_tokens=16 \
  --profile "kind=constant,rate=$RATE,max_concurrency=4,warmup=5" \
  --constraint kind=max_duration,seconds=30 \
  --output "kind=json,path=guidellm-${RATE}rps-repeat.json" \
  --disable-console-interactive
```

For the sequential baseline, use `--profile kind=synchronous` and limits `--constraint kind=max_requests,count=10` and `--constraint kind=max_duration,seconds=60`.

## November 24 follow-up on one A10 container

Ran 1, 2, and 4 req/s in order on the same A10 container. The workload, client cap, and serving flags stayed fixed. The duration limit increased to 60 seconds per stage with the same five-second requested warm-up.

| Metric | 1 req/s | 2 req/s | 4 req/s |
|---|---:|---:|---:|
| Measured duration | 56.0 s | 55.5 s | 55.3 s |
| Completed requests | 56 | 111 | 222 |
| Incomplete requests | 1 | 2 | 3 |
| Request errors | 0 | 0 | 0 |
| Completed requests/sec | 1.00 | 2.00 | 4.02 |
| Peak recorded client concurrency | 1 | 2 | 4 |
| Client TTFT p50 | 165.9 ms | 199.4 ms | 205.7 ms |
| Client TTFT p95 | 172.0 ms | 221.8 ms | 226.7 ms |
| Client TTFT p99 | 173.2 ms | 227.7 ms | 232.7 ms |
| Client ITL p50 | 34.5 ms | 35.6 ms | 37.8 ms |

Every completed request used 72 input tokens after formatting and generated 16 output tokens. GuideLLM recorded TTFT samples for 56, 110, and 219 requests in the three stages.

The server `/metrics` samples at roughly four-second intervals during the 2 and 4 req/s stages showed zero waiting requests and a preemption counter of zero. Sampling started after the 1 req/s stage. At 4 req/s, throughput matched the target and GuideLLM reached its four-request in-flight cap. Server capacity remains undetermined. A higher-load run should raise the client cap and continue sampling the server queue.

The reports are in [`raw/`](raw/) as `week5-guidellm-20261124-1rps.json`, `week5-guidellm-20261124-2rps.json`, and `week5-guidellm-20261124-4rps.json`.

## November 24 extension through 32 req/s

Ran 8, 16, and 32 req/s in order on one new A10 container. After three sequential warm-up requests, each stage used the same 60-second duration limit, five-second requested warm-up, and vLLM configuration. GuideLLM's in-flight cap increased to 64 across all three stages. Each completed request used 72 formatted input tokens and generated 16 output tokens.

| Metric | 8 req/s | 16 req/s | 32 req/s |
|---|---:|---:|---:|
| Measured duration | 55.1 s | 55.1 s | 55.0 s |
| Completed requests | 442 | 881 | 1,136 |
| Incomplete requests | 7 | 14 | 62 |
| Request errors | 0 | 0 | 0 |
| Completed requests/sec (count ÷ measured duration) | 8.02 | 16.00 | 20.64 |
| Peak recorded client concurrency | 8 | 17 | 64 |
| Client TTFT p50 | 274.1 ms | 282.7 ms | 2,419.6 ms |
| Client TTFT p95 | 295.8 ms | 307.2 ms | 2,514.6 ms |
| Client TTFT p99 | 322.0 ms | 313.9 ms | 2,570.2 ms |
| Client ITL p50 | 39.4 ms | 43.7 ms | 45.2 ms |
| Peak sampled vLLM running requests | 6 | 12 | 16 |
| Peak sampled vLLM waiting requests | 0 | 0 | 47 |
| vLLM preemption counter change | 0 | 0 | 0 |

At 32 req/s, vLLM reached its configured 16-running-request limit and the waiting queue grew. GuideLLM also reached its 64-request in-flight cap and reported an 18.9-second p95 root dispatch delay, a client-side scheduling metric separate from TTFT. The run measures about 20.6 completed req/s and 2.51 s p95 TTFT under this configuration. The client did not sustain the nominal 32 req/s arrival schedule under backpressure. Using **p95 TTFT below 500 ms** as the acceptance target for this interactive workload, 16 req/s passed at 307.2 ms with an empty sampled server queue. The 32 req/s stage failed at 2,514.6 ms, about five times the target. GuideLLM also recorded 62 incomplete requests out of 1,198 in its measurement window, alongside zero request errors. The nominal rate exceeded the number of arrivals the client actually dispatched.

The [MLPerf LoadGen configuration](https://github.com/mlcommons/inference/blob/master/loadgen/mlperf.conf) defines two Llama 3.1 8B server workloads with different latency limits. The Interactive workload uses 500 ms TTFT and 30 ms time per output token (TPOT). The standard Server workload uses 2,000 ms TTFT and 100 ms TPOT. This Qwen3 experiment uses 500 ms at p95 as its own TTFT criterion. The MLPerf benchmarks provide context for that threshold and have their own workload and TPOT requirements.

The server metrics were sampled about every three seconds, with 24, 25, and 24 samples respectively and zero failed scrapes. The reports in [`raw/`](raw/) use the stem `week5-guidellm-20261124-high-2b2d35` followed by `-8rps.json`, `-16rps.json`, or `-32rps.json`.

## vLLM CLI cross-check

Ran `vllm bench serve` 0.21.0 over loopback inside the serving container on a Modal A10 with the same engine settings. The random prompt generator used seed 42, fixed lengths, and `--ignore-eos`. Both stages returned 72 formatted input tokens and 16 output tokens per completed request, matching the GuideLLM lengths. The CLI used near-uniform arrivals (`--burstiness 10000`), a 64-request client concurrency limit, and `--goodput ttft:500`. At each rate, the CLI sent a fixed number of requests and waited for them to complete.

| Metric | 16 req/s target | 32 req/s target |
|---|---:|---:|
| Duration including drain | 60.68 s | 93.81 s |
| Completed requests | 960 | 1,920 |
| Failed requests | 0 | 0 |
| Completed requests/sec | 15.82 | 20.47 |
| Goodput within the 500 ms TTFT threshold (req/s) | 15.82 | 0.23 |
| Client TTFT p50 | 112.4 ms | 2,432.0 ms |
| Client TTFT p95 | 128.8 ms | 2,442.9 ms |
| Client TTFT p99 | 131.0 ms | 2,443.7 ms |
| Client ITL p50 | 42.6 ms | 44.7 ms |

Both clients passed the 500 ms p95 TTFT target at 16 req/s and missed it at 32 req/s. The native run reproduced roughly 20.5 completed req/s and over 2.4 seconds p95 TTFT at the higher rate. The native CLI ran over loopback with different synthetic prompt text. It also drained its request set, while GuideLLM ended after a fixed measurement window. The first native 32 req/s attempt ended when the container shut down mid-run. The table uses the completed retry on a new A10 container with external health probes during the run.

## November 24 SGLang comparison

Ran GuideLLM 0.8.1 from the Pi over HTTPS against SGLang 0.5.9 on one warm A10 in Modal us-east. Both engines used the same pinned Qwen3-8B revision, BF16, 8,192-token context limit, and 16-running-request limit with CUDA graphs disabled. SGLang used `--mem-fraction-static 0.85` and vLLM used `--gpu-memory-utilization 0.90`. Both servers reported temperature 0.6, top-k 20, and top-p 0.95. SGLang also logged a repetition penalty of 1.0.

After three sequential streaming warm-up requests, the measured stages used the same synthetic workload and seed, 60-second duration limit, and five-second requested warm-up. GuideLLM capped in-flight requests at 4 for the 1–4 req/s stages and 64 for the 8–32 req/s stages, matching the vLLM runs. Every completed SGLang request used 72 formatted input tokens and generated 16 output tokens.

| Target req/s | SGLang completed req/s | vLLM completed req/s | SGLang TTFT p50 / p95 / p99 (ms) | vLLM TTFT p95 (ms) | SGLang / vLLM median ITL (ms) | SGLang / vLLM incomplete |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1.02 | 1.00 | 241.5 / 254.2 / 258.0 | 172.0 | 34.9 / 34.5 | 1 / 1 |
| 2 | 2.00 | 2.00 | 267.1 / 291.6 / 301.1 | 221.8 | 38.7 / 35.6 | 2 / 2 |
| 4 | 4.00 | 4.02 | 262.4 / 291.1 / 296.8 | 226.7 | 42.8 / 37.8 | 4 / 3 |
| 8 | 7.98 | 8.02 | 268.0 / 293.6 / 305.2 | 295.8 | 56.0 / 39.4 | 10 / 7 |
| 16 | 16.02 | 16.00 | 927.3 / 1,325.7 / 1,356.9 | 307.2 | 41.7 / 43.7 | 20 / 14 |
| 32 | 16.52 | 20.64 | 3,298.7 / 3,385.9 / 3,507.4 | 2,514.6 | 39.3 / 45.2 | 64 / 62 |

Both engines reported zero request errors at all six rates. SGLang completed 57, 111, 221, 440, 882, and 909 requests, respectively. The highest tested rate under the project target of 500 ms p95 TTFT was 8 req/s on SGLang and 16 req/s on vLLM. At 16 req/s, SGLang still matched the scheduled throughput, with up to 14 requests waiting in its scheduler logs. At 32 req/s its observed throughput rose only to 16.52 req/s, with up to 48 waiting and 16 running. GuideLLM hit its 64-request cap and reported 26.4 s p95 root dispatch delay at that stage.

SGLang returned HTTP 404 on `/metrics` with this launcher, so the queue observations come from timestamped scheduler logs. Retraction counts were unavailable. The measured queue peaks at 8, 16, and 32 req/s were 0, 14, and 48. The engine runs used separate containers and their respective serving configurations, so the results describe these deployments and this workload.

The SGLang reports in [`raw/`](raw/) use the stem `week5-guidellm-sglang-20261124-210146` followed by `-warmup.json` or `-<rate>rps.json`. See the [SGLang runbook](../../variants/sglang/README.md#try-the-api-from-bruno-or-postman).

## November 24 SGLang rerun with Prometheus metrics

Repeated all six rate points on one fresh SGLang A10 container with `--enable-metrics`, using the same workload and client limits. The vLLM columns reuse the original GuideLLM results from the separate vLLM sessions above. SGLang reported zero request errors in all six stages.

| Target req/s | SGLang / vLLM completed req/s | SGLang / vLLM p95 TTFT (ms) | SGLang / vLLM incomplete | SGLang peak running / waiting |
|---:|---:|---:|---:|---:|
| 1 | 1.02 / 1.00 | 267.8 / 172.0 | 0 / 1 | 1 / 0 |
| 2 | 2.00 / 2.00 | 312.0 / 221.8 | 1 / 2 | 2 / 0 |
| 4 | 4.00 / 4.02 | 317.1 / 226.7 | 4 / 3 | 3 / 0 |
| 8 | 7.98 / 8.02 | 317.5 / 295.8 | 10 / 7 | 7 / 0 |
| 16 | 16.09 / 16.00 | 1,183.0 / 307.2 | 17 / 14 | 16 / 12 |
| 32 | 17.14 / 20.64 | 3,275.6 / 2,514.6 | 64 / 62 | 16 / 48 |

The highest tested rate below 500 ms p95 TTFT was 8 req/s for SGLang and 16 req/s for vLLM. At the 32 req/s target, SGLang completed 943 requests and GuideLLM reported 25.4 s p95 root dispatch delay. That client scheduling delay is measured separately from TTFT.

The SGLang queue and running peaks come from `sglang:num_queue_reqs` and `sglang:num_running_reqs`. Each stage had 20–22 scrapes inside its measured window and zero scrape failures. The `sglang:num_retracted_reqs` gauge was zero at every sampled point in those windows, and the `sglang:num_retractions_sum` histogram series had zero change across every stage. The GuideLLM reports and metrics snapshots in [`raw/`](raw/) use the stem `week5-guidellm-sglang-metrics-20261124-233454`.
