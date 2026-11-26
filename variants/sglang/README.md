# SGLang

SGLang deployment verified on Modal on 2026-11-22. The shared-prefix comparison with vLLM is pending.

## Verified deployment

| Setting | Value |
|---|---|
| Engine | SGLang 0.5.9 (`lmsysorg/sglang:v0.5.9`) |
| Model | `Qwen/Qwen3-8B` at `b968826d9c46dd6066d109eabc6255188de91218`, BF16, served as `qwen3-8b` |
| Hardware | One NVIDIA A10 (24 GB), Modal us-east |
| Limits | 8,192-token context, 16 running requests, `--mem-fraction-static 0.85`, CUDA graphs disabled |

Bruno reached the authenticated `modal serve` endpoint. `GET /v1/models` listed `qwen3-8b` with `owned_by: sglang` and `max_model_len: 8192`. A non-streaming `POST /v1/chat/completions` returned assistant text with 22 prompt tokens, 50 completion tokens, and `finish_reason: stop`. A separate request without credentials returned HTTP 401, confirming that the proxy requires authentication.

The final serving logs show shutdown and exit code 0. The cause of the earlier intermittent failures is unresolved. During another run, capture Bruno's HTTP status and the matching server logs to distinguish container startup from a failure while the engine is already warm.

## Try the API from Bruno or Postman

The [Modal app](../../deploy/modal/sglang_qwen3_8b.py) has two modes, like the [vLLM app](../../deploy/modal/README.md#deploy-and-verify). `modal run` runs an automated check and closes the app. `modal serve` gives you an HTTPS URL and keeps it available while the command is running. Both launchers default to the A10 profile; prefix either command with `GPU_PROFILE=H100` for the H100 profile.

From the repository root, run this in a terminal and leave it open.

```bash
modal serve deploy/modal/sglang_qwen3_8b.py
```

Copy the dev URL printed for `Server` into your Bruno or Postman environment as `base_url`, without a trailing slash. Select No Auth and remove the `Modal-Key` and `Modal-Secret` headers used in the earlier Bruno test. Anyone with the current dev URL can make requests while the session is running. Keep the URL private and stop the session when finished.

Send these requests in order.

| Method and URL | What a successful response shows |
|---|---|
| `GET {{base_url}}/health` | HTTP 200 means the server is ready. The first request can return 503 while Modal starts the A10. Watch the terminal logs and retry after startup. |
| `GET {{base_url}}/v1/models` | The JSON lists `qwen3-8b`, the name to use in chat requests. |
| `POST {{base_url}}/v1/chat/completions` | The JSON contains an assistant reply at `choices[0].message.content`. |

For the POST, set `Content-Type: application/json` and send this body.

```json
{
  "model": "qwen3-8b",
  "messages": [{"role": "user", "content": "Explain prefix caching in one sentence."}],
  "max_tokens": 64,
  "temperature": 0,
  "chat_template_kwargs": {"enable_thinking": false},
  "stream": false
}
```

After the JSON request works, set `stream` to `true` to request server-sent events instead of a single JSON response. The stream ends with `data: [DONE]`.

Modal provides the HTTPS URL and starts one A10 on demand. SGLang loads Qwen3-8B on that GPU and implements the OpenAI-compatible API. The URL stays available while `modal serve` is running, even if Modal idles the GPU after 120 seconds without traffic. Press Ctrl-C in the serving terminal to end the session.
