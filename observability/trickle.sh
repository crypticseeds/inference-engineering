#!/usr/bin/env bash
# Light load so the Grafana panels move. Not a benchmark.
# Usage: BASE_URL=https://<modal-host> ./trickle.sh [parallel] [seconds]
set -euo pipefail

: "${BASE_URL:?set BASE_URL to the Modal endpoint, without /v1}"
PARALLEL="${1:-2}"
DURATION="${2:-120}"
MODEL="${MODEL:-qwen3-8b}"

worker() {
  local end=$((SECONDS + DURATION))
  while ((SECONDS < end)); do
    curl -sS -o /dev/null -w "%{http_code} %{time_total}s\n" \
      "$BASE_URL/v1/chat/completions" \
      -H "Content-Type: application/json" \
      -d "{\"model\": \"$MODEL\", \"max_tokens\": 128,
           \"messages\": [{\"role\": \"user\", \"content\": \"Explain KV caching in three sentences.\"}],
           \"chat_template_kwargs\": {\"enable_thinking\": false}}"
    sleep 1
  done
}

for _ in $(seq "$PARALLEL"); do worker & done
wait
