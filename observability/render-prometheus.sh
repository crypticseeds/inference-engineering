#!/usr/bin/env bash
# Render prometheus.yaml with live Modal hosts into a private file outside the
# repo and print its path. Pass "-" to leave an engine unscraped.
# Usage: export PROMETHEUS_CONFIG="$(./render-prometheus.sh <vllm-host|-> <sglang-host|->)"
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 <vllm-host|-> <sglang-host|->" >&2
  exit 2
fi

target() {
  local host="${1#https://}"
  host="${host%/}"
  if [[ "$host" == "-" ]]; then echo '[]'; else echo "[\"${host}:443\"]"; fi
}

umask 022
out="$(mktemp -d "${TMPDIR:-/tmp}/prometheus.XXXXXX")/prometheus.yaml"
sed -e "s|\[\"VLLM_HOST:443\"\]|$(target "$1")|" \
    -e "s|\[\"SGLANG_HOST:443\"\]|$(target "$2")|" \
    "$(dirname "$0")/prometheus.yaml" > "$out"
echo "$out"
