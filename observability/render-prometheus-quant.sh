#!/usr/bin/env bash
# Render prometheus-quant.yaml with the three live Modal hosts into a private
# file outside the repo and print its path.
# Usage: export PROMETHEUS_CONFIG="$(./render-prometheus-quant.sh <bf16-host> <fp8-host> <awq-host>)"
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 <bf16-host> <fp8-host> <awq-host>" >&2
  exit 2
fi

host() {
  local h="${1#https://}"
  echo "${h%/}"
}

umask 022
out="$(mktemp -d "${TMPDIR:-/tmp}/prometheus.XXXXXX")/prometheus.yaml"
sed -e "s|BF16_HOST|$(host "$1")|" \
    -e "s|FP8_HOST|$(host "$2")|" \
    -e "s|AWQ_HOST|$(host "$3")|" \
    "$(dirname "$0")/prometheus-quant.yaml" > "$out"
echo "$out"
