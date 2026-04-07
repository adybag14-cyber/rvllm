#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export RVLLM_BATCHED_PIPELINE_V2=1

exec "${RVLLM_BIN:-target/release/rvllm}" benchmark "$@"
