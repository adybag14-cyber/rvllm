#!/bin/bash
# Head-to-head benchmark: rvLLM v2 vs vLLM 0.19, both direct engine (no HTTP).
#
# Both engines use identical parameters:
#   - Same model (Qwen2.5-7B)
#   - Same output length (512 tokens)
#   - Same concurrency levels (1,32,64,128)
#   - temperature=0, ignore_eos=True
#   - Same GPU
#
# Usage:
#   bash deploy/bench_vs_vllm.sh
#   MODEL=Qwen/Qwen2.5-7B MAX_TOKENS=512 bash deploy/bench_vs_vllm.sh

set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-7B}"
MAX_TOKENS="${MAX_TOKENS:-512}"
CONCURRENCY="${CONCURRENCY:-1,32,64,128}"
ITERS="${ITERS:-3}"
RESULTS_DIR="${RESULTS_DIR:-/tmp/bench_results}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${SCRIPT_DIR}/.."

mkdir -p "${RESULTS_DIR}"

echo "============================================"
echo "  rvLLM v2 vs vLLM 0.19 -- Direct Engine"
echo "============================================"
echo "Model:       ${MODEL}"
echo "Max tokens:  ${MAX_TOKENS}"
echo "Concurrency: ${CONCURRENCY}"
echo "Iterations:  ${ITERS}"
echo ""

# --- Find model path (local snapshots) ---
MODEL_PATH=""
for search_dir in /workspace/models /workspace/hf_cache /root/.cache/huggingface; do
    found=$(find "${search_dir}" -path "*Qwen2.5-7B*" -name 'config.json' 2>/dev/null | head -1)
    if [ -n "${found}" ]; then
        MODEL_PATH=$(dirname "${found}")
        break
    fi
done
if [ -z "${MODEL_PATH}" ]; then
    MODEL_PATH="${MODEL}"
    echo "Using HF model ID directly: ${MODEL_PATH}"
else
    echo "Model path: ${MODEL_PATH}"
fi

# --- 1. rvLLM benchmark ---
echo ""
echo "=== rvLLM v2 Benchmark ==="
RVLLM_BINARY="${REPO_DIR}/target/release/rvllm-v2-bench"
if [ ! -f "${RVLLM_BINARY}" ]; then
    echo "Building rvLLM..."
    cd "${REPO_DIR}"
    source "$HOME/.cargo/env" 2>/dev/null || true
    cargo build --release -p rvllm-v2 --features cuda-graphs --bin rvllm-v2-bench 2>&1 | tail -3
fi

"${RVLLM_BINARY}" --fp8 --model "${MODEL_PATH}" \
    --output-len "${MAX_TOKENS}" \
    --n "${CONCURRENCY}" --iters "${ITERS}" --json \
    > "${RESULTS_DIR}/rvllm.json" 2>"${RESULTS_DIR}/rvllm.log"

# Print summary from stderr
cat "${RESULTS_DIR}/rvllm.log"

# --- 2. vLLM benchmark ---
echo ""
echo "=== vLLM 0.19 Benchmark ==="

# Check if vLLM is installed
if ! python3 -c "import vllm; print(f'vLLM {vllm.__version__}')" 2>/dev/null; then
    echo "Installing vLLM..."
    pip install vllm 2>&1 | tail -3
fi

cd "${REPO_DIR}"
python3 deploy/vllm_direct_bench.py \
    --model "${MODEL}" \
    --max-tokens "${MAX_TOKENS}" \
    --concurrency "${CONCURRENCY}" \
    --output "${RESULTS_DIR}/vllm.json"

# --- 3. Side-by-side comparison ---
echo ""
echo "============================================"
echo "  Head-to-Head Comparison"
echo "============================================"

python3 -c "
import json, sys

with open('${RESULTS_DIR}/rvllm.json') as f:
    rvllm = json.load(f)
with open('${RESULTS_DIR}/vllm.json') as f:
    vllm_data = json.load(f)

print(f'Model: ${MODEL}, max_tokens=${MAX_TOKENS}')
print()
print(f'{\"N\":>6} | {\"rvLLM tok/s\":>12} | {\"vLLM tok/s\":>12} | {\"Speedup\":>8}')
print('-' * 50)

rvllm_map = {r['n']: r['mean_tok_per_sec'] for r in rvllm['results']}
vllm_map = {r['n']: r['tok_per_sec'] for r in vllm_data['results']}

all_ns = sorted(set(list(rvllm_map.keys()) + list(vllm_map.keys())))
for n in all_ns:
    rv = rvllm_map.get(n, 0)
    vl = vllm_map.get(n, 0)
    speedup = rv / vl if vl > 0 else float('inf')
    marker = ' <<' if speedup > 1.0 else ''
    print(f'{n:>6} | {rv:>12,.1f} | {vl:>12,.1f} | {speedup:>7.2f}x{marker}')

print()
# Overall summary
if all_ns:
    rv_max = max(rvllm_map.get(n, 0) for n in all_ns)
    vl_max = max(vllm_map.get(n, 0) for n in all_ns)
    print(f'Peak throughput: rvLLM {rv_max:,.1f} vs vLLM {vl_max:,.1f} tok/s ({rv_max/vl_max:.2f}x)')
    rv_min = rvllm_map.get(min(all_ns), 0)
    vl_min = vllm_map.get(min(all_ns), 0)
    if vl_min > 0:
        print(f'N=1 latency:     rvLLM {rv_min:,.1f} vs vLLM {vl_min:,.1f} tok/s ({rv_min/vl_min:.2f}x)')
"

echo ""
echo "Raw results: ${RESULTS_DIR}/rvllm.json ${RESULTS_DIR}/vllm.json"
