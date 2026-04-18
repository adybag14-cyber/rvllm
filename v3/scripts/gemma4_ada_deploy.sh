#!/usr/bin/env bash
set -euo pipefail

# Gemma 4 31B FP8 on RTX 6000 Ada (SM89) -- vast.ai deploy
#
# Usage:
#   ./scripts/gemma4_ada_deploy.sh <vast_instance_id>
#
# Builds SM89 attention kernel, compiles PTX for sm_89,
# builds rvLLM v3 with Gemma 4 support. Uses cuBLASLt for
# all FP8 GEMMs (native Ada support, no CUTLASS rebuild).

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SHA="$(cd "$REPO_ROOT" && git rev-parse HEAD)"

MODEL_ID="google/gemma-3-27b-it"
MODEL_DIR="/workspace/models/gemma-3-27b-it"
RUN_DIR="/workspace/runs/${SHA}"
KERNELS_DIR="${RUN_DIR}/kernels/sm_89"

echo "=== Gemma 4 31B FP8 -- RTX 6000 Ada Deploy ==="
echo "  SHA: ${SHA}"
echo "  Model: ${MODEL_ID}"

if [ -z "${1:-}" ]; then
    echo "ERROR: instance ID required"
    echo "Usage: $0 <vast_instance_id>"
    exit 1
fi
INSTANCE_ID="$1"
echo "  Instance: ${INSTANCE_ID}"

# --- Wait for SSH ---
echo "  Waiting for SSH..."
SSH_INFO=""
for i in $(seq 1 60); do
    SSH_INFO=$(vastai ssh-url "$INSTANCE_ID" 2>/dev/null || true)
    if [ -n "$SSH_INFO" ]; then break; fi
    sleep 10
done

if [ -z "$SSH_INFO" ]; then
    echo "ERROR: Instance did not become ready after 10 minutes"
    exit 1
fi

SSH_HOST=$(echo "$SSH_INFO" | cut -d@ -f2 | cut -d: -f1)
SSH_PORT=$(echo "$SSH_INFO" | cut -d: -f3)
SSH_CMD="ssh -o StrictHostKeyChecking=no -p ${SSH_PORT} root@${SSH_HOST}"

echo "  SSH: ${SSH_CMD}"

# --- Package and upload ---
echo "  Packaging tarball..."
TARBALL="/tmp/rvllm-ada-${SHA}.tar.gz"
(cd "$REPO_ROOT/.." && tar czf "$TARBALL" \
    --exclude='.git' \
    --exclude='target' \
    --exclude='*.pdf' \
    v3/)

echo "  Uploading to instance..."
scp -o StrictHostKeyChecking=no -P "$SSH_PORT" "$TARBALL" "root@${SSH_HOST}:/workspace/"

# --- Remote setup ---
echo "  Setting up remote environment..."
$SSH_CMD << 'REMOTE_SCRIPT'
set -euo pipefail

SHA="$(cat /workspace/REVISION 2>/dev/null || echo unknown)"

# Unpack
RUN_DIR="/workspace/runs/deploy"
rm -rf "${RUN_DIR}"
mkdir -p "${RUN_DIR}"
cd "${RUN_DIR}"
tar xzf /workspace/rvllm-ada-*.tar.gz
echo "deployed" > REVISION

nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader
echo "=== GPU detected ==="

# Install Rust
if ! command -v rustup &>/dev/null; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
fi
source "$HOME/.cargo/env"

# Download model
MODEL_DIR="/workspace/models/gemma-3-27b-it"
if [ ! -d "${MODEL_DIR}" ]; then
    echo "  Downloading model..."
    pip install -q huggingface-hub
    huggingface-cli download google/gemma-3-27b-it --local-dir "${MODEL_DIR}" \
        --exclude "*.gguf" --exclude "*.bin"
fi
echo "  Model size:"
du -sh "${MODEL_DIR}"

# Build SM89 attention .so
echo "  Building SM89 paged attention .so..."
KERNELS_DIR="${RUN_DIR}/kernels/sm_89"
mkdir -p "${KERNELS_DIR}"

cd "${RUN_DIR}/v3/kernels"
nvcc -shared -o "${KERNELS_DIR}/libfa_sm89_kernels.so" \
    paged_attention_sm89.cu \
    -arch=sm_89 -O3 --use_fast_math -Xcompiler -fPIC \
    -I/usr/local/cuda/include
echo "  SM89 attention .so built"

# Compile PTX kernels for SM89
echo "  Compiling fused PTX kernels for SM89..."
for cu in fused_rmsnorm_fp8_quant.cu fused_rope_cache_fp8kv.cu \
          fused_silu_fp8_quant.cu argmax.cu add_bias_f16.cu fp8_rescale.cu \
          fused_gelu_mul_fp8_quant.cu fused_qk_rmsnorm.cu \
          fused_rope_partial_fp8kv.cu logit_softcap.cu; do
    if [ -f "${cu}" ]; then
        ptx_name="${cu%.cu}.ptx"
        echo "    ${cu} -> ${ptx_name}"
        nvcc -ptx -arch=sm_89 -O3 --use_fast_math \
            -I/usr/local/cuda/include "${cu}" -o "${KERNELS_DIR}/${ptx_name}"
    fi
done

# Build kernel manifest
echo "  Building kernel manifest..."
cd "${RUN_DIR}/v3/kernels"
if [ -f make_manifest.py ]; then
    python3 make_manifest.py "${KERNELS_DIR}" sm_89
fi

# Create minimal policy.json (cuBLASLt handles all GEMMs, policy
# entries just need to exist with valid variant IDs)
cat > "${KERNELS_DIR}/policy.json" << 'POLICY'
{
  "arch": "sm_89",
  "entries": {}
}
POLICY

# Build CUTLASS stub .so (cuBLASLt handles all GEMMs on Ada,
# but CutlassLib::load() needs variant symbols to exist)
echo "  Building CUTLASS stub .so..."
gcc -shared -fPIC -o "${KERNELS_DIR}/libcutlass_kernels.so" \
    "${RUN_DIR}/v3/kernels/cutlass_stub_sm89.c"

# Build rvLLM
echo "  Building rvLLM v3..."
cd "${RUN_DIR}/v3"
cargo build --release --features cuda 2>&1 | tail -5

echo ""
echo "=== RTX 6000 Ada deploy complete ==="
echo "  Run dir:  ${RUN_DIR}"
echo "  Model:    ${MODEL_DIR}"
echo "  Kernels:  ${KERNELS_DIR}"
echo ""
echo "Run with:"
echo "  export RVLLM_MODEL_DIR=${MODEL_DIR}"
echo "  export RVLLM_KERNELS_DIR=${KERNELS_DIR}"
echo "  export RVLLM_CUTLASS_SO=${KERNELS_DIR}/libcutlass_kernels.so"
echo "  export RVLLM_FA3_SO=${KERNELS_DIR}/libfa_sm89_kernels.so"
echo "  export RVLLM_POLICY=${KERNELS_DIR}/policy.json"
echo "  export RVLLM_BATCH=32"
echo "  export RVLLM_NO_GRAPH=1"
echo "  ./target/release/rvllm-bench"

REMOTE_SCRIPT

echo ""
echo "=== Deploy complete ==="
echo "Instance: ${INSTANCE_ID}"
echo "SSH: ${SSH_CMD}"
