#!/usr/bin/env bash
set -euo pipefail

# Gemma 4 31B on H200 -- vast.ai provisioning + deploy
#
# Usage:
#   ./scripts/gemma4_h200_deploy.sh [vast_instance_id]
#
# If no instance ID given, searches for cheapest H200 and creates one.
# Downloads google/gemma-4-31B-it, builds FA3 for head_dim=256,
# compiles rvLLM v3 with Gemma 4 support.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SHA="$(cd "$REPO_ROOT" && git rev-parse HEAD)"

MODEL_ID="google/gemma-4-31B-it"
MODEL_DIR="/workspace/models/gemma-4-31B-it"
RUN_DIR="/workspace/runs/${SHA}"

echo "=== Gemma 4 31B H200 Deploy ==="
echo "  SHA: ${SHA}"
echo "  Model: ${MODEL_ID}"

# --- Provision or use existing instance ---
if [ -n "${1:-}" ]; then
    INSTANCE_ID="$1"
    echo "  Using existing instance: ${INSTANCE_ID}"
else
    echo "  Searching for cheapest H200..."
    INSTANCE_ID=$(vastai search offers \
        'gpu_name=H200 num_gpus=1 rentable=true cuda_vers>=12.4 disk_space>=500' \
        --order 'dph_total' --raw 2>/dev/null | \
        python3 -c "import sys,json; d=json.load(sys.stdin); print(d[0]['id'])" 2>/dev/null || true)

    if [ -z "$INSTANCE_ID" ]; then
        echo "ERROR: No H200 instances found on vast.ai"
        exit 1
    fi
    echo "  Creating instance from offer ${INSTANCE_ID}..."
    INSTANCE_ID=$(vastai create instance "$INSTANCE_ID" \
        --image pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel \
        --disk 600 \
        --onstart-cmd "apt-get update && apt-get install -y git cmake" \
        --raw 2>/dev/null | python3 -c "import sys,json; print(json.load(sys.stdin)['new_contract'])")
    echo "  Instance created: ${INSTANCE_ID}"
    echo "  Waiting for instance to start..."
    sleep 30
fi

# --- Wait for SSH ---
echo "  Waiting for SSH..."
for i in $(seq 1 60); do
    SSH_INFO=$(vastai ssh-url "$INSTANCE_ID" 2>/dev/null || true)
    if [ -n "$SSH_INFO" ]; then
        break
    fi
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
TARBALL="/tmp/rvllm-gemma4-${SHA}.tar.gz"
(cd "$REPO_ROOT" && tar czf "$TARBALL" \
    --exclude='.git' \
    --exclude='target' \
    --exclude='*.pdf' \
    Cargo.toml Cargo.lock \
    v3/)

echo "  Uploading to instance..."
scp -o StrictHostKeyChecking=no -P "$SSH_PORT" "$TARBALL" "root@${SSH_HOST}:/workspace/"

# --- Remote setup ---
echo "  Setting up remote environment..."
$SSH_CMD << REMOTE_SCRIPT
set -euo pipefail

# Unpack to SHA-pinned run dir
rm -rf "${RUN_DIR}"
mkdir -p "${RUN_DIR}"
cd "${RUN_DIR}"
tar xzf "/workspace/rvllm-gemma4-${SHA}.tar.gz"
echo "${SHA}" > REVISION

# Verify
echo "=== Remote SHA: \$(cat REVISION) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Install Rust
if ! command -v rustup &>/dev/null; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "\$HOME/.cargo/env"
fi
source "\$HOME/.cargo/env"

# Download model
if [ ! -d "${MODEL_DIR}" ]; then
    echo "  Downloading ${MODEL_ID}..."
    pip install -q huggingface-hub
    huggingface-cli download "${MODEL_ID}" --local-dir "${MODEL_DIR}" \
        --exclude "*.gguf" --exclude "*.bin"
fi

echo "  Model size:"
du -sh "${MODEL_DIR}"

# Build FA3 for head_dim=256
echo "  Building FA3 kernels for head_dim=128,256..."
if [ ! -f "/workspace/fa3/libfa3_kernels_hd256.so" ]; then
    pip install -q flash-attn --no-build-isolation 2>/dev/null || true
    # FA3 build from source would go here -- requires flash-attention-3 repo
    echo "  WARNING: FA3 .so for head_dim=256 needs manual build"
    echo "  Clone flash-attention-3, cd hopper/, python setup.py build --head-dims 128,256"
fi

# Build PTX kernels
echo "  Compiling Gemma 4 PTX kernels..."
cd "${RUN_DIR}/v3/kernels"
for cu in fused_gelu_mul_fp8_quant.cu fused_qk_rmsnorm.cu \
          fused_rope_partial_fp8kv.cu logit_softcap.cu; do
    ptx_name="\${cu%.cu}.ptx"
    echo "    nvcc \${cu} -> \${ptx_name}"
    nvcc -ptx -arch=sm_90a -O3 \
        --use_fast_math \
        -I/usr/local/cuda/include \
        "\${cu}" -o "\${ptx_name}"
done

# Build rvLLM
echo "  Building rvLLM v3..."
cd "${RUN_DIR}/v3"
cargo build --release --features cuda 2>&1 | tail -5

echo ""
echo "=== Gemma 4 H200 deploy complete ==="
echo "  Run dir: ${RUN_DIR}"
echo "  Model:   ${MODEL_DIR}"
echo "  SHA:     ${SHA}"
echo ""
echo "Next steps:"
echo "  1. Build FA3 .so for head_dim=256 (if not done)"
echo "  2. Run autotune for Gemma 4 GEMM shapes"
echo "  3. Run bench: RVLLM_MODEL=${MODEL_DIR} cargo run --release --bin rvllm_bench"

REMOTE_SCRIPT

echo ""
echo "=== Deploy complete ==="
echo "Instance: ${INSTANCE_ID}"
echo "SSH: ${SSH_CMD}"
echo "Run dir: ${RUN_DIR}"
