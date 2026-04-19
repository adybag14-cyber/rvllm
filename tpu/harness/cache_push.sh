#!/usr/bin/env bash
# Push compiled XLA cache to HF for reuse across deploys.
# Usage: ./cache_push.sh
set -euo pipefail

CACHE_DIR="${HOME}/.jax_cache"
HF_REPO="and-y/rvllm-kernels"
ARTIFACT="xla-v6e-cache"
JAX_VER=$(python3 -c "import jax; print(jax.__version__)" 2>/dev/null || echo "unknown")
SHA=$(cd /tmp && git -C ~/runs/tpu rev-parse --short HEAD 2>/dev/null || echo "local")

if [ ! -d "$CACHE_DIR" ] || [ -z "$(ls -A $CACHE_DIR)" ]; then
    echo "no XLA cache at $CACHE_DIR"
    exit 1
fi

TAR="/tmp/${ARTIFACT}-${SHA}.tar.gz"
echo "packaging cache: $(du -sh $CACHE_DIR | cut -f1)"
tar -czf "$TAR" -C "$(dirname $CACHE_DIR)" "$(basename $CACHE_DIR)"
echo "uploading to $HF_REPO"
huggingface-cli upload "$HF_REPO" "$TAR" "xla/v6e/${ARTIFACT}-jax${JAX_VER}-${SHA}.tar.gz"
echo "done: xla/v6e/${ARTIFACT}-jax${JAX_VER}-${SHA}.tar.gz"
