#!/usr/bin/env bash
# Pull compiled XLA cache from HF to skip cold compilation.
# Usage: ./cache_pull.sh [artifact-name]
set -euo pipefail

HF_REPO="and-y/rvllm-kernels"
CACHE_DIR="${HOME}/.jax_cache"
JAX_VER=$(python3 -c "import jax; print(jax.__version__)" 2>/dev/null || echo "unknown")

if [ -n "${1:-}" ]; then
    ARTIFACT="$1"
else
    echo "fetching latest xla/v6e cache for jax ${JAX_VER}..."
    ARTIFACT=$(huggingface-cli repo info "$HF_REPO" --files 2>/dev/null \
        | grep "xla/v6e/.*jax${JAX_VER}" | sort | tail -1 || true)
    if [ -z "$ARTIFACT" ]; then
        echo "no matching cache found for jax ${JAX_VER}"
        exit 1
    fi
fi

echo "downloading $ARTIFACT"
huggingface-cli download "$HF_REPO" "$ARTIFACT" --local-dir /tmp/xla_cache_dl
TAR=$(find /tmp/xla_cache_dl -name "*.tar.gz" | head -1)
if [ -z "$TAR" ]; then
    echo "no tarball found"
    exit 1
fi

mkdir -p "$CACHE_DIR"
tar -xzf "$TAR" -C "$(dirname $CACHE_DIR)"
echo "restored XLA cache to $CACHE_DIR ($(du -sh $CACHE_DIR | cut -f1))"
