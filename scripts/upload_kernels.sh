#!/bin/bash
# Upload pre-compiled kernel artifacts to HuggingFace.
#
# Run this ON the GPU instance after compiling kernels, CUTLASS .so, and autotune.
# Artifacts are uploaded to m0at/rvllm-kernels organized by GPU arch.
#
# Usage:
#   bash scripts/upload_kernels.sh [arch]
#   bash scripts/upload_kernels.sh sm_90
#
# Prerequisites:
#   - huggingface-cli installed (pip install huggingface-hub)
#   - HF_TOKEN set or logged in via `huggingface-cli login`
#   - Kernels compiled (bash kernels/build.sh)
#   - CUTLASS .so built (bash kernels/build_cutlass_so.sh $ARCH)
#   - Autotune run (cargo run --release -p rvllm-gpu --bin autotune-cutlass --features cuda)

set -euo pipefail

ARCH=${1:-sm_90}
REPO="m0at/rvllm-kernels"
SHA=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
STAGE_DIR="/tmp/rvllm-kernels-stage/${ARCH}"

echo "=== rvLLM Kernel Upload ==="
echo "Arch: ${ARCH}"
echo "SHA: ${SHA}"
echo "Repo: ${REPO}"

# Clean staging area
rm -rf "${STAGE_DIR}"
mkdir -p "${STAGE_DIR}/ptx" "${STAGE_DIR}/cubin" "${STAGE_DIR}/cutlass" "${STAGE_DIR}/autotune"

# Stage PTX files
PTX_SRC=""
if [ -d "kernels/${ARCH}" ]; then
    PTX_SRC="kernels/${ARCH}"
elif [ -d "kernels" ]; then
    PTX_SRC="kernels"
fi

if [ -n "${PTX_SRC}" ]; then
    PTX_COUNT=$(find "${PTX_SRC}" -maxdepth 1 -name '*.ptx' | wc -l | tr -d ' ')
    if [ "${PTX_COUNT}" -gt 0 ]; then
        cp "${PTX_SRC}"/*.ptx "${STAGE_DIR}/ptx/"
        echo "Staged ${PTX_COUNT} PTX files from ${PTX_SRC}"
    else
        echo "WARNING: No PTX files found in ${PTX_SRC}"
    fi
fi

# Stage cubin files
CUBIN_COUNT=0
for src in "kernels/${ARCH}" "kernels"; do
    if [ -d "${src}" ]; then
        found=$(find "${src}" -maxdepth 1 -name '*.cubin' 2>/dev/null | wc -l | tr -d ' ')
        if [ "${found}" -gt 0 ]; then
            cp "${src}"/*.cubin "${STAGE_DIR}/cubin/"
            CUBIN_COUNT=$((CUBIN_COUNT + found))
        fi
    fi
done
echo "Staged ${CUBIN_COUNT} cubin files"

# Stage CUTLASS .so
for so_name in libcutlass_kernels.so libfa3_kernels.so; do
    SO_PATH="kernels/${ARCH}/${so_name}"
    if [ -f "${SO_PATH}" ]; then
        SZ=$(stat -c%s "${SO_PATH}" 2>/dev/null || stat -f%z "${SO_PATH}")
        if [ "${SZ}" -lt 1000000 ]; then
            echo "WARNING: ${SO_PATH} is only ${SZ} bytes -- likely a Mac stub, skipping"
        else
            cp "${SO_PATH}" "${STAGE_DIR}/cutlass/"
            echo "Staged ${so_name} (${SZ} bytes)"
        fi
    fi
done

# Stage autotune cache
AUTOTUNE_PATH="${HOME}/.cache/rvllm/cutlass_autotune.json"
if [ -f "${AUTOTUNE_PATH}" ]; then
    cp "${AUTOTUNE_PATH}" "${STAGE_DIR}/autotune/"
    echo "Staged autotune cache"
else
    echo "WARNING: No autotune cache found at ${AUTOTUNE_PATH}"
fi

# Generate manifest with checksums
echo "Generating manifest..."
python3 -c "
import json, hashlib, os, datetime

stage = '${STAGE_DIR}'
files = {}
for root, dirs, filenames in os.walk(stage):
    for f in filenames:
        if f == 'manifest.json':
            continue
        path = os.path.join(root, f)
        rel = os.path.relpath(path, stage)
        h = hashlib.sha256(open(path, 'rb').read()).hexdigest()
        files[rel] = {'sha256': h, 'size': os.path.getsize(path)}

manifest = {
    'git_sha': '${SHA}',
    'build_timestamp': datetime.datetime.utcnow().isoformat() + 'Z',
    'cuda_version': os.popen('nvcc --version 2>/dev/null | grep release | sed \"s/.*release //;s/,.*//\"').read().strip() or 'unknown',
    'gpu_arch': '${ARCH}',
    'files': files
}
json.dump(manifest, open(os.path.join(stage, 'manifest.json'), 'w'), indent=2)
print(f'Manifest: {len(files)} files')
"

# Show what we're uploading
echo ""
echo "Staging directory contents:"
find "${STAGE_DIR}" -type f | sort | while read f; do
    SZ=$(stat -c%s "$f" 2>/dev/null || stat -f%z "$f")
    REL=$(echo "$f" | sed "s|${STAGE_DIR}/||")
    printf "  %-60s %s\n" "${REL}" "$(numfmt --to=iec ${SZ} 2>/dev/null || echo ${SZ})"
done

# Upload
echo ""
echo "Uploading to ${REPO}..."
huggingface-cli upload "${REPO}" "${STAGE_DIR}" "${ARCH}/" --repo-type model

echo ""
echo "=== Upload complete ==="
echo "Other instances can now pull kernels automatically."
echo "To test: unset RVLLM_KERNEL_DIR RVLLM_PTX_DIR && run bench"
