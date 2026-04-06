#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODEL="${MODEL:-/root/models/Qwen2.5-7B}"
N_VALUES="${N_VALUES:-1,64,128}"
OUTPUT_LEN="${OUTPUT_LEN:-128}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.9}"
FEATURES="${FEATURES:-cuda,cublaslt}"
PROFILE_NS="${PROFILE_NS:-1,64,128}"
NSYS_OUTPUT_LEN="${NSYS_OUTPUT_LEN:-16}"
BUILD="${BUILD:-1}"
RUN_NSYS="${RUN_NSYS:-1}"
NSYS_BIN="${NSYS_BIN:-}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) MODEL="$2"; shift 2 ;;
        --n) N_VALUES="$2"; shift 2 ;;
        --output-len) OUTPUT_LEN="$2"; shift 2 ;;
        --gpu-mem) GPU_MEM_UTIL="$2"; shift 2 ;;
        --profile-ns) PROFILE_NS="$2"; shift 2 ;;
        --nsys-output-len) NSYS_OUTPUT_LEN="$2"; shift 2 ;;
        --skip-build) BUILD=0; shift ;;
        --skip-nsys) RUN_NSYS=0; shift ;;
        *) echo "unknown flag: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "$NSYS_BIN" ]]; then
    if command -v nsys >/dev/null 2>&1; then
        NSYS_BIN="$(command -v nsys)"
    elif [[ -x /opt/nvidia/nsight-compute/2025.1.0/host/target-linux-x64/nsys ]]; then
        NSYS_BIN=/opt/nvidia/nsight-compute/2025.1.0/host/target-linux-x64/nsys
    fi
fi

timestamp="$(date +%Y%m%d_%H%M%S)"
out_dir="${ROOT_DIR}/results/autokernel/${timestamp}"
mkdir -p "$out_dir"

log() {
    printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"
}

if [[ "$BUILD" == "1" ]]; then
    log "building rvllm (--release, features=${FEATURES})"
    cargo build --release --features "${FEATURES}" -p rvllm
fi

bin="${ROOT_DIR}/target/release/rvllm"
if [[ ! -x "$bin" ]]; then
    echo "missing binary: ${bin}" >&2
    exit 1
fi

log "running fixed benchmark n=${N_VALUES} output_len=${OUTPUT_LEN}"
RVLLM_PROFILE=1 "$bin" benchmark \
    --model "$MODEL" \
    --n "$N_VALUES" \
    --output-len "$OUTPUT_LEN" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --json | tee "${out_dir}/benchmark.jsonl"

python3 - "${out_dir}/benchmark.jsonl" "${out_dir}/summary.txt" <<'PY'
import json, sys, pathlib
src = pathlib.Path(sys.argv[1])
dst = pathlib.Path(sys.argv[2])
rows = []
for line in src.read_text().splitlines():
    line = line.strip()
    if not line.startswith("{"):
        continue
    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        continue
    if "n" in obj and "tok_per_sec" in obj:
        rows.append((obj["n"], obj["tok_per_sec"]))
with dst.open("w") as f:
    for n, tps in rows:
        f.write(f"N={n}: {tps:.1f} tok/s\n")
print(dst.read_text(), end="")
PY

if [[ "$RUN_NSYS" == "1" ]] && [[ -n "$NSYS_BIN" ]]; then
    OLDIFS="$IFS"
    IFS=',' read -r -a prof_ns <<< "$PROFILE_NS"
    IFS="$OLDIFS"
    for n in "${prof_ns[@]}"; do
        base="${out_dir}/nsys_n${n}"
        log "profiling n=${n} with nsys"
        "$NSYS_BIN" profile --stats=true --force-overwrite=true -o "$base" \
            "$bin" benchmark \
            --model "$MODEL" \
            --n "$n" \
            --output-len "$NSYS_OUTPUT_LEN" \
            --gpu-memory-utilization "$GPU_MEM_UTIL" \
            --json > "${base}.stdout" 2> "${base}.stderr" || true
        "$NSYS_BIN" stats --force-export=true -r cuda_gpu_kern_sum "${base}.nsys-rep" \
            > "${base}.kern.txt" 2>/dev/null || true
        "$NSYS_BIN" stats --force-export=true -r cuda_gpu_mem_time_sum "${base}.nsys-rep" \
            > "${base}.mem.txt" 2>/dev/null || true
    done
fi

cat > "${out_dir}/run.txt" <<EOF
model=${MODEL}
n_values=${N_VALUES}
output_len=${OUTPUT_LEN}
gpu_mem_util=${GPU_MEM_UTIL}
profile_ns=${PROFILE_NS}
nsys_output_len=${NSYS_OUTPUT_LEN}
EOF

log "artifacts saved to ${out_dir}"
