#!/usr/bin/env bash
# race.sh — spin up a fresh vast.ai H100 and run the rvLLM vs vLLM lifecycle benchmark
# Usage: ./race.sh [vast_api_key]
# VASTAI_API_KEY env var is used if no argument is given.
set -euo pipefail

VASTAI_API_KEY="${1:-${VASTAI_API_KEY:-}}"
if [[ -z "$VASTAI_API_KEY" ]]; then
  echo "error: set VASTAI_API_KEY or pass it as first argument" >&2
  exit 1
fi

export VASTAI_API_KEY

# ---------------------------------------------------------------------------
# 1. Find cheapest available H100 with enough disk and network
# ---------------------------------------------------------------------------
echo "==> searching for H100 on vast.ai..."
OFFER=$(vastai search offers \
  'gpu_name=H100 disk_space>80 inet_up>200' \
  --order dph_total --limit 1 --raw 2>/dev/null | python3 -c "
import sys, json
rows = json.load(sys.stdin)
if not rows: sys.exit('no H100 offers found')
r = rows[0]
print(r['id'], r['dph_total'], r.get('gpu_name','?'))
")
OFFER_ID=$(echo "$OFFER" | awk '{print $1}')
echo "    offer $OFFER_ID selected ($OFFER)"

# ---------------------------------------------------------------------------
# 2. Launch instance
# ---------------------------------------------------------------------------
echo "==> launching instance..."
INSTANCE_ID=$(vastai create instance "$OFFER_ID" \
  --image pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel \
  --disk 80 \
  --ssh \
  --direct \
  --raw 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['new_contract'])")
echo "    instance $INSTANCE_ID created"

# ---------------------------------------------------------------------------
# 3. Wait for SSH
# ---------------------------------------------------------------------------
echo "==> waiting for SSH..."
SSH_HOST=""
SSH_PORT=""
for i in $(seq 1 60); do
  INFO=$(vastai show instance "$INSTANCE_ID" --raw 2>/dev/null)
  STATUS=$(echo "$INFO" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('actual_status',''))" 2>/dev/null || true)
  if [[ "$STATUS" == "running" ]]; then
    SSH_HOST=$(echo "$INFO" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('ssh_host',''))" 2>/dev/null || true)
    SSH_PORT=$(echo "$INFO" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('ssh_port',''))" 2>/dev/null || true)
    if [[ -n "$SSH_HOST" && -n "$SSH_PORT" ]]; then break; fi
  fi
  echo "    status=$STATUS, retry $i/60..."
  sleep 10
done

if [[ -z "$SSH_HOST" ]]; then
  echo "error: timed out waiting for SSH" >&2
  exit 1
fi
echo "    ssh root@$SSH_HOST -p $SSH_PORT"

SSH="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 root@$SSH_HOST -p $SSH_PORT"

# wait for sshd to accept connections
for i in $(seq 1 20); do
  $SSH 'echo ok' &>/dev/null && break || true
  sleep 5
done

# ---------------------------------------------------------------------------
# 4. Bootstrap: install Rust, deps, clone repo, download model
# ---------------------------------------------------------------------------
echo "==> bootstrapping instance..."
$SSH 'bash -s' <<'REMOTE'
set -euo pipefail

# System deps
apt-get update -qq
apt-get install -y -qq pkg-config libssl-dev

# Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable
. "$HOME/.cargo/env"

# Repo
git clone https://github.com/m0at/rvllm.git /root/rvllm

# Build rvllm binary and install vLLM in parallel
(
  cd /root/rvllm
  cargo build --release 2>&1
) &
BUILD_PID=$!

(
  python3 -m venv /root/venv
  /root/venv/bin/pip install --quiet uv
  /root/venv/bin/uv pip install vllm aiohttp 2>&1
) &
VENV_PID=$!

# Download model
/root/venv/bin/pip install --quiet huggingface_hub 2>/dev/null || pip install --quiet huggingface_hub
huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir /root/models/Qwen2.5-7B 2>&1 &
MODEL_PID=$!

wait $BUILD_PID
echo "--- rvllm build done ---"
wait $VENV_PID
echo "--- venv done ---"
wait $MODEL_PID
echo "--- model download done ---"

# aiohttp for harness (system python3)
pip install --quiet aiohttp
REMOTE

# ---------------------------------------------------------------------------
# 5. Run the benchmark
# ---------------------------------------------------------------------------
echo "==> running lifecycle benchmark..."
$SSH 'cd /root/rvllm && python3 bench_harness.py 2>&1'

# ---------------------------------------------------------------------------
# 6. Fetch results
# ---------------------------------------------------------------------------
echo "==> fetching results..."
mkdir -p bench
scp -P "$SSH_PORT" -o StrictHostKeyChecking=no \
  "root@$SSH_HOST:/root/bench_results/combined_results.json" \
  bench/combined_results_h100_lifecycle.json

echo ""
echo "==> done. results in bench/combined_results_h100_lifecycle.json"
echo "    instance $INSTANCE_ID is still running — destroy with:"
echo "    vastai destroy instance $INSTANCE_ID"
