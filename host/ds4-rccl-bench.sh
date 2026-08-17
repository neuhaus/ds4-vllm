#!/usr/bin/env bash
# ds4-rccl-bench.sh — RCCL all-reduce latency bench across BOTH DS4 boxes over
# the InfiniBand fabric. Run on box1 (the ray head); box2 is driven over ssh.
#
# Measures the per-op all-reduce latency RCCL delivers for the shapes the vLLM
# communicator sees (decode ~48 KiB, prefill ~4 MiB). The number to beat is the
# custom USB4 tbv_ar2's ~105 us/op on the decode collective; the decode step
# chains ~160 all-reduces per token, so per-op latency is the TP bottleneck.
#
# Requires: ds4-config.yaml (head_ip/worker_ip/container/rdma_hca) and the
# rdma cluster-env on BOTH boxes (repo host/, same path), same layout as the
# serving bring-up. container-heal.sh starts the containers if they are down.
set -uo pipefail

# Repo-relative: sibling files (config, env, bench, heal) resolve next to this
# script; box2 must have the same repo at the same path.
SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

eval "$("$SELF_DIR/ds4-config" "$SELF_DIR/ds4-config.yaml")"
HEAD_IP=${DS4_HEAD_IP:?ds4-config.yaml: head_ip missing}
WORKER_IP=${DS4_WORKER_IP:?ds4-config.yaml: worker_ip missing}
CTR=${DS4_CONTAINER:-vllm}
PORT=${DS4_BENCH_PORT:-29600}
ITERS=${DS4_BENCH_ITERS:-2000}
BENCH=$SELF_DIR/ds4-rccl-bench.py
HEAL=$SELF_DIR/container-heal.sh

box2() { timeout "${2:-120}" ssh -o BatchMode=yes "$WORKER_IP" "$1"; }
inbox() { timeout "${2:-120}" podman exec -u 1000:1000 -w "$HOME" "$CTR" bash -lc "$1"; }

# Env values that must match on both ranks (sourced from the rdma cluster-env
# inside each container; these two come from the site config).
ENVPASS="export DS4_RDMA_HCA=${DS4_RDMA_HCA:-} DS4_NET_IFACE=${DS4_NET_IFACE:-};"

# The bench must be present on BOTH boxes at the same repo path (box2 is
# driven over ssh; sync the repo there).
[ -f "$BENCH" ] || { echo "!! $BENCH missing on box1 (scp/sync the repo to box2)"; exit 1; }

# Containers running on both boxes (they do not start at boot on their own).
"$HEAL" "$CTR" >/dev/null 2>&1 || true
box2 "\$HEAL $CTR" 60 >/dev/null 2>&1 || true
inbox true 20 >/dev/null 2>&1 || { echo "!! box1 $CTR not exec-able"; exit 1; }
box2 "podman exec $CTR true" 20 >/dev/null 2>&1 || { echo "!! box2 $CTR not exec-able"; exit 1; }

echo "== RCCL bench: $HEAD_IP:$PORT, $ITERS iters, HCA=${DS4_RDMA_HCA:-ibp195s0} =="

# rank1 on box2 in the background, rank0 on box1 in the foreground; both source
# the rdma cluster-env so NCCL_IB_HCA/NCCL_IB_GID_INDEX point at the fabric.
box2 "podman exec -u 1000:1000 -w \$HOME $CTR bash -lc '$ENVPASS source $SELF_DIR/ds4-cluster-env.rdma.sh; exec python3 $BENCH --rank 1 --master $HEAD_IP --port $PORT --iters $ITERS'" 600 2>/dev/null \
  >/tmp/ds4-rccl-bench-rank1.log &
RANK1=$!

inbox "$ENVPASS source $SELF_DIR/ds4-cluster-env.rdma.sh; exec python3 $BENCH --rank 0 --master $HEAD_IP --port $PORT --iters $ITERS" 600 2>/dev/null
RC0=$?
wait "$RANK1" 2>/dev/null
echo "--- rank1 (box2) ---"
cat /tmp/ds4-rccl-bench-rank1.log
rm -f /tmp/ds4-rccl-bench-rank1.log

[ "$RC0" -eq 0 ] || { echo "!! rank0 (box1) bench failed"; exit 1; }
echo "== done — compare med us/op vs tbv_ar2's ~105 (decode collective) =="