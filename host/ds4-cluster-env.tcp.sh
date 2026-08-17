#!/usr/bin/env bash
# TCP-transport override for the DS4 vLLM cluster (`transport: tcp` in
# ds4-config.yaml). Runs the same 2-box cluster over sockets on the management
# interface with IB/RDMA and the tbv_ar all-reduce disabled -- the
# correctness/fallback profile for when the fabric is unavailable. Much slower
# decode (the all-reduce is latency-bound; see README).
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/ds4-cluster-env.sh"
export NCCL_IB_DISABLE=1
unset  NCCL_IB_HCA NCCL_IB_GID_INDEX
export NCCL_PROTO=Simple
export NCCL_SOCKET_IFNAME=${DS4_NET_IFACE:-}
export GLOO_SOCKET_IFNAME=${DS4_NET_IFACE:-}
export DS4_TBV_AR=0
export DS4_TBV_AR2=0
