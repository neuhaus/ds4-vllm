# AGENTS.md — bring up DeepSeek-V4-Flash on the 2-box gfx1151 vLLM cluster

You are an agent setting this up on a fresh pair of machines. This file is the
runbook: build the pieces, wire them together, get the model serving, and verify
it. Read it top to bottom **before** running anything — several steps are
hard-to-reverse and order matters.

## 0. What you are building

DeepSeek-V4-Flash served by a **patched vLLM**, tensor-parallel across **two AMD
Strix Halo (gfx1151) boxes**, with the inter-GPU all-reduce carried over a
**native InfiniBand** link (ConnectX-3 / mlx4) handled by **RCCL**.

```
        ┌────────────── box1 (ray HEAD, gfx1151) ──────────────┐
        │  toolbox "vllm"  ──►  vllm serve  TP rank 0           │
        │  ds4-vllm.service → ds4-cluster-restart.sh            │
        └───────────────┬───────────────────────────────────────┘
                        │  InfiniBand cable (QSFP)
                        │  HCA = ibp195s0  (ConnectX-3; udev-named after netdev)
                        │  control plane = head_ip / worker_ip
        ┌───────────────┴───────────────────────────────────────┐
        │  toolbox "vllm"  ──►  ray worker  TP rank 1          │
        └────────────── box2 (ray WORKER, gfx1151) ─────────────┘
```

The **interconnect latency is the TP bottleneck**: the decode step chains ~160
all-reduces per token, so per-op latency dominates (the USB4 custom path this
replaced measured ~105 µs/op; RCCL-over-IB should land well under that —
`host/ds4-rccl-bench.sh` measures it). Bandwidth is secondary; ConnectX-3
(40 Gb/s) has thin headroom at peak concurrency but is not the limiter at
realistic load.

Three layers, build/verify them in this order:

1. **InfiniBand fabric** — stock `mlx4_core`/`mlx4_ib` in-tree drivers, OpenSM
   (subnet manager), both ports Active with a LID. No custom kernel modules.
2. **vLLM engine** (`container/`) — rebuild the patched image, one toolbox container per box.
3. **Host orchestration** (`host/`) — the launch scripts, env, model weights.

> **USB4/Thunderbolt alternative.** The original interconnect — a custom
> Thunderbolt/USB4 soft-RDMA stack (`tbv/`) with its own kernel modules,
> provider, and `tbv_ar`/`tbv_ar2` all-reduce — is still documented in
> [`tbv/README.md`](tbv/README.md). It is NOT built or used on this path:
> `DS4_TBV_AR*` default to 0 and RCCL runs the all-reduce over the fabric.

## 0.1 Prerequisites (verify these exist; do NOT try to synthesize them)

- **2× AMD Strix Halo / gfx1151**, ~128 GB unified memory each, on the same LAN.
- A **ConnectX-3 (mlx4)** card in each box, direct-attached (QSFP cable) and
  **OpenSM running** on one of them (ConnectX-3 has no embedded subnet manager).
  `ibv_devinfo` must show the port **Active** with a LID on both boxes.
- Linux with **kernel headers/devel** for the running kernel on each box, `podman`,
  `toolbox`, `rdma-core`/`libibverbs`, `git`, build toolchain. mlx4 uses the
  **stock in-tree drivers** — nothing out-of-tree to build.
- The model weights **`deepseek-ai/DeepSeek-V4-Flash-0731`** (~150 GB) downloaded
  on **both** boxes (`hf download deepseek-ai/DeepSeek-V4-Flash-0731`).
- Root/sudo on both boxes (systemd units, RDMA memlock).

Pick roles now and keep them consistent everywhere: **box1 = ray head**,
**box2 = worker**. Site values (IPs, container name, transport, HCA pin, disk
KV) live in `host/ds4-config.yaml`, deployed as `~/ds4-config.yaml` on box1
(see §3); paths in the scripts are `$HOME`-relative.

---

## 0.2 Recall / context integrity — fixed, gated

Long-context recall on this stack is correct for the deployed profile. What
keeps it correct:

- `DS4_IDX_OFFICIAL=1` in `host/ds4-cluster-env.sh` -- the sparse indexer's
  official Hadamard128 + FP4 QAT scoring graph. Must be engaged on BOTH TP
  ranks (it is exported from the shared env, so keep the env identical).
- The `deepseek_v4_encoding.py` patch -- the chat encoder no longer strips
  prior assistant reasoning on tool conversations.

Any change to the indexer, MTP, kernels, or tuning knobs must re-pass
needle/recall probes at your target context depth before it ships (see §5).

Kernel/fusion changes must also pass `host/ds4-kernel-harness.py` (bit-exact
fused-vs-stock comparison with crash-shaped + stress inputs, run inside the
container with `HIP_LAUNCH_BLOCKING=1`). Trap: Triton 3.7.0 / ROCm 7.14
miscompiles runtime-trip-count `scf.for` loops (garbage loads/stores, sporadic
illegal memory accesses — the 2026-08-19 `_topk_ragged_decode_kernel` fault);
the AMDGPU backend drops the trip-count branch of the guarded loop form (no
`tl.assume`). Fix: keep the loop dynamic and add `tl.assume(num_tokens > 0)`
— that removes the guard structure and the lowering is bit-exact (fallback:
`tl.static_range` + masked inactive iterations). See the kernel in
`ds4_fused_glue.py` and the upstream report for details.

---

## 1. InfiniBand fabric (do this first)

Native InfiniBand on the **stock in-tree drivers** — no custom kernel modules,
no provider builds, no coordinated reboots. This is the connect-the-dots step,
run once per box. Full detail in the distro/OpenSM docs; the gates below are
what the cluster actually needs.

### 1.1 Drivers and firmware

- **`mlx4_core` / `mlx4_ib`** are in-tree. Load them: `modprobe mlx4_core
  mlx4_ib`. Nothing out-of-tree to build (unlike the old `tbv/` USB4 stack).
- Port mode: **InfiniBand** (`LINK_TYPE_P1=1`), set with `mlxconfig`/`mstconfig`
  + `mstflint`. (This deployment is native IB with OpenSM — **not** RoCE.)
- **OpenSM** running on one box (ConnectX-3 has no embedded subnet manager);
  it assigns the LIDs and brings the ports to Active.

### 1.2 Memlock (both boxes)

`sudo tbv/bringup/fix-memlock.sh` — the only file kept from the USB4 stack; it
raises the RDMA memlock limit (`/etc/security/limits.d/99-rdma-memlock.conf` +
systemd `DefaultLimitMEMLOCK`), which RCCL/`ibv_reg_mr` need. Re-login (or a new
ssh session) after running.

### 1.3 Configure the IPoIB netdev (both boxes)

The control plane (ray, NCCL/GLOO sockets) runs over **IPoIB**, whose netdev
here is `ibp195s0` (predictable naming; find yours with `ls /sys/class/net/`).
Give it the `head_ip`/`worker_ip` from `ds4-config.yaml` and bring it up at
boot (NetworkManager/nmtui or an `nmcli con add type infiniband` profile).
**Those IPs must actually be configured on this interface** — ray binds to them.

### 1.4 Verify RDMA (gate)

```bash
rdma link                                          # ibp195s0 state ACTIVE / LinkUp
ibv_devinfo -d ibp195s0                            # port state Active, with a LID
ls /sys/class/infiniband/                          # -> ibp195s0 (udev-named after netdev)
ip -br addr show ibp195s0                          # has head_ip / worker_ip
```
Then inside the serving container: `toolbox enter vllm -- ibv_devices` must
list `ibp195s0` (the image guarantees the mlx4 libibverbs provider). The HCA
name matches the netdev on this stack; on another box use whatever
`ibv_devices` prints and set `rdma_hca` in the config to match.

**De-risk option:** RDMA is a performance layer, not a correctness gate — set
`transport: tcp` in `~/ds4-config.yaml` to run the same cluster over sockets on
`net_iface` (much slower decode). If you want to validate the model path first,
skip to §2–§4 on TCP now and return to finish IB once tokens are flowing.

> **USB4/Thunderbolt alternative.** The original interconnect (`tbv/` — custom
> kernel modules, provider, `tbv_ar`/`tbv_ar2` all-reduce) is documented in
> [`tbv/README.md`](tbv/README.md). Not used here: `DS4_TBV_AR*` default to 0
> and RCCL runs the all-reduce over the mlx4 fabric. `host/ds4-rccl-bench.sh`
> measures the per-op latency to beat (USB4 was ~105 µs; the decode step chains
> ~160 all-reduces per token).

---

## 2. Build the vLLM engine

See [`container/`](container/). On **each** box:

```bash
cd container && ./build.sh                # -> ds4-vllm-patched:local  (base ~35 GB pulled once)
```
This is `FROM kyuz0/vllm-therock-gfx1151@<pinned digest>` + the DS4 patch-set
(36 modified files as `patches/vllm-upstream.patch`, 16 new — see
`container/patches/MANIFEST.md`). Then create the serving container, named per
`container:` in `ds4-config.yaml` (default **`vllm`**) with **toolbox** — the
`--` separator forwards the remaining args to `podman create` (toolbox ≥ 0.3):

```bash
toolbox create vllm --image ds4-vllm-patched:local -- \
  --device /dev/kfd --device /dev/dri --device /dev/infiniband \
  --group-add keep-groups --security-opt seccomp=unconfined
toolbox enter vllm -- vllm --version          # gate: prints a version
toolbox enter vllm -- ibv_devices             # gate: lists ibp195s0 (if §1 done)
```

`--group-add keep-groups` keeps the user's supplementary groups (video/render)
so the GPU nodes stay reachable without `--privileged`; the explicit
`/dev/infiniband` device grant covers the mlx4 HCA for the RDMA path.

## 3. Host orchestration + config

Set the site values in `host/ds4-config.yaml` (head/worker IPs, container
name, `transport: rdma|tcp`, RDMA HCA pin, disk KV). Everything runs **from
the repo checkout** — no `$HOME` deployment copies: the scripts resolve their
sibling files relative to their own location, and the systemd unit
(`host/systemd/ds4-vllm.service`, installed into `~/.config/systemd/user/`)
points at the repo's `host/ds4-cluster-restart.sh`. Three rules that bite:

- `ds4-cluster-env*.sh` **must be byte-identical on both boxes** — the two TP
  ranks silently diverge otherwise. Keep box2's repo synced to the same
  commit at the same absolute path (box1 → box2 over ssh).
- Box1 needs passwordless ssh to the worker IP: the cluster scripts drive
  box2's container over ssh.
- **`loginctl enable-linger` must be set on BOTH boxes.** The serving
  container runs under the user's rootless-podman systemd manager
  (`user@1000.service`); with linger off, logind deactivates that manager when
  the last session closes and the box2 container is SIGTERMed with it. The
  cluster then silently drops to a 1-GPU placement group and serve hangs until
  the API timeout. `ds4-cluster-restart.sh` now gates on this and fails fast.

## 4. Start serving

```bash
systemctl --user start ds4-vllm      # box1; ~5 min warm
```

`ds4-cluster-restart.sh` (the unit's ExecStart) does the whole sequence:
teardown + stranded-process reap, container heal on both boxes, ray head +
box2 worker (2 GPUs gate), then `vllm serve` via `ds4-vllm-manual-serve.sh`
as the transient `ds4-vllm-manual` unit (MTP speculative decode,
`deepseek_v4` tokenizer/reasoning/tool parsers, fp8 KV, eager, disk KV
tier), verifies the API and the IB fabric (`rdma link` must show the pinned
HCA ACTIVE), and dispatches the warmup (`ds4-vllm-warmup.py`, `warmup_ctx` in
the yaml) before reporting success. `systemctl --user stop ds4-vllm` tears
everything down. Run `host/ds4-rccl-bench.sh` afterwards to measure the RCCL
all-reduce per-op latency (the number to beat: USB4 tbv_ar2 ~105 µs/op).
