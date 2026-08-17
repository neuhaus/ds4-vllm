# DeepSeek-V4-Flash on vLLM — gfx1151 2-box rebuild kit

A reproducible rebuild of the hand-patched vLLM engine that serves
**DeepSeek-V4-Flash** across **two AMD Strix Halo (gfx1151) boxes**, tensor-parallel
(TP=2), with the inter-GPU all-reduce carried over a **native InfiniBand** link
(ConnectX-3 / mlx4) by RCCL. It contains everything needed to reconstruct the
*software* from a public base image plus the host-side scripts that launch and
drive it.

This code and stack was pretty much entirely put together by AI, I probably can not help too much outside of prompting my agent.
PRs are welcome for performance improvements.

## Performance

Single-stream, TP=2 across the two boxes over InfiniBand, with the
**DFlash parallel drafter** (DSpark MTP speculative decoding) enabled. Decode
speed depends on how often the drafter's parallel tokens are accepted, so
prose and code generate at different rates. Measured on the reference rig
(2× Ryzen AI Max+ 395 / Radeon 8060S, 128 GB UMA each): fresh uncached
prompts, temperature 0, thinking disabled, 300-token generations.

| context | prefill tok/s | decode — prose | decode — code |
|---|---|---|---|
| 512 | ~300 | 23 | 32 |
| 10k | ~270 | 23 | 27 |
| 50k | ~239 | 19 | 27 |
| 100k | ~191 | 19 | 22 |

Prefill is content-agnostic (prose and code measured within a few percent).
A fresh bringup warms its own kernels/caches automatically
(`warmup_ctx`), so these rates hold from the first real request.

---

## TL;DR — minimal bring-up

Hardware: **2× AMD Strix Halo (gfx1151)** boxes (~128 GB unified memory each), a
**ConnectX-3 (mlx4) InfiniBand card** in each, direct-attached (QSFP cable), with
**OpenSM** running on one of them. Then, in order:

```bash
# 0. model weights, ~150 GB, on BOTH boxes (https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)
hf download deepseek-ai/DeepSeek-V4-Flash-0731

# 1. InfiniBand fabric, on BOTH boxes (stock in-tree drivers; no kernel builds)
modprobe mlx4_core mlx4_ib          # verify: ibv_devinfo -d ibp195s0 -> Active + LID
sudo tbv/bringup/fix-memlock.sh     # RDMA memlock; re-login after
#    IPoIB netdev (ibp195s0) gets head_ip/worker_ip; OpenSM on one box assigns LIDs

# 2. the patched vLLM image (box1; copy to box2 with podman save | podman load)
container/build.sh                                  # then create the distrobox — §2 below

# 3. site config + host scripts
#    edit host/ds4-config.yaml (IPs, net_iface, transport, HCA, memory) and deploy per §3

# 4. launch (box1) — full 2-box bringup, OpenAI API on :1234 when done
systemctl --user start ds4-vllm
```

Full ordered runbook with the gates and gotchas: [`AGENTS.md`](AGENTS.md).

---

> **Read this first — what "rebuildable" means here.** The **container rebuilds
> deterministically on any machine** with podman (`container/` below). *Serving*
> the model, however, needs the matching rig: 2× gfx1151 boxes, a working
> Thunderbolt-4 RDMA fabric, ROCm 7, and the model weights. This is a
> hardware-specific research build, not a general-purpose vLLM package. See
> **Prerequisites**.
>
> **Setting it up? Follow [`AGENTS.md`](AGENTS.md)** — the ordered end-to-end
> runbook (RDMA → container → serve) written for a person or agent doing the
> bring-up on a fresh pair of boxes.

---

## License & attribution

Original work here is **Apache-2.0** ([LICENSE](LICENSE)). This project
builds on **vLLM** (Apache-2.0), the **Linux kernel thunderbolt drivers** and
**hellas-ai/thunderbolt-ibverbs** (GPL-2.0), and **rdma-core** — their
sources are fetched at pinned revisions at build time rather than
redistributed; the patches shipped here are derivative works licensed like
the code they modify. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)
for the full component/license table and upstream references.

---

## Layout

```
ds4-vllm-share/
├── README.md                     ← this file
├── AGENTS.md                     ← ordered bring-up runbook (RDMA → container → serve)
├── container/                    ← rebuild the patched vLLM engine (route 2)
│   ├── Dockerfile                ← FROM kyuz0 gfx1151 base + COPY the patch-set
│   ├── build.sh                  ← podman build helper (runs the packaging tests first)
│   ├── verify-patches.sh         ← prove patches/ really is base → rootfs
│   ├── rootfs/                   ← the 12 NEW files at their real paths (modified files ship as the patch)
│   └── patches/                  ← vllm-upstream.patch (base → patched) + MANIFEST.md
├── tbv/                          ← USB4/Thunderbolt soft-RDMA stack (NOT used on the IB path;
│                                    the original interconnect, kept for reference)
│   ├── ibverbs-local.patch       ← our diff on the pinned upstream thunderbolt-ibverbs
│   ├── nhi-throttle-mod/         ← NHI IRQ-throttle module source
│   ├── bringup/                  ← fix-memlock.sh (still used) + the USB4 RoCE bring-up
│   ├── systemd/                  ← boot units (matched core+net; RoCE bring-up)
├── host/                         ← host-side orchestration (run outside the container)
│   ├── ds4-config.yaml, ds4-config ← site config (IPs, transport, disk KV) + loader
│   ├── ds4-cluster-restart.sh    ← full validated bringup (ExecStart of ds4-vllm.service)
│   ├── ds4-cluster-down.sh       ← full teardown (ExecStop/StopPost)
│   ├── ds4-vllm-manual-serve.sh  ← the vllm serve invocation + all serving flags
│   ├── ds4-vllm-warmup.py        ← post-start JIT/prefill-cache warmer (warmup_ctx)
│   ├── ds4-rccl-bench.{sh,py}    ← RCCL all-reduce latency probe across both boxes (IB)
│   ├── ds4-cluster-env*.sh       ← canonical env + DS4_* tuning knobs (rdma/tcp variants)
│   ├── container-heal.sh         ← reconcile/start a wedged podman container
│   └── systemd/                  ← ds4-vllm.service
```

---

## Prerequisites (to actually serve)

- **2× AMD Strix Halo / gfx1151** boxes, ~128 GB unified memory each. box1 is the
  ray head; box2 joins as a worker.
- **ROCm 7** (provided inside the container via the kyuz0 base — you do not
  install it on the host).
- **podman** + **distrobox** on both hosts (rootless is fine; the live setup uses it).
- **Native InfiniBand** between the boxes (ConnectX-3 / mlx4, OpenSM running).
  The scripts expect an RDMA HCA named `ibp195s0` here (matches the netdev;
  set `rdma_hca` in the config to whatever `ibv_devices` prints) and an
  **IPoIB netdev** carrying `head_ip`/`worker_ip`.
  This uses the **stock in-tree `mlx4_core`/`mlx4_ib` drivers** — nothing
  out-of-tree to build (the old custom `tbv/` USB4 stack is documented in
  [`tbv/README.md`](tbv/README.md) but not used here). Without RDMA, the stack
  still runs on `transport: tcp` (much slower decode).
- **Model weights**: [`deepseek-ai/DeepSeek-V4-Flash-0731`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)
  (~150 GB checkpoint). Not included — `hf download deepseek-ai/DeepSeek-V4-Flash-0731`
  on both boxes (the `model:` key in `ds4-config.yaml` takes an HF id or a local
  path). Served as `deepseek-v4-flash`.

---

## 1. Rebuild the container

```bash
cd container
./build.sh                       # -> ds4-vllm-patched:local
```

This does `FROM docker.io/kyuz0/vllm-therock-gfx1151@<pinned-digest>`, applies
`container/patches/vllm-upstream.patch` to the base's own vLLM sources (31
files), overlays the 12 new files from `container/rootfs/`
(see [`container/patches/MANIFEST.md`](container/patches/MANIFEST.md)), guarantees
the stock **mlx4** (ConnectX-3) libibverbs provider via `rdma-core`, and rebuilds
ROCr with the idle-wait fix. The custom USB4 `tbv_ar`/`tbv_ar2` all-reduce
natives are **not** built — on this IB stack RCCL runs the TP all-reduce (the
`tbv_ar*.py` wrappers still ship, inert). The
base is ~35 GB and is pulled on first build; network is needed on the first
build for the pinned source fetches.

**Base image drift.** The Dockerfile pins the base by **digest** so the rebuild
matches the engine the patches were developed against (vLLM commit `470229c`). If
that digest is ever unpullable, replace it with `:latest` — but be aware kyuz0's
`latest` moves, and a newer base could carry a different vLLM whose files the
patches assume. Prefer the pinned digest.

**Keeping the patch honest.** Two checks:

```bash
python3 -m unittest discover -s tests -v   # manifest vs rootfs vs patch vs Dockerfile
container/verify-patches.sh                # patch applies cleanly to the pinned base
```

`build.sh` runs the first automatically; the second needs the base image
locally. To regenerate the patch after editing engine files, point
`DS4_PATCH_SRC` at a tree holding the desired files and run
`container/verify-patches.sh --write`.

## 2. Create the serving distrobox

The cluster scripts `podman exec` into a container named per
`ds4-config.yaml` (default **`vllm`**), so create one from the image you just
built (on **both** boxes):

```bash
distrobox create --name vllm --image ds4-vllm-patched:local --additional-flags \
  '--privileged --ipc host --pid host \
   --device /dev/kfd --device /dev/dri --device /dev/infiniband \
   --group-add video --group-add render --security-opt seccomp=unconfined'
distrobox enter vllm -- vllm --version
```

**Do not pass `--network host` in `--additional-flags`.** distrobox already
selects host networking, and passing it again fails with
`cannot set multiple networks without bridge network mode`. The resulting
container still gets `net=host`; verify with
`podman inspect vllm --format '{{.HostConfig.NetworkMode}}'`.

## 3. Run

Site specifics live in **`host/ds4-config.yaml`** — deploy it (edited for your
site) as `~/ds4-config.yaml` on box1 next to the scripts:

```yaml
model: deepseek-ai/DeepSeek-V4-Flash-0731   # HF id or local path; weights on BOTH boxes
transport: rdma        # rdma | tcp — which ds4-cluster-env.<transport>.sh the cluster sources
head_ip: 192.168.100.1 # ray head control-plane address (site value, on the IPoIB/LAN iface)
worker_ip: 192.168.100.2 # ray worker address (site value)
container: vllm        # podman container name (same on both boxes)
rdma_hca: ibp195s0    # NCCL_IB_HCA pin, rdma transport only (your ConnectX-3 port; name matches the netdev)
net_iface: ibp195s0    # IPoIB netdev carrying head_ip/worker_ip; NCCL/GLOO sockets bind it
api_port: 1234
disk_kv: true          # NVMe prefix-KV tier (fs_lru); prefixes survive restarts
disk_kv_gib: 30        # per-NODE disk cap; check df on BOTH boxes before raising
max_ctx: 524288        # --max-model-len (512K, the validated profile)
kv_pin_gib: 6          # pinned GPU KV pool; sized against MemAvailable, not grown by max_ctx
gpu_mem_util: 0.83     # vLLM --gpu-memory-utilization; IGNORED while the KV pin above is set
warmup_ctx: 2048       # post-start warmup prefill size; 0 disables
```

`host/ds4-config` (stdlib Python, no pyyaml) turns it into `DS4_*` exports for
the scripts. `transport: tcp` runs the same cluster without RDMA (correctness /
fallback profile; sockets over `net_iface`, no fabric).

Deploy (paths are `$HOME`-relative, same layout on both boxes):

- **box1**: `host/ds4-config{,.yaml}`, `ds4-cluster-restart.sh`,
  `ds4-cluster-down.sh`, `ds4-vllm-manual-serve.sh`, `ds4-vllm-warmup.py`,
  `container-heal.sh`, all three `ds4-cluster-env*.sh`, and
  `host/systemd/ds4-vllm.service` into `~/.config/systemd/user/`.
- **box2**: `ds4-cluster-env*.sh` and `container-heal.sh` only — box2 is driven
  over ssh (key auth box1→box2 required).

Then `systemctl --user start ds4-vllm` brings up the whole 2-box cluster
(teardown → container heal → ray on both boxes → `vllm serve` → API/RDMA
verify); `stop` tears it down. The env files must stay **identical on both
boxes** — the two TP ranks silently diverge otherwise. 

---

## What was patched, and why

Full table in [`container/patches/MANIFEST.md`](container/patches/MANIFEST.md).
The themes:

- **DeepSeek-V4 model on gfx1151** — the AMD/ROCm DSpark model path, MLA
  attention, fp8 (UE8M0) KV-latent compress/quant, and the MTP drafter.
- **Mid-context retrieval** — the sparse indexer runs the *official* QAT graph
  (Hadamard128 + FP4 sim) before top-512 scoring (`DS4_IDX_OFFICIAL`), which the
  stock FP8 indexer skipped; plus a ROCm sparse-MLA attention rewrite.
- **MoE / GEMM tuning** — decode-scoped MXFP4 `matmul_ogs` knobs
  (`DS4_MOE_BN/NW/NS/BK/WPE`, the `block_k` bandwidth lever), a tuned gfx1151
  A8W8 GEMM config, and a `DS4_W8A8_BF16` fast bf16 path.
- **InfiniBand all-reduce via RCCL** — the TP=2 all-reduce runs over the
  native-IB fabric (`NCCL_IB_HCA=ibp195s0`, GID index 0); the custom USB4
  `tbv_ar`/`tbv_ar2` all-reduce (the `tbv/` stack) is kept but inert
  (`DS4_TBV_AR*` default 0). `host/ds4-rccl-bench.sh` measures the RCCL
  per-op latency.
- **Disk KV cache** an `fs_lru` secondary
  tier gives the KV offloader a byte cap with LRU eviction, which the stock `fs`
  tier has no mechanism for, so it can point at a filesystem shared with
  everything else. Alongside it, the offloading scheduler now bounds each store
  batch: stock asks for every un-offloaded block at once and does not advance its
  cursor when the tier refuses, so a single refusal ratchets the ask past the
  tier and stores stop for the rest of the request. Bounding it lets a few
  hundred MiB of staging carry a full-length prefill.

---

## Provenance & notes

- Base: `docker.io/kyuz0/vllm-therock-gfx1151@sha256:25fd294f…`, vLLM commit `470229c`.
- The original container also had `py-spy` pip-installed (a profiler) and some
  incidental OS packages under `/usr/lib/python3.14`; neither affects serving and
  both are intentionally omitted. Re-add `py-spy` with `pip install py-spy` inside
  the container if you want it.
