#!/usr/bin/env python3
"""Standalone harness: bit-exact fused-vs-stock gates for the decode kernels.

Two suites:

1. ds4_fused_glue.topk_ragged_decode vs the stock chain. Reproduces the
   2026-08-19 crash shapes (45k-token prefix, MTP-5: 6 decode tokens, topk=512,
   compressed block_size=32) plus stress sweeps. For each case the fused kernel
   (ds4_fused_glue) and the stock 4-launch chain (rocm.py
   compute_global_topk_ragged_indices_and_indptr) get the SAME inputs; outputs
   must be bitwise equal. A HIP memory fault in the fused kernel under
   HIP_LAUNCH_BLOCKING=1 raises synchronously and is caught per-case.

2. ds4_moe_hip (MXFP4 MoE decode): the shipped libds4moe.so decodes e2m1
   weights through an fp16-subnormal bit trick; the same source compiled with
   -DFP4_ARITH=1 uses the straightforward arithmetic decode. The trick is
   exact (tests/test_mxfp4_decode.py pins it in pure python), so the two
   builds must agree BITWISE on the GPU, in the layout the engine actually
   hands the wrapper. Also pins: deterministic repeat runs, all-slots-written
   (NaN canary in the output buffer), zero-gamma slots, int32/int64 ids,
   P=M*topk=256 launch boundary, and the eligible() gates (per-call skips must
   not latch the module off).

Run inside the serving container (has triton/torch/GPU/hipcc):
  podman exec vllm bash -lc 'cd /home/sn/git/ds4-vllm && \
    HIP_LAUNCH_BLOCKING=1 python3 host/ds4-kernel-harness.py'

Optional filter: pass a substring to only run matching cases ("moe" runs only
the MoE suite).

NOTE on stock-path comparison for suite 2: ds4_moe_hip replaces matmul_ogs and
is documented as within 1 bf16 ULP of it (different summation order). That is
the AGENTS.md tolerance exception -- it is intentionally NOT bit-exact and must
not be gated as such; what IS gated bit-exact here is kernel-vs-reference.
"""

import os
import random
import subprocess
import sys
import tempfile
import types
import zlib

os.environ.setdefault("DS4_FUSE_RAGGED", "1")
os.environ.setdefault("DS4_FUSE_IDXGATHER", "1")

import torch  # noqa: E402

from vllm.models.deepseek_v4.amd import rocm as _rocm  # noqa: E402

import ds4_fused_glue as _glue  # noqa: E402

DEV = "cuda:0"


def stock(topk_indices, t2r, block_table, block_size, is_valid):
    saved = _rocm._DS4_FUSE_RAGGED
    _rocm._DS4_FUSE_RAGGED = None
    try:
        return _rocm.compute_global_topk_ragged_indices_and_indptr(
            topk_indices, t2r, block_table, block_size, is_valid
        )
    finally:
        _rocm._DS4_FUSE_RAGGED = saved


def fused(topk_indices, t2r, block_table, block_size, is_valid):
    return _glue.topk_ragged_decode(
        topk_indices, t2r, block_table, block_size, is_valid
    )


def build_idx_row(nt, topk, seq_len_kv, valid_count=None, interleave=None,
                  pad_extra=0, inject_above=None, spec_pos=None):
    """One token's topk row: valid indices below seq_len_kv, rest -1.

    interleave: if set, put (interleave) -1 gaps among the first valid entries.
    inject_above: list of extra index values to append (before -1 pad) that may
    exceed seq_len_kv-1 (speculative / beyond-table case).
    pad_extra: extra -1 padding slots to reserve.
    """
    n_valid = topk - pad_extra
    if valid_count is not None:
        n_valid = min(valid_count, n_valid)
    pos = seq_len_kv - 1
    row = torch.full((topk,), -1, dtype=torch.int32)
    vals = (torch.arange(n_valid, dtype=torch.int32) - n_valid + 1 + pos)
    vals = vals.clamp(min=0)
    if interleave:
        vals[::interleave + 1] = -1
    if inject_above:
        vals[:len(inject_above)] = torch.tensor(inject_above, dtype=torch.int32)
    row[:vals.numel()] = vals
    return row


def make_table(num_seqs, row_len, base=100):
    tab = torch.arange(base, base + num_seqs * row_len,
                       dtype=torch.int32, device=DEV)
    return tab.reshape(num_seqs, row_len).contiguous()


def run_case(name, topk_indices, t2r, block_table, block_size, is_valid,
             iters=10):
    print(f"START {name}", flush=True)
    try:
        for i in range(iters):
            r_f, i_f, l_f = fused(topk_indices, t2r, block_table, block_size,
                                  is_valid)
            torch.cuda.synchronize()
            r_s, i_s, l_s = stock(topk_indices, t2r, block_table, block_size,
                                  is_valid)
            torch.cuda.synchronize()
            total = int(i_s[-1])
            for label, a, b in (("ragged", r_f[:total], r_s[:total]),
                                ("indptr", i_f, i_s),
                                ("lens", l_f, l_s)):
                if a.shape != b.shape or not torch.equal(a, b):
                    n = int((a != b).sum()) if a.shape == b.shape else -1
                    print(f"FAIL {name} iter={i} tensor={label} mismatches={n}")
                    print("  fused:", a.flatten()[:24].tolist())
                    print("  stock:", b.flatten()[:24].tolist())
                    return False
        print(f"OK   {name} ({iters} iters, bit-exact)", flush=True)
        return True
    except Exception as e:
        print(f"CRASH {name}: {type(e).__name__}: {e}", flush=True)
        return False


def cases():
    c = []

    # 1. exact crash shape: 45,081 tokens -> compressed seq 11,271, bs=32,
    #    table row = ceil(11271/32)=353 entries, 6 tokens (1 target+5 drafts),
    #    topk 512.
    def crash_shape():
        nt, topk, bs, slk = 6, 512, 32, 11271
        row_len = (slk + bs - 1) // bs
        idx = torch.full((nt, topk), -1, dtype=torch.int32, device=DEV)
        idx[0] = build_idx_row(nt, topk, slk)
        for d in range(1, nt):
            idx[d] = build_idx_row(nt, topk, slk + d, valid_count=topk - d)
        t2r = torch.zeros(nt, dtype=torch.int32, device=DEV)
        tab = make_table(1, row_len)
        valid = torch.ones(nt, dtype=torch.int32, device=DEV)
        return idx, t2r, tab, bs, valid

    c.append(("crash_shape", crash_shape))

    # 2. speculative indices beyond table coverage: draft rows include index
    #    values whose bidx exceeds row_len (idx//bs >= row_len). Masked load
    #    must survive (bnum=0) and match stock bitwise.
    def spec_beyond_table():
        nt, topk, bs, slk = 6, 512, 32, 11271
        row_len = (slk + bs - 1) // bs
        idx = torch.full((nt, topk), -1, dtype=torch.int32, device=DEV)
        idx[0] = build_idx_row(nt, topk, slk)
        for d in range(1, nt):
            idx[d] = build_idx_row(nt, topk, slk, inject_above=[slk + d * 100])
        t2r = torch.zeros(nt, dtype=torch.int32, device=DEV)
        tab = make_table(1, row_len)
        valid = torch.ones(nt, dtype=torch.int32, device=DEV)
        return idx, t2r, tab, bs, valid

    c.append(("spec_beyond_table", spec_beyond_table))

    # 3. interleaved -1 gaps among valid indices.
    def interleaved():
        nt, topk, bs, slk = 6, 512, 32, 11271
        row_len = (slk + bs - 1) // bs
        idx = torch.full((nt, topk), -1, dtype=torch.int32, device=DEV)
        idx[0] = build_idx_row(nt, topk, slk, interleave=3)
        t2r = torch.zeros(nt, dtype=torch.int32, device=DEV)
        tab = make_table(1, row_len)
        valid = torch.ones(nt, dtype=torch.int32, device=DEV)
        return idx, t2r, tab, bs, valid

    c.append(("interleaved_gaps", interleaved))

    # 4. invalid draft tokens (is_valid=0 rows) mixed in.
    def invalid_tokens():
        nt, topk, bs, slk = 6, 512, 32, 11271
        row_len = (slk + bs - 1) // bs
        idx = torch.full((nt, topk), -1, dtype=torch.int32, device=DEV)
        idx[0] = build_idx_row(nt, topk, slk)
        for d in range(1, nt):
            idx[d] = build_idx_row(nt, topk, slk + d, valid_count=topk - 40)
        t2r = torch.zeros(nt, dtype=torch.int32, device=DEV)
        tab = make_table(1, row_len)
        valid = torch.tensor([1, 0, 1, 0, 1, 0], dtype=torch.int32, device=DEV)
        return idx, t2r, tab, bs, valid

    c.append(("invalid_tokens", invalid_tokens))

    # 5. table row-length sweep (bidx near and beyond each row).
    def table_lengths():
        nt, topk, bs = 6, 512, 32
        slk = 11271
        row_len = 1024
        idx = torch.full((nt, topk), -1, dtype=torch.int32, device=DEV)
        idx[0] = build_idx_row(nt, topk, slk)
        t2r = torch.zeros(nt, dtype=torch.int32, device=DEV)
        tab = make_table(1, row_len)
        valid = torch.ones(nt, dtype=torch.int32, device=DEV)
        return idx, t2r, tab, bs, valid

    c.append(("table_len_1024", table_lengths))

    def table_lengths_short():
        nt, topk, bs = 6, 512, 32
        slk = 11271
        row_len = 64  # far shorter than needed: bidx up to 352 >> 64
        idx = torch.full((nt, topk), -1, dtype=torch.int32, device=DEV)
        idx[0] = build_idx_row(nt, topk, slk)
        t2r = torch.zeros(nt, dtype=torch.int32, device=DEV)
        tab = make_table(1, row_len)
        valid = torch.ones(nt, dtype=torch.int32, device=DEV)
        return idx, t2r, tab, bs, valid

    c.append(("table_len_64_short", table_lengths_short))

    # 6. block-size sweep.
    def block_sizes():
        nt, topk, slk = 6, 512, 11271
        bs = 128
        row_len = (slk + bs - 1) // bs
        idx = torch.full((nt, topk), -1, dtype=torch.int32, device=DEV)
        idx[0] = build_idx_row(nt, topk, slk)
        t2r = torch.zeros(nt, dtype=torch.int32, device=DEV)
        tab = make_table(1, row_len)
        valid = torch.ones(nt, dtype=torch.int32, device=DEV)
        return idx, t2r, tab, bs, valid

    c.append(("block_size_128", block_sizes))

    def block_sizes_16():
        nt, topk, slk = 6, 512, 11271
        bs = 16
        row_len = (slk + bs - 1) // bs
        idx = torch.full((nt, topk), -1, dtype=torch.int32, device=DEV)
        idx[0] = build_idx_row(nt, topk, slk)
        t2r = torch.zeros(nt, dtype=torch.int32, device=DEV)
        tab = make_table(1, row_len)
        valid = torch.ones(nt, dtype=torch.int32, device=DEV)
        return idx, t2r, tab, bs, valid

    c.append(("block_size_16", block_sizes_16))

    # 7. topk sweep (TOPK_PAD next_pow2 changes).
    def topk_sweep():
        topk = 1024
        nt, bs, slk = 6, 32, 11271
        row_len = (slk + bs - 1) // bs
        idx = torch.full((nt, topk), -1, dtype=torch.int32, device=DEV)
        idx[0] = build_idx_row(nt, topk, slk)
        t2r = torch.zeros(nt, dtype=torch.int32, device=DEV)
        tab = make_table(1, row_len)
        valid = torch.ones(nt, dtype=torch.int32, device=DEV)
        return idx, t2r, tab, bs, valid

    c.append(("topk_1024", topk_sweep))

    # 8. num_tokens sweep incl. the fusion cutoff boundary.
    def num_tokens_sweep():
        nt = 64
        topk, bs, slk = 512, 32, 11271
        row_len = (slk + bs - 1) // bs
        idx = torch.full((nt, topk), -1, dtype=torch.int32, device=DEV)
        for t in range(nt):
            idx[t] = build_idx_row(nt, topk, min(slk + t, slk + 8))
        t2r = torch.zeros(nt, dtype=torch.int32, device=DEV)
        tab = make_table(1, row_len)
        valid = torch.ones(nt, dtype=torch.int32, device=DEV)
        return idx, t2r, tab, bs, valid

    c.append(("num_tokens_64", num_tokens_sweep))

    # 9. multi-request mapping: several requests, tokens mapping via t2r.
    def multi_req():
        nreq = 3
        nt, topk, bs, slk = 9, 512, 32, 11271
        row_len = (slk + bs - 1) // bs
        idx = torch.full((nt, topk), -1, dtype=torch.int32, device=DEV)
        for t in range(nt):
            idx[t] = build_idx_row(nt, topk, slk + (t % 3))
        t2r = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=torch.int32,
                           device=DEV)
        tab = make_table(nreq, row_len)
        valid = torch.ones(nt, dtype=torch.int32, device=DEV)
        return idx, t2r, tab, bs, valid

    c.append(("multi_req", multi_req))

    # 10. 1D flat block_table layout (as some metadata builders produce).
    def flat_table():
        nt, topk, bs, slk = 6, 512, 32, 11271
        row_len = (slk + bs - 1) // bs
        idx = torch.full((nt, topk), -1, dtype=torch.int32, device=DEV)
        idx[0] = build_idx_row(nt, topk, slk)
        t2r = torch.zeros(nt, dtype=torch.int32, device=DEV)
        tab = torch.arange(100, 100 + row_len, dtype=torch.int32, device=DEV)
        valid = torch.ones(nt, dtype=torch.int32, device=DEV)
        return idx, t2r, tab, bs, valid

    c.append(("flat_1d_table", flat_table))

    # 11. bool is_valid dtype.
    def bool_valid():
        nt, topk, bs, slk = 6, 512, 32, 11271
        row_len = (slk + bs - 1) // bs
        idx = torch.full((nt, topk), -1, dtype=torch.int32, device=DEV)
        idx[0] = build_idx_row(nt, topk, slk)
        t2r = torch.zeros(nt, dtype=torch.int32, device=DEV)
        tab = make_table(1, row_len)
        valid = torch.tensor([True, False, True, True, False, True],
                             device=DEV)
        return idx, t2r, tab, bs, valid

    c.append(("bool_valid", bool_valid))

    # 12. randomized stress across the parameter space.
    def stress():
        rng = random.Random(1234)
        for it in range(25):
            nt = rng.choice([1, 2, 3, 5, 6, 8, 16, 32, 64])
            topk = rng.choice([128, 256, 512, 1024])
            bs = rng.choice([16, 32, 64, 128])
            slk = rng.choice([256, 1024, 5000, 11271])
            row_len = max(1, (slk + bs - 1) // bs)
            if rng.random() < 0.3:
                row_len = rng.choice([1, 4, row_len // 2, row_len])
            idx = torch.full((nt, topk), -1, dtype=torch.int32, device=DEV)
            for t in range(nt):
                nval = rng.randrange(0, topk + 1)
                inter = rng.choice([None, 1, 3, 7])
                above = None
                if rng.random() < 0.2:
                    above = [slk + rng.randrange(1, 4000)]
                idx[t] = build_idx_row(nt, topk, slk + t, valid_count=nval,
                                       interleave=inter, inject_above=above)
            t2r = torch.randint(0, max(1, nt // 3), (nt,), dtype=torch.int32,
                                device=DEV)
            tab = make_table(max(1, nt // 3), row_len,
                             base=rng.randrange(0, 100000))
            valid = torch.randint(0, 2, (nt,), dtype=torch.int32, device=DEV)
            yield f"stress[{it}]", (idx, t2r, tab, bs, valid)

    c.append(("stress", stress, True))

    return c


def ragged_suite(only):
    results = []
    for name, builder, *_ in cases():
        is_gen = builder.__code__.co_flags & 0x20  # generator flag
        if only and only not in name:
            continue
        if is_gen:
            for sub_name, args in builder():
                if only and only not in f"{name}_{sub_name}":
                    continue
                results.append(run_case(f"{name}/{sub_name}", *args, iters=5))
        else:
            results.append(run_case(name, *builder(), iters=10))
    return results


# --------------------------------------------------------- ds4_moe_hip suite

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _moe_quant(s1, s2, clamp=10.0):
    qc = types.SimpleNamespace()
    qc.gemm1_clamp_limit = clamp
    qc.w1_bias = None
    qc.w2_bias = None
    qc.w1_precision = types.SimpleNamespace(weight_scale=s1)
    qc.w2_precision = types.SimpleNamespace(weight_scale=s2)
    return qc


def _moe_weights(g, E, K1, N1, K2, N2):
    v1 = torch.randint(0, 256, (E, K1 // 2, N1), generator=g).to(torch.uint8).to(DEV)
    v2 = torch.randint(0, 256, (E, K2 // 2, N2), generator=g).to(torch.uint8).to(DEV)
    # e8m0 exponents kept near 1.0 so f32 accumulators cannot overflow
    s1 = torch.randint(118, 137, (E, K1 // 32, N1), generator=g).to(torch.uint8).to(DEV)
    s2 = torch.randint(118, 137, (E, K2 // 32, N2), generator=g).to(torch.uint8).to(DEV)
    return v1, v2, s1, s2


def _moe_inputs(g, M, topk, E, K1, ids_dtype=torch.int32, same_experts=False):
    # distinct experts per token (a token picks an expert at most once -- the
    # entry builder relies on it, exactly like the router)
    if same_experts:
        perm = torch.randperm(E, generator=g)[:topk]
        ids = perm.unsqueeze(0).repeat(M, 1)
    else:
        ids = torch.stack([torch.randperm(E, generator=g)[:topk]
                           for _ in range(M)])
    tw = torch.rand(M, topk, generator=g, dtype=torch.float32)
    tw = tw / tw.sum(-1, keepdim=True)
    x = torch.randn(M, K1, generator=g, dtype=torch.float32).to(torch.bfloat16)
    return ids.to(ids_dtype).to(DEV), tw.to(DEV), x.to(DEV)


def _swap_lib(m, paths):
    m._LIB_PATHS = list(paths)
    m._lib = None
    m.disabled_reason = None
    return m._load()


def moe_run(name, M, topk, E, K1=4096, N1=2048, clamp=10.0, gamma_zero=0.0,
            ids_dtype=torch.int32, same_experts=False, iters=3, ref_so=None,
            ship_paths=None):
    import ds4_moe_hip as m
    K2, N2 = N1 // 2, K1
    print(f"START {name}", flush=True)
    try:
        g = torch.Generator().manual_seed(zlib.crc32(name.encode()) & 0xffff)
        v1, v2, s1, s2 = _moe_weights(g, E, K1, N1, K2, N2)
        qc = _moe_quant(s1, s2, clamp)
        for i in range(iters):
            ids, tw, x = _moe_inputs(g, M, topk, E, K1, ids_dtype)
            if gamma_zero > 0.0:
                tw = tw * (torch.rand(M, topk, generator=g) > gamma_zero).float().to(DEV)
            out = torch.full((M, N2), float("nan"), dtype=torch.bfloat16,
                             device=DEV)
            act = types.SimpleNamespace(name="SILU")
            if not m.eligible(M, ids, x, v1, v2, out, qc, activation=act):
                print(f"FAIL {name}: eligible() refused "
                      f"({m._skip_reason or m.disabled_reason})")
                return False
            # shipped lib (fp16 bit-trick decode)
            m.fused_experts(out, x, v1, v2, tw, ids, qc, swiglu_limit=clamp,
                            tag="harness")
            torch.cuda.synchronize()
            if torch.isnan(out.float()).any().item():
                print(f"FAIL {name}: NaN survived in the output buffer "
                      f"(unwritten element?)")
                return False
            # determinism: same inputs, fresh buffer, bitwise-same result
            out2 = torch.full_like(out, float("nan"))
            m.fused_experts(out2, x, v1, v2, tw, ids, qc, swiglu_limit=clamp,
                            tag="harness")
            torch.cuda.synchronize()
            if not torch.equal(out, out2):
                print(f"FAIL {name}: repeat run is not deterministic")
                return False
            # reference lib (-DFP4_ARITH=1, arithmetic e2m1 decode): must be
            # bit-identical to the fp16 bit-trick build
            if _swap_lib(m, [ref_so]) is None:
                print("FAIL moe/ref_load: reference lib failed to load")
                return False
            out3 = torch.full_like(out, float("nan"))
            try:
                m.fused_experts(out3, x, v1, v2, tw, ids, qc,
                                swiglu_limit=clamp, tag="harness")
                torch.cuda.synchronize()
            finally:
                _swap_lib(m, ship_paths)
            if not torch.equal(out, out3):
                n = int((out != out3).sum())
                print(f"FAIL {name}: {n} element(s) differ between the fp16 "
                      f"bit-trick build and the -DFP4_ARITH=1 reference")
                return False
        print(f"OK   {name} ({iters} iters, bit-exact vs reference)",
              flush=True)
        return True
    except Exception as e:
        print(f"CRASH {name}: {type(e).__name__}: {e}", flush=True)
        return False


def moe_eligibility_gates():
    """eligible() must refuse the shapes the C wrappers would reject, and the
    per-call skips must NOT latch the module off."""
    import ds4_moe_hip as m
    print("START moe/eligible_gates", flush=True)
    try:
        K1, N1 = 4096, 2048
        K2, N2 = N1 // 2, K1
        g = torch.Generator().manual_seed(7)
        E, M, topk = 8, 8, 8
        v1, v2, s1, s2 = _moe_weights(g, E, K1, N1, K2, N2)
        qc = _moe_quant(s1, s2, 10.0)
        ids, tw, x = _moe_inputs(g, M, topk, E, K1)
        out = torch.empty(M, N2, dtype=torch.bfloat16, device=DEV)
        silu = types.SimpleNamespace(name="SILU")

        # P = M*topk = 512 exceeds the 256-slot build kernel: skip, no latch
        big_ids, big_tw, _ = _moe_inputs(g, 8, 64, 64, K1)
        if m.eligible(8, big_ids, x, v1, v2, out, qc, activation=silu):
            print("FAIL eligible_gates: accepted P=512 (> 256 build slots)")
            return False
        if m.disabled_reason is not None:
            print("FAIL eligible_gates: P-overflow latched disabled_reason")
            return False
        # int16 ids would be misread as int32 pairs by the build kernel
        if m.eligible(M, ids.to(torch.int16), x, v1, v2, out, qc,
                      activation=silu):
            print("FAIL eligible_gates: accepted int16 topk_ids")
            return False
        # K=1536 passes K%512==0 but gives wpr=3 waves: the C wrapper would
        # return -2 and fused_experts would raise; eligible must refuse first
        w1b = torch.randint(0, 256, (E, 1536 // 2, N1), generator=g).to(torch.uint8)
        s1b = torch.randint(118, 137, (E, 1536 // 32, N1), generator=g).to(torch.uint8)
        qcb = _moe_quant(s1b, s2, 10.0)
        if m.eligible(M, ids, torch.randn(M, 1536).to(torch.bfloat16).to(DEV),
                      w1b, v2, torch.empty(M, N2, dtype=torch.bfloat16,
                                           device=DEV), qcb, activation=silu):
            print("FAIL eligible_gates: accepted K1=1536 (wpr=3)")
            return False
        # activation mismatch is a per-call skip, not a latch
        if m.eligible(M, ids, x, v1, v2, out, qc,
                      activation=types.SimpleNamespace(name="SWIGLUOAI")):
            print("FAIL eligible_gates: accepted SWIGLUOAI activation")
            return False
        if m.disabled_reason is not None:
            print("FAIL eligible_gates: activation mismatch latched "
                  "disabled_reason")
            return False
        if not m.eligible(M, ids, x, v1, v2, out, qc, activation=silu):
            print(f"FAIL eligible_gates: SILU refused after skips "
                  f"({m._skip_reason or m.disabled_reason})")
            return False
        print("OK   moe/eligible_gates (all gates refuse, no latch)",
              flush=True)
        return True
    except Exception as e:
        print(f"CRASH moe/eligible_gates: {type(e).__name__}: {e}", flush=True)
        return False


def moe_suite():
    results = []
    try:
        import ds4_moe_hip as m
    except Exception as e:
        print(f"SKIP moe suite: ds4_moe_hip unavailable ({e})")
        return results
    ship_paths = list(m._LIB_PATHS)
    if _swap_lib(m, ship_paths) is None:
        print(f"SKIP moe suite: {m.disabled_reason}")
        return results
    src = os.path.join(REPO, "container/native/ds4_moe_mxfp4.cpp")
    if not os.path.exists(src):
        print("SKIP moe suite: native source not in the repo checkout")
        return results
    ref_so = tempfile.mktemp(prefix="ds4_moe_ref_", suffix=".so")
    try:
        subprocess.run(
            ["/opt/rocm/bin/hipcc", "-O3", "--offload-arch=gfx1151",
             "-shared", "-fPIC", "-DFP4_ARITH=1", "-o", ref_so, src],
            check=True)
    except Exception as e:
        print(f"FAIL moe/ref_build: hipcc -DFP4_ARITH=1 failed: {e}")
        _swap_lib(m, ship_paths)
        return [False]
    try:
        results.append(moe_run("moe/basic_M1_topk8", 1, 8, 8,
                               ref_so=ref_so, ship_paths=ship_paths))
        results.append(moe_run("moe/basic_M8_topk8", 8, 8, 8,
                               ref_so=ref_so, ship_paths=ship_paths))
        results.append(moe_run("moe/same_expert_set_M8", 8, 8, 8,
                               same_experts=True,
                               ref_so=ref_so, ship_paths=ship_paths))
        results.append(moe_run("moe/wide_E_distinct", 8, 8, 64,
                               ref_so=ref_so, ship_paths=ship_paths))
        results.append(moe_run("moe/zero_gamma_slots", 8, 8, 8, gamma_zero=0.3,
                               ref_so=ref_so, ship_paths=ship_paths))
        results.append(moe_run("moe/no_clamp", 4, 8, 8, clamp=0.0,
                               ref_so=ref_so, ship_paths=ship_paths))
        results.append(moe_run("moe/ids_int64", 8, 8, 8,
                               ids_dtype=torch.int64,
                               ref_so=ref_so, ship_paths=ship_paths))
        results.append(moe_run("moe/p256_boundary", 8, 32, 64,
                               ref_so=ref_so, ship_paths=ship_paths))
        results.append(moe_eligibility_gates())
    finally:
        _swap_lib(m, ship_paths)
        try:
            os.unlink(ref_so)
        except OSError:
            pass
    return results


def main():
    only = sys.argv[1] if len(sys.argv) > 1 else None
    torch.manual_seed(0)
    results = ragged_suite(only)
    if not only or "moe" in only:
        results.extend(moe_suite())
    n_fail = sum(1 for r in results if not r)
    print(f"\n=== {len(results) - n_fail}/{len(results)} cases OK ===", flush=True)
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
