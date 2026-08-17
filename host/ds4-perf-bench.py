#!/usr/bin/env python3
"""ds4-perf-bench.py — single-stream throughput probe for the DS4 vLLM server.

Measures the same three numbers per context as the USB4-era table:

    context | prefill (t/s) | prose decode (t/s) | code decode (t/s)

Streams one chat-completion per (context, scenario) and derives:
  prefill t/s = usage.prompt_tokens   / TTFT          (time to first token)
  decode  t/s = usage.completion_tokens / (total - TTFT)

Prefill is prompt-tokens-per-second into the first token (so it falls with
context as indexing cost grows); decode is tokens/s across the generation
(so MTP speculative acceptance is included). The context is calibrated to
land near the target with a cheap probe request (reported prompt_tokens is
the measured request's, which is the honest number).

Prose scenario = story context + "continue"; code = a large code file +
"implement process(data)". max_tokens 128, temperature 0.

Run on box1 once the API answers (it is live, ds4-vllm.service):
    python3 ds4-perf-bench.py                          # 512 10k 50k 100k
    python3 ds4-perf-bench.py --contexts 512 10000     # subset
    python3 ds4-perf-bench.py --max-tokens 256 --repeat 3
"""
import argparse
import itertools
import json
import os
import statistics
import sys
import time
import urllib.request

import urllib.error

_MODEL = "deepseek-v4-flash"  # served-model-name
_URL = "http://127.0.0.1:{port}/v1/chat/completions"
_MARKER = itertools.count()  # unique prompt prefix: defeats the prefix/KV cache

# A line whose token cost is roughly the average for its scenario; the probe
# pass refines the count, so these only need to be ballpark.
_PROSE_LINE = ("The quiet harbor at dusk smelled of salt and diesel as the "
               "fishermen hauled in the last nets of the day.")
_CODE_LINE = ("def merge_intervals(intervals):\n"
              "    intervals.sort(key=lambda x: x[0])  # sort by start time\n"
              "    merged = [intervals[0]]\n"
              "    for lo, hi in intervals[1:]:\n"
              "        if lo <= merged[-1][1]:\n"
              "            merged[-1][1] = max(merged[-1][1], hi)\n"
              "        else:\n"
              "            merged.append([lo, hi])\n"
              "    return merged\n")


def log(m):
    print(m, flush=True)


def stream_chat(prompt, max_tokens, timeout_s):
    """POST a streaming chat completion; return (ttft_s, total_s, usage)."""
    body = json.dumps({
        "model": _MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens, "temperature": 0.0, "stream": True,
        "stream_options": {"include_usage": True},
    }).encode()
    req = urllib.request.Request(_URL.format(port=PORT), data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    t_first = None
    usage = {}
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                ev = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if t_first is None and ev.get("choices") and ev["choices"][0].get("delta", {}):
                # Anchor TTFT on the first emitted token of ANY kind: this serve
                # config forces thinking (reasoning_content) at high effort, so
                # content may not appear within max_tokens at all.
                if ev["choices"][0]["delta"].get("content") or ev["choices"][0]["delta"].get("reasoning_content"):
                    t_first = time.perf_counter()
            if ev.get("usage"):
                usage = ev["usage"]
    t_end = time.perf_counter()
    ttft = t_first - t0 if t_first is not None else t_end - t0
    return ttft, t_end - t0, usage


def build_prompt(kind, target_tokens):
    """Calibrate a prompt to ~target_tokens by firing one prefill-only probe."""
    unit = _PROSE_LINE if kind == "prose" else _CODE_LINE
    ask = ("\n\nContinue the story from here, three paragraphs." if kind == "prose"
           else "\n\nNow implement `process(data)` over this module and return it.")
    n = max(1, target_tokens // 18)
    prev_n = -1
    while True:
        # Unique leading marker (fresh tokens every request) so vLLM's prefix
        # cache and the disk-KV tier can never short-circuit the prefill --
        # otherwise the calibration probe warms the exact prompt the measured
        # request then reuses and prefill t/s reads absurdly high.
        marker = f"<unique {next(_MARKER)} {os.urandom(4).hex()}>"
        prompt = marker + "\n" + "\n".join([unit] * n) + ask
        try:
            _, _, usage = stream_chat(prompt, 1, 600)
        except Exception as e:
            sys.exit(f"[bench] calibration probe failed at {target_tokens}: {e}")
        actual = usage.get("prompt_tokens", 0)
        if actual == 0:
            sys.exit("[bench] no prompt_tokens from calibration probe")
        if abs(actual - target_tokens) <= max(32, target_tokens // 20) or n <= 1:
            return prompt
        # Granularity floor: if recomputing n yields the same value, no integer
        # line count lands closer (e.g. the code unit is ~89 tokens, so a 512
        # target is unreachable within tolerance) -- take what we have; the
        # measured request reports its real prompt_tokens anyway.
        if n == prev_n:
            return prompt
        prev_n = n
        n = max(1, round(n * target_tokens / actual))
        if n > 2_000_000:
            sys.exit(f"[bench] calibration diverged (target {target_tokens})")


def run_one(kind, target_ctx, max_tokens):
    prompt = build_prompt(kind, target_ctx)
    ttft, total, usage = stream_chat(prompt, max_tokens, 900)
    pt = usage.get("prompt_tokens", 0)
    ct = usage.get("completion_tokens", 0)
    if pt == 0 or ct == 0:
        sys.exit(f"[bench] empty usage for ctx={target_ctx} {kind}: {usage}")
    prefill_tps = pt / ttft if ttft > 0 else float("nan")
    decode_s = total - ttft
    decode_tps = ct / decode_s if decode_s > 0 else float("nan")
    return pt, prefill_tps, decode_tps


def main():
    global PORT
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--contexts", type=int, nargs="+", default=[512, 10000, 50000, 100000])
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--repeat", type=int, default=1)
    args = ap.parse_args()
    PORT = args.port

    log(f"[bench] model={_MODEL} max_tokens={args.max_tokens} repeat={args.repeat}")
    log(f"{'context':>8} | {'prefill t/s':>11} | {'prose dec t/s':>13} | {'code dec t/s':>12}")
    log("-" * 52)
    for ctx in args.contexts:
        prefill, prose, code = [], [], []
        for _ in range(args.repeat):
            _, p1, d1 = run_one("prose", ctx, args.max_tokens)
            _, p2, d2 = run_one("code", ctx, args.max_tokens)
            prefill.append(p1); prose.append(d1); code.append(d2)
        med = lambda xs: statistics.median(xs) if len(xs) > 1 else xs[0]
        log(f"{ctx:>8} | {med(prefill):>11.1f} | {med(prose):>13.1f} | {med(code):>12.1f}")
    log("-" * 52)
    log("[bench] prefill = prompt_tokens/TTFT-to-first-token; decode = completion_tokens/(total-TTFT); "
        "thinking is ON (high effort), so TTFT anchors on the first reasoning token and decode includes "
        "the reasoning phase.")


if __name__ == "__main__":
    main()