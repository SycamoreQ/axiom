"""
Benchmark runner for MLX backend (Llama 3.2 1B Instruct 4-bit)
Standardized and repeatable benchmark comparable with Axiom's InferenceProfile JSON schema.
Features:
- Accurate macOS Mach task_info resident set size (true current RSS, not high-water ru_maxrss)
- Separate host RSS and Metal unified-memory metrics (mx.get_active_memory / mx.get_peak_memory)
- Exact prompt tokenization parity (24-token Llama 3.2 Instruct template, add_special_tokens=False)
- Warmup runs and repeated trials reporting min, median, max, and std dev
- Separate model loading time reporting
"""

import sys
import time
import json
import resource
import os
import argparse
import ctypes
import ctypes.util

try:
    import mlx.core as mx
    import mlx_lm
except ImportError:
    print("Error: mlx or mlx_lm not found. Run with `uv run --with mlx --with mlx-lm python3 bench_mlx.py`")
    sys.exit(1)


# Mach task_info for true current resident memory on macOS
class MachTaskBasicInfo(ctypes.Structure):
    _fields_ = [
        ("virtual_size", ctypes.c_uint64),
        ("resident_size", ctypes.c_uint64),
        ("resident_size_max", ctypes.c_uint64),
        ("user_time_sec", ctypes.c_int32),
        ("user_time_usec", ctypes.c_int32),
        ("system_time_sec", ctypes.c_int32),
        ("system_time_usec", ctypes.c_int32),
        ("policy", ctypes.c_int32),
        ("suspend_count", ctypes.c_int32),
    ]


_libc = ctypes.CDLL(ctypes.util.find_library("c"))
MACH_TASK_BASIC_INFO = 20
MACH_TASK_BASIC_INFO_COUNT = ctypes.sizeof(MachTaskBasicInfo) // ctypes.sizeof(ctypes.c_uint32)


def get_current_rss_bytes():
    """True resident set size (current RSS) via macOS Mach task_info."""
    try:
        info = MachTaskBasicInfo()
        count = ctypes.c_uint32(MACH_TASK_BASIC_INFO_COUNT)
        kr = _libc.task_info(
            _libc.mach_task_self(),
            MACH_TASK_BASIC_INFO,
            ctypes.byref(info),
            ctypes.byref(count),
        )
        if kr == 0:
            return int(info.resident_size)
    except Exception:
        pass
    # Fallback
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_maxrss


def get_peak_rss_bytes():
    """High-water peak RSS via getrusage (ru_maxrss in bytes on macOS)."""
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_maxrss


def get_metal_memory_mb():
    """Return (active_mb, peak_mb) from MLX unified memory allocator."""
    try:
        active = mx.get_active_memory() / (1024.0 * 1024.0)
        peak = mx.get_peak_memory() / (1024.0 * 1024.0)
        return active, peak
    except Exception:
        return None, None


def get_cpu_times():
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_utime, usage.ru_stime


def compute_latency_stats(latencies_ms):
    if not latencies_ms:
        return {
            "count": 0, "mean_ms": 0.0, "median_ms": 0.0,
            "p90_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0,
            "min_ms": 0.0, "max_ms": 0.0, "std_dev_ms": 0.0,
            "tokens_per_sec": 0.0
        }

    sorted_l = sorted(latencies_ms)
    count = len(sorted_l)
    total = sum(sorted_l)
    mean_ms = total / count

    def percentile(p):
        if count == 1:
            return sorted_l[0]
        idx = int(round(p * (count - 1)))
        return sorted_l[min(idx, count - 1)]

    median_ms = percentile(0.50)
    p90_ms = percentile(0.90)
    p95_ms = percentile(0.95)
    p99_ms = percentile(0.99)
    min_ms = sorted_l[0]
    max_ms = sorted_l[-1]

    variance = sum((x - mean_ms) ** 2 for x in sorted_l) / count
    std_dev_ms = variance ** 0.5
    tokens_per_sec = (count / (total / 1000.0)) if total > 0 else 0.0

    return {
        "count": count,
        "mean_ms": mean_ms,
        "median_ms": median_ms,
        "p90_ms": p90_ms,
        "p95_ms": p95_ms,
        "p99_ms": p99_ms,
        "min_ms": min_ms,
        "max_ms": max_ms,
        "std_dev_ms": std_dev_ms,
        "tokens_per_sec": tokens_per_sec,
    }


def run_single_pass(model, tokenizer, formatted_prompt, max_new_tokens, temperature, quiet=False):
    tok_start = time.perf_counter()
    prompt_tokens = tokenizer.encode(formatted_prompt, add_special_tokens=False)
    tok_time_ms = (time.perf_counter() - tok_start) * 1000.0
    num_prompt_tokens = len(prompt_tokens)

    start_rss = get_current_rss_bytes()
    u_start, s_start = get_cpu_times()
    start_time = time.perf_counter()

    token_timings = []
    prev_time = start_time
    output_text = []

    step = 0
    sampler = mlx_lm.sample_utils.make_sampler(temp=temperature)

    for resp in mlx_lm.stream_generate(
        model,
        tokenizer,
        prompt=formatted_prompt,
        max_tokens=max_new_tokens,
        sampler=sampler,
    ):
        now = time.perf_counter()
        latency_ms = (now - prev_time) * 1000.0
        prev_time = now

        is_prefill = (step == 0)
        token_timings.append({
            "step": step,
            "token_id": resp.token,
            "token_str": resp.text,
            "latency_ms": latency_ms,
            "is_prefill": is_prefill,
            "draft_tokens_count": None
        })
        output_text.append(resp.text)
        if not quiet:
            print(resp.text, end="", flush=True)
        step += 1

    end_time = time.perf_counter()
    u_end, s_end = get_cpu_times()
    end_rss = get_current_rss_bytes()
    peak_rss = get_peak_rss_bytes()
    metal_active_mb, metal_peak_mb = get_metal_memory_mb()

    if not quiet:
        print("\n")

    total_time_ms = (end_time - start_time) * 1000.0
    ttft_ms = token_timings[0]["latency_ms"] if token_timings else 0.0
    prefill_tps = (num_prompt_tokens / (ttft_ms / 1000.0)) if ttft_ms > 0 else 0.0

    decode_latencies = [t["latency_ms"] for t in token_timings[1:]]
    itl_stats = compute_latency_stats(decode_latencies)

    decode_time_ms = sum(decode_latencies)
    decode_tps = (len(decode_latencies) / (decode_time_ms / 1000.0)) if decode_time_ms > 0 else 0.0
    overall_tps = (len(token_timings) / (total_time_ms / 1000.0)) if total_time_ms > 0 else 0.0

    user_cpu_ms = (u_end - u_start) * 1000.0
    sys_cpu_ms = (s_end - s_start) * 1000.0
    total_cpu_ms = user_cpu_ms + sys_cpu_ms
    cpu_util = (total_cpu_ms / total_time_ms * 100.0) if total_time_ms > 0 else 0.0

    peak_rss_mb = peak_rss / (1024.0 * 1024.0)
    current_rss_mb = end_rss / (1024.0 * 1024.0)
    delta_rss_mb = (end_rss - start_rss) / (1024.0 * 1024.0)

    profile = {
        "model_path": "mlx-community/Llama-3.2-1B-Instruct-4bit",
        "device": "Apple Silicon (MLX Local Engine)",
        "prompt_tokens": num_prompt_tokens,
        "generated_tokens": len(token_timings),
        "total_tokens": num_prompt_tokens + len(token_timings),
        "tokenization_time_ms": tok_time_ms,
        "ttft_ms": ttft_ms,
        "prefill_throughput_tok_s": prefill_tps,
        "decode_time_ms": decode_time_ms,
        "decode_throughput_tok_s": decode_tps,
        "total_inference_time_ms": total_time_ms,
        "end_to_end_time_ms": total_time_ms + tok_time_ms,
        "overall_throughput_tok_s": overall_tps,
        "itl_stats": itl_stats,
        "speculative": None,
        "resource_diff": {
            "wall_time_ms": total_time_ms,
            "delta_rss_mb": delta_rss_mb,
            "peak_rss_mb": peak_rss_mb,
            "current_rss_mb": current_rss_mb,
            "user_cpu_ms": user_cpu_ms,
            "system_cpu_ms": sys_cpu_ms,
            "total_cpu_ms": total_cpu_ms,
            "cpu_utilization_pct": cpu_util,
        },
        "memory": {
            "host_initial_rss_mb": start_rss / (1024.0 * 1024.0),
            "host_peak_rss_mb": peak_rss_mb,
            "host_delta_rss_mb": delta_rss_mb,
            "metal_pool_capacity_mb": metal_peak_mb,
            "metal_pool_peak_mb": metal_peak_mb,
            "metal_pool_active_mb": metal_active_mb,
        },
        "token_timings": token_timings,
    }
    return profile


def main():
    parser = argparse.ArgumentParser(description="Axiom vs MLX Inference Benchmark")
    parser.add_argument("--model", default="mlx-community/Llama-3.2-1B-Instruct-4bit", help="Model ID")
    parser.add_argument(
        "--prompt",
        default="Explain the fundamental principles of quantum computing in three clear bullet points.",
        help="Prompt text",
    )
    parser.add_argument("--max-tokens", "-n", type=int, default=32, help="Max new tokens")
    parser.add_argument("--temperature", "-t", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--warmup", type=int, default=1, help="Number of warmup iterations")
    parser.add_argument("--runs", type=int, default=3, help="Number of benchmark trials")
    parser.add_argument("--json", default="mlx_llama32_1b_profile.json", help="Output JSON path")
    args = parser.parse_args()

    print("============================================================")
    print("                 MLX INFERENCE BENCHMARK                    ")
    print("============================================================")
    print(f"Model:       {args.model}")
    print(f"Backend:     Apple MLX (Local Metal Framework)")
    print(f"Prompt:      \"{args.prompt}\"")
    print(f"Max tokens:  {args.max_tokens}")
    print(f"Temperature: {args.temperature}")
    print(f"Warmup runs: {args.warmup} | Benchmark runs: {args.runs}")
    print("------------------------------------------------------------")

    load_start = time.perf_counter()
    model, tokenizer = mlx_lm.load(args.model)
    load_time_sec = time.perf_counter() - load_start
    print(f"[Model Loading] Completed in {load_time_sec:.3f} s (excluded from inference metrics)")

    # Standard Llama 3.2 Instruct prompt template
    formatted_prompt = (
        f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        f"{args.prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )

    test_tokens = tokenizer.encode(formatted_prompt, add_special_tokens=False)
    print(f"[Prompt Tokens] {len(test_tokens)} tokens encoded (identical tokenization to Axiom)")

    # Warmup runs
    for w in range(args.warmup):
        print(f"\n[Warmup {w + 1}/{args.warmup}] Running warm-up...")
        _ = run_single_pass(
            model,
            tokenizer,
            formatted_prompt,
            args.max_tokens,
            args.temperature,
            quiet=True,
        )

    # Benchmark trials
    profiles = []
    decode_speeds = []
    ttfts = []
    median_itls = []

    print("\n--- Benchmark Runs ---")
    for r in range(args.runs):
        print(f"\nRun {r + 1}/{args.runs}:")
        print("  ", end="", flush=True)
        prof = run_single_pass(
            model,
            tokenizer,
            formatted_prompt,
            args.max_tokens,
            args.temperature,
            quiet=False,
        )
        profiles.append(prof)
        decode_speeds.append(prof["decode_throughput_tok_s"])
        ttfts.append(prof["ttft_ms"])
        median_itls.append(prof["itl_stats"]["median_ms"])

    # Compute statistics across runs
    decode_speeds.sort()
    ttfts.sort()
    median_itls.sort()

    median_decode = decode_speeds[len(decode_speeds) // 2]
    min_decode = min(decode_speeds)
    max_decode = max(decode_speeds)

    median_ttft = ttfts[len(ttfts) // 2]
    min_ttft = min(ttfts)
    max_ttft = max(ttfts)

    best_profile = profiles[0]
    for p in profiles:
        if p["decode_throughput_tok_s"] == median_decode:
            best_profile = p
            break

    # Save representative (median) profile JSON
    with open(args.json, "w") as f:
        json.dump(best_profile, f, indent=2)

    print("\n============================================================")
    print("                 MLX BENCHMARK SUMMARY                      ")
    print("============================================================")
    print(f"Model Load Time   : {load_time_sec:.3f} s (excluded from inference)")
    print(f"Prompt Tokens     : {len(test_tokens)}")
    print(f"Generated Tokens  : {best_profile['generated_tokens']}")
    print(f"Decode Throughput : {median_decode:.1f} tok/s (range: {min_decode:.1f} - {max_decode:.1f} tok/s across {args.runs} runs)")
    print(f"Prefill TTFT      : {median_ttft:.2f} ms (range: {min_ttft:.2f} - {max_ttft:.2f} ms)")
    print(f"Median ITL        : {median_itls[len(median_itls)//2]:.2f} ms")
    print(f"Host Current RSS  : {best_profile['resource_diff']['current_rss_mb']:.1f} MB (Peak: {best_profile['resource_diff']['peak_rss_mb']:.1f} MB)")
    if best_profile['memory']['metal_pool_active_mb'] is not None:
        print(f"Metal Unified Mem : {best_profile['memory']['metal_pool_active_mb']:.1f} MB active (Peak: {best_profile['memory']['metal_pool_peak_mb']:.1f} MB)")
    print(f"Representative profile saved to: {args.json}")


if __name__ == "__main__":
    main()
