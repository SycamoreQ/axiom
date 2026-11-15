"""
Benchmark runner for MLX backend (Llama 3.2 1B Instruct 4-bit)
Exports results matching Axiom's InferenceProfile JSON schema for comparative evaluation.
"""

import sys
import time
import json
import resource
import os

try:
    import mlx.core as mx
    import mlx_lm
except ImportError:
    print("Error: mlx or mlx_lm not found.")
    sys.exit(1)


def get_current_rss_bytes():
    # macOS ru_maxrss is in bytes
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_maxrss


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


def main():
    model_id = "mlx-community/Llama-3.2-1B-Instruct-4bit"
    prompt = "The future of artificial intelligence is"
    max_new_tokens = 32
    temperature = 0.0
    out_json = "mlx_llama32_1b_profile.json"

    print("============================================================")
    print("                 MLX INFERENCE BENCHMARK                    ")
    print("============================================================")
    print(f"Model:       {model_id}")
    print(f"Backend:     Apple MLX (Metal)")
    print(f"Prompt:      \"{prompt}\"")
    print(f"Max tokens:  {max_new_tokens}")
    print(f"Temperature: {temperature}")
    print("------------------------------------------------------------")

    load_start = time.perf_counter()
    model, tokenizer = mlx_lm.load(model_id)
    print(f"Model loaded in {time.perf_counter() - load_start:.2f}s")

    # Format prompt for Llama 3.2 Instruct
    formatted_prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

    # Measure tokenization
    tok_start = time.perf_counter()
    prompt_tokens = tokenizer.encode(formatted_prompt)
    tok_time_ms = (time.perf_counter() - tok_start) * 1000.0
    num_prompt_tokens = len(prompt_tokens)

    # Resource snapshot before inference
    start_rss = get_current_rss_bytes()
    u_start, s_start = get_cpu_times()
    start_time = time.perf_counter()

    token_timings = []
    prev_time = start_time
    output_text = []

    print("\nOutput Generation:")
    print("  ", end="", flush=True)

    step = 0
    # Stream generation to capture each token's latency
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
        print(resp.text, end="", flush=True)
        step += 1

    end_time = time.perf_counter()
    u_end, s_end = get_cpu_times()
    end_rss = get_current_rss_bytes()
    print("\n")

    # Compute milestones
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

    peak_rss_mb = end_rss / (1024.0 * 1024.0)
    delta_rss_mb = (end_rss - start_rss) / (1024.0 * 1024.0)

    profile = {
        "model_path": model_id,
        "device": "Apple Silicon (MLX)",
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
            "current_rss_mb": peak_rss_mb,
            "user_cpu_ms": user_cpu_ms,
            "system_cpu_ms": sys_cpu_ms,
            "total_cpu_ms": total_cpu_ms,
            "cpu_utilization_pct": cpu_util,
        },
        "memory": {
            "host_initial_rss_mb": start_rss / (1024.0 * 1024.0),
            "host_peak_rss_mb": peak_rss_mb,
            "host_delta_rss_mb": delta_rss_mb,
            "metal_pool_capacity_mb": None,
            "metal_pool_peak_mb": None,
            "metal_pool_active_mb": None,
        },
        "token_timings": token_timings,
    }

    with open(out_json, "w") as f:
        json.dump(profile, f, indent=2)

    print(f"Profile results exported to: {out_json}")
    print(f"TTFT:        {ttft_ms:.2f} ms ({prefill_tps:.1f} tok/s)")
    print(f"Decode:      {decode_tps:.1f} tok/s (Mean ITL: {itl_stats['mean_ms']:.2f} ms, P50: {itl_stats['median_ms']:.2f} ms)")
    print(f"Host Peak:   {peak_rss_mb:.2f} MB")
    print(f"CPU Util:    {cpu_util:.1f} %")


if __name__ == "__main__":
    main()

