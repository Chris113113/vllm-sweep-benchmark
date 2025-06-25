import asyncio
import time
import aiohttp
import argparse
import json
import sys
import numpy as np

# --- Configuration ---
VLLM_HOST = "http://localhost:8000"
LOG_INTERVAL = 100

# ==============================================================================
# PHASE 1: LATENCY-FOCUSED FUNCTIONS (STREAMING, SERIAL)
# ==============================================================================

async def measure_single_prompt_latency(session, prompt_data, timeout_config):
    """
    Sends a single streaming request and measures detailed latency metrics.
    Returns a dictionary of metrics in milliseconds.
    """
    url = f"{VLLM_HOST}/v1/completions"
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": prompt_data["model"],
        "prompt": prompt_data["prompt"],
        "max_tokens": prompt_data["max_output_tokens"],
        "temperature": 0.7,
        "stream": True,
    }

    metrics = {
        "ttft_ms": -1,
        "tbst_ms": -1,
        "mean_itl_ms": -1,
        "output_tokens": 0,
        "error": None
    }

    request_start_time = time.monotonic()
    first_token_time = None
    second_token_time = None
    last_token_time = None
    token_count = 0

    try:
        async with session.post(url, headers=headers, json=payload, timeout=timeout_config) as response:
            if response.status != 200:
                metrics["error"] = f"HTTP Error {response.status}: {await response.text()}"
                return metrics

            async for line in response.content:
                line = line.strip()
                if not line or not line.startswith(b'data:'):
                    continue
                
                if line.startswith(b'data: [DONE]'):
                    break
                
                try:
                    data = json.loads(line[len(b'data:'):].strip())
                    if not first_token_time:
                        first_token_time = time.monotonic()
                        metrics["ttft_ms"] = (first_token_time - request_start_time) * 1000
                    elif not second_token_time:
                         second_token_time = time.monotonic()
                         metrics["tbst_ms"] = (second_token_time - first_token_time) * 1000
                    token_count += 1
                    last_token_time = time.monotonic()
                except (json.JSONDecodeError, KeyError):
                    continue

    except asyncio.TimeoutError:
        metrics["error"] = f"Request timed out after {timeout_config.total} seconds"
        return metrics
    except aiohttp.ClientConnectorError as e:
        metrics["error"] = f"Connection Error: {e}"
        return metrics

    metrics["output_tokens"] = token_count
    if token_count > 2 and last_token_time and second_token_time:
        duration_ms = (last_token_time - second_token_time) * 1000
        metrics["mean_itl_ms"] = duration_ms / (token_count - 2) if token_count > 2 else 0

    return metrics

# ==============================================================================
# PHASE 2: THROUGHPUT-FOCUSED FUNCTIONS (BATCH, CONCURRENT)
# ==============================================================================

async def send_throughput_request(session, prompt_data, timeout_config):
    """Sends a single non-streaming request for the throughput test."""
    url = f"{VLLM_HOST}/v1/completions"
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": prompt_data["model"],
        "prompt": prompt_data["prompt"],
        "max_tokens": prompt_data["max_output_tokens"],
    }
    try:
        async with session.post(url, headers=headers, json=payload, timeout=timeout_config) as response:
            if response.status == 200:
                resp_json = await response.json()
                return resp_json["usage"]["completion_tokens"], None
            else:
                return 0, (response.status, await response.text())
    except asyncio.TimeoutError:
        return 0, (408, f"Request timed out after {timeout_config.total} seconds")
    except aiohttp.ClientConnectorError as e:
        return 0, (999, f"Connection Error: {e}")

async def throughput_worker(queue, session, results, progress_counter, progress_lock, timeout_config):
    """Worker for the high-concurrency throughput phase."""
    while True:
        try:
            prompt_data = await queue.get()
            tokens, error = await send_throughput_request(session, prompt_data, timeout_config)
            results.append((tokens, error))
            
            async with progress_lock:
                progress_counter['completed'] += 1
                if progress_counter['completed'] % LOG_INTERVAL == 0:
                    print(f"Client (Throughput): Completed {progress_counter['completed']}/{progress_counter['total']} requests...", file=sys.stderr)
            queue.task_done()
        except asyncio.CancelledError:
            break

# ==============================================================================
# MAIN ORCHESTRATOR
# ==============================================================================

async def main(args):
    # --- Generate Synthetic Prompts ---
    base_prompt = "This is a synthetic prompt for benchmarking. "
    synthetic_prompt = (base_prompt * (args.max_input_length // len(base_prompt) + 1))[:args.max_input_length]
    
    timeout = aiohttp.ClientTimeout(total=args.request_timeout)
    
    # --- PHASE 1: LATENCY BENCHMARK ---
    print("\n" + "="*80, file=sys.stderr)
    print(f"PHASE 1: Running Latency Probe ({args.latency_test_prompts} serial requests)...", file=sys.stderr)
    print("="*80, file=sys.stderr)
    
    latency_results = []
    async with aiohttp.ClientSession() as session:
        for i in range(args.latency_test_prompts):
            prompt_data = {"model": args.model, "prompt": synthetic_prompt, "max_output_tokens": args.max_output_length}
            result = await measure_single_prompt_latency(session, prompt_data, timeout)
            latency_results.append(result)
            print(f"  Latency probe {i+1}/{args.latency_test_prompts} complete.", file=sys.stderr, end="\r")

    successful_latency_runs = [r for r in latency_results if r["error"] is None and r["ttft_ms"] > -1]
    
    # --- PHASE 2: THROUGHPUT BENCHMARK ---
    print("\n" + "="*80, file=sys.stderr)
    print(f"PHASE 2: Running Throughput Test ({args.throughput_test_prompts} requests at concurrency {args.concurrency})...", file=sys.stderr)
    print("="*80, file=sys.stderr)
    
    throughput_queue = asyncio.Queue()
    for _ in range(args.throughput_test_prompts):
        await throughput_queue.put({"model": args.model, "prompt": synthetic_prompt, "max_output_tokens": args.max_output_length})

    throughput_results_list = []
    progress_lock = asyncio.Lock()
    progress_counter = {"completed": 0, "total": args.throughput_test_prompts}
    
    throughput_start_time = time.monotonic()
    async with aiohttp.ClientSession() as session:
        worker_tasks = [
            asyncio.create_task(throughput_worker(throughput_queue, session, throughput_results_list, progress_counter, progress_lock, timeout))
            for _ in range(args.concurrency)
        ]
        await throughput_queue.join()
        for task in worker_tasks:
            task.cancel()
        await asyncio.gather(*worker_tasks, return_exceptions=True)
    throughput_duration = time.monotonic() - throughput_start_time

    # --- AGGREGATE AND OUTPUT FINAL JSON ---
    
    # Process latency results into a single summary dictionary using percentiles
    latency_summary = {}
    if successful_latency_runs:
        ttfts = [r['ttft_ms'] for r in successful_latency_runs]
        tbsts = [r['tbst_ms'] for r in successful_latency_runs if r['tbst_ms'] > -1]
        itls = [r['mean_itl_ms'] for r in successful_latency_runs if r['mean_itl_ms'] > -1]

        latency_summary = {
            "p1_ttft_ms": round(np.percentile(ttfts, 1), 2) if ttfts else 0,
            "p50_ttft_ms": round(np.percentile(ttfts, 50), 2) if ttfts else 0,
            "p95_ttft_ms": round(np.percentile(ttfts, 95), 2) if ttfts else 0,
            "p99_ttft_ms": round(np.percentile(ttfts, 99), 2) if ttfts else 0,
            
            "p1_tbst_ms": round(np.percentile(tbsts, 1), 2) if tbsts else 0,
            "p50_tbst_ms": round(np.percentile(tbsts, 50), 2) if tbsts else 0,
            "p95_tbst_ms": round(np.percentile(tbsts, 95), 2) if tbsts else 0,
            "p99_tbst_ms": round(np.percentile(tbsts, 99), 2) if tbsts else 0,

            "p1_mean_itl_ms": round(np.percentile(itls, 1), 2) if itls else 0,
            "p50_mean_itl_ms": round(np.percentile(itls, 50), 2) if itls else 0,
            "p95_mean_itl_ms": round(np.percentile(itls, 95), 2) if itls else 0,
            "p99_mean_itl_ms": round(np.percentile(itls, 99), 2) if itls else 0,
        }

    # Process throughput results
    total_throughput_tokens = sum(r[0] for r in throughput_results_list)
    throughput_errors = [r[1] for r in throughput_results_list if r[1] is not None]
    successful_throughput_reqs = args.throughput_test_prompts - len(throughput_errors)
    
    throughput_summary = {}
    if successful_throughput_reqs > 0 and throughput_duration > 0:
        throughput_summary = {
            "total_duration_sec": round(throughput_duration, 2),
            "successful_requests": successful_throughput_reqs,
            "failed_requests": len(throughput_errors),
            "total_tokens_generated": total_throughput_tokens,
            "throughput_req_per_sec": round(successful_throughput_reqs / throughput_duration, 2),
            "throughput_tokens_per_sec": round(total_throughput_tokens / throughput_duration, 2),
            "errors": [f"Status: {e[0]}, Body: {e[1][:100]}" for e in throughput_errors[:5]]
        }

    # --- Final Human-Readable Summary to stderr ---
    print("\n\n" + "="*80, file=sys.stderr)
    print("Benchmark Complete: Final Summary", file=sys.stderr)
    print("="*80, file=sys.stderr)
    if latency_summary:
        print("--- Latency (ms) ---", file=sys.stderr)
        print(f"  Time to First Token (TTFT): p50: {latency_summary.get('p50_ttft_ms', 'N/A'):.2f}, p95: {latency_summary.get('p95_ttft_ms', 'N/A'):.2f}, p99: {latency_summary.get('p99_ttft_ms', 'N/A'):.2f}", file=sys.stderr)
        print(f"  Inter-Token Latency (ITL):  p50: {latency_summary.get('p50_mean_itl_ms', 'N/A'):.2f}, p95: {latency_summary.get('p95_mean_itl_ms', 'N/A'):.2f}, p99: {latency_summary.get('p99_mean_itl_ms', 'N/A'):.2f}", file=sys.stderr)
    if throughput_summary:
        print("\n--- Throughput ---", file=sys.stderr)
        print(f"  Requests per second: {throughput_summary.get('throughput_req_per_sec', 'N/A'):.2f}", file=sys.stderr)
        print(f"  Tokens per second:   {throughput_summary.get('throughput_tokens_per_sec', 'N/A'):.2f}", file=sys.stderr)
    print("="*80, file=sys.stderr)

    # Combine into a single final JSON object
    output_data = {
        "config": vars(args),
        "results": {
            **latency_summary,
            **throughput_summary
        }
    }
    
    # Print final JSON to stdout
    print(json.dumps(output_data, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Comprehensive Latency and Throughput Benchmark Client for vLLM.")
    parser.add_argument("--model", type=str, required=True, help="The name of the model to target on the server.")
    parser.add_argument("--latency-test-prompts", type=int, default=20, help="Number of serial prompts for latency test.")
    parser.add_argument("--throughput-test-prompts", type=int, default=500, help="Total number of prompts for throughput test.")
    parser.add_argument("--concurrency", type=int, default=100, help="Number of constant concurrent requests for throughput test.")
    parser.add_argument("--max-input-length", type=int, default=64, help="Characters for the synthetic input prompt.")
    parser.add_argument("--max-output-length", type=int, default=512, help="Max tokens to generate per request.")
    parser.add_argument("--request-timeout", type=int, default=180, help="Timeout in seconds for each individual web request.")
    args = parser.parse_args()
    
    asyncio.run(main(args))
