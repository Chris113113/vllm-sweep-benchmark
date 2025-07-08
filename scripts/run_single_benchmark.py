import subprocess
import time
import sys
import threading
import os
import signal
import shlex
import json
import argparse
import re
import io
from datetime import datetime

# --- Configuration ---
VLLM_BENCHMARK_SCRIPT_PATH = "vllm-repo-benchmark/benchmarks/benchmark_vllm_throughput.py"
FATAL_ERROR_STRINGS = ["EngineCore failed to start."]
RETRYABLE_ERROR_STRINGS = ["uncorrectable NVLink error"]
PROCESS_CLEANUP_TIMEOUT = 15

# --- Utility Functions ---
def sanitize_filename(name):
    s = re.sub(r'[\\/*?:"<>|]', "", name)
    s = s.replace(" ", "_")
    return s

def stream_and_capture_output(pipe, prefix, captured_lines_list, fatal_error_event=None, fatal_error_strings=None, retryable_error_event=None, retryable_error_strings=None):
    try:
        with io.TextIOWrapper(pipe, encoding='utf-8', errors='replace') as text_stream:
            on_new_line = True
            for char in iter(lambda: text_stream.read(1), ''):
                if not char:
                    break
                if on_new_line:
                    timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
                    sys.stdout.write(f"[{timestamp}] [{prefix}]: ")
                    on_new_line = False
                sys.stdout.write(char)
                if 'line_buffer' not in locals():
                    line_buffer = ""
                line_buffer += char
                if char == '\n':
                    on_new_line = True
                    captured_lines_list.append(line_buffer)
                    if fatal_error_event and fatal_error_strings and any(e in line_buffer for e in fatal_error_strings):
                        fatal_error_event.set()
                    if retryable_error_event and retryable_error_strings and any(e in line_buffer for e in retryable_error_strings):
                        retryable_error_event.set()
                    line_buffer = ""
                sys.stdout.flush()
            if 'line_buffer' in locals() and line_buffer:
                captured_lines_list.append(line_buffer)
                if fatal_error_event and fatal_error_strings and any(e in line_buffer for e in fatal_error_strings):
                    fatal_error_event.set()
                if retryable_error_event and retryable_error_strings and any(e in line_buffer for e in retryable_error_strings):
                    retryable_error_event.set()
                sys.stdout.write('\n')
                sys.stdout.flush()
    except Exception as e:
        timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
        print(f"[{timestamp}] [STREAM_ERROR]: An error occurred in the stream reader: {e}", flush=True)

def execute_and_monitor_process(command, job_name, log_prefix, logs_dir, log_filename_base, gpu_ids=None, fatal_error_strings=None, retryable_error_strings=None):
    fatal_error_event = threading.Event()
    retryable_error_event = threading.Event()
    env = os.environ.copy()
    if gpu_ids:
        env['CUDA_VISIBLE_DEVICES'] = ",".join(map(str, gpu_ids))
        job_name_with_gpu = f"{job_name}_GPUs{gpu_ids}"
    else:
        job_name_with_gpu = job_name
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, preexec_fn=os.setsid, env=env)
    stdout_lines, stderr_lines = [], []
    stdout_prefix = f"{job_name_with_gpu}:{log_prefix}_STDOUT"
    stderr_prefix = f"{job_name_with_gpu}:{log_prefix}_STDERR"
    stream_args = (process.stdout, stdout_prefix, stdout_lines, fatal_error_event, fatal_error_strings, retryable_error_event, retryable_error_strings)
    stdout_thread = threading.Thread(target=stream_and_capture_output, args=stream_args, daemon=True)
    stderr_thread = threading.Thread(target=stream_and_capture_output, args=(process.stderr, stderr_prefix, stderr_lines, fatal_error_event, fatal_error_strings, retryable_error_event, retryable_error_strings), daemon=True)
    stdout_thread.start()
    stderr_thread.start()
    while process.poll() is None:
        if retryable_error_event.is_set():
            print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] [ORCHESTRATOR]: RETRYABLE ERROR DETECTED on {job_name_with_gpu}. Terminating process group {process.pid} with SIGKILL.")
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            break
        if fatal_error_event.is_set():
            print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] [ORCHESTRATOR]: FATAL ERROR DETECTED on {job_name_with_gpu}. Terminating process group {process.pid} with SIGKILL.")
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            break
        time.sleep(0.1)
    try:
        process.wait(timeout=PROCESS_CLEANUP_TIMEOUT)
    except subprocess.TimeoutExpired:
        print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] [WARNING]: Subprocess {process.pid} ({job_name_with_gpu}) did not terminate gracefully after {PROCESS_CLEANUP_TIMEOUT}s. Forcing final kill.")
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
    stdout_thread.join(timeout=PROCESS_CLEANUP_TIMEOUT)
    stderr_thread.join(timeout=PROCESS_CLEANUP_TIMEOUT)
    log_filepath = os.path.join(logs_dir, log_filename_base)
    with open(log_filepath, 'w') as f:
        f.write("--- STDOUT ---\n")
        f.write("".join(stdout_lines))
        f.write("\n--- STDERR ---\n")
        f.write("".join(stderr_lines))
    print(f"   Full logs for {job_name_with_gpu} saved to '{log_filepath}'")
    status = "success"
    if retryable_error_event.is_set():
        status = "retryable_error"
    elif fatal_error_event.is_set():
        status = "fatal_error"
    return "".join(stdout_lines), status

def run_vllm_throughput_benchmark(run_config, run_index, logs_dir, gpu_ids=None):
    run_name = run_config.get('name', f'vLLM_Official_Run_{run_index}')
    args_list = shlex.split(run_config.get('args', ''))
    print("\n" + "-"*80)
    print(f"🏁 ({run_index}) Running Official vLLM Throughput Benchmark: '{run_name}' on GPUs {gpu_ids}")
    command = ['python', VLLM_BENCHMARK_SCRIPT_PATH] + args_list
    log_filename = f"{run_index}_{sanitize_filename(run_name)}_vllm_script.log"
    full_output, exec_status = execute_and_monitor_process(command, run_name, "VLLM_BENCH", logs_dir, log_filename, gpu_ids=gpu_ids, fatal_error_strings=FATAL_ERROR_STRINGS, retryable_error_strings=RETRYABLE_ERROR_STRINGS)
    if exec_status == "fatal_error":
        return {"run_name": run_name, "status": "TERMINATED_DUE_TO_FATAL_ERROR"}
    if exec_status == "retryable_error":
        return {"run_name": run_name, "status": "RETRYABLE_ERROR"}
    results = {}
    status = "FAILED"
    throughput_match = re.search(r"Throughput: ([\\d.]+) requests/s, ([\\d.]+) total tokens/s, ([\\d.]+) output tokens/s", full_output)
    if throughput_match:
        results["throughput_req_per_sec"] = float(throughput_match.group(1))
        results["throughput_total_tokens_per_sec"] = float(throughput_match.group(2))
        results["throughput_output_tokens_per_sec"] = float(throughput_match.group(3))
        status = "SUCCESS"
    ttft_avg_match = re.search(r"Average TTFT \\(s\\): ([\\d.]+)", full_output)
    if ttft_avg_match:
        results["avg_ttft_s"] = float(ttft_avg_match.group(1))
    ttft_p99_match = re.search(r"P99 TTFT \\(s\\): ([\\d.]+)", full_output)
    if ttft_p99_match:
        results["p99_ttft_s"] = float(ttft_p99_match.group(1))
    tpot_avg_match = re.search(r"Average TPOT \\(tokens/s\\): ([\\d.]+)", full_output)
    if tpot_avg_match:
        results["avg_tpot_tokens_per_s"] = float(tpot_avg_match.group(1))
    tpot_p99_match = re.search(r"P99 TPOT \\(tokens/s\\): ([\\d.]+)", full_output)
    if tpot_p99_match:
        results["p99_tpot_tokens_per_s"] = float(tpot_p99_match.group(1))
    for line in full_output.splitlines():
        if "Total num prompt tokens:" in line: results["total_prompt_tokens"] = int(line.split(":")[1].strip())
        if "Total num output tokens:" in line: results["total_output_tokens"] = int(line.split(":")[1].strip())
    return {"run_name": run_name, "mode": "vllm_throughput", "config": {arg.strip('-'): val for arg, val in zip(args_list[::2], args_list[1::2])}, "results": results, "client_log_file": log_filename, "status": status}

def main():
    parser = argparse.ArgumentParser(description="VLLM Single Benchmark Runner")
    parser.add_argument("--run-name", required=True, help="A descriptive name for the benchmark run.")
    parser.add_argument("--output-dir", default=f"benchmark_results_{int(time.time())}", help="Directory to save the final results and logs.")
    parser.add_argument("--gpu-ids", help="Comma-separated list of GPU IDs to use.")
    parser.add_argument('benchmark_args', nargs=argparse.REMAINDER, help="Arguments to pass to the vLLM benchmark script.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    logs_dir = os.path.join(args.output_dir, f"client_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(logs_dir, exist_ok=True)
    print(f"Results will be saved to directory: '{args.output_dir}'")
    print(f"Logs will be saved to directory: '{logs_dir}'")

    gpu_ids = [int(gid.strip()) for gid in args.gpu_ids.split(',')] if args.gpu_ids else None

    run_config = {
        'name': args.run_name,
        'args': ' '.join(args.benchmark_args)
    }

    result = run_vllm_throughput_benchmark(run_config, 1, logs_dir, gpu_ids=gpu_ids)

    report_filepath = os.path.join(args.output_dir, "summary_report.json")
    with open(report_filepath, 'w') as f:
        json.dump([result], f, indent=4)

    print("\n" + "-"*80)
    print(f"✅ Benchmark run complete. Final report is at '{report_filepath}'")
    print("#"*80)

if __name__ == "__main__":
    main()
