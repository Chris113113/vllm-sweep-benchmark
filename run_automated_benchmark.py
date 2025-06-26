import subprocess
import time
import sys
import threading
import requests
import os
import signal
import yaml
import shlex
import json
import argparse
import re
import io
from datetime import datetime

# --- Configuration ---
SERVER_STARTUP_TIMEOUT = 300
SERVER_HEALTH_CHECK_URL = "http://localhost:8000/v1/models"
VLLM_BENCHMARK_SCRIPT_PATH = "vllm-repo-benchmark/benchmarks/benchmark_vllm_throughput.py"
FATAL_ERROR_STRINGS = ["EngineCore failed to start."]
RETRYABLE_ERROR_STRINGS = ["uncorrectable NVLink error"]
# Increase the cleanup timeout to give the OS more time to reap zombie processes
PROCESS_CLEANUP_TIMEOUT = 15

# --- Utility Functions ---
def sanitize_filename(name):
    s = re.sub(r'[\\/*?:"<>|]', "", name)
    s = s.replace(" ", "_")
    return s

def is_server_ready():
    try:
        response = requests.get(SERVER_HEALTH_CHECK_URL, timeout=1)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

def wait_for_server(timeout):
    start_time = time.monotonic()
    print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] Waiting for server to become ready...", end="")
    while time.monotonic() - start_time < timeout:
        if is_server_ready():
            print(f"\n[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] Server is ready.")
            return True
        print(".", end="", flush=True)
        time.sleep(1)
    print(f"\n[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] Error: Server did not become ready in time.")
    return False

def stream_and_capture_output(pipe, prefix, captured_lines_list, fatal_error_event=None, fatal_error_strings=None, retryable_error_event=None, retryable_error_strings=None):
    """
    Reads a subprocess's output stream, buffering until a newline or carriage return.
    This allows progress bars (which use \r) to update in place, while still capturing
    and prefixing full lines of output for logging.
    """
    try:
        # Use a wrapper to treat the binary pipe as a text stream.
        with io.TextIOWrapper(pipe, encoding='utf-8', errors='replace') as text_stream:
            buffer = ""
            for char in iter(lambda: text_stream.read(1), ''):
                if not char:
                    break  # End of stream
                
                buffer += char
                
                # Process the buffer when a line-ending character is found.
                # This handles both normal log lines (\n) and updating lines like progress bars (\r).
                if char == '\n' or char == '\r':
                    timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
                    
                    # Write the prefix and the buffered content. The buffer already contains the
                    # correct line ending (\n or \r), so we write it directly.
                    sys.stdout.write(f"[{timestamp}] [{prefix}]: {buffer}")
                    sys.stdout.flush()

                    captured_lines_list.append(buffer)
                    
                    # Check for error strings in the completed line.
                    if fatal_error_event and fatal_error_strings:
                        if any(e in buffer for e in fatal_error_strings):
                            fatal_error_event.set()
                    if retryable_error_event and retryable_error_strings:
                        if any(e in buffer for e in retryable_error_strings):
                            retryable_error_event.set()
                    
                    # Reset buffer after processing the line.
                    buffer = ""

            # Handle any remaining text in the buffer that doesn't end with a newline.
            if buffer:
                timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
                sys.stdout.write(f"[{timestamp}] [{prefix}]: {buffer}\n") # Add a final newline for clarity
                sys.stdout.flush()
                captured_lines_list.append(buffer)

    except Exception as e:
        timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
        print(f"[{timestamp}] [STREAM_ERROR]: An error occurred in the stream reader: {e}", flush=True)

def execute_and_monitor_process(command, job_name, log_prefix, output_dir, log_filename_base, fatal_error_strings=None, retryable_error_strings=None):
    """
    A helper function to execute a subprocess, stream its logs,
    monitor for errors, and terminate early if needed.
    Returns the full output and a status indicating success, fatal error, or retryable error.
    """
    fatal_error_event = threading.Event()
    retryable_error_event = threading.Event()
    
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, preexec_fn=os.setsid)
    
    stdout_lines, stderr_lines = [], []
    
    # Combine job name and log prefix for a more descriptive log output
    stdout_prefix = f"{job_name}:{log_prefix}_STDOUT"
    stderr_prefix = f"{job_name}:{log_prefix}_STDERR"
    
    stream_args = (
        process.stdout, stdout_prefix, stdout_lines,
        fatal_error_event, fatal_error_strings,
        retryable_error_event, retryable_error_strings
    )
    
    stdout_thread = threading.Thread(target=stream_and_capture_output, args=stream_args, daemon=True)
    stderr_thread = threading.Thread(target=stream_and_capture_output, args=(process.stderr, stderr_prefix, stderr_lines, fatal_error_event, fatal_error_strings, retryable_error_event, retryable_error_strings), daemon=True)
    
    stdout_thread.start()
    stderr_thread.start()

    while process.poll() is None:
        if retryable_error_event.is_set():
            print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] [ORCHESTRATOR]: RETRYABLE ERROR DETECTED. Terminating process group {process.pid} with SIGKILL.")
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            break
        if fatal_error_event.is_set():
            print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] [ORCHESTRATOR]: FATAL ERROR DETECTED. Terminating process group {process.pid} with SIGKILL.")
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            break
        time.sleep(0.1)

    try:
        process.wait(timeout=PROCESS_CLEANUP_TIMEOUT)
    except subprocess.TimeoutExpired:
        print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] [WARNING]: Subprocess {process.pid} did not terminate gracefully after {PROCESS_CLEANUP_TIMEOUT}s. Forcing final kill.")
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)

    stdout_thread.join(timeout=PROCESS_CLEANUP_TIMEOUT)
    stderr_thread.join(timeout=PROCESS_CLEANUP_TIMEOUT)

    log_filepath = os.path.join(output_dir, log_filename_base)
    with open(log_filepath, 'w') as f:
        f.write("--- STDOUT ---\n")
        f.write("".join(stdout_lines))
        f.write("\n--- STDERR ---\n")
        f.write("".join(stderr_lines))
    
    print(f"   Full logs for this run saved to '{log_filepath}'")
    
    status = "success"
    if retryable_error_event.is_set():
        status = "retryable_error"
    elif fatal_error_event.is_set():
        status = "fatal_error"
        
    return "".join(stdout_lines), status


# --- Benchmark Execution Functions ---
def run_client_benchmark(client_config, run_index, output_dir):
    run_name = client_config.get('name', f'Unnamed_Client_Run_{run_index}')
    client_args = shlex.split(client_config.get('client_args', ''))
    
    print("\n" + "-"*80)
    print(f"🏁 ({run_index}) Running Client Benchmark: '{run_name}'")
    
    command = ['python', 'run_benchmark_client.py'] + client_args
    log_filename = f"{run_index}_{sanitize_filename(run_name)}_client.log"
    
    client_stdout, exec_status = execute_and_monitor_process(
        command, run_name, "CLIENT", output_dir, log_filename, 
        fatal_error_strings=FATAL_ERROR_STRINGS,
        retryable_error_strings=RETRYABLE_ERROR_STRINGS
    )

    if exec_status == "fatal_error":
        return {"run_name": run_name, "status": "TERMINATED_DUE_TO_FATAL_ERROR"}
    if exec_status == "retryable_error":
        return {"run_name": run_name, "status": "RETRYABLE_ERROR"}
    
    if client_stdout:
        try:
            result = json.loads(client_stdout)
            result['run_name'] = run_name
            result['client_log_file'] = log_filename
            return result
        except json.JSONDecodeError:
            return {"run_name": run_name, "status": "CLIENT_JSON_ERROR"}
    return {"run_name": run_name, "status": "CLIENT_NO_OUTPUT"}

def run_vllm_throughput_benchmark(run_config, run_index, output_dir):
    run_name = run_config.get('name', f'vLLM_Official_Run_{run_index}')
    args_list = shlex.split(run_config.get('args', ''))
    
    print("\n" + "-"*80)
    print(f"🏁 ({run_index}) Running Official vLLM Throughput Benchmark: '{run_name}'")
    
    command = ['python', VLLM_BENCHMARK_SCRIPT_PATH] + args_list
    log_filename = f"{run_index}_{sanitize_filename(run_name)}_vllm_script.log"

    full_output, exec_status = execute_and_monitor_process(
        command, run_name, "VLLM_BENCH", output_dir, log_filename, 
        fatal_error_strings=FATAL_ERROR_STRINGS,
        retryable_error_strings=RETRYABLE_ERROR_STRINGS
    )

    if exec_status == "fatal_error":
        return {"run_name": run_name, "status": "TERMINATED_DUE_TO_FATAL_ERROR"}
    if exec_status == "retryable_error":
        return {"run_name": run_name, "status": "RETRYABLE_ERROR"}

    results = {}
    status = "COMPLETED"
    throughput_match = re.search(r"Throughput: ([\d.]+) requests/s, ([\d.]+) total tokens/s, ([\d.]+) output tokens/s", full_output)
    if throughput_match:
        results["throughput_req_per_sec"] = float(throughput_match.group(1))
        results["throughput_total_tokens_per_sec"] = float(throughput_match.group(2))
        results["throughput_output_tokens_per_sec"] = float(throughput_match.group(3))
        status = "SUCCESS"

    # Parse new metrics
    ttft_avg_match = re.search(r"Average TTFT \(s\): ([\d.]+)", full_output)
    if ttft_avg_match:
        results["avg_ttft_s"] = float(ttft_avg_match.group(1))
    ttft_p99_match = re.search(r"P99 TTFT \(s\): ([\d.]+)", full_output)
    if ttft_p99_match:
        results["p99_ttft_s"] = float(ttft_p99_match.group(1))
    tpot_avg_match = re.search(r"Average TPOT \(tokens/s\): ([\d.]+)", full_output)
    if tpot_avg_match:
        results["avg_tpot_tokens_per_s"] = float(tpot_avg_match.group(1))
    tpot_p99_match = re.search(r"P99 TPOT \(tokens/s\): ([\d.]+)", full_output)
    if tpot_p99_match:
        results["p99_tpot_tokens_per_s"] = float(tpot_p99_match.group(1))

    for line in full_output.splitlines():
        if "Total num prompt tokens:" in line: results["total_prompt_tokens"] = int(line.split(":")[1].strip())
        if "Total num output tokens:" in line: results["total_output_tokens"] = int(line.split(":")[1].strip())

    return {
        "run_name": run_name, "mode": "vllm_throughput",
        "config": {arg.strip('-'): val for arg, val in zip(args_list[::2], args_list[1::2])},
        "results": results, "client_log_file": log_filename,
        "status": status
    }


def update_summary_file(filepath, new_result):
    """Reads the summary JSON, updates the entry for the given run, and writes it back."""
    try:
        with open(filepath, 'r') as f: all_results = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError): all_results = []
    
    run_name_to_update = new_result.get('run_name')
    found = False
    for i, result in enumerate(all_results):
        if result.get('run_name') == run_name_to_update:
            all_results[i] = new_result; found = True; break
    if not found: all_results.append(new_result)
    with open(filepath, 'w') as f: json.dump(all_results, f, indent=4)

def main():
    parser = argparse.ArgumentParser(description="VLLM Automated Benchmark Runner")
    parser.add_argument("config_file", help="Path to the YAML configuration file for benchmarks.")
    parser.add_argument("--output-dir", help="Directory to save the final results and logs.", default=f"benchmark_results_{int(time.time())}")
    parser.add_argument("--max-retries", type=int, default=1, help="Maximum number of retries for a job that fails with a retryable error.")
    parser.add_argument("--retry-delay", type=int, default=10, help="Seconds to wait before retrying a failed job.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Results will be saved to directory: '{args.output_dir}'")
    with open(args.config_file, 'r') as f: config = yaml.safe_load(f)
    
    # --- Generate full list of all planned jobs ---
    planned_jobs = []
    if 'server_config' in config and 'client_runs' in config:
        for client_config in config.get('client_runs', []):
            planned_jobs.append({'type': 'custom', 'config': client_config})
    if 'standalone_runs' in config:
        for run_config in config.get('standalone_runs', []):
            if run_config.get('mode') != 'vllm_throughput': continue
            base_name, base_args_str = run_config.get('name', 'Standalone_Run'), run_config.get('args', '')
            tp_sizes, length_configs = run_config.get('tensor_parallel_sizes', [None]), run_config.get('length_configs', [None])
            for length_conf in length_configs:
                for tp in tp_sizes:
                    name_parts, args_parts = [base_name], [base_args_str]
                    if tp: name_parts.append(f"TP-{tp}"); args_parts.append(f"--tensor-parallel-size {tp}")
                    if length_conf:
                        input_len, output_len = length_conf.get('input_len'), length_conf.get('output_len')
                        if input_len is not None and output_len is not None:
                            max_len = input_len + output_len
                            name_parts.append(f"ISL-{input_len}_OSL-{output_len}")
                            args_parts.append(f"--input-len {input_len} --output-len {output_len} --max-model-len {max_len}")
                        else: continue
                    final_name, final_args = "_".join(name_parts), " ".join(filter(None, args_parts))
                    if final_args: planned_jobs.append({'type': 'standalone', 'config': {'name': final_name, 'args': final_args, 'mode': 'vllm_throughput'}})
    
    if not planned_jobs: print("No valid benchmark runs found in config file."); return

    # --- Initialize summary report ---
    report_filepath = os.path.join(args.output_dir, "summary_report.json")
    initial_results = [{"run_name": job['config'].get('name'), "status": "NOT_STARTED"} for job in planned_jobs]
    with open(report_filepath, 'w') as f: json.dump(initial_results, f, indent=4)
    print(f"✅ Initial summary report created at '{report_filepath}' with {len(initial_results)} planned runs.")

    # --- Execute all planned jobs ---
    run_counter = 0
    server_process = None
    try:
        for i, job in enumerate(planned_jobs):
            run_counter += 1
            
            current_retry = 0
            while current_retry <= args.max_retries:
                print("\n" + "#"*80); print(f"# Starting Benchmark {run_counter}/{len(planned_jobs)} (Attempt {current_retry + 1}/{args.max_retries + 1})"); print("#"*80)
                
                result = None
                if job['type'] == 'custom':
                    if server_process is None: # Start server if not already running
                        server_args = shlex.split(config['server_config'].get('server_args', ''))
                        server_process = subprocess.Popen(['./start_server.sh'] + server_args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, preexec_fn=os.setsid)
                        log_thread = threading.Thread(target=stream_and_capture_output, args=(server_process.stdout, "SERVER", []), daemon=True); log_thread.start()
                        if not wait_for_server(SERVER_STARTUP_TIMEOUT):
                            print("🚨 Server failed to start. Cannot run custom benchmarks."); break
                    result = run_client_benchmark(job['config'], run_counter, args.output_dir)
                
                elif job['type'] == 'standalone':
                    result = run_vllm_throughput_benchmark(job['config'], run_counter, args.output_dir)

                if result:
                    update_summary_file(report_filepath, result)
                
                if result and result.get("status") == "RETRYABLE_ERROR":
                    current_retry += 1
                    if current_retry <= args.max_retries:
                        print(f"🚨 Run failed with a retryable error. Retrying in {args.retry_delay} seconds...")
                        time.sleep(args.retry_delay)
                    else:
                        print("🚨 Run failed with a retryable error. Max retries reached.")
                        break 
                else:
                    # Break the retry loop if the run was successful or failed with a non-retryable error
                    break

    finally:
        if server_process and server_process.poll() is None:
            print("\n" + "="*80); print("🛑 Shutting down server.");
            os.killpg(os.getpgid(server_process.pid), signal.SIGTERM); server_process.wait(timeout=PROCESS_CLEANUP_TIMEOUT)

    print("\n" + "#"*80)
    print(f"✅ All benchmark runs complete. Final report is at '{report_filepath}'")
    print("#"*80)

if __name__ == "__main__":
    main()