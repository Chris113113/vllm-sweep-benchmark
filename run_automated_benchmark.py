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
import queue
import concurrent.futures
import queue
import concurrent.futures

# --- Configuration ---
SERVER_STARTUP_TIMEOUT = 300
SERVER_HEALTH_CHECK_URL = "http://localhost:8000/v1/models"
VLLM_BENCHMARK_SCRIPT_PATH = "vllm-repo-benchmark/benchmarks/benchmark_vllm_throughput.py"
FATAL_ERROR_STRINGS = ["EngineCore failed to start."]
RETRYABLE_ERROR_STRINGS = ["uncorrectable NVLink error"]
# Increase the cleanup timeout to give the OS more time to reap zombie processes
PROCESS_CLEANUP_TIMEOUT = 15

# --- Utility Functions ---
def get_gpu_count():
    """Returns the number of available GPUs by querying nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=count", "--format=csv,noheader"],
            capture_output=True, text=True, check=True
        )
        # The command can return the count for each GPU on a new line.
        # We just need the value from the first line.
        gpu_count_str = result.stdout.strip().split('\n')[0]
        return int(gpu_count_str)
    except (FileNotFoundError, subprocess.CalledProcessError, ValueError, IndexError) as e:
        print(f"[WARNING] Could not query nvidia-smi for GPU count: {e}. Falling back to sequential execution.")
        return 0

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

# --- GPU Resource Management ---
class GPUManager:
    """A thread-safe manager to allocate and release GPUs."""
    def __init__(self, num_gpus):
        self.num_gpus = num_gpus
        self.gpus = list(range(num_gpus))
        self.available = [True] * num_gpus
        self.lock = threading.Lock()

    def acquire(self, count):
        """Acquires a contiguous block of 'count' GPUs."""
        with self.lock:
            for i in range(self.num_gpus - count + 1):
                if all(self.available[i:i+count]):
                    self.available[i:i+count] = [False] * count
                    gpu_ids = list(range(i, i + count))
                    print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] [GPUManager]: Acquired GPUs {gpu_ids}")
                    return gpu_ids
            return None

    def release(self, gpu_ids):
        """Releases a list of GPUs, making them available again."""
        with self.lock:
            for gpu_id in gpu_ids:
                self.available[gpu_id] = True
            print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] [GPUManager]: Released GPUs {gpu_ids}")




def stream_and_capture_output(pipe, prefix, captured_lines_list, fatal_error_event=None, fatal_error_strings=None, retryable_error_event=None, retryable_error_strings=None):
    """
    Reads a subprocess's output stream, intelligently handling carriage returns (\r)
    to allow in-place updates for progress bars, while correctly prefixing new lines.
    """
    try:
        with io.TextIOWrapper(pipe, encoding='utf-8', errors='replace') as text_stream:
            on_new_line = True
            for char in iter(lambda: text_stream.read(1), ''):
                if not char:
                    break  # End of stream

                # If we are on a new line, print the prefix.
                if on_new_line:
                    timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
                    sys.stdout.write(f"[{timestamp}] [{prefix}]: ")
                    on_new_line = False
                
                # Write the character from the subprocess.
                sys.stdout.write(char)
                
                # Buffer the character for the log file.
                # We handle the buffer on a line-by-line basis for error checking.
                if 'line_buffer' not in locals():
                    line_buffer = ""
                line_buffer += char

                # If the character is a newline, we are back to a new line.
                # Also process the buffered line for logs and errors.
                if char == '\n':
                    on_new_line = True
                    captured_lines_list.append(line_buffer)
                    # Error checking
                    if fatal_error_event and fatal_error_strings and any(e in line_buffer for e in fatal_error_strings):
                        fatal_error_event.set()
                    if retryable_error_event and retryable_error_strings and any(e in line_buffer for e in retryable_error_strings):
                        retryable_error_event.set()
                    line_buffer = ""

                sys.stdout.flush()

            # After the loop, if there's any content left in the buffer,
            # it means the process output didn't end with a newline.
            if 'line_buffer' in locals() and line_buffer:
                captured_lines_list.append(line_buffer)
                # Final error check on the remaining buffer
                if fatal_error_event and fatal_error_strings and any(e in line_buffer for e in fatal_error_strings):
                    fatal_error_event.set()
                if retryable_error_event and retryable_error_strings and any(e in line_buffer for e in retryable_error_strings):
                    retryable_error_event.set()
                # Add a final newline to the console for clean termination.
                sys.stdout.write('\n')
                sys.stdout.flush()

    except Exception as e:
        timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
        print(f"[{timestamp}] [STREAM_ERROR]: An error occurred in the stream reader: {e}", flush=True)

def execute_and_monitor_process(command, job_name, log_prefix, output_dir, log_filename_base, gpu_ids=None, fatal_error_strings=None, retryable_error_strings=None):
    """
    A helper function to execute a subprocess on a specific set of GPUs, stream its logs,
    monitor for errors, and terminate early if needed.
    """
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

    log_filepath = os.path.join(output_dir, log_filename_base)
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

def run_vllm_throughput_benchmark(run_config, run_index, output_dir, gpu_ids=None):
    run_name = run_config.get('name', f'vLLM_Official_Run_{run_index}')
    args_list = shlex.split(run_config.get('args', ''))
    
    print("\n" + "-"*80)
    print(f"🏁 ({run_index}) Running Official vLLM Throughput Benchmark: '{run_name}' on GPUs {gpu_ids}")
    
    command = ['python', VLLM_BENCHMARK_SCRIPT_PATH] + args_list
    log_filename = f"{run_index}_{sanitize_filename(run_name)}_vllm_script.log"

    full_output, exec_status = execute_and_monitor_process(
        command, run_name, "VLLM_BENCH", output_dir, log_filename, 
        gpu_ids=gpu_ids,
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

def print_run_plan(jobs):
    """Prints a formatted table of the benchmark execution plan."""
    print("\n" + "#"*80)
    print("# Benchmark Execution Plan")
    print("#"*80)
    
    if not jobs:
        print("# No jobs planned.")
        print("#"*80)
        return
        
    headers = ["Num", "Type", "TP", "Name", "Arguments"]
    max_name_len = max([len(j['config'].get('name', 'N/A')) for j in jobs] + [len(headers[3])])

    # Print table header
    header_line = f"# {headers[0]:<4}| {headers[1]:<11}| {headers[2]:<3}| {headers[3]:<{max_name_len}} | {headers[4]}"
    print(header_line)
    print("#" + "-" * (len(header_line) - 1))

    # Print each job as a row
    for i, job in enumerate(jobs):
        config = job['config']
        job_type = job['type']
        name = config.get('name', 'N/A')
        
        if job_type == 'standalone':
            tp_size = config.get('tensor_parallel_size', 1)
            details = config.get('args', '')
        elif job_type == 'custom':
            tp_size = '-'
            details = config.get('client_args', '')
        else:
            tp_size = '-'
            details = ''
            
        print(f"# {i+1:<4}| {job_type:<11}| {tp_size:<3}| {name:<{max_name_len}} | {details}")

    print("#"*80)

def run_job_wrapper(job, run_index, gpu_manager, args):

    """
    A wrapper function that acquires GPUs, runs a benchmark job,
    and ensures the GPUs are released.
    """
    tp_size = job['config'].get('tensor_parallel_size', 1)
    
    gpu_ids = None
    while gpu_ids is None:
        gpu_ids = gpu_manager.acquire(tp_size)
        if gpu_ids is None:
            time.sleep(1) # Wait for GPUs to become available

    try:
        result = run_vllm_throughput_benchmark(job['config'], run_index, args.output_dir, gpu_ids=gpu_ids)
        return result
    finally:
        gpu_manager.release(gpu_ids)


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

    # --- Job Planning ---
    server_jobs = [
        {'type': 'custom', 'config': client_config}
        for client_config in config.get('client_runs', [])
        if 'server_config' in config
    ]
    
    standalone_jobs = []
    for run_config in config.get('standalone_runs', []):
        if run_config.get('mode') != 'vllm_throughput':
            continue
        
        base_name = run_config.get('name', 'Standalone_Run')
        base_args_str = run_config.get('args', '')
        tp_sizes = run_config.get('tensor_parallel_sizes', [1]) # Default to 1 if not specified
        length_configs = run_config.get('length_configs', [{}])

        for length_conf in length_configs:
            for tp in tp_sizes:
                name_parts = [base_name, f"TP-{tp}"]
                args_parts = [base_args_str, f"--tensor-parallel-size {tp}"]
                
                input_len = length_conf.get('input_len')
                output_len = length_conf.get('output_len')

                if input_len is not None and output_len is not None:
                    max_len = input_len + output_len
                    name_parts.append(f"ISL-{input_len}_OSL-{output_len}")
                    args_parts.append(f"--input-len {input_len} --output-len {output_len} --max-model-len {max_len}")

                final_name = "_".join(name_parts)
                final_args = " ".join(filter(None, args_parts))
                
                standalone_jobs.append({
                    'type': 'standalone',
                    'config': {
                        'name': final_name,
                        'args': final_args,
                        'mode': 'vllm_throughput',
                        'tensor_parallel_size': tp
                    }
                })

    # Sort standalone jobs by tensor parallel size (ascending)
    standalone_jobs.sort(key=lambda j: j['config'].get('tensor_parallel_size', 1))
    
    all_jobs = standalone_jobs + server_jobs
    if not all_jobs:
        print("No valid benchmark runs found in config file.")
        return

    # --- Print Run Plan ---
    print_run_plan(all_jobs)

    # --- Initialize Summary Report ---
    report_filepath = os.path.join(args.output_dir, "summary_report.json")
    initial_results = [{"run_name": job['config'].get('name'), "status": "NOT_STARTED"} for job in all_jobs]
    with open(report_filepath, 'w') as f:
        json.dump(initial_results, f, indent=4)
    print(f"✅ Initial summary report created at '{report_filepath}' with {len(initial_results)} planned runs.")

    # --- Execute Standalone Jobs in Parallel ---
    num_gpus = get_gpu_count()
    if num_gpus > 0 and standalone_jobs:
        print(f"\n" + "#"*80)
        print(f"# Detected {num_gpus} GPUs. Running {len(standalone_jobs)} standalone jobs in parallel.")
        print("#"*80)
        
        gpu_manager = GPUManager(num_gpus)
        # Use number of GPUs as max_workers to ensure we can saturate the hardware
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_gpus) as executor:
            # Map each job to the wrapper function
            future_to_job = {
                executor.submit(run_job_wrapper, job, i + 1, gpu_manager, args): job
                for i, job in enumerate(standalone_jobs)
            }
            
            for future in concurrent.futures.as_completed(future_to_job):
                job = future_to_job[future]
                try:
                    result = future.result()
                    if result:
                        update_summary_file(report_filepath, result)
                except Exception as exc:
                    print(f"🚨 Job '{job['config']['name']}' generated an exception: {exc}")
                    update_summary_file(report_filepath, {"run_name": job['config']['name'], "status": "EXCEPTION"})

    # --- Execute Server/Client Jobs Sequentially ---
    if server_jobs:
        print("\n" + "#"*80)
        print(f"# Running {len(server_jobs)} server/client jobs sequentially.")
        print("#"*80)
        
        server_process = None
        try:
            # Start the server once
            server_args = shlex.split(config['server_config'].get('server_args', ''))
            server_process = subprocess.Popen(['./start_server.sh'] + server_args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, preexec_fn=os.setsid)
            log_thread = threading.Thread(target=stream_and_capture_output, args=(server_process.stdout, "SERVER", []), daemon=True)
            log_thread.start()
            
            if not wait_for_server(SERVER_STARTUP_TIMEOUT):
                print("🚨 Server failed to start. Cannot run custom benchmarks.")
            else:
                # Run all client benchmarks against the single server
                for i, job in enumerate(server_jobs):
                    run_counter = len(standalone_jobs) + i + 1
                    result = run_client_benchmark(job['config'], run_counter, args.output_dir)
                    if result:
                        update_summary_file(report_filepath, result)
        finally:
            if server_process and server_process.poll() is None:
                print("\n" + "="*80)
                print("🛑 Shutting down server.")
                os.killpg(os.getpgid(server_process.pid), signal.SIGTERM)
                server_process.wait(timeout=PROCESS_CLEANUP_TIMEOUT)

    print("\n" + "#"*80)
    print(f"✅ All benchmark runs complete. Final report is at '{report_filepath}'")
    print("#"*80)


if __name__ == "__main__":
    main()