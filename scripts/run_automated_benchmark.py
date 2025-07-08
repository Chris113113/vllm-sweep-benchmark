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

# --- Thread-safe lock for summary file ---
summary_file_lock = threading.Lock()

# --- Configuration ---
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

# --- GPU Resource Management ---
class GPUManager:
    """A thread-safe manager to allocate and release GPUs."""
    def __init__(self, num_gpus):
        self.num_gpus = num_gpus
        self.gpus = list(range(num_gpus))
        self.available = [True] * num_gpus
        self.lock = threading.Lock()

    def acquire(self, count):
        """Acquires 'count' available GPUs, not necessarily contiguous."""
        with self.lock:
            available_gpus = [i for i, is_free in enumerate(self.available) if is_free]
            if len(available_gpus) >= count:
                gpu_ids_to_acquire = available_gpus[:count]
                for gpu_id in gpu_ids_to_acquire:
                    self.available[gpu_id] = False
                print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] [GPUManager]: Acquired GPUs {gpu_ids_to_acquire}")
                return gpu_ids_to_acquire
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

def execute_and_monitor_process(command, job_name, log_prefix, logs_dir, log_filename_base, gpu_ids=None, fatal_error_strings=None, retryable_error_strings=None):
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


# --- Benchmark Execution Functions ---



def update_summary_file(filepath, new_result, status=None):
    """
    Reads the summary JSON, updates the entry for the given run, and writes it back.
    This function is now thread-safe.
    """
    with summary_file_lock:
        try:
            with open(filepath, 'r') as f:
                all_results = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            all_results = []

        run_name_to_update = new_result.get('run_name')
        found = False
        for i, result in enumerate(all_results):
            if result.get('run_name') == run_name_to_update:
                if status:
                    all_results[i]['status'] = status
                else:
                    # Merge new result with existing data, preserving original fields
                    all_results[i] = {**result, **new_result}
                found = True
                break
        
        if not found and not status:
            all_results.append(new_result)

        with open(filepath, 'w') as f:
            json.dump(all_results, f, indent=4)

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
        else:
            tp_size = '-'
            details = ''
            
        print(f"# {i+1:<4}| {job_type:<11}| {tp_size:<3}| {name:<{max_name_len}} | {details}")

    print("#"*80)

def run_job_wrapper(job, run_index, gpu_manager, args, logs_dir):
    """
    A wrapper function that acquires GPUs, runs a benchmark job,
    and ensures the GPUs are released.
    """
    tp_size = job['config'].get('tensor_parallel_size', 1)
    run_name = job['config'].get('name')
    report_filepath = os.path.join(args.output_dir, "summary_report.json")

    gpu_ids = None
    while gpu_ids is None:
        gpu_ids = gpu_manager.acquire(tp_size)
        if gpu_ids is None:
            time.sleep(1)  # Wait for GPUs to become available

    # Mark the job as RUNNING now that it has acquired GPUs
    update_summary_file(report_filepath, {"run_name": run_name}, status="RUNNING")

    try:
        result = run_vllm_throughput_benchmark(job['config'], run_index, logs_dir, gpu_ids=gpu_ids)
        return result
    finally:
        gpu_manager.release(gpu_ids)


def main():
    parser = argparse.ArgumentParser(description="VLLM Automated Benchmark Runner")
    parser.add_argument("config_file", help="Path to the YAML configuration file for benchmarks.")
    parser.add_argument("--output-dir", help="Directory to save the final results and logs.", default=f"benchmark_results_{int(time.time())}")
    parser.add_argument("--max-retries", type=int, default=1, help="Maximum number of retries for a job that fails with a retryable error.")
    parser.add_argument("--retry-delay", type=int, default=10, help="Seconds to wait before retrying a failed job.")
    parser.add_argument("--prompt-for-confirmation", action="store_true", help="If set, prompt for confirmation before running the benchmark.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    logs_dir = os.path.join(args.output_dir, f"client_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(logs_dir, exist_ok=True)
    print(f"Results will be saved to directory: '{args.output_dir}'")
    print(f"Logs will be saved to directory: '{logs_dir}'")
    with open(args.config_file, 'r') as f: config = yaml.safe_load(f)

    # --- Load existing summary to support resuming ---
    report_filepath = os.path.join(args.output_dir, "summary_report.json")
    existing_results = []
    completed_runs = set()
    if os.path.exists(report_filepath):
        try:
            with open(report_filepath, 'r') as f:
                existing_results = json.load(f)
            for result in existing_results:
                if result.get('status') == 'SUCCESS':
                    completed_runs.add(result.get('run_name'))
            print(f"Found existing summary report. Will skip {len(completed_runs)} completed runs.")
        except (json.JSONDecodeError, IOError) as e:
            print(f"[WARNING] Could not read existing summary report: {e}. Starting fresh.")
            existing_results = []

    # --- Job Planning ---
    all_planned_jobs = []
    for run_config in config.get('standalone_runs', []):
        if run_config.get('mode') != 'vllm_throughput':
            continue
        
        base_name = run_config.get('name', 'Standalone_Run')
        base_args_str = run_config.get('args', '')
        tp_sizes = run_config.get('tensor_parallel_sizes', [1])
        length_configs = run_config.get('length_configs', [{}])

        for length_conf in length_configs:
            for tp in tp_sizes:
                name_parts = [base_name, f"TP-{tp}"]
                args_parts = [base_args_str, f"--tensor-parallel-size {tp}"]
                
                input_len = length_conf.get('input_len')
                output_len = length_conf.get('output_len')
                prefix_len = length_conf.get('prefix_len')

                # Append length-related arguments and build name
                if input_len is not None:
                    name_parts.append(f"ISL-{input_len}")
                    args_parts.append(f"--input-len {input_len}")
                if output_len is not None:
                    name_parts.append(f"OSL-{output_len}")
                    args_parts.append(f"--output-len {output_len}")
                if prefix_len is not None:
                    name_parts.append(f"PL-{prefix_len}")
                    args_parts.append(f"--prefix-len {prefix_len}")

                # Calculate and append max_model_len if not manually specified
                if input_len is not None and output_len is not None and "--max-model-len" not in base_args_str:
                    max_len = input_len + output_len + (prefix_len or 0) + 1
                    args_parts.append(f"--max-model-len {max_len}")

                final_name = "_".join(name_parts)
                final_args = " ".join(filter(None, args_parts))
                
                all_planned_jobs.append({
                    'type': 'standalone',
                    'config': {
                        'name': final_name,
                        'args': final_args,
                        'mode': 'vllm_throughput',
                        'tensor_parallel_size': tp
                    }
                })

    # --- Initialize Summary Report ---
    # Create a map of existing results for easy lookup
    results_map = {res.get('run_name'): res for res in existing_results}
    # Add all planned jobs to the map, preserving existing results
    for job in all_planned_jobs:
        run_name = job['config'].get('name')
        if run_name not in results_map:
            results_map[run_name] = {"run_name": run_name, "status": "NOT_STARTED"}
    
    # Convert map back to a list for saving
    final_summary_list = list(results_map.values())
    with open(report_filepath, 'w') as f:
        json.dump(final_summary_list, f, indent=4)
    print(f"✅ Summary report updated at '{report_filepath}' with {len(all_planned_jobs)} total runs.")

    # Filter out completed jobs
    standalone_jobs = [job for job in all_planned_jobs if job['config']['name'] not in completed_runs]
    
    # Sort standalone jobs by tensor parallel size (ascending)
    standalone_jobs.sort(key=lambda j: j['config'].get('tensor_parallel_size', 1))
    
    if not standalone_jobs:
        print("No new benchmark runs to execute.")
        return

    # --- Print Run Plan ---
    print_run_plan(standalone_jobs)

    # --- Prompt for confirmation if the flag is set ---
    if args.prompt_for_confirmation:
        response = input("Continue with the benchmark execution? (y/n): ").lower()
        if response != 'y':
            print("Benchmark execution cancelled by user.")
            return

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
                executor.submit(run_job_wrapper, job, i + 1, gpu_manager, args, logs_dir): job
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

    print("\n" + "#"*80)
    print(f"✅ All benchmark runs complete. Final report is at '{report_filepath}'")
    print("#"*80)


if __name__ == "__main__":
    main()