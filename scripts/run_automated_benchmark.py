import subprocess
import time
import sys
import threading
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
import shutil

# --- Thread-safe lock for summary file ---
summary_file_lock = threading.Lock()

# --- Configuration ---
PROCESS_CLEANUP_TIMEOUT = 15

# --- Utility Functions ---
def get_gpu_count():
    """Returns the number of available GPUs by querying nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=count", "--format=csv,noheader"],
            capture_output=True, text=True, check=True
        )
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
        """Acquires 'count' available GPUs."""
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
        """Releases a list of GPUs."""
        with self.lock:
            for gpu_id in gpu_ids:
                self.available[gpu_id] = True
            print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] [GPUManager]: Released GPUs {gpu_ids}")

def update_summary_file(filepath, new_result, status=None):
    """Reads the summary JSON, updates the entry for the given run, and writes it back."""
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
    header_line = f"# {headers[0]:<4}| {headers[1]:<11}| {headers[2]:<3}| {headers[3]:<{max_name_len}} | {headers[4]}"
    print(header_line)
    print("#" + "-" * (len(header_line) - 1))
    for i, job in enumerate(jobs):
        config = job['config']
        job_type = job['type']
        name = config.get('name', 'N/A')
        tp_size = config.get('tensor_parallel_size', 1)
        details = config.get('args', '')
        print(f"# {i+1:<4}| {job_type:<11}| {tp_size:<3}| {name:<{max_name_len}} | {details}")
    print("#"*80)

def run_job_wrapper(job, run_index, gpu_manager, args, logs_dir):
    """
    A wrapper function that acquires GPUs, runs a benchmark job by calling
    the single-run script, and ensures the GPUs are released.
    """
    tp_size = job['config'].get('tensor_parallel_size', 1)
    run_name = job['config'].get('name')
    report_filepath = os.path.join(args.output_dir, "summary_report.json")

    gpu_ids = None
    while gpu_ids is None:
        gpu_ids = gpu_manager.acquire(tp_size)
        if gpu_ids is None:
            time.sleep(1)

    update_summary_file(report_filepath, {"run_name": run_name}, status="RUNNING")
    job_tmp_dir = os.path.join(args.output_dir, f"tmp_{run_index}_{sanitize_filename(run_name)}")
    os.makedirs(job_tmp_dir, exist_ok=True)

    try:
        command = [
            sys.executable,
            'scripts/run_single_benchmark.py',
            '--run-name', run_name,
            '--output-dir', job_tmp_dir,
            '--gpu-ids', ",".join(map(str, gpu_ids)),
            '--'
        ]
        benchmark_args = shlex.split(job['config'].get('args', ''))
        command.extend(benchmark_args)

        print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] [ORCHESTRATOR]: Launching job '{run_name}' with command: {' '.join(command)}")
        process = subprocess.run(command, capture_output=True, text=True)

        job_summary_path = os.path.join(job_tmp_dir, "summary_report.json")
        if os.path.exists(job_summary_path):
            with open(job_summary_path, 'r') as f:
                result_data = json.load(f)
            if result_data:
                result = result_data[0]
                log_sub_dirs = [d for d in os.listdir(job_tmp_dir) if d.startswith('client_logs_')]
                if log_sub_dirs:
                    job_logs_dir = os.path.join(job_tmp_dir, log_sub_dirs[0])
                    for log_file in os.listdir(job_logs_dir):
                        new_log_filename = f"{run_index}_{log_file}"
                        shutil.move(os.path.join(job_logs_dir, log_file), os.path.join(logs_dir, new_log_filename))
                        # Update the log file path to be relative to the output directory
                        result['client_log_file'] = os.path.join(os.path.basename(logs_dir), new_log_filename)
                return result
        
        print(f"🚨 Job '{run_name}' failed to produce a summary report.")
        print("--- STDOUT ---\n", process.stdout)
        print("--- STDERR ---\n", process.stderr)
        return {"run_name": run_name, "status": "EXECUTION_FAILED", "error_message": process.stderr}

    except Exception as e:
        print(f"🚨 An exception occurred while running job '{run_name}': {e}")
        return {"run_name": run_name, "status": "EXCEPTION", "error_message": str(e)}
    finally:
        gpu_manager.release(gpu_ids)
        if os.path.exists(job_tmp_dir):
            shutil.rmtree(job_tmp_dir)

def main():
    parser = argparse.ArgumentParser(description="VLLM Automated Benchmark Runner")
    parser.add_argument("config_file", help="Path to the YAML configuration file for benchmarks.")
    parser.add_argument("--output-dir", help="Directory to save the final results and logs.", default=f"benchmark_results_{int(time.time())}")
    parser.add_argument("--max-retries", type=int, default=1, help="Maximum number of retries for a job that fails with a retryable error.")
    parser.add_argument("--retry-delay", type=int, default=10, help="Seconds to wait before retrying a failed job.")
    parser.add_argument("--prompt-for-confirmation", action="store_true", help="If set, prompt for confirmation before running the benchmark.")
    args = parser.parse_args()

    args.output_dir = os.path.join("results", args.output_dir)

    os.makedirs(args.output_dir, exist_ok=True)
    logs_dir = os.path.join(args.output_dir, f"client_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(logs_dir, exist_ok=True)
    print(f"Results will be saved to directory: '{args.output_dir}'")
    print(f"Logs will be saved to directory: '{logs_dir}'")
    with open(args.config_file, 'r') as f: config = yaml.safe_load(f)

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

                if input_len is not None:
                    name_parts.append(f"ISL-{input_len}")
                    args_parts.append(f"--input-len {input_len}")
                if output_len is not None:
                    name_parts.append(f"OSL-{output_len}")
                    args_parts.append(f"--output-len {output_len}")
                if prefix_len is not None:
                    name_parts.append(f"PL-{prefix_len}")
                    args_parts.append(f"--prefix-len {prefix_len}")

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

    results_map = {res.get('run_name'): res for res in existing_results}
    for job in all_planned_jobs:
        run_name = job['config'].get('name')
        if run_name not in results_map:
            results_map[run_name] = {"run_name": run_name, "status": "NOT_STARTED"}
    
    final_summary_list = list(results_map.values())
    with open(report_filepath, 'w') as f:
        json.dump(final_summary_list, f, indent=4)
    print(f"✅ Summary report updated at '{report_filepath}' with {len(all_planned_jobs)} total runs.")

    standalone_jobs = [job for job in all_planned_jobs if job['config']['name'] not in completed_runs]
    standalone_jobs.sort(key=lambda j: j['config'].get('tensor_parallel_size', 1))
    
    if not standalone_jobs:
        print("No new benchmark runs to execute.")
        return

    print_run_plan(standalone_jobs)

    if args.prompt_for_confirmation:
        response = input("Continue with the benchmark execution? (y/n): ").lower()
        if response != 'y':
            print("Benchmark execution cancelled by user.")
            return

    num_gpus = get_gpu_count()
    if num_gpus > 0 and standalone_jobs:
        print(f"\n" + "#"*80)
        print(f"# Detected {num_gpus} GPUs. Running {len(standalone_jobs)} standalone jobs in parallel.")
        print("#"*80)
        
        gpu_manager = GPUManager(num_gpus)
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_gpus) as executor:
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
