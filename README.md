# vLLM Automated Benchmarking Suite

This project provides a powerful and flexible suite of scripts to automate the performance benchmarking of Large Language Models (LLMs) served with [vLLM](https://github.com/vllm-project/vllm).

The suite is designed to measure the **raw generation (decoding) throughput** of a model. It achieves this by launching a single, persistent server instance and running a series of different client workloads against it. This "hot cache" methodology is highly efficient and simulates a real-world production environment where the server is continuously handling requests.

## Features

* **Efficient Single-Server Workflow**: Starts one server and runs all defined client benchmarks against it, avoiding costly model reloads between tests.
* **Config-File Driven**: Easily define an entire benchmark suite in a single, human-readable `benchmarks.yml` file.
* **Rate-Controlled Load Generation**: The client uses an asyncio worker queue to maintain a constant, stable level of concurrency, preventing request timeouts and providing a realistic measure of sustained throughput.
* **Automated Orchestration**: A master script handles starting the server, waiting for it to be ready, executing the series of client runs, and cleanly shutting down the server.
* **Comprehensive Results Output**: All results are saved to a clean output directory containing:
    * A `summary_report.json` with structured metrics from all runs.
    * Detailed `*.log` files for each individual client run.
    * Publication-quality plots (`.png`) comparing performance across runs.

## Prerequisites

* Python 3.8+
* An NVIDIA GPU with a compatible CUDA toolkit installed.
* The `bash` shell (for the `start_server.sh` script).

## Installation

1.  **Clone the Repository (or place the scripts in a directory)**
    ```bash
    # git clone <your-repo-url>
    # cd <your-repo-directory>
    ```

2.  **Create and Activate a Virtual Environment (Recommended)**
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```
    *(On Windows, use `venv\Scripts\activate`)*

3.  **Install Dependencies**
    Install all the necessary Python packages from the `requirements.txt` file.
    ```bash
    pip install -r requirements.txt
    ```

## Usage

The primary workflow involves two steps: configuring your benchmark suite and running the orchestrator.

### 1. Configure Your Benchmarks

Edit the `benchmarks.yml` file to define the server you want to test and the list of client workloads you want to run against it.

**`benchmarks.yml` Example:**
```yaml
# ==============================================================================
# VLLM Automated Benchmark Configuration (Single Server, Multi-Client)
# ==============================================================================

# Define the single server configuration to be used for all client runs.
server_config:
  server_args: >
    --model "meta-llama/Meta-Llama-3-8B-Instruct"
    --tensor-parallel-size 1
    --gpu-memory-utilization 0.95
    --max-num-seqs 512

# A list of different client workloads to test against the server.
client_runs:
  - name: "Gen-focused - 64 concurrent requests"
    client_args: >
      --model "meta-llama/Meta-Llama-3-8B-Instruct"
      --num-prompts 500
      --concurrency 64
      --max-input-length 32
      --max-output-length 1024

  - name: "Gen-focused - 128 concurrent requests"
    client_args: >
      --model "meta-llama/Meta-Llama-3-8B-Instruct"
      --num-prompts 1000
      --concurrency 128
      --max-input-length 32
      --max-output-length 1024
````

### 2\. Run the Automated Suite

Execute the `run_automated_benchmark.py` script, passing the path to your configuration file.

```bash
# This will start one server and run all defined client_runs against it.
python run_automated_benchmark.py benchmarks.yml --output-dir ./my_benchmark_results
```

### 3\. Review the Output

After the run completes, the output directory will be structured as follows:

```
my_benchmark_results/
│
├── 1_Gen-focused-64-concurrent-requests_client.log
├── 2_Gen-focused-128-concurrent-requests_client.log
├── summary_report.json
└── visualization_report.png
```

  * **`summary_report.json`**: The main file containing structured JSON data for all completed runs, perfect for programmatic analysis.
  * **`*_client.log`**: Detailed log files for each client run, containing its full `stdout` and `stderr`.
  * **`visualization_report.png`**: A set of plots automatically generated to compare the results of the runs.

### 4\. Visualize Results

If you want to re-run the visualization on an existing report, you can use the offline script:

```bash
python visualize_results.py ./my_benchmark_results/summary_report.json
```

This will generate a new `visualization_report.png` inside that directory.

```