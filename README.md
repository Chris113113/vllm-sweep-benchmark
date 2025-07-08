# vLLM Automated Benchmarking Suite

This project provides a powerful and flexible script to automate the performance benchmarking of Large Language Models (LLMs) using the official `vllm.entrypoints.benchmark_throughput` script.

The suite is designed to measure the raw generation (decoding) throughput of a model across various configurations. It achieves this by orchestrating a series of standalone benchmark runs, each with its own set of parameters, such as tensor parallel size and input/output lengths.

## Features

- **Standalone Benchmark Runs**: Each benchmark is executed as a separate, independent process, ensuring clean and isolated measurements.
- **Config-File Driven**: Easily define an entire benchmark suite in a single, human-readable YAML file.
- **Automated Orchestration**: A master script handles the generation and execution of all benchmark combinations defined in the configuration file.
- **Resumable Runs**: The script can be stopped and restarted, and it will automatically skip any jobs that have already completed successfully.
- **Comprehensive Results Output**: All results are saved to a clean output directory containing:
    - A `summary_report.json` with structured metrics from all runs.
    - A `client_logs_{timestamp}` directory containing detailed `*.log` files for each individual benchmark run.
    - Publication-quality plots (`.png`) comparing performance across different configurations.
    - A `benchmark_summary.csv` file containing a tabular representation of the results.

## Prerequisites

- Python 3.8+
- An NVIDIA GPU with a compatible CUDA toolkit installed.
- The `bash` shell.

## Installation

1.  **Clone the Repository**
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

Edit a YAML configuration file (e.g., `benchmarks/full_run.yaml`) to define the benchmark runs you want to execute. The `standalone_runs` section allows you to specify a list of models and their corresponding configurations.

**`benchmarks.yml` Example:**
```yaml
standalone_runs:
  - name: "Llama-3.3-70B-Instruct"
    mode: "vllm_throughput"
    tensor_parallel_sizes: [2, 4, 8]
    args: >
      --model "meta-llama/Llama-3.3-70B-Instruct"
      --num-prompts 200
      --gpu-memory-utilization 0.9
      --quantization fp8
    length_configs:
      - { input_len: 256, output_len: 256, prefix_len: 0 }
      - { input_len: 1024, output_len: 1024, prefix_len: 100 }

  - name: "Llama-3.1-8B-Instruct"
    mode: "vllm_throughput"
    tensor_parallel_sizes: [1, 2]
    args: >
      --model "meta-llama/Llama-3.1-8B-Instruct"
      --num-prompts 500
    length_configs:
      - { input_len: 256, output_len: 256, prefix_len: 0 }
```

### 2. Run the Automated Suite

Execute the `run_automated_benchmark.py` script, passing the path to your configuration file.

```bash
python scripts/run_automated_benchmark.py benchmarks/full_run.yaml --output-dir ./my_benchmark_results
```

### 3. Review the Output

After the run completes, the output directory will be structured as follows:

```
my_benchmark_results/
│
├── summary_report.json
├── benchmark_summary.csv
├── client_logs_{timestamp}/
│   ├── 1_Llama-3.3-70B-Instruct_TP-2_ISL-256_OSL-256_vllm_script.log
│   ├── 2_Llama-3.3-70B-Instruct_TP-4_ISL-256_OSL-256_vllm_script.log
│   └── ...
└── meta-llama_Llama-3.3-70B-Instruct/
    └── throughput_by_seqlen_and_tp.png
```

- **`summary_report.json`**: The main file containing structured JSON data for all completed runs, perfect for programmatic analysis.
- **`benchmark_summary.csv`**: A CSV file containing a tabular representation of the results.
- **`client_logs_{timestamp}/*.log`**: Detailed log files for each benchmark run.
- **`throughput_by_seqlen_and_tp.png`**: A set of plots automatically generated to compare the results of the runs for each model.

### 4. Visualize Results

If you want to re-run the visualization on an existing report, you can use the offline script:

```bash
python scripts/visualize_result.py ./my_benchmark_results/summary_report.json
```

This will generate new visualizations inside the output directory.

