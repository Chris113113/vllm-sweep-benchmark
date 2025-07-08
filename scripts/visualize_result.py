import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import json
import os
import re

def sanitize_filename(name):
    """Removes invalid characters from a string to make it a valid filename."""
    name = name.replace('/', '_')
    s = re.sub(r'[\\/*?:"<>|]', "", name)
    s = s.replace(" ", "_")
    return s

def save_to_csv(df, output_path, model_name):
    """
    Saves the benchmark results to a CSV file with a specific schema.
    """
    if df.empty:
        return

    # Create a new DataFrame with the desired schema
    csv_df = pd.DataFrame()

    # Map the data from the input DataFrame to the new schema
    csv_df['run_id'] = df['run_name']
    csv_df['model_id'] = df['model']
    csv_df['inference_software_id'] = 'vllm'  # Assuming vLLM is the inference software
    csv_df['hardware_id'] = 'a4'  # Placeholder for hardware ID
    csv_df['workload_type'] = 'offline_benchmark'
    csv_df['workload_checkpoint_path'] = ''  # Placeholder
    csv_df['workload_tokenizer_name_or_path'] = df['model']
    csv_df['workload_max_output_length'] = df['output-len']
    csv_df['workload_max_input_length'] = df['input-len']
    csv_df['workload_precision_config'] = df.get('quantization', '') 
    csv_df['hardware_total_chips_used'] = pd.to_numeric(df['tp_size'], errors='coerce').fillna(0).astype(int)
    csv_df['result_success'] = df['status'] == 'SUCCESS'
    csv_df['result_error_message'] = ''  # Placeholder
    csv_df['metrics_output_tokens_per_sec'] = df.get('throughput_output_tokens_per_sec')
    csv_df['metrics_e2e_latency_p90_ms'] = ''  # Placeholder, as this metric is not in the source data
    csv_df['metrics_ttft_avg_ms'] = df.get('avg_ttft_s', 0) * 1000
    csv_df['metrics_tpot_avg_ms'] = df.get('avg_tpot_tokens_per_s')
    
    # Add prefix length to comments if it exists and is not 0
    if 'prefix-len' in df.columns:
        csv_df['logs_comments_string'] = df['prefix-len'].apply(
            lambda x: f"Prefix length: {x}" if pd.notna(x) and x != 0 else ''
        )
    else:
        csv_df['logs_comments_string'] = ''

    csv_df['run_type'] = 'benchmark'
    csv_df['update_person_ldap'] = 'pirillo'  # Placeholder
    csv_df['uploaded_to_bq'] = ''  # Placeholder

    csv_filepath = os.path.join(output_path, f'{sanitize_filename(model_name)}_benchmark_results.csv')
    csv_df.to_csv(csv_filepath, index=False)
    print(f"CSV results saved to: {csv_filepath}")

def plot_latency_metrics(df, output_path, model_name):
    """
    Generates and saves dot plots for latency percentiles (TTFT, ITL).
    """
    latency_metrics = ['p1_ttft_ms', 'p50_ttft_ms', 'p95_ttft_ms', 'p99_ttft_ms']
    if not any(metric in df.columns for metric in latency_metrics):
        print("Latency percentile metrics (pXX_ttft_ms) not found in results. Skipping latency plot.")
        return

    found_metrics = [m for m in latency_metrics if m in df.columns]

    df_melted = df.melt(
        id_vars='run_name',
        value_vars=found_metrics,
        var_name='Metric',
        value_name='Latency (ms)'
    )

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(14, 8))

    sns.stripplot(
        data=df_melted,
        x='Latency (ms)',
        y='run_name',
        hue='Metric',
        jitter=0.1,
        dodge=True,
        ax=ax,
        size=8,
        palette='viridis'
    )
    
    ax.set_title(f'Latency Percentiles for {model_name}', fontsize=18, pad=20)
    ax.set_xlabel('Latency (ms)', fontsize=12)
    ax.set_ylabel('Benchmark Run', fontsize=12)
    ax.legend(title='Metric', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(axis='x', linestyle='--')

    fig.tight_layout()
    
    latency_plot_path = os.path.join(output_path, 'latency_percentile_report.png')
    fig.savefig(latency_plot_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Latency visualization saved to: {latency_plot_path}")

def plot_throughput_metrics(df, output_path, model_name):
    """
    Generates and saves a dot plot for throughput vs. sequence length,
    colored by Tensor Parallel size, with annotated values.
    """
    required_cols = ['throughput_tokens_per_sec', 'seq_len_label', 'tp_size']
    if not all(col in df.columns for col in required_cols):
        print(f"Required columns for throughput plot not found. Needed: {required_cols}. Skipping.")
        return
        
    df_sorted = df.sort_values(by=['tp_size', 'seq_len_label'])

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(16, 9))

    sns.scatterplot(
        data=df_sorted,
        x='seq_len_label',
        y='throughput_tokens_per_sec',
        hue='tp_size',
        size='tp_size',
        sizes=(100, 250),
        palette='viridis_r',
        ax=ax,
        style='tp_size',
        s=150
    )
    
    for index, row in df_sorted.iterrows():
        ax.text(
            row['seq_len_label'], 
            row['throughput_tokens_per_sec'] + (df['throughput_tokens_per_sec'].max() * 0.01), # Offset slightly above the dot
            f"{row['throughput_tokens_per_sec']:.0f}", 
            horizontalalignment='center',
            size='small',
            color='black',
            weight='semibold'
        )

    ax.set_title(f'Throughput for {model_name}\n(Tokens/Second vs. Sequence Length by Tensor Parallel Size)', fontsize=20, pad=20)
    ax.set_xlabel('Sequence Length (Input / Output / Prefix Tokens)', fontsize=14)
    ax.set_ylabel('Throughput (Generated Tokens / Second)', fontsize=14)
    ax.tick_params(axis='x', rotation=45, labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.legend(title='TP Size', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
    ax.grid(axis='y', linestyle='--')

    ax.set_ylim(0, df['throughput_tokens_per_sec'].max() * 1.1)

    fig.tight_layout()
    
    throughput_plot_path = os.path.join(output_path, 'throughput_by_seqlen_and_tp.png')
    fig.savefig(throughput_plot_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Throughput visualization saved to: {throughput_plot_path}")

def main():
    """Main function to load data and generate visualizations."""
    parser = argparse.ArgumentParser(description="Visualize results from a benchmark summary JSON file.")
    parser.add_argument("summary_file", help="Path to the summary_report.json file.")
    args = parser.parse_args()
    
    output_path = os.path.dirname(args.summary_file)
    if not output_path: output_path = "."

    try:
        with open(args.summary_file, 'r') as f: data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found at '{args.summary_file}'"); return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{args.summary_file}'."); return

    if not data:
        print("No benchmark results found in the file."); return

    flat_data = []
    for run in data:
        if run.get('status', 'SUCCESS') == 'SUCCESS' and 'config' in run and 'results' in run:
            flat_run = {**run['config'], **run['results'], 'status': run.get('status', "NONE"), 'run_name': run.get('run_name', 'Unnamed Run')}
            
            tp_size = flat_run.get('tensor-parallel-size', flat_run.get('tp_size', 1))
            flat_run['tp_size'] = int(tp_size)

            input_len = flat_run.get('input-len', flat_run.get('max_input_length'))
            output_len = flat_run.get('output-len', flat_run.get('max_output_length'))
            prefix_len = flat_run.get('prefix-len')

            if input_len is not None and output_len is not None:
                label = f"{input_len} / {output_len}"
                if prefix_len is not None:
                    label += f" / {prefix_len}"
                flat_run['seq_len_label'] = label
            else:
                flat_run['seq_len_label'] = 'N/A'

            tokens_per_sec = flat_run.get('throughput_total_tokens_per_sec', flat_run.get('throughput_tokens_per_sec'))
            if tokens_per_sec is not None:
                flat_run['throughput_tokens_per_sec'] = float(tokens_per_sec)

            flat_data.append(flat_run)
        else:
            flat_data.append({
                'run_name': run.get('run_name', 'Unnamed Failed Run'),
                'status': run.get('status', 'UNKNOWN_FAILURE')
            })
            
    df = pd.DataFrame(flat_data)
    successful_runs_df = df[df['status'] == 'SUCCESS'].copy()
    
    if successful_runs_df.empty:
        print("No successful benchmark runs to visualize."); return

    # --- Generate a single CSV for all successful runs ---
    save_to_csv(successful_runs_df, output_path, "benchmark_summary")

    model_col = 'model' if 'model' in successful_runs_df.columns else 'run_name'
    for model_identifier, group_df in successful_runs_df.groupby(model_col):
        print(f"--- Generating plots for group: {model_identifier} ---")
        
        model_output_path = os.path.join(output_path, sanitize_filename(model_identifier))
        os.makedirs(model_output_path, exist_ok=True)
        
        plot_latency_metrics(group_df, model_output_path, model_identifier)
        plot_throughput_metrics(group_df, model_output_path, model_identifier)

if __name__ == "__main__":
    main()
