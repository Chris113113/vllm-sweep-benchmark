import argparse
import json
import os
import yaml
from transformers import AutoTokenizer
from benchmark_dataset import RandomDataset, SampleRequest
from typing import List


def generate_dataset(args, tokenizer, length_config):
    """Generates a dataset for a given length configuration using RandomDataset."""
    num_requests = args.num_prompts
    input_len = length_config["input_len"]
    output_len = length_config["output_len"]
    prefix_len = length_config.get("prefix_len")

    # We are explicitly using RandomDataset as per the problem description's focus
    # on length_configs which are used with RandomDataset.
    dataset = RandomDataset(random_seed=args.seed)

    sample_kwargs = {
        "tokenizer": tokenizer,
        "num_requests": num_requests,
        "input_len": input_len,
        "output_len": output_len,
    }
    if prefix_len is not None:
        sample_kwargs["prefix_len"] = prefix_len

    requests = dataset.sample(**sample_kwargs)
    return requests


def save_requests(requests: List[SampleRequest], output_path: str):
    """Saves a list of SampleRequest objects to a JSON file."""
    data_to_save = []
    for req in requests:
        # Manually construct a dictionary from the SampleRequest object
        # to ensure serializability and to select only the needed fields.
        # The structure is inferred from benchmark_vllm_throughput.py
        record = {
            "prompt": req.prompt,
            "prompt_len": req.prompt_len,
            "expected_output_len": req.expected_output_len,
        }
        data_to_save.append(record)

    with open(output_path, "w") as f:
        json.dump(data_to_save, f, indent=2)
    print(f"Saved dataset with {len(requests)} requests to {output_path}")


def main():
    """
    Main function to generate and save datasets based on a YAML configuration.
    """
    parser = argparse.ArgumentParser(
        description="Generate and save test datasets based on a YAML file with length configurations."
    )
    parser.add_argument(
        "--yaml-file",
        type=str,
        required=True,
        help="Path to the YAML config file (e.g., benchmarks/gemma.yaml).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save the generated dataset files.",
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=1000,
        help="Number of prompts to generate for each dataset.",
    )
    parser.add_argument(
        "--seed", type=int, default=420, help="Random seed for dataset generation."
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code when loading tokenizer.",
    )

    args = parser.parse_args()

    print(f"Loading YAML config from: {args.yaml_file}")
    with open(args.yaml_file, "r") as f:
        config = yaml.safe_load(f)

    os.makedirs(args.output_dir, exist_ok=True)

    standalone_runs = config.get("standalone_runs", [])
    if not standalone_runs:
        print("No 'standalone_runs' found in the YAML file.")
        return

    current_tokenizer_name = None
    tokenizer = None

    for run_config in standalone_runs:
        model_name = run_config.get("name", "unknown_model")
        sanitized_model_name = model_name.replace("/", "_")
        length_configs = run_config.get("length_configs")
        args_str = run_config.get("args", "")

        model_for_tokenizer = None
        # Using shlex to handle quoted arguments correctly
        import shlex
        try:
            args_list = shlex.split(args_str)
            if "--model" in args_list:
                model_index = args_list.index("--model")
                model_for_tokenizer = args_list[model_index + 1]
        except (ValueError, IndexError):
            model_for_tokenizer = None

        if not model_for_tokenizer:
            print(f"Warning: --model not found in args for run '{model_name}'. Skipping this run.")
            continue

        if model_for_tokenizer != current_tokenizer_name:
            print(f"\nLoading tokenizer for model: {model_for_tokenizer}")
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    model_for_tokenizer, trust_remote_code=args.trust_remote_code
                )
                current_tokenizer_name = model_for_tokenizer
            except Exception as e:
                print(f"    Failed to load tokenizer for {model_for_tokenizer}: {e}")
                tokenizer = None
                current_tokenizer_name = None
                continue

        if not tokenizer:
            print(f"  Skipping because tokenizer for {model_for_tokenizer} could not be loaded.")
            continue

        if not length_configs:
            continue

        model_output_dir = os.path.join(args.output_dir, sanitized_model_name)
        os.makedirs(model_output_dir, exist_ok=True)

        print(f"\nProcessing model: {model_name}")
        for i, length_config in enumerate(length_configs):
            print(f"  Generating dataset for config: {length_config}")

            try:
                requests = generate_dataset(args, tokenizer, length_config)

                input_len = length_config["input_len"]
                output_len = length_config["output_len"]
                prefix_len = length_config.get("prefix_len", 0)

                filename = f"inlen{input_len}_outlen{output_len}_prefixlen{prefix_len}.json"
                output_path = os.path.join(model_output_dir, filename)

                save_requests(requests, output_path)
            except Exception as e:
                print(
                    f"    Failed to generate dataset for config {length_config}: {e}"
                )




if __name__ == "__main__":
    main()
