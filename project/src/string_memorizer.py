"""
string_memorizer.py - Script to test the memorization capacity of GPT-2 using LoRA adapters.

This script can perform the following:
1. Run experiments to train GPT-2 models modified with LoRA to memorize random strings of varying lengths.
2. Analyze the results from a previous run and generate plots.

Usage:
- To run experiments with default settings:
    python string_memorizer.py
- To run experiments with custom settings:
    python string_memorizer.py [options]

  Options:
    --models               List of GPT-2 models to test (default: ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'])
    --string_lengths       List of string lengths to test (default: [10, 20, 30])
    --epochs               List of epoch counts to test (default: [1000])
    --lora_ranks           List of LoRA ranks to test (default: [1, 2, 4, 8, 16, 32])
    --batch_size           Batch size for training (default: 1)
    --num_samples          Number of samples to generate during evaluation (default: 100)
    --target_layers        List of transformer layer indices to apply LoRA to (default: all layers)
    --target_module_types  List of module types to apply LoRA to (choices: 'attn.c_attn', 'attn.c_proj', 'mlp.c_fc', 'mlp.c_proj')

- To analyze results:
    python string_memorizer.py --analyze
"""

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from peft import get_peft_model, LoraConfig, TaskType
import random
import string
import logging
import warnings
import csv
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch.utils.data as data
import numpy as np
import datetime

# Generate timestamp
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# Create filenames with timestamp
csv_file = f'results_{timestamp}.csv'
log_file = f'output_{timestamp}.log'

# Configure logging to write outputs to a file
logging.basicConfig(
    filename=log_file,   # Output file name with timestamp
    filemode='w',            # Overwrite the file each time the script is run
    level=logging.INFO,      # Set the logging level
    format='%(message)s'     # Simplify the log format
)

# Suppress specific warnings to keep the logs clean
warnings.filterwarnings(
    'ignore',
    message='Setting `pad_token_id` to `eos_token_id`:.*'
)
warnings.filterwarnings(
    'ignore',
    message='The attention mask and the pad token id were not set.*'
)

# Set the computation device to GPU if available, else CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

def generate_random_string(length):
    """
    Generate a random string of specified length.

    The string includes ASCII letters, digits, punctuation (excluding '|'), and spaces.

    Args:
        length (int): The desired length of the random string.

    Returns:
        str: Randomly generated string without the '|' character.
    """
    chars = string.ascii_letters + string.digits + string.punctuation.replace('|', '') + ' '
    random_string = ''.join(random.choice(chars) for _ in range(length))
    assert '|' not in random_string, "Random string contains '|' character"
    return random_string

def run_experiments(model_sizes, string_lengths, epochs_list, lora_ranks,
                    batch_size, num_samples, target_layers, target_module_types,
                    random_init, increase_string_length, string_length_increment, csv_file):
    """
    Run the experiments to train models and record results.

    Args:
        model_sizes (list): List of GPT-2 model sizes to test.
        string_lengths (list): List of string lengths to test.
        epochs_list (list): List of epoch counts to test.
        lora_ranks (list): List of LoRA ranks to test.
        batch_size (int): Batch size for training.
        num_samples (int): Number of samples to generate during evaluation.
        target_layers (list): List of transformer layer indices to apply LoRA to.
        target_module_types (list): List of nested module paths (e.g., ['attn', 'c_attn']) to apply LoRA to.
        random_init (bool): Initialize models with random weights instead of pre-trained weights.
        increase_string_length (bool): Increase string length incrementally until 100% accuracy is achieved.
        string_length_increment (int): Increment by which to increase the string length when --increase_string_length is set.
        csv_file (str): CSV file to store the results.
    """
    # CSV file to store the results
    csv_file = 'results.csv'

    # Check if CSV file exists; if not, create it and write the header
    if not os.path.exists(csv_file):
        with open(csv_file, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file, quoting=csv.QUOTE_MINIMAL)
            writer.writerow([
                'ModelSize',
                'StringLength',
                'Epochs',
                'LoRARank',
                'TargetLayers',
                'TargetModuleTypes',
                'SuccessRate',
                'TrueString',
                'GeneratedSamples'
            ])

    # Loop over each combination of hyperparameters
    for model_size in model_sizes:
        logging.info(f"\nTesting GPT-2 Model Size: {model_size}")

        # Load GPT-2 tokenizer for the specific model size
        tokenizer = GPT2Tokenizer.from_pretrained(model_size)
        tokenizer.pad_token = tokenizer.eos_token  # Set pad_token_id to eos_token_id to avoid warnings

        for lora_rank in lora_ranks:
            logging.info(f"\n      Testing LoRA Rank: {lora_rank}")

            for module_type in target_module_types:
                # Convert module_type to string if it's a list
                if isinstance(module_type, list):
                    module_type_str = '.'.join(module_type)
                else:
                    module_type_str = module_type

                logging.info(f"\nTesting Module Type: {module_type_str}")

                for layer_idx in target_layers:
                    logging.info(f"\nTesting Layer Index: {layer_idx}")

                    # Construct target_modules
                    target_modules = [f"transformer.h.{layer_idx}.{module_type_str}"]
                    logging.info(f"  Applying LoRA to modules: {target_modules}")

                    # Initialize string lengths
                    if increase_string_length:
                        current_string_length = min(string_lengths)
                        max_string_length = max(string_lengths)
                    else:
                        string_lengths_to_test = string_lengths.copy()

                    while True:
                        if increase_string_length:
                            char_length = current_string_length
                        else:
                            if string_lengths_to_test:
                                char_length = string_lengths_to_test.pop(0)
                            else:
                                break  # Exit the loop if no more lengths to test

                        logging.info(f"\n  Testing String Length (characters): {char_length}")

                        # Generate a random string for the model to memorize
                        random_string = generate_random_string(char_length)
                        logging.info(f"  Random string to memorize: {random_string}")

                        # Tokenize the random string and determine token length
                        inputs = tokenizer(random_string, return_tensors="pt").to(device)
                        input_ids = inputs["input_ids"]
                        labels = input_ids.clone()
                        token_length = input_ids.shape[1]
                        logging.info(f"  Tokenized input length (tokens): {token_length}")

                        # Create a dataset of repeated input_ids and labels for batching
                        dataset = data.TensorDataset(input_ids.repeat(batch_size, 1), labels.repeat(batch_size, 1))
                        dataloader = data.DataLoader(dataset, batch_size=batch_size)

                        for epochs in epochs_list:
                            logging.info(f"\n    Testing Number of Epochs: {epochs}")

                            # Re-instantiate the base GPT-2 model for each combination
                            if random_init:
                                # Initialize model with random weights
                                config = GPT2Config.from_pretrained(model_size)
                                model = GPT2LMHeadModel(config).to(device)
                                logging.info(f"      Initialized {model_size} with random weights")
                            else:
                                # Load pre-trained model
                                model = GPT2LMHeadModel.from_pretrained(model_size).to(device)
                            model.train()

                            # Set pad_token_id and eos_token_id in the model's configuration to avoid warnings
                            model.config.pad_token_id = tokenizer.eos_token_id
                            model.config.eos_token_id = tokenizer.eos_token_id  # Optional but recommended

                            # Determine the total number of layers in the model
                            total_layers = len(model.transformer.h)

                            # If target_layers is empty, apply LoRA to all layers
                            if not target_layers:
                                target_layers_indices = list(range(total_layers))
                            else:
                                target_layers_indices = target_layers

                            # Construct target_modules based on specified layers and module paths
                            target_modules = []
                            for layer_idx in target_layers_indices:
                                for module_path in target_module_types:
                                    # module_path is a list of module names
                                    if isinstance(module_path, list):
                                        module_pattern = f"transformer.h.{layer_idx}." + ".".join(module_path)
                                    else:
                                        module_pattern = f"transformer.h.{layer_idx}.{module_path}"
                                    target_modules.append(module_pattern)

                            logging.info(f"      Applying LoRA to modules: {target_modules}")

                            # Configure LoRA with the current rank
                            lora_config = LoraConfig(
                                r=lora_rank,
                                lora_alpha=16,
                                target_modules=target_modules,
                                lora_dropout=0.05,
                                bias="none",
                                task_type=TaskType.CAUSAL_LM
                            )

                            # Apply LoRA to the GPT-2 model to create a modified model
                            lora_model = get_peft_model(model, lora_config).to(device)
                            lora_model.train()

                            # Set up the optimizer for training
                            optimizer = torch.optim.AdamW(lora_model.parameters(), lr=1e-4)

                            # Training loop using DataLoader
                            for epoch in range(epochs):
                                for batch_input_ids, batch_labels in dataloader:
                                    outputs = lora_model(input_ids=batch_input_ids, labels=batch_labels)
                                    loss = outputs.loss
                                    loss.backward()
                                    optimizer.step()
                                    optimizer.zero_grad()
                                # Evaluate and log accuracy at regular intervals
                                if (epoch + 1) % max(1, epochs // 10) == 0 or epoch == 0:
                                    logging.info(f"      Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
                                    # Evaluation during training
                                    lora_model.eval()
                                    with torch.no_grad():
                                        exact_match_count = 0
                                        generated_sample_text = ''  # To store one sample output
                                        for idx in range(10):  # Estimate training accuracy using 10 samples
                                            generated_ids = lora_model.generate(
                                                input_ids=input_ids[:, :1],
                                                max_new_tokens=token_length - 1,
                                                attention_mask=inputs['attention_mask'][:, :1],
                                                do_sample=True,          # Enable stochastic sampling
                                                top_k=50,
                                                temperature=1.0
                                            )
                                            generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()
                                            if idx == 0:
                                                generated_sample_text = generated_text  # Store the first sample
                                            if generated_text == random_string:
                                                exact_match_count += 1
                                        # Calculate approximate training accuracy
                                        training_accuracy = (exact_match_count / 10) * 100
                                        logging.info(f"      Training Accuracy: {training_accuracy:.2f}%")
                                        # Log the first generated sample
                                        logging.info(f"      Sample Output: {generated_sample_text}")
                                    lora_model.train()

                            # Evaluation phase
                            lora_model.eval()
                            with torch.no_grad():
                                success_count = 0           # Counter for successful reproductions
                                generated_samples = []      # List to store generated samples

                                # Generate samples to evaluate the model's memorization
                                for i in range(num_samples):
                                    generated_ids = lora_model.generate(
                                        input_ids=input_ids[:, :1],
                                        max_new_tokens=token_length - 1,
                                        attention_mask=inputs['attention_mask'][:, :1],
                                        do_sample=True,
                                        top_k=50,
                                        temperature=1.0
                                    )
                                    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()

                                    # Store up to 10 samples for logging
                                    if i < 10:
                                        generated_samples.append(generated_text)

                                    # Increment success count if the generated text matches the target string
                                    if generated_text == random_string:
                                        success_count += 1

                                # Calculate the success rate for this combination
                                success_fraction = success_count / num_samples
                                success_rate_percent = success_fraction * 100
                                logging.info(f"      Success rate: {success_count}/{num_samples} ({success_rate_percent:.2f}%)")

                                # Log the first 10 generated samples for inspection
                                logging.info("\n      Sampled outputs (first 10 samples):")
                                for idx, sample in enumerate(generated_samples, 1):
                                    logging.info(f"        Sample {idx}:")
                                    logging.info(f"          Generated: {sample}")
                                    logging.info(f"          Target   : {random_string}\n")

                                # Process random_string to escape newline characters
                                processed_random_string = random_string.replace('\n', '\\n')

                                # Process generated_samples to replace strings containing '|' or '\n'
                                processed_generated_samples = []
                                for sample in generated_samples:
                                    # Replace newline characters in the sample
                                    sample = sample.replace('\n', '\\n')
                                    if '|' in sample:
                                        processed_generated_samples.append('error')
                                    else:
                                        processed_generated_samples.append(sample)

                                # Join the processed generated samples into a single string (separated by '|')
                                samples_str = ' | '.join(processed_generated_samples)

                                # Process target layers and modules for CSV
                                processed_target_layers = ','.join(map(str, target_layers)) if target_layers else 'All'
                                processed_target_module_types = ','.join(['.'.join(module_path) if isinstance(module_path, list) else module_path for module_path in target_module_types])

                                # Save the results to the CSV file
                                with open(csv_file, mode='a', newline='', encoding='utf-8') as file:
                                    writer = csv.writer(file, quoting=csv.QUOTE_MINIMAL)
                                    writer.writerow([
                                        model_size,
                                        char_length,
                                        epochs,
                                        lora_rank,
                                        processed_target_layers,
                                        processed_target_module_types,
                                        f"{success_rate_percent:.2f}",
                                        processed_random_string,
                                        samples_str
                                    ])

                            # Clean up to free memory
                            del lora_model
                            del model
                            torch.cuda.empty_cache()

                            # After evaluation
                            if increase_string_length:
                                if success_rate_percent >= 100.0:
                                    logging.info(f"Achieved 100% accuracy at string length {char_length}.")
                                    break  # Stop increasing string length
                                else:
                                    current_string_length += string_length_increment
                                    if current_string_length > max_string_length:
                                        logging.info("Reached maximum string length without achieving 100% accuracy.")
                                        break
                            else:
                                continue  # Proceed to next string length

    # Log a message indicating the completion of the experiment
    logging.info(f"\nExperiment completed. Results have been saved to '{csv_file}'.")

def analyze_results():
    """
    Analyze the results from the CSV file and generate plots.
    """
    # CSV file to read the results from
    csv_file = 'results.csv'

    # Check if the CSV file exists
    if not os.path.exists(csv_file):
        print(f"The file '{csv_file}' does not exist. Please run experiments first.")
        return

    # Load the data into a pandas DataFrame
    df = pd.read_csv(csv_file)

    # Convert columns to appropriate data types
    df['StringLength'] = df['StringLength'].astype(int)
    df['LoRARank'] = df['LoRARank'].astype(int)
    df['SuccessRate'] = df['SuccessRate'].astype(float)
    df['ModelSize'] = df['ModelSize'].astype(str)

    # Get unique string lengths
    string_lengths = sorted(df['StringLength'].unique())

    # Calculate 'n' as the ceiling of the square root of the number of string lengths
    num_lengths = len(string_lengths)
    n = int(np.ceil(np.sqrt(num_lengths)))

    # Set up the plotting style
    sns.set(style='whitegrid')

    # Create subplots in an n x n grid
    fig, axes = plt.subplots(n, n, figsize=(6 * n, 5 * n), sharey=True)

    # Flatten the axes array for easy iteration
    axes = axes.flatten()

    # Iterate over the axes and string lengths
    for ax, length in zip(axes, string_lengths):
        # Subset data for the current string length
        subset = df[df['StringLength'] == length]

        # Pivot the data to have LoRARank on x-axis, SuccessRate on y-axis, and lines for each ModelSize
        pivot_df = subset.pivot_table(index='LoRARank', columns='ModelSize', values='SuccessRate')

        # Plot each model size
        for model_size in df['ModelSize'].unique():
            if model_size in pivot_df.columns:
                ax.plot(
                    pivot_df.index,
                    pivot_df[model_size],
                    marker='o',
                    label=model_size
                )

        ax.set_title(f'String Length: {length} characters')
        ax.set_xlabel('LoRA Rank')
        ax.set_ylabel('Success Rate (%)')
        ax.set_xticks(df['LoRARank'].unique())
        ax.legend(title='Model Size')

    # Hide any unused subplots
    for i in range(len(string_lengths), len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.savefig('analysis_plot.png')
    plt.show()
    print("Analysis complete. Plot saved as 'analysis_plot.png'.")

if __name__ == "__main__":
    # Generate timestamped filenames
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file = f'results_{timestamp}.csv'
    log_file = f'output_{timestamp}.log'

    # Configure logging
    logging.basicConfig(
        filename=log_file,
        filemode='w',
        level=logging.INFO,
        format='%(message)s'
    )

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='String Memorization Experiment')
    parser.add_argument('--analyze', action='store_true', help='Analyze the results and generate plots')
    parser.add_argument('--models', nargs='+', default=['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'],
                        help='List of GPT-2 models to test')
    parser.add_argument('--string_lengths', nargs='+', type=int, default=[10, 20, 30],
                        help='List of string lengths to test')
    parser.add_argument('--epochs', nargs='+', type=int, default=[1000],
                        help='List of epoch counts to test')
    parser.add_argument('--lora_ranks', nargs='+', type=int, default=[1, 2, 4, 8, 16, 32],
                        help='List of LoRA ranks to test')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for training')
    parser.add_argument('--num_samples', type=int, default=100, help='Number of samples to generate during evaluation')
    parser.add_argument('--target_layers', nargs='+', type=int, default=[],
                        help='List of transformer layer indices to apply LoRA to (default: all layers)')
    parser.add_argument('--target_module_types', nargs='+', default=['attn.c_attn'],
                        help="List of module types to apply LoRA to (e.g., 'attn.c_attn')")
    parser.add_argument('--random_init', action='store_true',
                        help='Initialize models with random weights instead of pre-trained weights')
    parser.add_argument('--increase_string_length', action='store_true',
                        help='Increase string length incrementally until 100% accuracy is achieved')
    parser.add_argument('--string_length_increment', type=int, default=10,
                        help='Increment by which to increase the string length when --increase_string_length is set (default: 10)')
    args = parser.parse_args()

    # Parse nested module paths
    if args.target_module_types:
        # Each module type is a dot-separated string; split into a list
        target_module_types = [module_path.strip().split('.') for module_path in args.target_module_types]
    else:
        target_module_types = [['attn', 'c_attn']]  # Default to 'attn.c_attn'

    if args.analyze:
        analyze_results()
    else:
        run_experiments(
            model_sizes=args.models,
            string_lengths=args.string_lengths,
            epochs_list=args.epochs,
            lora_ranks=args.lora_ranks,
            batch_size=args.batch_size,
            num_samples=args.num_samples,
            target_layers=args.target_layers,
            target_module_types=target_module_types,
            random_init=args.random_init,
            increase_string_length=args.increase_string_length,
            string_length_increment=args.string_length_increment,
            csv_file=csv_file
        )