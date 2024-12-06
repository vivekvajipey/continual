"""
string_memorizer.py - Script to test the memorization capacity of GPT-2 using LoRA adapters.

This script can perform the following:
1. Run experiments to train GPT-2 models modified with LoRA to memorize random strings of varying lengths.
2. Analyze the results from a previous run and generate plots.

Usage:
- To run experiments:
    python string_memorizer.py --run_experiments
- To analyze results:
    python string_memorizer.py --analyze
"""

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
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

# Configure logging to write outputs to a file
logging.basicConfig(
    filename='output.log',   # Output file name
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

    The string includes ASCII letters, digits, punctuation, and spaces.

    Args:
        length (int): The desired length of the random string.

    Returns:
        str: Randomly generated string.
    """
    chars = string.ascii_letters + string.digits + string.punctuation + ' '
    return ''.join(random.choice(chars) for _ in range(length))

def run_experiments():
    """
    Run the experiments to train models and record results.
    """
    # Parameters for the experiment
    model_sizes = ['gpt2', 'gpt2-medium']  # List of GPT-2 model sizes to test
    string_lengths = [10, 20, 30]          # List of string lengths (in characters) to test
    epochs = 1000                          # Number of training epochs for each model
    num_samples = 100                      # Number of samples to generate during evaluation
    lora_ranks = [1, 2, 4, 8, 16, 32]      # List of LoRA ranks to test

    # CSV file to store the results
    csv_file = 'results.csv'

    # Check if CSV file exists, if not, create it and write the header
    if not os.path.exists(csv_file):
        with open(csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                'ModelSize',
                'StringLength',
                'LoRARank',
                'SuccessRate',
                'TrueString',
                'GeneratedSamples'
            ])

    # Loop over each GPT-2 model size
    for model_size in model_sizes:
        logging.info(f"\nTesting GPT-2 Model Size: {model_size}")

        # Load GPT-2 tokenizer for the specific model size
        tokenizer = GPT2Tokenizer.from_pretrained(model_size)
        tokenizer.pad_token = tokenizer.eos_token  # Set pad_token_id to eos_token_id to avoid warnings

        # Loop over each string length
        for char_length in string_lengths:
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

            # Loop over each LoRA rank
            for lora_rank in lora_ranks:
                logging.info(f"\n    Testing LoRA Rank: {lora_rank}")

                # Re-instantiate the base GPT-2 model for each LoRA rank to avoid adapter accumulation
                model = GPT2LMHeadModel.from_pretrained(model_size).to(device)
                model.train()

                # Set pad_token_id and eos_token_id in the model's configuration to avoid warnings
                model.config.pad_token_id = tokenizer.eos_token_id
                model.config.eos_token_id = tokenizer.eos_token_id  # Optional but recommended

                # Configure LoRA with the current rank
                lora_config = LoraConfig(
                    r=lora_rank,               # Rank of the LoRA adaptation matrices
                    lora_alpha=16,             # Scaling factor for LoRA weights
                    target_modules=["c_attn"], # Modules to apply LoRA to (attention layers)
                    lora_dropout=0.05,         # Dropout probability for LoRA layers
                    bias="none",               # Not using bias in LoRA layers
                    task_type=TaskType.CAUSAL_LM  # Task type: Causal Language Modeling
                )

                # Apply LoRA to the GPT-2 model to create a modified model
                lora_model = get_peft_model(model, lora_config).to(device)
                lora_model.train()

                # Set up the optimizer for training
                optimizer = torch.optim.AdamW(lora_model.parameters(), lr=1e-4)

                # Calculate logging interval (logging every 1/10th of the total epochs)
                logging_interval = max(1, epochs // 10)

                # Training loop
                for epoch in range(epochs):
                    # Forward pass: Compute predicted outputs by passing inputs to the model
                    outputs = lora_model(input_ids=input_ids, labels=labels)
                    loss = outputs.loss  # Compute the loss
                    loss.backward()      # Backpropagate the gradients
                    optimizer.step()     # Update model parameters
                    optimizer.zero_grad()# Zero the gradients for the next iteration

                    # Log the loss at specified intervals
                    if (epoch + 1) % logging_interval == 0 or epoch == 0:
                        logging.info(f"      Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

                # Evaluation phase
                lora_model.eval()
                with torch.no_grad():
                    success_count = 0           # Counter for successful reproductions
                    generated_samples = []      # List to store generated samples

                    # Generate samples to evaluate the model's memorization
                    for i in range(num_samples):
                        generated_ids = lora_model.generate(
                            input_ids=input_ids[:, :1],      # Start generation from the first token
                            max_new_tokens=token_length - 1, # Generate enough tokens to match the input length
                            attention_mask=inputs['attention_mask'][:, :1],  # Attention mask for the input_ids
                            do_sample=True,                  # Enable stochastic sampling
                            top_k=50,                        # Consider the top K tokens at each step
                            temperature=1.0                  # Sampling temperature for randomness
                        )
                        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()

                        # Store up to 10 samples for logging
                        if i < 10:
                            generated_samples.append(generated_text)

                        # Increment success count if the generated text matches the target string
                        if generated_text == random_string:
                            success_count += 1

                    # Calculate the success rate for this LoRA rank
                    success_fraction = success_count / num_samples
                    success_rate_percent = success_fraction * 100
                    logging.info(f"      Success rate: {success_count}/{num_samples} ({success_rate_percent:.2f}%)")

                    # Log the first 10 generated samples for inspection
                    logging.info("\n      Sampled outputs (first 10 samples):")
                    for idx, sample in enumerate(generated_samples, 1):
                        logging.info(f"        Sample {idx}:")
                        logging.info(f"          Generated: {sample}")
                        logging.info(f"          Target   : {random_string}\n")

                    # Save the results to the CSV file
                    with open(csv_file, mode='a', newline='') as file:
                        writer = csv.writer(file)
                        # Join the generated samples into a single string (separated by '|')
                        samples_str = ' | '.join(generated_samples)
                        writer.writerow([
                            model_size,
                            char_length,
                            lora_rank,
                            f"{success_rate_percent:.2f}",
                            random_string,
                            samples_str
                        ])

                # Clean up to free memory
                del lora_model
                del model                  # Ensure the original model is deleted
                torch.cuda.empty_cache()   # Clear CUDA cache to free up memory

    # Log a message indicating the completion of the experiment
    logging.info("\nExperiment completed. Results have been saved to 'results.csv'.")

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

    # Set up the plotting style
    sns.set(style='whitegrid')

    # Create subplots for each string length
    num_lengths = len(string_lengths)
    fig, axes = plt.subplots(1, num_lengths, figsize=(6 * num_lengths, 5), sharey=True)

    if num_lengths == 1:
        axes = [axes]  # When there's only one subplot, make it iterable

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

    plt.tight_layout()
    plt.savefig('analysis_plot.png')
    plt.show()
    print("Analysis complete. Plot saved as 'analysis_plot.png'.")

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='String Memorization Experiment')
    parser.add_argument('--run_experiments', action='store_true', help='Run the experiments')
    parser.add_argument('--analyze', action='store_true', help='Analyze the results and generate plots')
    args = parser.parse_args()

    if args.run_experiments:
        run_experiments()
    elif args.analyze:
        analyze_results()
    else:
        print("Please specify an action: --run_experiments or --analyze")