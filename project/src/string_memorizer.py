"""
string_memorizer.py - Script to test the memorization capacity of GPT-2 using LoRA adapters.

This script can perform the following:
1. Run experiments to train GPT-2 models modified with LoRA to memorize random strings of varying lengths.
2. Analyze the results from a previous run and generate plots.

Usage:
- To run experiments with default settings:
    python string_memorizer.py
- To run experiments with custom settings:
    python src/string_memorizer.py [options]
- Hyperparameter sweep
    python src/string_memorizer.py --num_batches 10000 --string_lengths 50 100 --models gpt2 gpt2-medium --layer_stride 2 --increase_string_length --string_length_increment 5 --target_module_types mlp.c_fc attn.c_attn

  Options:
    --models               List of GPT-2 models to test (default: ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'])
    --string_lengths       List of string lengths to test (default: [10, 20, 30])
    --num_batches          Number of batches to train for (default: 1000)
    --lora_ranks           List of LoRA ranks to test (default: [1, 2, 4, 8, 16, 32])
    --batch_size           Batch size for training (default: 1)
    --num_samples          Number of samples to generate during evaluation (default: 100)
    --target_layers        List of transformer layer indices to apply LoRA to (default: all layers)
    --target_module_types  List of module types to apply LoRA to (choices: 'attn.c_attn', 'attn.c_proj', 'mlp.c_fc', 'mlp.c_proj')

- To analyze results:
    python src/string_memorizer.py --analyze
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

# At the top of the file, after imports
OUTPUT_DIR = './string_memo_results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Generate timestamp
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# Create filenames with timestamp
csv_file = f'results_{timestamp}.csv'
log_file = f'output_{timestamp}.log'

# Configure logging to write outputs to a file
logging.basicConfig(
    filename=os.path.join(OUTPUT_DIR, log_file),   # Output file name with timestamp
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

    The string includes ASCII letters, digits, punctuation (excluding '|', newlines, quotation marks, and backticks), and spaces.

    Args:
        length (int): The desired length of the random string.

    Returns:
        str: Randomly generated string without forbidden characters.
    """
    # Define the set of characters to use, excluding '|', newlines, quotes, and backtick '`'
    chars = (
        string.ascii_letters +
        string.digits +
        string.punctuation.replace('|', '').replace('\n', '').replace('\r', '').replace('"', '').replace("'", '').replace('`', '') +
        ' '
    )
    random_string = ''.join(random.choice(chars) for _ in range(length))
    # Ensure that no forbidden characters are included
    forbidden_chars = ['|', '\n', '\r', '"', "'", '`']
    assert not any(fc in random_string for fc in forbidden_chars), "Random string contains forbidden characters"
    return random_string

def sanitize_string(s):
    """
    Sanitize the string by escaping or removing problematic characters.

    Args:
        s (str): The string to sanitize.

    Returns:
        str: Sanitized string.
    """
    # Replace newlines with literal '\\n' and carriage returns with '\\r'
    s = s.replace('\n', '\\n').replace('\r', '\\r')
    # Escape quotes and backticks
    s = s.replace('"', '\\"').replace("'", "\\'").replace('`', '\\`')
    return s

def evaluate_model(model, tokenizer, input_ids, attention_mask, target_string, num_samples, batch_size_eval=32):
    """
    Evaluate the model's ability to generate the target string.

    Args:
        model (PreTrainedModel): The trained model to evaluate.
        tokenizer (PreTrainedTokenizer): The tokenizer used for decoding outputs.
        input_ids (Tensor): The input token IDs.
        attention_mask (Tensor): The attention mask for the input.
        target_string (str): The target string to compare against.
        num_samples (int): Number of samples to generate for evaluation.
        batch_size_eval (int): Batch size to use during evaluation.

    Returns:
        Tuple[float, List[str]]: A tuple containing the success rate (as a percentage)
        and a list of the first 10 generated samples.
    """
    model.eval()
    with torch.no_grad():
        # Prepare batched inputs for generation
        eval_input_ids = input_ids[:, :1].repeat(num_samples, 1)
        eval_attention_mask = attention_mask[:, :1].repeat(num_samples, 1)

        # Generate samples in batches
        all_generated_texts = []
        for i in range(0, num_samples, batch_size_eval):
            batch_input_ids = eval_input_ids[i:i+batch_size_eval]
            batch_attention_mask = eval_attention_mask[i:i+batch_size_eval]

            generated_ids = model.generate(
                input_ids=batch_input_ids,
                max_new_tokens=input_ids.size(1) - 1,
                attention_mask=batch_attention_mask,
                do_sample=True,
                top_k=50,
                temperature=1.0
            )
            batch_generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            batch_generated_texts = [text.strip() for text in batch_generated_texts]
            all_generated_texts.extend(batch_generated_texts)

        # Sanitize generated texts
        sanitized_texts = [sanitize_string(text) for text in all_generated_texts]

        # Evaluate success rate
        success_count = sum(1 for text in all_generated_texts if text == target_string)

        # Collect the first 10 samples for logging
        generated_samples = sanitized_texts[:10]

        # Calculate the success rate
        success_fraction = success_count / num_samples
        success_rate_percent = success_fraction * 100

    return success_rate_percent, generated_samples

def create_bias_adapter(model, target_modules):
    """
    Create a custom bias-only adapter by freezing all parameters except biases in target modules.
    
    Args:
        model: The PyTorch model
        target_modules: List of module names to apply bias training to
    """
    # First freeze all parameters
    for param in model.parameters():
        param.requires_grad = False
    
    # Then unfreeze only the bias parameters in target modules
    for name, module in model.named_modules():
        if any(target in name for target in target_modules):
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias.requires_grad = True
                print(f"Enabled bias training for: {name}")
    
    return model

def run_experiments(model_sizes, string_lengths, num_batches, lora_ranks,
                    batch_size, num_samples, target_layers, target_module_types,
                    random_init, increase_string_length, string_length_increment,
                    csv_file):
    """
    Run the experiments to train models and record results.

    Args:
        model_sizes (list): List of GPT-2 model sizes to test.
        string_lengths (list): List of string lengths to test.
        num_batches (int): Number of batches to train for.
        lora_ranks (list): List of LoRA ranks to test.
        batch_size (int): Batch size for training.
        num_samples (int): Number of samples to generate during evaluation.
        target_layers (list): List of transformer layer indices to apply LoRA to.
        target_module_types (list): List of module types to apply LoRA to.
        random_init (bool): Whether to initialize models with random weights.
        increase_string_length (bool): Whether to increase string length incrementally.
        string_length_increment (int): Increment value for increasing string length.
        csv_file (str): Filename of the CSV file to save results.

    """
    # Check if CSV file exists; if not, create it and write the header
    if not os.path.exists(csv_file):
        with open(csv_file, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file, quoting=csv.QUOTE_MINIMAL)
            writer.writerow([
                'ModelSize',
                'StringLength',
                'NumBatches',
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

        # Determine the list of target layers
        if target_layers:
            target_layers_list = target_layers
        else:
            # Apply to all layers if target_layers is empty
            # For GPT-2, layers are named from 0 to n_layers - 1
            if random_init:
                config = GPT2Config.from_pretrained(model_size)
            else:
                config = GPT2LMHeadModel.from_pretrained(model_size).config
            total_layers = config.n_layer
            target_layers_list = list(range(total_layers))

        for lora_rank in lora_ranks:
            logging.info(f"\n  Testing LoRA Rank: {lora_rank}")

            for module_type in target_module_types:
                # Convert module_type to string if it's a list
                if isinstance(module_type, list):
                    module_type_str = '.'.join(module_type)
                else:
                    module_type_str = module_type

                logging.info(f"\n  Testing Module Type: {module_type_str}")

                for layer_idx in target_layers_list:
                    logging.info(f"\n    Testing Layer Index: {layer_idx}")

                    # Re-instantiate the base model for each layer_idx to avoid multiple adapters
                    if random_init:
                        # Initialize model with random weights
                        config = GPT2Config.from_pretrained(model_size)
                        model = GPT2LMHeadModel(config).to(device)
                        logging.info(f"          Re-initialized {model_size} with random weights for layer {layer_idx}")
                    else:
                        # Load pre-trained model
                        model = GPT2LMHeadModel.from_pretrained(model_size).to(device)

                    model.train()

                    # Set pad_token_id and eos_token_id
                    model.config.pad_token_id = tokenizer.eos_token_id
                    model.config.eos_token_id = tokenizer.eos_token_id

                    # Construct the module name relative to the base module
                    module_pattern = f"transformer.h.{layer_idx}.{module_type_str}"

                    # Use the exact module name as the target
                    target_modules = [module_pattern]

                    logging.info(f"          Applying adapter to modules: {target_modules}")

                    # Check if lora_rank is zero
                    if lora_rank == 0:
                        # Use custom bias-only training
                        model = create_bias_adapter(model, target_modules)
                        peft_model = model  # No need for peft wrapper
                        logging.info(f"          Using bias-only fine-tuning for modules: {target_modules}")
                    else:
                        # Configure LoRA with the current rank and exact module
                        peft_config = LoraConfig(
                            r=lora_rank,
                            lora_alpha=16,
                            target_modules=target_modules,
                            lora_dropout=0.05,
                            bias="none",
                            task_type=TaskType.CAUSAL_LM
                        )
                        peft_model = get_peft_model(model, peft_config).to(device)
                        logging.info(f"          Using LoRA with rank {lora_rank} for modules: {target_modules}")

                    # Apply the PEFT model with the appropriate configuration
                    peft_model.train()

                    # Initialize optimizer
                    optimizer = torch.optim.AdamW(peft_model.parameters(), lr=1e-4)

                    # Determine the range of string lengths
                    if increase_string_length:
                        start_length = min(string_lengths)
                        end_length = max(string_lengths) + 1  # Include max length in range
                        string_length_list = list(range(start_length, end_length, string_length_increment))
                    else:
                        string_length_list = string_lengths  # Use provided list of string lengths

                    stop_increasing_length = False  # Reset flag at start of string lengths
                    
                    for char_length in string_length_list:
                        if stop_increasing_length:
                            break  # Skip remaining string lengths if we failed to achieve target accuracy
                            
                        logging.info(f"\n      Testing String Length: {char_length}")
                        achieved_20_percent_training = False  # Reset flag at start of training

                        random_string = generate_random_string(char_length)
                        processed_random_string = sanitize_string(random_string)
                        logging.info(f"        Random String: {processed_random_string}")
                        inputs = tokenizer(random_string, return_tensors="pt")
                        input_ids = inputs['input_ids'].to(device)
                        attention_mask = inputs['attention_mask'].to(device)  # Move attention_mask to device
                        labels = input_ids.clone()  # Labels are the same as input_ids for language modeling

                        # Create TensorDataset and DataLoader
                        dataset = data.TensorDataset(input_ids, attention_mask, labels)  # Include attention_mask
                        dataloader = data.DataLoader(dataset, batch_size=batch_size)

                        # Training loop
                        for batch_idx in range(num_batches):
                            for batch_input_ids, batch_attention_mask, batch_labels in dataloader:
                                # Move tensors to device if not already moved
                                batch_input_ids = batch_input_ids.to(device)
                                batch_attention_mask = batch_attention_mask.to(device)
                                batch_labels = batch_labels.to(device)

                                outputs = peft_model(
                                    input_ids=batch_input_ids,
                                    attention_mask=batch_attention_mask,
                                    labels=batch_labels
                                )
                                loss = outputs.loss
                                loss.backward()
                                optimizer.step()
                                optimizer.zero_grad()
                            # Evaluate and log accuracy at regular intervals
                            if (batch_idx + 1) % max(1, num_batches // 10) == 0 or batch_idx == 0:
                                current_loss = loss.item()

                                # Determine the number of samples for training evaluation
                                evaluation_samples = 10
                                if num_batches >= 1000:
                                    evaluation_samples = 100

                                # Evaluation during training using the evaluate_model function
                                training_accuracy, training_samples = evaluate_model(
                                    model=peft_model,
                                    tokenizer=tokenizer,
                                    input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    target_string=random_string,
                                    num_samples=evaluation_samples,
                                    batch_size_eval=32  # You can adjust this as needed
                                )
                                # Use the first sample for logging
                                generated_sample_text = training_samples[0] if training_samples else ''

                                # Logging all info on the same line
                                logging.info(f"          Batch {batch_idx+1}/{num_batches}, Loss: {current_loss:.4f}, Training Accuracy: {training_accuracy:.2f}%, Sample Output: {generated_sample_text}")

                                # Early stopping if 20% training accuracy was achieved
                                if training_accuracy >= 20.0:
                                    logging.info(f"          Achieved 20% training accuracy at batch {batch_idx+1}")
                                    achieved_20_percent_training = True
                                    break

                        # Final evaluation and logging
                        success_rate_percent, generated_samples = evaluate_model(
                            model=peft_model,
                            tokenizer=tokenizer,
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            target_string=random_string,
                            num_samples=num_samples,
                            batch_size_eval=32  # You can adjust this as needed
                        )
                        logging.info(f"          Success rate: {success_rate_percent:.2f}%")
                        # Log the 10 generated samples on one line
                        samples_for_log = ' | '.join(generated_samples)
                        logging.info(f"          Sampled outputs (first 10 samples): {samples_for_log}")

                        # Process generated_samples for CSV
                        processed_generated_samples = []
                        for sample in generated_samples:
                            if '|' in sample:
                                processed_generated_samples.append('error')
                            else:
                                processed_generated_samples.append(sample)

                        # Join the processed generated samples into a single string (separated by '|')
                        samples_str = ' | '.join(processed_generated_samples)

                        # Save the results to the CSV file
                        with open(csv_file, mode='a', newline='', encoding='utf-8') as file:
                            writer = csv.writer(file, quoting=csv.QUOTE_MINIMAL)
                            writer.writerow([
                                model_size,
                                char_length,
                                num_batches,
                                lora_rank,
                                str(layer_idx),
                                module_type_str,
                                f"{success_rate_percent:.2f}",
                                processed_random_string,
                                samples_str
                            ])

                        # Only check for 20% threshold if we didn't achieve early stopping
                        if increase_string_length and not achieved_20_percent_training and success_rate_percent < 20.0:
                            logging.info(f"          Achieved less than 20% success rate after full training; stopping further attempts with longer strings.")
                            stop_increasing_length = True
                            break  # Exit the string length loop

                        # Save the learned bias vector(s) after training
                        if lora_rank == 0:
                            # We need to save the bias vector from the targeted module
                            bias_vectors = {}
                            for module_name in target_modules:
                                # Access the module
                                module = dict(model.named_modules())[module_name]
                                if hasattr(module, 'bias') and module.bias is not None:
                                    bias_vector = module.bias.detach().cpu()
                                    bias_vectors[module_name] = bias_vector
                                    # Create a filename for the bias vector
                                    bias_filename = f"bias_{model_size}_layer{layer_idx}_{module_type_str.replace('.', '_')}.pt"
                                    bias_filepath = os.path.join(OUTPUT_DIR, bias_filename)
                                    # Save the bias vector
                                    torch.save(bias_vector, bias_filepath)
                                    logging.info(f"          Saved bias vector to {bias_filepath}")
                        else:
                            # Optionally, save the LoRA weights
                            pass  # You can implement similar saving for LoRA weights if desired

                        # Clean up to free memory after processing all string lengths for current layer_idx
                        del peft_model
                        torch.cuda.empty_cache()

                # Clean up after all layers are processed for current module_type
                del model
                torch.cuda.empty_cache()

    logging.info(f"\nAll experiments completed. Results have been saved to '{csv_file}'.")

def analyze_results(analysis_mode='default', csv_file='results.csv'):
    """
    Analyze the results from the CSV file and generate plots.

    Args:
        analysis_mode (str): The analysis mode to use ('default' or 'max_string_length').
        csv_file (str): The path to the CSV file containing the results.
    """
    # Check if the CSV file exists
    if not os.path.exists(csv_file):
        print(f"The file '{csv_file}' does not exist. Please run experiments first.")
        return

    # Extract timestamp from csv_file name (assuming format 'results_YYYYMMDD_HHMMSS.csv')
    timestamp = os.path.basename(csv_file).replace('results_', '').replace('.csv', '')

    # Load the data into a pandas DataFrame
    df = pd.read_csv(csv_file)

    # Convert columns to appropriate data types
    df['StringLength'] = df['StringLength'].astype(int)
    df['LoRARank'] = df['LoRARank'].astype(int)
    df['SuccessRate'] = df['SuccessRate'].astype(float)
    df['ModelSize'] = df['ModelSize'].astype(str)
    df['TargetLayers'] = df['TargetLayers'].astype(str)
    df['TargetModuleTypes'] = df['TargetModuleTypes'].astype(str)

    if analysis_mode == 'default':
        # Default analysis plots
        plt.figure(figsize=(12, 8))
        sns.boxplot(data=df, x='StringLength', y='SuccessRate')
        plt.title('Success Rate Distribution by String Length')
        plot_path = os.path.join(OUTPUT_DIR, f'success_rate_by_length_{timestamp}.png')
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()
        print(f"Default analysis plot saved as '{plot_path}'")

    elif analysis_mode == 'max_string_length':
        # Filter data for desired success rate
        threshold = 95.0
        df_threshold = df[df['SuccessRate'] >= threshold]

        # Exclude rows where TargetLayers is 'All' or NaN
        df_threshold = df_threshold[~df_threshold['TargetLayers'].isin(['All', 'NaN'])]

        # Split 'TargetLayers' by commas and explode
        df_threshold['TargetLayers'] = df_threshold['TargetLayers'].str.split(',')
        df_threshold = df_threshold.explode('TargetLayers')

        # Strip whitespace and convert 'TargetLayers' to integers
        df_threshold['TargetLayers'] = df_threshold['TargetLayers'].str.strip()
        df_threshold['TargetLayers'] = df_threshold['TargetLayers'].astype(int)

        # Group by ModelSize, TargetModuleTypes, TargetLayers, and find the maximum StringLength
        grouped = df_threshold.groupby(['ModelSize', 'TargetModuleTypes', 'TargetLayers'])['StringLength'].max().reset_index()

        # Create a pivot table for plotting
        pivot_table = grouped.pivot_table(
            index='TargetLayers',
            columns=['ModelSize', 'TargetModuleTypes'],
            values='StringLength',
            fill_value=0
        )

        # Sort the TargetLayers
        pivot_table = pivot_table.sort_index()

        # Plotting
        plt.figure(figsize=(12, 8))
        for (model_size, target_module), data_series in pivot_table.items():
            plt.plot(
                data_series.index,
                data_series.values,
                marker='o',
                label=f"{model_size}, {target_module}"
            )

        plt.xlabel('Layer Index')
        plt.ylabel('Maximum String Length')
        plt.title(f"Maximum String Length per Layer Index\n(Success Rate >= {threshold}%)")
        plt.legend(title='Model, Target Module', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        plot_path = os.path.join(OUTPUT_DIR, f'max_string_length_per_layer_{timestamp}.png')
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()
        print(f"Analysis complete. Plot saved as '{plot_path}'")

    else:
        print(f"Unknown analysis mode: '{analysis_mode}'. Available modes: 'default', 'max_string_length'")

def load_bias_vector_into_model(model, bias_filepath, module_name):
    """
    Load a bias vector from a file and insert it into the specified module of the model.

    Args:
        model: The GPT-2 model instance.
        bias_filepath (str): Path to the saved bias vector file.
        module_name (str): Name of the module to insert the bias into.
    """
    # Load the bias vector
    bias_vector = torch.load(bias_filepath).to(model.device)

    # Access the module in the model
    module = dict(model.named_modules())[module_name]

    # Check if the module has a bias attribute
    if hasattr(module, 'bias') and module.bias is not None:
        # Replace the module's bias with the loaded bias vector
        module.bias.data = bias_vector.clone()
        print(f"Loaded bias vector into module: {module_name}")
    else:
        print(f"Module {module_name} does not have a bias attribute.")

    return model

def plot_bias_vector(bias_vector, title='Bias Vector Visualization'):
    """
    Plot the bias vector.

    Args:
        bias_vector (torch.Tensor or numpy.ndarray): The bias vector to plot.
        title (str): Title of the plot.
    """
    if isinstance(bias_vector, torch.Tensor):
        bias_vector = bias_vector.cpu().numpy()
    elif isinstance(bias_vector, np.ndarray):
        pass
    else:
        raise TypeError("bias_vector must be a torch.Tensor or numpy.ndarray")

    # Create the plot
    plt.figure(figsize=(12, 6))
    plt.plot(bias_vector, marker='o', linestyle='-')
    plt.title(title)
    plt.xlabel('Dimension Index')
    plt.ylabel('Bias Value')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Generate timestamped filenames in the output directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    default_csv_file = os.path.join(OUTPUT_DIR, f'results_{timestamp}.csv')
    default_log_file = os.path.join(OUTPUT_DIR, f'output_{timestamp}.log')

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='String Memorization Experiment')
    parser.add_argument('--analyze', action='store_true', help='Analyze the results and generate plots')
    parser.add_argument('--analysis_mode', type=str, default='default',
                        help="Analysis mode to use ('default' or 'max_string_length')")
    parser.add_argument('--csv_file', type=str, default=default_csv_file, help='CSV file to save or read results')
    parser.add_argument('--log_file', type=str, default=default_log_file, help='Log file to save outputs')
    parser.add_argument('--models', nargs='+', default=['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'],
                        help='List of GPT-2 models to test')
    parser.add_argument('--string_lengths', nargs='+', type=int, default=[10, 20, 30],
                        help='List of string lengths to test')
    parser.add_argument('--num_batches', type=int, default=1000, help='Number of batches to train for')
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
                        help='Increase string length incrementally until 98% accuracy is achieved')
    parser.add_argument('--string_length_increment', type=int, default=10,
                        help='Increment by which to increase the string length when --increase_string_length is set (default: 10)')
    parser.add_argument('--layer_stride', type=int, help='Generate target layers as multiples of this number (e.g., 0, n, 2n, ...)')
    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        filename=args.log_file,
        filemode='w',
        level=logging.INFO,
        format='%(message)s'
    )

    # Parse nested module paths
    if args.target_module_types:
        # Each module type is a dot-separated string; split into a list
        target_module_types = [module_path.strip().split('.') for module_path in args.target_module_types]
    else:
        target_module_types = [['attn', 'c_attn']]  # Default to 'attn.c_attn'

    # Generate target layers if stride is specified
    if args.layer_stride is not None:
        if args.target_layers:
            print("Warning: --layer_stride overrides --target_layers")
        
        # Get number of layers from model config
        config = GPT2Config.from_pretrained(args.models[0])
        n_layers = config.n_layer
        
        # Generate layers: 0, n, 2n, ... while k*n < n_layers
        stride = args.layer_stride
        args.target_layers = list(range(0, n_layers, stride))
        print(f"Generated target layers with stride {stride}: {args.target_layers}")

    if args.analyze:
        analyze_results(analysis_mode=args.analysis_mode, csv_file=args.csv_file)
    else:
        run_experiments(
            model_sizes=args.models,
            string_lengths=args.string_lengths,
            num_batches=args.num_batches,
            lora_ranks=args.lora_ranks,
            batch_size=args.batch_size,
            num_samples=args.num_samples,
            target_layers=args.target_layers,
            target_module_types=target_module_types,
            random_init=args.random_init,
            increase_string_length=args.increase_string_length,
            string_length_increment=args.string_length_increment,
            csv_file=args.csv_file
        )