import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType
import logging
import os
import datetime

# Configure logging
OUTPUT_DIR = './doc_compression_results'
os.makedirs(OUTPUT_DIR, exist_ok=True)
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
logging.basicConfig(
    filename=os.path.join(OUTPUT_DIR, f'output_{timestamp}.log'),
    level=logging.INFO,
    format='%(message)s'
)

# Determine device - now including MPS support
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
logging.info(f"Using device: {device}")

def prepare_document():
    """Prepare the example document split into first and second half."""
    document = """Cheese is a type of dairy product produced in a range of flavors, textures, and forms by coagulation of the milk protein casein. It comprises proteins and fat from milk (usually the milk of cows, buffalo, goats or sheep). During production, milk is usually acidified and either the enzymes of rennet or bacterial enzymes with similar activity are added to cause the casein to coagulate. The solid curds are then separated from the liquid whey and pressed into finished cheese. Some cheeses have aromatic molds on the rind, the outer layer, or throughout."""
    
    # Split document in half (rough approximation)
    split_idx = len(document) // 2
    first_half = document[:split_idx]
    second_half = document[split_idx:]
    
    return first_half, second_half

def setup_model_and_lora():
    """Initialize model and configure LoRA."""
    model_name = "meta-llama/Llama-2-7b-hf"
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Set model dtype based on device
    if device.type == "cuda":
        dtype = torch.float16
    else:
        dtype = torch.float32  # MPS and CPU work better with float32
    
    # Initialize model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto"
    )
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model, tokenizer

def prepare_inputs(tokenizer, second_half):
    """
    Prepare input and target tensors for training.
    Both input_ids and labels will be same length, with labels shifted one position
    """
    # Tokenize second half
    second_half_tokens = tokenizer(second_half, return_tensors="pt", padding=True, truncation=True)
    second_half_ids = second_half_tokens["input_ids"]
    
    # Create input sequence: start token + all but last token of second half
    start_token = tokenizer("<s>", return_tensors="pt")["input_ids"]
    input_sequence = torch.cat([start_token, second_half_ids[:, :-1]], dim=1).to(device)
    
    # Create labels: pad token + second half tokens (except last)
    # -100 is the ignore_index for loss calculation
    labels = torch.cat([torch.full_like(start_token, -100), second_half_ids[:, :-1]], dim=1).to(device)
    
    # Create attention mask
    attention_mask = torch.ones_like(input_sequence).to(device)
    
    # Log shapes and content for debugging
    logging.info(f"Input shape: {input_sequence.shape}, Labels shape: {labels.shape}")
    logging.info(f"Input text: {tokenizer.decode(input_sequence[0])}")
    logging.info(f"Labels text (excluding padding): {tokenizer.decode(second_half_ids[0, :-1])}")
    
    return {
        "input_ids": input_sequence,
        "attention_mask": attention_mask,
        "labels": labels
    }

def train_lora(model, tokenizer, second_half, num_epochs=50):
    """Train LoRA to generate second half without first half context."""
    model.train()
    
    # Prepare inputs and targets
    tensors = prepare_inputs(tokenizer, second_half)
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    try:
        # Training loop
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(
                input_ids=tensors["input_ids"],
                attention_mask=tensors["attention_mask"],
                labels=tensors["labels"]
            )
            
            loss = outputs.loss
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            # Log progress every 5 epochs
            if (epoch + 1) % 5 == 0:
                logging.info(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}")
                
                # Generate sample during training to monitor progress
                with torch.no_grad():
                    generated = model.generate(
                        input_ids=tensors["input_ids"][:, :1],  # Only use start token for generation
                        max_new_tokens=tensors["labels"].size(1),
                        temperature=0.7,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        do_sample=True,
                        top_k=50
                    )
                    generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
                    logging.info(f"Sample generation:\n{generated_text}\n")
                    logging.info(f"Target text:\n{second_half}\n")
    
    except Exception as e:
        logging.error(f"Error during training: {str(e)}")
        raise e

def evaluate_generation(model, tokenizer, first_half, second_half, num_samples=5):
    """Evaluate the trained model's ability to generate the second half."""
    model.eval()
    
    logging.info("\nEvaluation:")
    logging.info(f"Original second half:\n{second_half}\n")
    
    # Prepare input tokens
    start_tokens = tokenizer("<s>", return_tensors="pt").to(device)
    target_length = len(tokenizer(second_half)["input_ids"][0])
    
    try:
        with torch.no_grad():
            for i in range(num_samples):
                generated = model.generate(
                    input_ids=start_tokens["input_ids"],
                    max_new_tokens=target_length,
                    temperature=0.7,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    do_sample=True,
                    top_k=50
                )
                generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
                logging.info(f"Generated sample {i+1}:\n{generated_text}\n")
    
    except Exception as e:
        logging.error(f"Error during evaluation: {str(e)}")
        raise e

def main():
    try:
        # 1. Prepare document
        first_half, second_half = prepare_document()
        logging.info(f"First half:\n{first_half}\n")
        logging.info(f"Second half:\n{second_half}\n")
        
        # 2. Setup model and LoRA
        model, tokenizer = setup_model_and_lora()
        
        # 3. Train LoRA
        train_lora(model, tokenizer, second_half)
        
        # 4. Evaluate results
        evaluate_generation(model, tokenizer, first_half, second_half)
    
    except Exception as e:
        logging.error(f"Error in main: {str(e)}")
        raise e

if __name__ == "__main__":
    main()