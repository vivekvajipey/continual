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
    """Prepare the example document."""
    document = """Cheese is a type of dairy product produced in a range of flavors, textures, and forms by coagulation of the milk protein casein. It comprises proteins and fat from milk (usually the milk of cows, buffalo, goats or sheep). During production, milk is usually acidified and either the enzymes of rennet or bacterial enzymes with similar activity are added to cause the casein to coagulate. The solid curds are then separated from the liquid whey and pressed into finished cheese. Some cheeses have aromatic molds on the rind, the outer layer, or throughout."""
    
    # Use the entire document as both input and target
    input_text = document
    target_text = document
    
    return input_text, target_text


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

def prepare_inputs(tokenizer, text):
    """
    Prepare input and target tensors for training.
    Input is the entire text except the last token.
    Labels are the entire text except the first token.
    """
    # Tokenize the text
    tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    input_ids_full = tokens["input_ids"].to(device)          # Shape: [1, 138]
    attention_mask_full = tokens["attention_mask"].to(device)  # Shape: [1, 138]
    
    # Shift input_ids and labels
    input_ids = input_ids_full[:, :-1]  # Shape: [1, 137]
    labels = input_ids_full[:, 1:].clone()  # Shape: [1, 137]
    
    # Replace pad token id's in labels by -100 so they are ignored by the loss
    labels[labels == tokenizer.pad_token_id] = -100
    
    # Log shapes and content for debugging
    logging.info(f"Input shape: {input_ids.shape}, Labels shape: {labels.shape}")
    logging.info(f"Input text: {tokenizer.decode(input_ids[0], skip_special_tokens=True)}")
    logging.info(f"Labels text (excluding padding): {tokenizer.decode(labels[0], skip_special_tokens=True)}")
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask_full[:, :-1],
        "labels": labels
    }

def train_lora(model, tokenizer, text, num_epochs=50, early_stopping_threshold=0.99):
    """Train LoRA to memorize and reproduce the input text with early stopping."""
    model.train()
    
    # Prepare inputs and targets
    tensors = prepare_inputs(tokenizer, text)
    
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
                        input_ids=tensors["input_ids"],  # Input sequence excluding last token
                        attention_mask=tensors["attention_mask"],
                        max_new_tokens=len(tensors["labels"][0]),  # Generate tokens equal to labels length
                        temperature=1.0,  # Greedy decoding
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        do_sample=False  # Deterministic generation
                    )
                    # Concatenate input and generated tokens
                    full_generated_ids = torch.cat([tensors["input_ids"][0], generated[0]], dim=0)
                    full_generated_text = tokenizer.decode(full_generated_ids, skip_special_tokens=True)
                    logging.info(f"Sample generation:\n{full_generated_text}\n")
                    logging.info(f"Target text:\n{text}\n")
                    
                    # Check if the generated text matches the target text
                    accuracy = full_generated_text == text
                    if accuracy:
                        logging.info(f"Exact match achieved at epoch {epoch + 1}. Stopping training.")
                        break
    
    except Exception as e:
        logging.error(f"Error during training: {str(e)}")
        raise e

def evaluate_generation(model, tokenizer, input_text, target_text, num_samples=5):
    """Evaluate the trained model's ability to generate the target text."""
    model.eval()
    
    logging.info("\nEvaluation:")
    logging.info(f"Target text:\n{target_text}\n")
    
    # Prepare input tokens (entire input sequence excluding last token)
    tensors = prepare_inputs(tokenizer, input_text)
    input_ids = tensors["input_ids"].to(device)            # Shape: [1, 137]
    attention_mask = tensors["attention_mask"].to(device)  # Shape: [1, 137]
    
    target_length = tensors["labels"].shape[1]  # Number of tokens to generate (137)
    
    try:
        with torch.no_grad():
            for i in range(num_samples):
                generated = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=target_length,  # Generate tokens equal to labels length
                    temperature=1.0,  # Greedy decoding
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    do_sample=False  # Deterministic generation
                )
                # Concatenate input and generated tokens
                full_generated_ids = torch.cat([input_ids[0], generated[0]], dim=0)
                generated_text = tokenizer.decode(full_generated_ids, skip_special_tokens=True)
                logging.info(f"Generated sample {i+1}:\n{generated_text}\n")
    
    except Exception as e:
        logging.error(f"Error during evaluation: {str(e)}")
        raise e

def save_model(model, tokenizer, save_dir):
    """Save the trained model and tokenizer."""
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    logging.info(f"Model and tokenizer saved to {save_dir}")

def main():
    try:
        logging.info("=== Starting the Fine-Tuning Process ===")
        
        # 1. Prepare document
        input_text, target_text = prepare_document()
        logging.info("=== Document Prepared ===")
        logging.info(f"Input text:\n{input_text}\n")
        logging.info(f"Target text:\n{target_text}\n")
        
        # 2. Setup model and LoRA
        model, tokenizer = setup_model_and_lora()
        logging.info("=== Model and LoRA Setup Complete ===")
        
        # 3. Train LoRA
        logging.info("=== Starting Training ===")
        train_lora(model, tokenizer, input_text)
        logging.info("=== Training Completed ===")
        
        # 4. Evaluate results
        logging.info("=== Starting Evaluation ===")
        evaluate_generation(model, tokenizer, input_text, target_text)
        logging.info("=== Evaluation Completed ===")
        
        # 5. Save the trained model
        logging.info("=== Saving the Trained Model ===")
        save_dir = os.path.join(OUTPUT_DIR, f"trained_model_{timestamp}")
        save_model(model, tokenizer, save_dir)
        logging.info("=== Model Saved Successfully ===")
    
    except Exception as e:
        logging.error(f"Error in main: {str(e)}")
        raise e

if __name__ == "__main__":
    main()
