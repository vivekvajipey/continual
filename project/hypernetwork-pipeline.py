import glob
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

# ============
# Configuration
# ============
config = {
    "model_name_or_path": "meta-llama/llama-3.2-1B",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "num_epochs": 3,
    "learning_rate": 1e-4,
    "batch_size": 4,            # adjust depending on your GPU
    "max_length": 512,          # max token length per sample (will be truncated/padded)
    "lora_rank": 4,
    "eval_interval": 50,        # steps between qualitative evaluations
    "checkpoint_interval": 200, # checkpoint every 200 steps (adjust as needed)
    "wandb_project": "lora-compression-head",
    "checkpoint_dir": "/scr/tadimeti/checkpoints"  # local directory to store checkpoints
}

# Create checkpoint directory if it doesn't exist
os.makedirs(config["checkpoint_dir"], exist_ok=True)

# Initialize wandb
wandb.init(project=config["wandb_project"], config=config)
cfg = wandb.config

# ============
# Compression Head Definition
# ============
class CompressionHead(nn.Module):
    """
    A simple compression head that takes a vector (e.g. the last hidden state
    of the first half of a document) and outputs a weight delta (LoRA update)
    for a target linear layer.
    
    The target linear layer (e.g. a query projection) has weight shape (out_features, in_features).
    """
    def __init__(self, hidden_dim, target_in_features, target_out_features, rank=8):
        super().__init__()
        self.rank = rank
        self.linear1 = nn.Linear(hidden_dim, rank)
        self.linear2 = nn.Linear(rank, target_in_features * target_out_features)
        self.target_in_features = target_in_features
        self.target_out_features = target_out_features

    def forward(self, hidden):
        # hidden: (batch_size, hidden_dim); here we assume batch_size == 1 for simplicity.
        x = self.linear1(hidden)            # -> (batch_size, rank)
        x = torch.relu(x)
        x = self.linear2(x)                 # -> (batch_size, target_in_features * target_out_features)
        # Reshape to (batch_size, target_out_features, target_in_features)
        delta = x.view(-1, self.target_out_features, self.target_in_features)
        # For batch_size==1, squeeze the batch dimension.
        return delta.squeeze(0)           # -> (target_out_features, target_in_features)

# ============
# Load Model, Tokenizer, and Dataset
# ============
tokenizer = AutoTokenizer.from_pretrained(cfg.model_name_or_path)
if tokenizer.pad_token is None:
    # Option 1: Use the EOS token as the padding token.
    tokenizer.pad_token = tokenizer.eos_token
    # Option 2 (alternative): Add a dedicated pad token.
    # tokenizer.add_special_tokens({'pad_token': '[PAD]'})

model = AutoModelForCausalLM.from_pretrained(cfg.model_name_or_path)
model.to(cfg.device)
model.eval()  # keep base model frozen

# Freeze all base model parameters
for param in model.parameters():
    param.requires_grad = False

# Load WikiText-2 (raw version) using the Hugging Face datasets library
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

# Preprocessing function: tokenize and truncate long texts
def tokenize_function(ex):
    return tokenizer(ex["text"], truncation=True, max_length=cfg.max_length)

tokenized_dataset = dataset.map(tokenize_function, batched=False)
# Filter out examples that are too short to be split into two halves.
tokenized_dataset = tokenized_dataset.filter(lambda ex: ex["input_ids"] is not None and len(ex["input_ids"]) > 50)

# Create a DataLoader using fast tokenization (which handles padding internally)
def collate_fn(examples):
    texts = [ex["text"] for ex in examples]
    encodings = tokenizer(
        texts,
        padding="max_length",  # or padding=True for dynamic padding
        truncation=True,
        max_length=cfg.max_length,
        return_tensors="pt"
    )
    return encodings["input_ids"]

dataloader = DataLoader(tokenized_dataset, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_fn)

# ============
# Choose a Target Layer to "Inject" the LoRA delta
# ============
# Here we choose the query projection of the first transformer block.
target_layer = model.model.layers[0].self_attn.q_proj

# Save the original forward function so we can wrap it.
original_forward = target_layer.forward

def lora_forward(input):
    """
    A wrapped forward method that, if a delta (LoRA update) is stored on the layer,
    adds it to the base weight.
    """
    weight = target_layer.weight
    if hasattr(target_layer, "lora_delta") and target_layer.lora_delta is not None:
        weight = weight + target_layer.lora_delta
    return F.linear(input, weight, target_layer.bias)

# Override the forward of the target layer
target_layer.forward = lora_forward

# ============
# Helper Functions
# ============
def split_input_ids(input_ids):
    """
    Given input_ids of shape (batch, seq_len), split each sample into first and second halves.
    Assumes seq_len is even.
    """
    seq_len = input_ids.shape[1]
    half = seq_len // 2
    first_half = input_ids[:, :half]
    second_half = input_ids[:, half:]
    return first_half, second_half

def decode_tokens(updated_logits, base_logits, num_tokens=25):
    """
    Decodes the first `num_tokens` token predictions from both updated and baseline logits.
    """
    updated_token_ids = torch.argmax(updated_logits[0, :num_tokens, :], dim=-1).tolist()
    base_token_ids = torch.argmax(base_logits[0, :num_tokens, :], dim=-1).tolist()
    updated_decoded = tokenizer.decode(updated_token_ids)
    base_decoded = tokenizer.decode(base_token_ids)
    return {"updated": updated_decoded, "base": base_decoded}

def save_checkpoint(epoch, global_step, loss_value):
    """
    Saves a checkpoint locally and uploads it to wandb.
    """
    checkpoint = {
        "epoch": epoch,
        "global_step": global_step,
        "compression_head_state_dict": compression_head.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss_value
    }
    checkpoint_filename = f"checkpoint_epoch{epoch}_step{global_step}.pt"
    checkpoint_path = os.path.join(cfg.checkpoint_dir, checkpoint_filename)
    torch.save(checkpoint, checkpoint_path)
    wandb.save(checkpoint_path)
    print(f"Saved checkpoint: {checkpoint_path}")

def load_latest_checkpoint():
    """
    Loads the latest checkpoint from the checkpoint directory, if any exist.
    Returns the checkpoint dictionary and the filename, or (None, None) if no checkpoint is found.
    """
    checkpoint_files = glob.glob(os.path.join(cfg.checkpoint_dir, "checkpoint_epoch*_step*.pt"))
    if checkpoint_files:
        latest_checkpoint_file = max(checkpoint_files, key=os.path.getctime)
        checkpoint = torch.load(latest_checkpoint_file, map_location=cfg.device)
        print(f"Loaded checkpoint: {latest_checkpoint_file}")
        return checkpoint, latest_checkpoint_file
    else:
        print("No checkpoint found, starting from scratch.")
        return None, None

# ============
# Initialize Compression Head and Optimizer
# ============
hidden_dim = model.config.hidden_size  # e.g., 2048 or model-specific
target_out_features, target_in_features = target_layer.weight.shape
compression_head = CompressionHead(hidden_dim, target_in_features, target_out_features, rank=cfg.lora_rank)
compression_head.to(cfg.device)
optimizer = optim.Adam(compression_head.parameters(), lr=cfg.learning_rate)

# Try loading a checkpoint to resume training
start_epoch = 0
global_step = 0
checkpoint, ckpt_file = load_latest_checkpoint()
if checkpoint is not None:
    start_epoch = checkpoint["epoch"]
    global_step = checkpoint["global_step"]
    compression_head.load_state_dict(checkpoint["compression_head_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    print(f"Resuming training from epoch {start_epoch}, global step {global_step}.")

# ============
# Training Loop
# ============
print("Starting training loop...")
model.train()  # Set model to train mode for teacher forcing
for epoch in range(start_epoch, cfg.num_epochs):
    epoch_loss = 0.0
    for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
        # Process each sample in the batch independently (for simplicity).
        batch_loss = 0.0
        for sample in batch:
            sample = sample.unsqueeze(0).to(cfg.device)  # shape: (1, seq_len)
            first_half, second_half = split_input_ids(sample)
            
            # (A) Run first half to get summary hidden state.
            with torch.no_grad():
                outputs_first = model(first_half, output_hidden_states=True)
            last_hidden_state = outputs_first.hidden_states[-1][:, -1, :]  # shape: (1, hidden_dim)
            
            # (B) Compute LoRA delta from the compression head.
            delta = compression_head(last_hidden_state)  # shape: (target_out_features, target_in_features)
            
            # (C) Run modified model on second half (with injected delta).
            target_layer.lora_delta = delta
            outputs_mod = model(second_half, output_hidden_states=False)
            logits_mod = outputs_mod.logits  # shape: (1, second_half_length, vocab_size)
            
            # (D) Run baseline full-context model.
            target_layer.lora_delta = None
            full_context = torch.cat([first_half, second_half], dim=1)
            with torch.no_grad():
                outputs_full = model(full_context, output_hidden_states=False)
            logits_full = outputs_full.logits
            second_half_len = second_half.shape[1]
            logits_full_second = logits_full[:, -second_half_len:, :]
            
            # (E) Compute KL divergence loss.
            log_probs_mod = F.log_softmax(logits_mod, dim=-1)
            probs_full = F.softmax(logits_full_second, dim=-1)
            loss = F.kl_div(log_probs_mod, probs_full, reduction="batchmean")
            
            batch_loss += loss
            global_step += 1

            # Save a checkpoint every cfg.checkpoint_interval steps.
            if global_step % cfg.checkpoint_interval == 0:
                save_checkpoint(epoch+1, global_step, loss.item())
        
        # Average loss over samples in the batch.
        batch_loss = batch_loss / batch.shape[0]
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
        
        epoch_loss += batch_loss.item()
        wandb.log({"loss": batch_loss.item(), "epoch": epoch+1, "global_step": global_step})
    
    avg_loss = epoch_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{cfg.num_epochs} - Average Loss: {avg_loss:.4f}")
    # Optionally, save a checkpoint at the end of each epoch.
    save_checkpoint(epoch+1, global_step, avg_loss)

print("Training complete.")
