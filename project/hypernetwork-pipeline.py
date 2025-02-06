import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# ============
# Configuration
# ============
MODEL_NAME_OR_PATH = "meta-llama/llama-3.2-1B"  
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_EPOCHS = 3
LEARNING_RATE = 1e-4

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
# Load Model and Tokenizer
# ============
# (For LLaMA-style models you may need additional configuration.)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME_OR_PATH)
model.to(DEVICE)
model.eval()  # We keep the base model frozen

# Freeze all base model parameters
for param in model.parameters():
    param.requires_grad = False

# ============
# Choose a Target Layer to "Inject" the LoRA delta
# ============
# In this example we choose the query projection of the first transformer block.
# Adjust the attribute path as needed for your model.
# (For many LLaMA implementations, the module structure is like model.model.layers[i].self_attn.q_proj)
target_layer = model.model.layers[0].self_attn.q_proj

# Save the original forward function so we can wrap it.
original_forward = target_layer.forward

def lora_forward(input):
    """
    A wrapped forward method that, if a delta (LoRA update) is stored on the layer,
    adds it to the base weight.
    """
    weight = target_layer.weight
    # If the attribute 'lora_delta' exists and is not None, add it.
    if hasattr(target_layer, "lora_delta") and target_layer.lora_delta is not None:
        weight = weight + target_layer.lora_delta
    # Use a simple linear transformation (assumes no special dropout or other behavior)
    return F.linear(input, weight, target_layer.bias) # look into llama's forward pass?

# Override the forward of the target layer
target_layer.forward = lora_forward


def decode_first_token(updated_logits, base_logits, tokenizer=tokenizer):
    """
    Decodes the first token prediction from both the updated logits and the baseline logits.
    
    Args:
        updated_logits (torch.Tensor): Logits from the model with the injected LoRA update.
                                       Expected shape: (batch_size, seq_length, vocab_size).
        base_logits (torch.Tensor): Logits from the baseline full-context model.
                                    Expected shape: (batch_size, seq_length, vocab_size).
        tokenizer: The tokenizer used for decoding token IDs into text.
    
    Returns:
        dict: A dictionary containing the decoded first token from both the updated and the baseline logits.
              For example: {'updated': 'The', 'base': 'A'}.
    """
    # For debugging, we assume batch_size == 1 and decode the first token in the sequence.
    # Compute the token id with the highest logit for the first token position.
    updated_first_token_id = torch.argmax(updated_logits[0, 0, :]).item()
    base_first_token_id = torch.argmax(base_logits[0, 0, :]).item()

    # Decode the token ids into strings.
    updated_decoded = tokenizer.decode([updated_first_token_id])
    base_decoded = tokenizer.decode([base_first_token_id])

    # Print the decoded tokens for debugging purposes.
    print("Decoded first token (updated logits):", updated_decoded)
    print("Decoded first token (baseline logits):", base_decoded)

    return {"updated": updated_decoded, "base": base_decoded}

# ============
# Initialize Compression Head and Optimizer
# ============
hidden_dim = model.config.hidden_size  # e.g. 2048 or as defined in your model config
# Get target weight shape (out_features, in_features)
target_out_features, target_in_features = target_layer.weight.shape

compression_head = CompressionHead(hidden_dim, target_in_features, target_out_features, rank=4)
compression_head.to(DEVICE)
optimizer = optim.Adam(compression_head.parameters(), lr=LEARNING_RATE)

# ============
# Dummy Dataset (Replace with Your Actual Data)
# ============
# For demonstration we create a list of texts.
# In practice, choose documents where the second half clearly depends on the first half.
dummy_texts = [
    "The Apollo program was the third United States human spaceflight program carried out by NASA, "
    "which accomplished landing the first humans on the Moon from 1969 to 1972. It was first conceived "
    "during Dwight D. Eisenhower's administration as a three-person spacecraft to follow the one-person "
    "Mercury spacecraft. The Apollo spacecraft was composed of a command module, service module, and lunar module.",
    
    "Artificial intelligence has seen tremendous progress in recent years, with deep learning at the core "
    "of many state-of-the-art systems. Researchers are continually improving algorithms and architectures "
    "to better model complex data and to understand the underlying representations. This progress has enabled "
    "significant advances in computer vision, natural language processing, and reinforcement learning."
]

# ============
# Helper: Split Tokenized Input in Half
# ============
def split_input_ids(input_ids):
    """
    Given input_ids of shape (1, seq_len), split into first and second halves.
    """
    seq_len = input_ids.shape[1]
    half = seq_len // 2
    first_half = input_ids[:, :half]
    second_half = input_ids[:, half:]
    return first_half, second_half

# ============
# Training Loop
# ============
print("Starting training loop...")
model.train()  # (Even though we keep its parameters frozen; this is so that e.g. dropout, if any, behaves as in training)
for epoch in range(NUM_EPOCHS):
    epoch_loss = 0.0
    # Using tqdm for a progress bar over the dataset
    for text in tqdm(dummy_texts, desc=f"Epoch {epoch+1}"):
        # Tokenize and move tokens to DEVICE
        inputs = tokenizer(text, return_tensors="pt")
        input_ids = inputs.input_ids.to(DEVICE)
        
        # Split into first and second halves
        first_half, second_half = split_input_ids(input_ids)
        
        # ------
        # (A) Run the first half to obtain a “summary” hidden state.
        # We request hidden states so we can extract the last token’s representation.
        # ------
        with torch.no_grad():
            outputs_first = model(first_half, output_hidden_states=True)
        # Get the last token’s hidden state from the final layer; shape: (1, hidden_dim)
        last_hidden_state = outputs_first.hidden_states[-1][:, -1, :]

        # ------
        # (B) Compute the LoRA weight delta from the compression head.
        # ------
        delta = compression_head(last_hidden_state)  # shape: (target_out_features, target_in_features)
        
        # ------
        # (C) Run the “modified” model on the second half.
        # We set the target layer’s attribute 'lora_delta' so that its forward pass adds the update.
        # ------
        target_layer.lora_delta = delta  # inject the computed LoRA update
        # Run teacher forcing on the second-half tokens.
        outputs_mod = model(second_half, output_hidden_states=False)
        logits_mod = outputs_mod.logits  # shape: (1, second_half_length, vocab_size)
        
        # ------
        # (D) Run the baseline “full-context” model on the concatenation of (first_half + second_half)
        # (Make sure no delta is applied in this run.)
        # ------
        target_layer.lora_delta = None  # disable injection for the baseline run
        full_context = torch.cat([first_half, second_half], dim=1)
        with torch.no_grad():
            outputs_full = model(full_context, output_hidden_states=False)
        logits_full = outputs_full.logits  # shape: (1, full_seq_length, vocab_size)
        # Extract the logits corresponding to the second-half positions.
        second_half_len = second_half.shape[1]
        # (Assuming the predictions for the second half start at index "first_half_length")
        logits_full_second = logits_full[:, -second_half_len:, :]
        
        # Decode the first token predictions for debugging

        decoded_tokens = decode_first_token(logits_mod, logits_full_second, tokenizer)

        # ------
        # (E) Compute the loss.
        # We use KL divergence between the token distributions produced with and without the delta.
        # (The idea is that the compression head should generate a delta that “recovers” the full-context predictions.)
        # ------
        log_probs_mod = F.log_softmax(logits_mod, dim=-1)
        probs_full = F.softmax(logits_full_second, dim=-1)
        loss = F.kl_div(log_probs_mod, probs_full, reduction="batchmean")
        
        # ------
        # (F) Backpropagation: update the compression head parameters.
        # ------
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    avg_loss = epoch_loss / len(dummy_texts)
    print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Average Loss: {avg_loss:.4f}")

print("Training complete.")
