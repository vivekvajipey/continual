import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

# ========
# Configuration
# ========
config = {
    "model_name_or_path": "meta-llama/llama-3.2-1B",  # same model as training
    "device": "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu",
    "max_length": 512,
    "lora_rank": 4,
    "checkpoint_path": "checkpoints/checkpoint_epoch1_step4600.pt",
    "eval_texts": [
        "The history of artificial intelligence began in the 1950s when pioneers started thinking about machines that could mimic human reasoning. Over the decades, developments in hardware and smart algorithms propelled the field forward.",
        "In recent years, language models have become increasingly powerful. They are used for a wide range of tasks, including translation, summarization, and even creative writing."
    ],
    "generation_max_length": 50  # tokens for generation
}

# ========
# Load Tokenizer and Model
# ========
tokenizer = AutoTokenizer.from_pretrained(config["model_name_or_path"])
if tokenizer.pad_token is None:
    # Fallback: use EOS token as pad token if not defined.
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(config["model_name_or_path"])
model.to(config["device"])
model.eval()

# Freeze base model parameters
for param in model.parameters():
    param.requires_grad = False

# ========
# Identify and Override the Target Layer
# ========
# In our setup we use the query projection of the first transformer layer.
target_layer = model.model.layers[0].self_attn.q_proj

# Save the original forward (if needed)
original_forward = target_layer.forward

def lora_forward(input):
    weight = target_layer.weight
    if hasattr(target_layer, "lora_delta") and target_layer.lora_delta is not None:
        weight = weight + target_layer.lora_delta
    return F.linear(input, weight, target_layer.bias)

target_layer.forward = lora_forward

# ========
# Define the Compression Head (LoRA Generator)
# ========
class CompressionHead(nn.Module):
    """
    CompressionHead reduces the hidden state to a low-rank representation and
    projects it back to generate a delta update for the target layer.
    """
    def __init__(self, hidden_dim, target_in_features, target_out_features, rank=8):
        super().__init__()
        self.rank = rank
        self.linear1 = nn.Linear(hidden_dim, rank)
        self.linear2 = nn.Linear(rank, target_in_features * target_out_features)
        self.target_in_features = target_in_features
        self.target_out_features = target_out_features

    def forward(self, hidden):
        x = self.linear1(hidden)            # -> (batch_size, rank)
        x = torch.relu(x)
        x = self.linear2(x)                 # -> (batch_size, target_in_features * target_out_features)
        delta = x.view(-1, self.target_out_features, self.target_in_features)
        return delta.squeeze(0)           # assume batch_size==1

# ========
# Create Compression Head Instance and Load Checkpoint
# ========
hidden_dim = model.config.hidden_size  # e.g., 2048, model dependent
target_out_features, target_in_features = target_layer.weight.shape
compression_head = CompressionHead(hidden_dim, target_in_features, target_out_features, rank=config["lora_rank"])
compression_head.to(config["device"])

# Load in the trained compression head weights from the checkpoint.
checkpoint = torch.load(config["checkpoint_path"], map_location=config["device"])
compression_head.load_state_dict(checkpoint["compression_head_state_dict"])
compression_head.eval()

# ========
# Helper Functions
# ========
def split_input_ids(input_ids):
    """
    Splits input_ids of shape (1, seq_len) into two halves.
    """
    seq_len = input_ids.shape[1]
    half = seq_len // 2
    first_half = input_ids[:, :half]
    second_half = input_ids[:, half:]
    return first_half, second_half

def get_logits_with_delta(second_half, delta):
    """
    Injects the delta and runs a forward pass on the second half.
    """
    target_layer.lora_delta = delta
    with torch.no_grad():
        outputs = model(second_half, output_hidden_states=False)
    logits_mod = outputs.logits
    target_layer.lora_delta = None
    return logits_mod

def get_logits_full_context(full_input, second_half_len):
    """
    Runs a full-context (baseline) forward pass and extracts logits for second half tokens.
    """
    with torch.no_grad():
        outputs = model(full_input, output_hidden_states=False)
    logits_full = outputs.logits
    return logits_full[:, -second_half_len:, :]

# ========
# Evaluation on Qualitative Examples: Compare Second-Half Probabilities
# ========
print("=== Evaluating Second-Half Probabilities ===")
for text in config["eval_texts"]:
    print("\nInput text:")
    print(text)
    encoding = tokenizer(text, truncation=True, max_length=config["max_length"], return_tensors="pt")
    input_ids = encoding["input_ids"].to(config["device"])
    
    # Ensure text is long enough to split
    if input_ids.shape[1] < 10:
        print("Text too short to split, skipping.")
        continue

    # 1. Split into first half (prefix) and second half.
    first_half, second_half = split_input_ids(input_ids)
    
    # 2. Run first half to obtain the summary hidden state.
    with torch.no_grad():
        outputs_first = model(first_half, output_hidden_states=True)
    last_hidden_state = outputs_first.hidden_states[-1][:, -1, :]  # (1, hidden_dim)
    
    # 3. Compute the low-rank delta using the compression head.
    delta = compression_head(last_hidden_state)
    
    # 4. Run the modified model on the second half (delta injected).
    logits_mod = get_logits_with_delta(second_half, delta)
    probs_mod = F.softmax(logits_mod, dim=-1)
    
    # 5. Run the full-context baseline model (prefix+second half, no delta).
    full_input = torch.cat([first_half, second_half], dim=1)
    second_half_len = second_half.shape[1]
    logits_full = get_logits_full_context(full_input, second_half_len)
    probs_full = F.softmax(logits_full, dim=-1)
    
    # 6. Compare the top predicted token at the first few positions of the second half.
    print("\nComparing predictions for the first few tokens of the second half:")
    for pos in range(min(5, second_half_len)):
        token_mod_id = torch.argmax(probs_mod[0, pos, :]).item()
        token_full_id = torch.argmax(probs_full[0, pos, :]).item()
        token_mod = tokenizer.decode([token_mod_id]).strip()
        token_full = tokenizer.decode([token_full_id]).strip()
        print(f"Position {pos}: LoRA token: {token_mod}  |  Baseline token: {token_full}")
    print("-" * 50)

# ========
# Evaluation: Generation with No Context (Only BOS Token)
# ========
# print("\n=== Generation with No Context ===")
# bos_token_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
# if bos_token_id is None:
#     # Fallback if neither BOS nor EOS is set.
#     bos_token_id = 1
# input_ids = torch.tensor([[bos_token_id]]).to(config["device"])
# # Create an attention mask since it is required for generation.
# attention_mask = torch.ones_like(input_ids)

# # Determine a valid pad_token_id.
# pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else model.config.eos_token_id
# if pad_token_id is None:
#     pad_token_id = bos_token_id

# # To see if the compression head (LoRA updates) biases generation, we compute delta from the BOS token.
# with torch.no_grad():
#     outputs_first = model(input_ids, output_hidden_states=True)
# last_hidden_state = outputs_first.hidden_states[-1][:, -1, :]
# delta = compression_head(last_hidden_state)

# # Inject the delta and generate text.
# target_layer.lora_delta = delta
# generated_ids = model.generate(
#     input_ids,
#     max_length=config["generation_max_length"],
#     do_sample=True,
#     pad_token_id=pad_token_id,
#     attention_mask=attention_mask
# )
# generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
# print("LoRA model generated text with no context:")
# print(generated_text)
# target_layer.lora_delta = None