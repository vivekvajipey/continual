# from dotenv import load_dotenv
# load_dotenv()  # Load environment variables from .env

# import glob
# import os
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from transformers import AutoTokenizer, AutoModelForCausalLM
# from datasets import load_dataset
# from torch.utils.data import DataLoader
# from tqdm import tqdm
# import wandb

# # ============
# # Configuration
# # ============
# config = {
#     "model_name_or_path": "meta-llama/llama-3.2-1B",
#     "device": "cuda" if torch.cuda.is_available() else "cpu",
#     "num_epochs": 3,
#     "learning_rate": 1e-4,
#     "batch_size": 4,  # adjust depending on your GPU
#     "max_length": 512,  # max token length per sample (will be truncated/padded)
#     "lora_rank": 64,
#     "eval_interval": 50,  # steps between qualitative evaluations
#     "checkpoint_interval": 1000,  # checkpoint every 1000 steps (adjust as needed)
#     "wandb_project": "lora-compression-head",
#     "checkpoint_dir": os.getenv("CHECKPOINT_DIR", "/scr/tadimeti/checkpoints"),
# }

# # Create checkpoint directory if it doesn't exist.
# os.makedirs(config["checkpoint_dir"], exist_ok=True)

# # Initialize wandb.
# wandb.init(project=config["wandb_project"], config=config)
# cfg = wandb.config


# # ============
# # Compression Head Definition
# # ============
# class CompressionHead(nn.Module):
#     """
#     A simple compression head that takes a vector (e.g. the last hidden state
#     of the first half of a document) and outputs a weight delta (LoRA update)
#     for a target linear layer.

#     The target linear layer (e.g. a query or value projection) has weight shape (out_features, in_features).
#     """

#     def __init__(self, hidden_dim, target_in_features, target_out_features, rank=64):
#         super().__init__()
#         self.rank = rank
#         self.linear1 = nn.Linear(hidden_dim, rank)
#         self.linear2 = nn.Linear(
#             rank, (target_in_features + target_out_features) * rank
#         )
#         self.target_in_features = target_in_features
#         self.target_out_features = target_out_features

#     def forward(self, hidden):
#         # hidden: (batch_size, hidden_dim); assume batch_size == 1 for simplicity.
#         x = self.linear1(hidden)  # -> (batch_size, rank)
#         x = torch.nn.GELU()(x)  # Use GELU activation to avoid dead neurons.
#         x = self.linear2(x)  # -> (batch_size, target_in_features * target_out_features)
#         # Reshape to (batch_size, target_out_features, target_in_features)
#         delta_tall = x[:, : self.target_out_features * self.rank].reshape(
#             -1, self.target_out_features, self.rank
#         )
#         delta_wide = x[:, self.target_out_features * self.rank :].reshape(
#             -1, self.rank, self.target_in_features
#         )
#         # tall is of size (out, k) and wide is of size (k, in)
#         delta = delta_tall @ delta_wide
#         return delta.squeeze(0)  # -> (target_out_features, target_in_features)


# # ============
# # Load Model, Tokenizer, and Dataset
# # ============
# tokenizer = AutoTokenizer.from_pretrained(cfg.model_name_or_path)
# if tokenizer.pad_token is None:
#     # Use the EOS token as the padding token if none exists.
#     tokenizer.pad_token = tokenizer.eos_token

# model = AutoModelForCausalLM.from_pretrained(cfg.model_name_or_path)
# model.to(cfg.device)
# model.eval()  # keep base model frozen

# # Freeze all base model parameters.
# for param in model.parameters():
#     param.requires_grad = False

# # Load WikiText-2 (raw version) using the Hugging Face datasets library.
# dataset = load_dataset(
#     "wikitext",
#     "wikitext-2-raw-v1",
#     split="train",
#     cache_dir=os.getenv("HF_CACHE_DIR", "/scr/tadimeti/hf_cache/hub")
# )


# # Preprocessing: tokenize and truncate texts.
# def tokenize_function(ex):
#     return tokenizer(ex["text"], truncation=True, max_length=cfg.max_length)


# tokenized_dataset = dataset.map(tokenize_function, batched=False)
# # Filter out examples that are too short to split into two halves.
# tokenized_dataset = tokenized_dataset.filter(
#     lambda ex: ex["input_ids"] is not None and len(ex["input_ids"]) > 50
# )


# # DataLoader using fast tokenization (handles padding internally).
# def collate_fn(examples):
#     texts = [ex["text"] for ex in examples]
#     encodings = tokenizer(
#         texts,
#         padding="max_length",
#         truncation=True,
#         max_length=cfg.max_length,
#         return_tensors="pt",
#     )
#     return encodings["input_ids"]


# dataloader = DataLoader(
#     tokenized_dataset, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_fn
# )

# # ========
# # Load Validation Set
# # ========
# val_dataset = load_dataset(
#     "wikitext",
#     "wikitext-2-raw-v1",
#     split="validation",
#     cache_dir=os.getenv("HF_CACHE_DIR", "/scr/tadimeti/hf_cache/hub")
# )
# val_tokenized_dataset = val_dataset.map(tokenize_function, batched=False)
# val_tokenized_dataset = val_tokenized_dataset.filter(
#     lambda ex: ex["input_ids"] is not None and len(ex["input_ids"]) > 50
# )

# val_dataloader = DataLoader(
#     val_tokenized_dataset, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_fn
# )

# # ============
# # Choose Target Layers for LoRA Injection: q_proj and v_proj.
# # ============
# # For the query projection.
# target_layer_q = model.model.layers[0].self_attn.q_proj
# original_forward_q = target_layer_q.forward


# def lora_forward_q(input):
#     weight = target_layer_q.weight
#     if hasattr(target_layer_q, "lora_delta") and target_layer_q.lora_delta is not None:
#         weight = weight + target_layer_q.lora_delta
#     return F.linear(input, weight, target_layer_q.bias)


# target_layer_q.forward = lora_forward_q

# # For the value projection.
# target_layer_v = model.model.layers[0].self_attn.v_proj
# original_forward_v = target_layer_v.forward


# def lora_forward_v(input):
#     weight = target_layer_v.weight
#     if hasattr(target_layer_v, "lora_delta") and target_layer_v.lora_delta is not None:
#         weight = weight + target_layer_v.lora_delta
#     return F.linear(input, weight, target_layer_v.bias)


# target_layer_v.forward = lora_forward_v


# # ============
# # Helper Functions
# # ============
# def split_input_ids(input_ids):
#     """
#     Split each sample in a batch (shape: [batch, seq_len]) into first and second halves.
#     Assumes the sequence length is even.
#     """
#     seq_len = input_ids.shape[1]
#     half = seq_len // 2
#     first_half = input_ids[:, :half]
#     second_half = input_ids[:, half:]
#     return first_half, second_half


# def decode_tokens(updated_logits, base_logits, num_tokens=25):
#     """
#     Decode the first `num_tokens` token predictions from both updated and baseline logits.
#     """
#     updated_token_ids = torch.argmax(updated_logits[0, :num_tokens, :], dim=-1).tolist()
#     base_token_ids = torch.argmax(base_logits[0, :num_tokens, :], dim=-1).tolist()
#     updated_decoded = tokenizer.decode(updated_token_ids)
#     base_decoded = tokenizer.decode(base_token_ids)
#     return {"updated": updated_decoded, "base": base_decoded}


# def save_checkpoint(epoch, global_step, loss_value, best=False):
#     """
#     Save a checkpoint locally and upload it to wandb.
#     If best is True, save the checkpoint as the best model.
#     """
#     checkpoint = {
#         "epoch": epoch,
#         "global_step": global_step,
#         "compression_head_q_state_dict": compression_head_q.state_dict(),
#         "compression_head_v_state_dict": compression_head_v.state_dict(),
#         "optimizer_state_dict": optimizer.state_dict(),
#         "loss": loss_value,
#     }
#     prefix = "best_" if best else ""
#     checkpoint_filename = f"{prefix}checkpoint_epoch{epoch}_step{global_step}.pt"
#     checkpoint_path = os.path.join(cfg.checkpoint_dir, checkpoint_filename)
#     torch.save(checkpoint, checkpoint_path)
#     wandb.save(checkpoint_path)
#     print(f"Saved {'best ' if best else ''}checkpoint: {checkpoint_path}")


# def load_latest_checkpoint():
#     """
#     Load the latest checkpoint from the checkpoint directory, if any exist.
#     Returns (checkpoint, filename) or (None, None) if no checkpoint is found.
#     """
#     checkpoint_files = glob.glob(
#         os.path.join(cfg.checkpoint_dir, "checkpoint_epoch*_step*.pt")
#     )
#     if checkpoint_files:
#         latest_checkpoint_file = max(checkpoint_files, key=os.path.getctime)
#         checkpoint = torch.load(latest_checkpoint_file, map_location=cfg.device)
#         print(f"Loaded checkpoint: {latest_checkpoint_file}")
#         return checkpoint, latest_checkpoint_file
#     else:
#         print("No checkpoint found, starting from scratch.")
#         return None, None


# # ============
# # Initialize Compression Heads and Optimizer
# # ============
# hidden_dim = model.config.hidden_size  # e.g., 2048 or as defined by the model

# # For q_proj.
# q_out_features, q_in_features = target_layer_q.weight.shape
# compression_head_q = CompressionHead(
#     hidden_dim, q_in_features, q_out_features, rank=cfg.lora_rank
# )
# compression_head_q.to(cfg.device)

# # For v_proj.
# v_out_features, v_in_features = target_layer_v.weight.shape
# compression_head_v = CompressionHead(
#     hidden_dim, v_in_features, v_out_features, rank=cfg.lora_rank
# )
# compression_head_v.to(cfg.device)

# # Create an optimizer that updates both compression heads.
# optimizer = optim.Adam(
#     list(compression_head_q.parameters()) + list(compression_head_v.parameters()),
#     lr=cfg.learning_rate,
# )

# # Calculate total steps for cosine decay.
# total_steps = cfg.num_epochs * len(dataloader)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
#     optimizer, T_max=total_steps, eta_min=1e-6
# )

# # Try loading a checkpoint to resume training.
# start_epoch = 0
# global_step = 0
# checkpoint, ckpt_file = load_latest_checkpoint()
# if checkpoint is not None:
#     start_epoch = checkpoint["epoch"]
#     global_step = checkpoint["global_step"]
#     compression_head_q.load_state_dict(checkpoint["compression_head_q_state_dict"])
#     compression_head_v.load_state_dict(checkpoint["compression_head_v_state_dict"])
#     optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
#     print(f"Resuming training from epoch {start_epoch}, global step {global_step}.")

# best_val_loss = float('inf')

# # ============
# # Training Loop
# # ============
# print("Starting training loop...")
# model.train()  # Set model to train mode (for teacher forcing).
# for epoch in range(start_epoch, cfg.num_epochs):
#     epoch_loss = 0.0
#     for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
#         batch_loss = 0.0
#         for sample in batch:
#             sample = sample.unsqueeze(0).to(cfg.device)  # shape: (1, seq_len)
#             first_half, second_half = split_input_ids(sample)

#             # (A) Run first half to obtain summary hidden state.
#             with torch.no_grad():
#                 outputs_first = model(first_half, output_hidden_states=True)
#             last_hidden_state = outputs_first.hidden_states[-1][
#                 :, -1, :
#             ]  # shape: (1, hidden_dim)

#             # (B) Compute LoRA deltas for q_proj and v_proj.
#             q_delta = compression_head_q(
#                 last_hidden_state
#             )  # shape: (q_out_features, q_in_features)
#             v_delta = compression_head_v(
#                 last_hidden_state
#             )  # shape: (v_out_features, v_in_features)

#             # (C) Run modified model on second half (with injected deltas).
#             target_layer_q.lora_delta = q_delta
#             target_layer_v.lora_delta = v_delta
#             outputs_mod = model(second_half, output_hidden_states=False)
#             logits_mod = (
#                 outputs_mod.logits
#             )  # shape: (1, second_half_length, vocab_size)

#             # (D) Run baseline full-context model (no injection).
#             target_layer_q.lora_delta = None
#             target_layer_v.lora_delta = None
#             full_context = torch.cat([first_half, second_half], dim=1)
#             with torch.no_grad():
#                 outputs_full = model(full_context, output_hidden_states=False)
#             logits_full = outputs_full.logits
#             second_half_len = second_half.shape[1]
#             logits_full_second = logits_full[:, -second_half_len:, :]

#             # (E) Compute KL divergence loss.
#             log_probs_mod = F.log_softmax(logits_mod, dim=-1)
#             probs_full = F.softmax(logits_full_second, dim=-1)
#             loss = F.kl_div(log_probs_mod, probs_full, reduction="batchmean")

#             batch_loss += loss
#             global_step += 1

#             # Save a checkpoint every checkpoint_interval steps.
#             if global_step % cfg.checkpoint_interval == 0:
#                 save_checkpoint(epoch + 1, global_step, loss.item())

#             # ------------------------------------------------------------
#             # Evaluate on a subset of the validation set every eval_interval steps.
#             # ------------------------------------------------------------
#             if global_step % cfg.eval_interval == 0:
#                 subset_batches = 5  # Number of validation batches to use for evaluation.
#                 eval_loss_total = 0.0
#                 num_eval_samples = 0

#                 model.eval()
#                 with torch.no_grad():
#                     for i, val_batch in enumerate(val_dataloader):
#                         if i >= subset_batches:
#                             break
#                         for val_sample in val_batch:
#                             val_sample = val_sample.unsqueeze(0).to(cfg.device)
#                             fh, sh = split_input_ids(val_sample)
#                             outputs_first = model(fh, output_hidden_states=True)
#                             last_hidden_state = outputs_first.hidden_states[-1][:, -1, :]

#                             q_delta = compression_head_q(last_hidden_state)
#                             v_delta = compression_head_v(last_hidden_state)

#                             target_layer_q.lora_delta = q_delta
#                             target_layer_v.lora_delta = v_delta
#                             outputs_mod = model(sh, output_hidden_states=False)
#                             logits_mod = outputs_mod.logits

#                             target_layer_q.lora_delta = None
#                             target_layer_v.lora_delta = None
#                             full_context = torch.cat([fh, sh], dim=1)
#                             outputs_full = model(full_context, output_hidden_states=False)
#                             logits_full = outputs_full.logits
#                             sh_len = sh.shape[1]
#                             logits_full_second = logits_full[:, -sh_len:, :]

#                             log_probs_mod = F.log_softmax(logits_mod, dim=-1)
#                             probs_full = F.softmax(logits_full_second, dim=-1)
#                             eval_loss = F.kl_div(log_probs_mod, probs_full, reduction="batchmean")
#                             eval_loss_total += eval_loss.item()
#                             num_eval_samples += 1

#                 avg_eval_loss = eval_loss_total / num_eval_samples if num_eval_samples > 0 else float('inf')
#                 wandb.log({"eval_loss_subset": avg_eval_loss, "global_step": global_step})
#                 print(f"Step {global_step} - Eval subset loss: {avg_eval_loss:.4f}")
#                 model.train()
#             # ------------------------------------------------------------

#         # Average loss over samples in the batch.
#         batch_loss = batch_loss / batch.shape[0]
#         optimizer.zero_grad()
#         batch_loss.backward()
#         optimizer.step()
#         scheduler.step()  # Update the learning rate using cosine decay.

#         epoch_loss += batch_loss.item()
#         wandb.log(
#             {
#                 "loss": batch_loss.item(),
#                 "epoch": epoch + 1,
#                 "global_step": global_step,
#                 "learning_rate": optimizer.param_groups[0]["lr"],
#             }
#         )
    
#     avg_loss = epoch_loss / len(dataloader)
#     print(f"Epoch {epoch+1}/{cfg.num_epochs} - Average Training Loss: {avg_loss:.4f}")

#     # (Optional) Full validation evaluation at the end of the epoch if desired.
#     val_loss_total = 0.0
#     num_val_samples = 0
#     model.eval()
#     with torch.no_grad():
#         for val_batch in tqdm(val_dataloader, desc=f"Full Validation Epoch {epoch+1}"):
#             for sample in val_batch:
#                 sample = sample.unsqueeze(0).to(cfg.device)
#                 fh, sh = split_input_ids(sample)
#                 outputs_first = model(fh, output_hidden_states=True)
#                 last_hidden_state = outputs_first.hidden_states[-1][:, -1, :]

#                 q_delta = compression_head_q(last_hidden_state)
#                 v_delta = compression_head_v(last_hidden_state)

#                 target_layer_q.lora_delta = q_delta
#                 target_layer_v.lora_delta = v_delta
#                 outputs_mod = model(sh, output_hidden_states=False)
#                 logits_mod = outputs_mod.logits

#                 target_layer_q.lora_delta = None
#                 target_layer_v.lora_delta = None
#                 full_context = torch.cat([fh, sh], dim=1)
#                 outputs_full = model(full_context, output_hidden_states=False)
#                 logits_full = outputs_full.logits
#                 sh_len = sh.shape[1]
#                 logits_full_second = logits_full[:, -sh_len:, :]

#                 log_probs_mod = F.log_softmax(logits_mod, dim=-1)
#                 probs_full = F.softmax(logits_full_second, dim=-1)
#                 loss_val = F.kl_div(log_probs_mod, probs_full, reduction="batchmean")
#                 val_loss_total += loss_val.item()
#                 num_val_samples += 1
#     avg_val_loss = val_loss_total / num_val_samples if num_val_samples > 0 else float('inf')
#     wandb.log({"val_loss_full": avg_val_loss, "epoch": epoch + 1, "global_step": global_step})
#     print(f"Epoch {epoch+1}/{cfg.num_epochs} - Full Validation Loss: {avg_val_loss:.4f}")

#     # Save best full validation checkpoint if improved.
#     if avg_val_loss < best_val_loss:
#         best_val_loss = avg_val_loss
#         save_checkpoint(epoch + 1, global_step, avg_val_loss, best=True)

#     # Save a checkpoint at the end of the epoch (latest model).
#     save_checkpoint(epoch + 1, global_step, avg_loss)

#     model.train()

# print("Training complete.")

# from dotenv import load_dotenv
# load_dotenv()  # Load environment variables from .env

# import glob
# import os
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from transformers import AutoTokenizer, AutoModelForCausalLM
# from datasets import load_dataset
# from torch.utils.data import DataLoader
# from tqdm import tqdm
# import wandb

# # ============
# # Configuration
# # ============
# config = {
#     "model_name_or_path": "meta-llama/llama-3.2-1B",
#     "device": "cuda" if torch.cuda.is_available() else "cpu",
#     "num_epochs": 3,
#     "learning_rate": 1e-4,
#     "batch_size": 4,  # adjust depending on your GPU
#     "max_length": 512,  # max token length per sample (will be truncated/padded)
#     "lora_rank": 64,
#     "eval_interval": 50,  # steps between qualitative evaluations
#     "checkpoint_interval": 1000,  # checkpoint every 1000 steps (adjust as needed)
#     "wandb_project": "lora-compression-head",
#     "checkpoint_dir": os.getenv("CHECKPOINT_DIR", "/scr/tadimeti/checkpoints"),
# }

# # Create checkpoint directory if it doesn't exist.
# os.makedirs(config["checkpoint_dir"], exist_ok=True)

# # Initialize wandb.
# wandb.init(project=config["wandb_project"], config=config)
# cfg = wandb.config


# # ============
# # Compression Head Definition
# # ============
# class CompressionHead(nn.Module):
#     """
#     A simple compression head that takes a vector (e.g. the last hidden state
#     of the first half of a document) and outputs a weight delta (LoRA update)
#     for a target linear layer.

#     The target linear layer (e.g. a query or value projection) has weight shape (out_features, in_features).
#     """

#     def __init__(self, hidden_dim, target_in_features, target_out_features, rank=64):
#         super().__init__()
#         self.rank = rank
#         self.linear1 = nn.Linear(hidden_dim, rank)
#         self.linear2 = nn.Linear(
#             rank, (target_in_features + target_out_features) * rank
#         )
#         self.target_in_features = target_in_features
#         self.target_out_features = target_out_features

#     def forward(self, hidden):
#         # hidden: (batch_size, hidden_dim); assume batch_size == 1 for simplicity.
#         x = self.linear1(hidden)  # -> (batch_size, rank)
#         x = torch.nn.GELU()(x)  # Use GELU activation to avoid dead neurons.
#         x = self.linear2(x)  # -> (batch_size, target_in_features * target_out_features)
#         # Reshape to (batch_size, target_out_features, target_in_features)
#         delta_tall = x[:, : self.target_out_features * self.rank].reshape(
#             -1, self.target_out_features, self.rank
#         )
#         delta_wide = x[:, self.target_out_features * self.rank :].reshape(
#             -1, self.rank, self.target_in_features
#         )
#         # tall is of size (out, k) and wide is of size (k, in)
#         delta = delta_tall @ delta_wide
#         return delta.squeeze(0)  # -> (target_out_features, target_in_features)


# # ============
# # Load Model, Tokenizer, and Dataset
# # ============
# tokenizer = AutoTokenizer.from_pretrained(cfg.model_name_or_path)
# if tokenizer.pad_token is None:
#     # Use the EOS token as the padding token if none exists.
#     tokenizer.pad_token = tokenizer.eos_token

# model = AutoModelForCausalLM.from_pretrained(cfg.model_name_or_path)
# model.to(cfg.device)
# model.eval()  # keep base model frozen

# # Freeze all base model parameters.
# for param in model.parameters():
#     param.requires_grad = False

# # Load WikiText-2 (raw version) using the Hugging Face datasets library.
# dataset = load_dataset(
#     "wikitext",
#     "wikitext-2-raw-v1",
#     split="train",
#     cache_dir=os.getenv("HF_CACHE_DIR", "/scr/tadimeti/hf_cache/hub")
# )


# # Preprocessing: tokenize and truncate texts.
# def tokenize_function(ex):
#     return tokenizer(ex["text"], truncation=True, max_length=cfg.max_length)


# tokenized_dataset = dataset.map(tokenize_function, batched=False)
# # Filter out examples that are too short to split into two halves.
# tokenized_dataset = tokenized_dataset.filter(
#     lambda ex: ex["input_ids"] is not None and len(ex["input_ids"]) > 50
# )


# # DataLoader using fast tokenization (handles padding internally).
# def collate_fn(examples):
#     texts = [ex["text"] for ex in examples]
#     encodings = tokenizer(
#         texts,
#         padding="max_length",
#         truncation=True,
#         max_length=cfg.max_length,
#         return_tensors="pt",
#     )
#     return encodings["input_ids"]


# dataloader = DataLoader(
#     tokenized_dataset, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_fn
# )

# # ========
# # Load Validation Set
# # ========
# val_dataset = load_dataset(
#     "wikitext",
#     "wikitext-2-raw-v1",
#     split="validation",
#     cache_dir=os.getenv("HF_CACHE_DIR", "/scr/tadimeti/hf_cache/hub")
# )
# val_tokenized_dataset = val_dataset.map(tokenize_function, batched=False)
# val_tokenized_dataset = val_tokenized_dataset.filter(
#     lambda ex: ex["input_ids"] is not None and len(ex["input_ids"]) > 50
# )

# val_dataloader = DataLoader(
#     val_tokenized_dataset, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_fn
# )

# # ============
# # Choose Target Layers for LoRA Injection: q_proj and v_proj.
# # ============
# # For the query projection.
# target_layer_q = model.model.layers[0].self_attn.q_proj
# original_forward_q = target_layer_q.forward


# def lora_forward_q(input):
#     weight = target_layer_q.weight
#     if hasattr(target_layer_q, "lora_delta") and target_layer_q.lora_delta is not None:
#         weight = weight + target_layer_q.lora_delta
#     return F.linear(input, weight, target_layer_q.bias)


# target_layer_q.forward = lora_forward_q

# # For the value projection.
# target_layer_v = model.model.layers[0].self_attn.v_proj
# original_forward_v = target_layer_v.forward


# def lora_forward_v(input):
#     weight = target_layer_v.weight
#     if hasattr(target_layer_v, "lora_delta") and target_layer_v.lora_delta is not None:
#         weight = weight + target_layer_v.lora_delta
#     return F.linear(input, weight, target_layer_v.bias)


# target_layer_v.forward = lora_forward_v


# # ============
# # Helper Functions
# # ============
# def split_input_ids(input_ids):
#     """
#     Split each sample in a batch (shape: [batch, seq_len]) into first and second halves.
#     Assumes the sequence length is even.
#     """
#     seq_len = input_ids.shape[1]
#     half = seq_len // 2
#     first_half = input_ids[:, :half]
#     second_half = input_ids[:, half:]
#     return first_half, second_half


# def decode_tokens(updated_logits, base_logits, num_tokens=25):
#     """
#     Decode the first `num_tokens` token predictions from both updated and baseline logits.
#     """
#     updated_token_ids = torch.argmax(updated_logits[0, :num_tokens, :], dim=-1).tolist()
#     base_token_ids = torch.argmax(base_logits[0, :num_tokens, :], dim=-1).tolist()
#     updated_decoded = tokenizer.decode(updated_token_ids)
#     base_decoded = tokenizer.decode(base_token_ids)
#     return {"updated": updated_decoded, "base": base_decoded}


# def save_checkpoint(epoch, global_step, loss_value, best=False):
#     """
#     Save a checkpoint locally and upload it to wandb.
#     If best is True, save the checkpoint as the best model.
#     """
#     checkpoint = {
#         "epoch": epoch,
#         "global_step": global_step,
#         "compression_head_q_state_dict": compression_head_q.state_dict(),
#         "compression_head_v_state_dict": compression_head_v.state_dict(),
#         "optimizer_state_dict": optimizer.state_dict(),
#         "loss": loss_value,
#     }
#     prefix = "best_" if best else ""
#     checkpoint_filename = f"{prefix}checkpoint_epoch{epoch}_step{global_step}.pt"
#     checkpoint_path = os.path.join(cfg.checkpoint_dir, checkpoint_filename)
#     torch.save(checkpoint, checkpoint_path)
#     wandb.save(checkpoint_path)
#     print(f"Saved {'best ' if best else ''}checkpoint: {checkpoint_path}")


# def load_latest_checkpoint():
#     """
#     Load the latest checkpoint from the checkpoint directory, if any exist.
#     Returns (checkpoint, filename) or (None, None) if no checkpoint is found.
#     """
#     checkpoint_files = glob.glob(
#         os.path.join(cfg.checkpoint_dir, "checkpoint_epoch*_step*.pt")
#     )
#     if checkpoint_files:
#         latest_checkpoint_file = max(checkpoint_files, key=os.path.getctime)
#         checkpoint = torch.load(latest_checkpoint_file, map_location=cfg.device)
#         print(f"Loaded checkpoint: {latest_checkpoint_file}")
#         return checkpoint, latest_checkpoint_file
#     else:
#         print("No checkpoint found, starting from scratch.")
#         return None, None


# # ============
# # Initialize Compression Heads and Optimizer
# # ============
# hidden_dim = model.config.hidden_size  # e.g., 2048 or as defined by the model

# # For q_proj.
# q_out_features, q_in_features = target_layer_q.weight.shape
# compression_head_q = CompressionHead(
#     hidden_dim, q_in_features, q_out_features, rank=cfg.lora_rank
# )
# compression_head_q.to(cfg.device)

# # For v_proj.
# v_out_features, v_in_features = target_layer_v.weight.shape
# compression_head_v = CompressionHead(
#     hidden_dim, v_in_features, v_out_features, rank=cfg.lora_rank
# )
# compression_head_v.to(cfg.device)

# # Create an optimizer that updates both compression heads.
# optimizer = optim.Adam(
#     list(compression_head_q.parameters()) + list(compression_head_v.parameters()),
#     lr=cfg.learning_rate,
# )

# # Calculate total steps for cosine decay.
# total_steps = cfg.num_epochs * len(dataloader)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
#     optimizer, T_max=total_steps, eta_min=1e-6
# )

# # Try loading a checkpoint to resume training.
# start_epoch = 0
# global_step = 0
# checkpoint, ckpt_file = load_latest_checkpoint()
# if checkpoint is not None:
#     start_epoch = checkpoint["epoch"]
#     global_step = checkpoint["global_step"]
#     compression_head_q.load_state_dict(checkpoint["compression_head_q_state_dict"])
#     compression_head_v.load_state_dict(checkpoint["compression_head_v_state_dict"])
#     optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
#     print(f"Resuming training from epoch {start_epoch}, global step {global_step}.")

# best_val_loss = float('inf')

# # ============
# # Training Loop
# # ============
# print("Starting training loop...")
# model.train()  # Set model to train mode (for teacher forcing).
# for epoch in range(start_epoch, cfg.num_epochs):
#     epoch_loss = 0.0
#     for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
#         batch_loss = 0.0
#         for sample in batch:
#             sample = sample.unsqueeze(0).to(cfg.device)  # shape: (1, seq_len)
#             first_half, second_half = split_input_ids(sample)

#             # (A) Run first half to obtain summary hidden state.
#             with torch.no_grad():
#                 outputs_first = model(first_half, output_hidden_states=True)
#             last_hidden_state = outputs_first.hidden_states[-1][
#                 :, -1, :
#             ]  # shape: (1, hidden_dim)

#             # (B) Compute LoRA deltas for q_proj and v_proj.
#             q_delta = compression_head_q(
#                 last_hidden_state
#             )  # shape: (q_out_features, q_in_features)
#             v_delta = compression_head_v(
#                 last_hidden_state
#             )  # shape: (v_out_features, v_in_features)

#             # (C) Run modified model on second half (with injected deltas).
#             target_layer_q.lora_delta = q_delta
#             target_layer_v.lora_delta = v_delta
#             outputs_mod = model(second_half, output_hidden_states=False)
#             logits_mod = (
#                 outputs_mod.logits
#             )  # shape: (1, second_half_length, vocab_size)

#             # (D) Run baseline full-context model (no injection).
#             target_layer_q.lora_delta = None
#             target_layer_v.lora_delta = None
#             full_context = torch.cat([first_half, second_half], dim=1)
#             with torch.no_grad():
#                 outputs_full = model(full_context, output_hidden_states=False)
#             logits_full = outputs_full.logits
#             second_half_len = second_half.shape[1]
#             logits_full_second = logits_full[:, -second_half_len:, :]

#             # (E) Compute KL divergence loss.
#             log_probs_mod = F.log_softmax(logits_mod, dim=-1)
#             probs_full = F.softmax(logits_full_second, dim=-1)
#             loss = F.kl_div(log_probs_mod, probs_full, reduction="batchmean")

#             batch_loss += loss
#             global_step += 1

#             # Save a checkpoint every checkpoint_interval steps.
#             if global_step % cfg.checkpoint_interval == 0:
#                 save_checkpoint(epoch + 1, global_step, loss.item())

#             # ------------------------------------------------------------
#             # Evaluate on a subset of the validation set every eval_interval steps.
#             # ------------------------------------------------------------
#             if global_step % cfg.eval_interval == 0:
#                 subset_batches = 5  # Number of validation batches to use for evaluation.
#                 eval_loss_total = 0.0
#                 num_eval_samples = 0

#                 model.eval()
#                 with torch.no_grad():
#                     for i, val_batch in enumerate(val_dataloader):
#                         if i >= subset_batches:
#                             break
#                         for val_sample in val_batch:
#                             val_sample = val_sample.unsqueeze(0).to(cfg.device)
#                             fh, sh = split_input_ids(val_sample)
#                             outputs_first = model(fh, output_hidden_states=True)
#                             last_hidden_state = outputs_first.hidden_states[-1][:, -1, :]

#                             q_delta = compression_head_q(last_hidden_state)
#                             v_delta = compression_head_v(last_hidden_state)

#                             target_layer_q.lora_delta = q_delta
#                             target_layer_v.lora_delta = v_delta
#                             outputs_mod = model(sh, output_hidden_states=False)
#                             logits_mod = outputs_mod.logits

#                             target_layer_q.lora_delta = None
#                             target_layer_v.lora_delta = None
#                             full_context = torch.cat([fh, sh], dim=1)
#                             outputs_full = model(full_context, output_hidden_states=False)
#                             logits_full = outputs_full.logits
#                             sh_len = sh.shape[1]
#                             logits_full_second = logits_full[:, -sh_len:, :]

#                             log_probs_mod = F.log_softmax(logits_mod, dim=-1)
#                             probs_full = F.softmax(logits_full_second, dim=-1)
#                             eval_loss = F.kl_div(log_probs_mod, probs_full, reduction="batchmean")
#                             eval_loss_total += eval_loss.item()
#                             num_eval_samples += 1

#                 avg_eval_loss = eval_loss_total / num_eval_samples if num_eval_samples > 0 else float('inf')
#                 wandb.log({"eval_loss_subset": avg_eval_loss, "global_step": global_step})
#                 print(f"Step {global_step} - Eval subset loss: {avg_eval_loss:.4f}")
#                 model.train()
#             # ------------------------------------------------------------

#         # Average loss over samples in the batch.
#         batch_loss = batch_loss / batch.shape[0]
#         optimizer.zero_grad()
#         batch_loss.backward()
#         optimizer.step()
#         scheduler.step()  # Update the learning rate using cosine decay.

#         epoch_loss += batch_loss.item()
#         wandb.log(
#             {
#                 "loss": batch_loss.item(),
#                 "epoch": epoch + 1,
#                 "global_step": global_step,
#                 "learning_rate": optimizer.param_groups[0]["lr"],
#             }
#         )
    
#     avg_loss = epoch_loss / len(dataloader)
#     print(f"Epoch {epoch+1}/{cfg.num_epochs} - Average Training Loss: {avg_loss:.4f}")

#     # (Optional) Full validation evaluation at the end of the epoch if desired.
#     val_loss_total = 0.0
#     num_val_samples = 0
#     model.eval()
#     with torch.no_grad():
#         for val_batch in tqdm(val_dataloader, desc=f"Full Validation Epoch {epoch+1}"):
#             for sample in val_batch:
#                 sample = sample.unsqueeze(0).to(cfg.device)
#                 fh, sh = split_input_ids(sample)
#                 outputs_first = model(fh, output_hidden_states=True)
#                 last_hidden_state = outputs_first.hidden_states[-1][:, -1, :]

#                 q_delta = compression_head_q(last_hidden_state)
#                 v_delta = compression_head_v(last_hidden_state)

#                 target_layer_q.lora_delta = q_delta
#                 target_layer_v.lora_delta = v_delta
#                 outputs_mod = model(sh, output_hidden_states=False)
#                 logits_mod = outputs_mod.logits

#                 target_layer_q.lora_delta = None
#                 target_layer_v.lora_delta = None
#                 full_context = torch.cat([fh, sh], dim=1)
#                 outputs_full = model(full_context, output_hidden_states=False)
#                 logits_full = outputs_full.logits
#                 sh_len = sh.shape[1]
#                 logits_full_second = logits_full[:, -sh_len:, :]

#                 log_probs_mod = F.log_softmax(logits_mod, dim=-1)
#                 probs_full = F.softmax(logits_full_second, dim=-1)
#                 loss_val = F.kl_div(log_probs_mod, probs_full, reduction="batchmean")
#                 val_loss_total += loss_val.item()
#                 num_val_samples += 1
#     avg_val_loss = val_loss_total / num_val_samples if num_val_samples > 0 else float('inf')
#     wandb.log({"val_loss_full": avg_val_loss, "epoch": epoch + 1, "global_step": global_step})
#     print(f"Epoch {epoch+1}/{cfg.num_epochs} - Full Validation Loss: {avg_val_loss:.4f}")

#     # Save best full validation checkpoint if improved.
#     if avg_val_loss < best_val_loss:
#         best_val_loss = avg_val_loss
#         save_checkpoint(epoch + 1, global_step, avg_val_loss, best=True)

#     # Save a checkpoint at the end of the epoch (latest model).
#     save_checkpoint(epoch + 1, global_step, avg_loss)

#     model.train()

# print("Training complete.")
import glob
import os
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

# ============
# Configuration
# ============
config = {
    "model_name_or_path": "meta-llama/llama-3.2-3B",  # Generation model
    "bert_model_name": "bert-base-uncased",             # For encoding context
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "num_epochs": 3,
    "learning_rate": 1e-4,
    "batch_size": 4,         # adjust depending on your GPU
    "max_length": 512,       # max token length per sample (will be truncated/padded)
    "lora_rank": 64,
    "eval_interval": 50,     # steps between evaluations
    "checkpoint_interval": 1000,  # checkpoint every 1000 steps
    "wandb_project": "lora-compression-head",
    "checkpoint_dir": os.getenv("CHECKPOINT_DIR", "/scr/tadimeti/checkpoints"),
}

# ============
# CompressionHead with integrated Transformer layers (trainable)
# ============
class CompressionHead(nn.Module):
    """
    Processes a sequence (e.g. BERT's last hidden states) through a trainable mini transformer,
    then applies a final linear mapping that is factorized into two matrices (A and B). Their product
    forms the LoRA delta update.
    """
    def __init__(self, input_dim, target_in_features, target_out_features, rank=64,
                 num_transformer_layers=2, transformer_nhead=8, transformer_dim=None):
        super().__init__()
        self.rank = rank
        self.target_in_features = target_in_features
        self.target_out_features = target_out_features

        if transformer_dim is None:
            transformer_dim = input_dim
        self.transformer_dim = transformer_dim

        # Project input if needed.
        if input_dim != transformer_dim:
            self.input_proj = nn.Linear(input_dim, transformer_dim)
        else:
            self.input_proj = nn.Identity()

        # Define a TransformerEncoder (trainable)
        encoder_layer = nn.TransformerEncoderLayer(d_model=transformer_dim, nhead=transformer_nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)

        # Final linear layer that outputs a vector of size (target_in_features+target_out_features)*rank.
        self.fc = nn.Linear(transformer_dim, (target_in_features + target_out_features) * rank)

    def forward(self, x, attention_mask=None):
        # x: (batch, seq_len, input_dim)
        x = self.input_proj(x)  # -> (batch, seq_len, transformer_dim)
        # Permute to (seq_len, batch, transformer_dim) for TransformerEncoder
        x = x.transpose(0, 1)
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = attention_mask == 0
        x = self.transformer_encoder(x, src_key_padding_mask=key_padding_mask)
        # Permute back to (batch, seq_len, transformer_dim)
        x = x.transpose(0, 1)
        pooled = x.mean(dim=1)  # Mean pool over sequence -> (batch, transformer_dim)
        out = self.fc(pooled)   # (batch, (target_in+target_out)*rank)
        # Split output into two parts.
        part1 = out[:, :self.target_out_features * self.rank]
        part2 = out[:, self.target_out_features * self.rank:]
        delta_tall = part1.view(-1, self.target_out_features, self.rank)  # (batch, target_out, rank)
        delta_wide = part2.view(-1, self.rank, self.target_in_features)    # (batch, rank, target_in)
        delta = delta_tall @ delta_wide  # (batch, target_out, target_in)
        if delta.size(0) == 1:
            return delta.squeeze(0)
        return delta

# ============
# Helper Functions
# ============
def tokenize_function(ex, tokenizer, max_length):
    return tokenizer(ex["text"], truncation=True, max_length=max_length)

def collate_fn(examples, tokenizer, max_length):
    texts = [ex["text"] for ex in examples]
    encodings = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return encodings["input_ids"], encodings.get("attention_mask", None)

def split_input_ids(input_ids):
    seq_len = input_ids.shape[1]
    half = seq_len // 2
    return input_ids[:, :half], input_ids[:, half:]

def save_checkpoint(epoch, global_step, loss_value, comp_head_q, comp_head_v, optimizer, cfg, best=False):
    checkpoint = {
        "epoch": epoch,
        "global_step": global_step,
        "compression_head_q_state_dict": comp_head_q.state_dict(),
        "compression_head_v_state_dict": comp_head_v.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss_value,
    }
    prefix = "best_" if best else ""
    ckpt_filename = f"{prefix}checkpoint_epoch{epoch}_step{global_step}.pt"
    ckpt_path = os.path.join(cfg.checkpoint_dir, ckpt_filename)
    torch.save(checkpoint, ckpt_path)
    wandb.save(ckpt_path)
    print(f"Saved {'best ' if best else ''}checkpoint: {ckpt_path}")

def load_latest_checkpoint(cfg):
    ckpt_files = glob.glob(os.path.join(cfg.checkpoint_dir, "checkpoint_epoch*_step*.pt"))
    if ckpt_files:
        latest_ckpt = max(ckpt_files, key=os.path.getctime)
        checkpoint = torch.load(latest_ckpt, map_location=cfg.device)
        print(f"Loaded checkpoint: {latest_ckpt}")
        return checkpoint, latest_ckpt
    else:
        print("No checkpoint found, starting from scratch.")
        return None, None

# ============
# Main Training Pipeline
# ============
def main():
    wandb.init(project=config["wandb_project"], config=config)
    cfg = wandb.config
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    # Load LLaMA tokenizer (for the generation model)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load BERT tokenizer (for encoding context)
    bert_tokenizer = AutoTokenizer.from_pretrained(cfg.bert_model_name)

    # Load and freeze BERT (used only for encoding)
    bert_model = AutoModel.from_pretrained(cfg.bert_model_name)
    bert_model.to(cfg.device)
    bert_model.eval()
    for param in bert_model.parameters():
        param.requires_grad = False

    # Load and freeze the LLaMA generation model.
    model = AutoModelForCausalLM.from_pretrained(cfg.model_name_or_path)
    model.to(cfg.device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    # Load dataset and create dataloaders.
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train",
                             cache_dir=os.getenv("HF_CACHE_DIR", "/scr/tadimeti/hf_cache/hub"))
    tokenized_dataset = dataset.map(lambda ex: tokenize_function(ex, tokenizer, cfg.max_length), batched=False)
    tokenized_dataset = tokenized_dataset.filter(lambda ex: ex["input_ids"] is not None and len(ex["input_ids"]) > 50)
    dataloader = DataLoader(tokenized_dataset, batch_size=cfg.batch_size, shuffle=True,
                              collate_fn=lambda examples: collate_fn(examples, tokenizer, cfg.max_length))

    val_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation",
                               cache_dir=os.getenv("HF_CACHE_DIR", "/scr/tadimeti/hf_cache/hub"))
    val_tokenized_dataset = val_dataset.map(lambda ex: tokenize_function(ex, tokenizer, cfg.max_length), batched=False)
    val_tokenized_dataset = val_tokenized_dataset.filter(lambda ex: ex["input_ids"] is not None and len(ex["input_ids"]) > 50)
    val_dataloader = DataLoader(val_tokenized_dataset, batch_size=cfg.batch_size, shuffle=False,
                                collate_fn=lambda examples: collate_fn(examples, tokenizer, cfg.max_length))

    # Override target layers with LoRA injection.
    target_layer_q = model.model.layers[0].self_attn.q_proj
    def lora_forward_q(input):
        weight = target_layer_q.weight
        if hasattr(target_layer_q, "lora_delta") and target_layer_q.lora_delta is not None:
            weight = weight + target_layer_q.lora_delta
        return F.linear(input, weight, target_layer_q.bias)
    target_layer_q.forward = lora_forward_q

    target_layer_v = model.model.layers[0].self_attn.v_proj
    def lora_forward_v(input):
        weight = target_layer_v.weight
        if hasattr(target_layer_v, "lora_delta") and target_layer_v.lora_delta is not None:
            weight = weight + target_layer_v.lora_delta
        return F.linear(input, weight, target_layer_v.bias)
    target_layer_v.forward = lora_forward_v

    # Get hidden size of LLaMA.
    hidden_dim = model.config.hidden_size  # e.g., 2048

    # Get weight shapes for target layers.
    q_out_features, q_in_features = target_layer_q.weight.shape
    v_out_features, v_in_features = target_layer_v.weight.shape

    # Initialize the Compression Heads.
    # Here, input_dim is BERT's hidden size (768). You can adjust transformer_dim if desired.
    comp_head_q = CompressionHead(input_dim=768, target_in_features=q_in_features,
                                  target_out_features=q_out_features, rank=cfg.lora_rank,
                                  num_transformer_layers=2, transformer_nhead=8, transformer_dim=768)
    comp_head_q.to(cfg.device)
    comp_head_v = CompressionHead(input_dim=768, target_in_features=v_in_features,
                                  target_out_features=v_out_features, rank=cfg.lora_rank,
                                  num_transformer_layers=2, transformer_nhead=8, transformer_dim=768)
    comp_head_v.to(cfg.device)

    # Initialize optimizer for the trainable compression heads.
    optimizer = optim.Adam(list(comp_head_q.parameters()) + list(comp_head_v.parameters()),
                           lr=cfg.learning_rate)
    total_steps = cfg.num_epochs * len(dataloader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)

    # Set up AMP GradScaler.
    scaler = torch.cuda.amp.GradScaler()

    # Load checkpoint if available.
    start_epoch = 0
    global_step = 0
    checkpoint, _ = load_latest_checkpoint(cfg)
    if checkpoint is not None:
        start_epoch = checkpoint["epoch"]
        global_step = checkpoint["global_step"]
        comp_head_q.load_state_dict(checkpoint["compression_head_q_state_dict"])
        comp_head_v.load_state_dict(checkpoint["compression_head_v_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print(f"Resuming training from epoch {start_epoch}, global step {global_step}.")

    best_val_loss = float('inf')

    print("Starting training loop...")
    model.train()  # teacher forcing mode
    for epoch in range(start_epoch, cfg.num_epochs):
        epoch_loss = 0.0
        for batch_input_ids, _ in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
            batch_loss = 0.0
            optimizer.zero_grad()
            # Process samples in the batch one by one to reduce peak memory usage.
            for sample in batch_input_ids:
                sample = sample.unsqueeze(0).to(cfg.device)  # (1, seq_len)
                first_half, second_half = split_input_ids(sample)

                # --- Encode first half using frozen BERT ---
                text_first_half = tokenizer.decode(first_half[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
                # Optional debug print:
                # print("Decoded text:", text_first_half)
                bert_encoding = bert_tokenizer(
                    text_first_half,
                    return_tensors="pt",
                    max_length=cfg.max_length,
                    truncation=True,
                    padding="max_length",
                    add_special_tokens=True
                )
                bert_input_ids = bert_encoding["input_ids"].to(cfg.device)
                bert_attention_mask = bert_encoding.get("attention_mask", None)
                if bert_attention_mask is not None:
                    bert_attention_mask = bert_attention_mask.to(cfg.device)
                with torch.no_grad():
                    bert_outputs = bert_model(input_ids=bert_input_ids,
                                              attention_mask=bert_attention_mask,
                                              output_hidden_states=True)
                bert_seq = bert_outputs.last_hidden_state  # (1, seq_len, 768)

                # --- Compute LoRA deltas with AMP ---
                with torch.cuda.amp.autocast():
                    q_delta = comp_head_q(bert_seq, attention_mask=bert_attention_mask)
                    v_delta = comp_head_v(bert_seq, attention_mask=bert_attention_mask)
                    # Inject deltas and run modified model on second half.
                    target_layer_q.lora_delta = q_delta
                    target_layer_v.lora_delta = v_delta
                    outputs_mod = model(second_half, output_hidden_states=False)
                    logits_mod = outputs_mod.logits  # (1, seq_len_second, vocab_size)
                    # Baseline: full-context model (without injection).
                    target_layer_q.lora_delta = None
                    target_layer_v.lora_delta = None
                    full_context = torch.cat([first_half, second_half], dim=1)
                    with torch.no_grad():
                        outputs_full = model(full_context, output_hidden_states=False)
                    logits_full = outputs_full.logits
                    second_half_len = second_half.shape[1]
                    logits_full_second = logits_full[:, -second_half_len:, :]
                    log_probs_mod = F.log_softmax(logits_mod, dim=-1)
                    probs_full = F.softmax(logits_full_second, dim=-1)
                    loss = F.kl_div(log_probs_mod, probs_full, reduction="batchmean")

                # Accumulate loss and perform backward immediately per sample.
                scaler.scale(loss).backward()
                batch_loss += loss.detach().item()
                global_step += 1

                if global_step % cfg.checkpoint_interval == 0:
                    save_checkpoint(epoch + 1, global_step, loss.item(),
                                    comp_head_q, comp_head_v, optimizer, cfg)

                if global_step % cfg.eval_interval == 0:
                    # (Evaluation code omitted for brevity; similar modifications with AMP can be applied.)
                    pass

            # Step optimizer for the batch.
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            epoch_loss += batch_loss / batch_input_ids.shape[0]
            wandb.log({"loss": batch_loss / batch_input_ids.shape[0],
                       "epoch": epoch + 1,
                       "global_step": global_step,
                       "learning_rate": optimizer.param_groups[0]["lr"]})
        
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{cfg.num_epochs} - Average Training Loss: {avg_loss:.4f}")

        # (Full validation evaluation omitted for brevity; apply similar changes with AMP.)
        save_checkpoint(epoch + 1, global_step, avg_loss,
                        comp_head_q, comp_head_v, optimizer, cfg)
        model.train()

    print("Training complete.")

if __name__ == "__main__":
    # For debugging CUDA errors, you can set:
    # os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    main()
