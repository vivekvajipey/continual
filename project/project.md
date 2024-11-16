# Context Compression with LoRA

## Project Overview

### Inspiration
This project builds on Professor Noah Goodman's observation about human learning: while humans have limited working memory (context), we excel at continual learning by converting context into weight updates. The goal is to explore whether we can achieve similar capabilities in language models by compressing context information into LoRA weight patches.

### Core Idea
Develop methods to compress context information from an LLM's input window into compact LoRA parameter updates, allowing the model to maintain the context's information without keeping the full text in the input window.

### Related Work

#### Gist Tokens (Mu et al.)
The primary methodological inspiration comes from "Learning to Compress Prompts with Gist Tokens" which demonstrates:
- Successful compression of prompts into small sets of "gist tokens"
- Up to 26x compression rates while maintaining output quality
- Modified attention masking technique to force context compression
- Comprehensive evaluation framework using multiple metrics

#### Continual Learning in LLMs
Recent work highlights two relevant types of continuity:
1. Vertical Continuity: Adaptation from general to specific capabilities
2. Horizontal Continuity: Adaptation across time and domains

Our work primarily addresses vertical continuity by compressing specific context into weight updates, though there are potential applications to horizontal continuity when managing multiple context compressions over time.

## Technical Approach

### Phase 1: LoRA-based Replication of Gisting
The first phase will replicate the experiments from the Gist token paper but using LoRA parameters instead of gist tokens.

#### Data
Using Alpaca+ dataset (following Mu et al.):
- 130,321 total examples
- 104,664 unique tasks
- Validation splits:
  - 1000 Seen prompts (unseen inputs)
  - 1000 Unseen prompts
  - 252 Human prompts (OOD test)

#### Implementation Tools

1. UnSloth for efficient LoRA training:
```python
from unsloth import FastLanguageModel
from trl import SFTTrainer

# Model loading
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Meta-Llama-3.1-8B",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit
)

# Training setup
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        learning_rate = 2e-4,
        max_steps = 60,
        # Additional training arguments...
    )
)
```

### Experimental Design

#### Key Variables to Test
1. LoRA configurations:
   - Rank dimensions (r=2, 8, 16, etc.)
   - Learning rates
   - Training steps
   - Amount of training data needed

2. Compression effectiveness:
   - Compression rates achievable
   - Output quality preservation
   - Computational efficiency gains

#### Evaluation Metrics
Following Gist paper methodology:
1. ROUGE-L scores compared to original outputs
2. ChatGPT evaluation of output quality
3. Human evaluation on subset of examples
4. Computational metrics:
   - FLOPs reduction
   - Wall clock time improvements
   - Memory/storage savings

### Success Criteria
1. Compression Rate: Achieve compression rates comparable to gist tokens (up to 26x)
2. Quality: Maintain output quality within 5% of uncompressed baseline
3. Efficiency: Demonstrate meaningful reduction in computation costs
4. Generalization: Show effectiveness on unseen prompts

## Future Directions

### Phase 2: Direct LoRA Generation
Once basic compression is validated, explore Professor Goodman's proposal of generating LoRA patches directly:
1. Design architecture for LoRA patch generation
2. Develop training methodology
3. Compare efficiency vs. optimization-based approach