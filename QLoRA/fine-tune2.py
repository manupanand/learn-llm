# =============================================================================
# fine-tune.py — QLoRA Fine-tuning Pipeline for Llama 3.2 1B
# Hardware Target: NVIDIA GB10 Blackwell 120GB VRAM
# Dataset: Small Excel Q&A (20 rows) — Company Knowledge Bot
# =============================================================================
# BLOCK 1: IMPORTS
# These are the core libraries you need. Think of them as:
# - torch: the engine (like NumPy but for GPU + automatic differentiation)
# - transformers: Hugging Face library that wraps pretrained LLMs
# - peft: Parameter Efficient Fine-Tuning (LoRA lives here)
# - trl: Trainer specifically built for LLM fine-tuning (wraps transformers Trainer)
# - bitsandbytes: enables 4-bit quantization (loads heavy models in less VRAM)
# =============================================================================

import os
import torch
import pandas as pd
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, PeftModel, get_peft_model
from trl import SFTTrainer

# =============================================================================
# BLOCK 2: DATA PREPARATION
# You read Excel → convert to CSV → load as HuggingFace dataset
# Why CSV? HuggingFace's load_dataset works natively with CSV
# Your Excel must have columns: "Questions" and "Answers"
# =============================================================================

xlsx_file = pd.read_excel('training.xlsx')
xlsx_file.to_csv('output.csv', index=False)

# =============================================================================
# BLOCK 3: PROMPT TEMPLATE
# This is the instruction format you're teaching the model to follow.
# {} are placeholders — Python's .format() fills them at runtime.
# The model learns: "when I see ### Input, I respond after ### Response"
# EOS_TOKEN (end-of-sequence) tells the model where generation should STOP.
# Without it, the model generates forever — critical bug if missing.
# =============================================================================

input_prompt = """Below is a Human Input, write appropriate Response based on the input.

### Input:
{}

### Response:
{}"""

# =============================================================================
# BLOCK 4: CONFIGURATION — MODEL & TRAINING SETTINGS
# =============================================================================

# The pretrained base model from Hugging Face Hub
# 1B parameters = 1 billion weights — small but capable for Q&A
model_name = "unsloth/Llama-3.2-1B-Instruct"

# Name for your fine-tuned model (saved locally)
new_model = "llama3.2-fine-tuned"

# =============================================================================
# BLOCK 5: QLoRA / LoRA PARAMETERS
# LoRA = Low-Rank Adaptation. Instead of updating ALL 1B weights (expensive),
# you inject small trainable matrices into the model.
# Think of it like CFD mesh refinement — you only refine the important zones.
#
# lora_r: rank of the adapter matrices. Higher = more capacity but more VRAM.
#         For 20 rows, r=16 is enough. r=64 risks overfitting.
#         CORRECTED: 64 → 16
#
# lora_alpha: scaling factor. Rule of thumb: alpha = 2 * r
#             CORRECTED: 16 → 32
#
# lora_dropout: regularization to prevent overfitting.
#               With only 20 rows, keep dropout at 0.05 (small but present)
#               CORRECTED: 0.1 → 0.05
# =============================================================================

lora_r = 16              # CORRECTED from 64 — 20 rows does not need r=64
lora_alpha = 32          # CORRECTED from 16 — should be 2x lora_r
lora_dropout = 0.05      # CORRECTED from 0.1 — less dropout for tiny dataset

# =============================================================================
# BLOCK 6: BITSANDBYTES (QUANTIZATION) PARAMETERS
# On your GB10 with 120GB VRAM, a 1B model in bf16 = ~2GB VRAM.
# You do NOT need 4-bit quantization at all.
# BUT keeping it doesn't hurt and shows you understand the pattern.
# nf4 = NormalFloat4 — better than fp4 for LLM weight distributions
# double quantization = quantize the quantization constants (saves ~0.4GB/B params)
# =============================================================================

use_4bit = True                          # Can set False on GB10 — you have headroom
bnb_4bit_compute_dtype = "bfloat16"     # CORRECTED: float16 → bfloat16 (Blackwell supports bf16)
bnb_4bit_quant_type = "nf4"             # Keep nf4 — better than fp4
use_nested_quant = True                  # CORRECTED: False → True — free VRAM saving

# =============================================================================
# BLOCK 7: TRAINING ARGUMENTS
# These control the training loop — how long, how fast, how to optimize.
# =============================================================================

output_dir = "./results"

# CORRECTED: 20 epochs → 5 epochs
# With 20 rows and 20 epochs the model will memorize, not learn.
# 3-5 epochs is sufficient for tiny instruction datasets.
num_train_epochs = 5

# CORRECTED: fp16=False, bf16=False → bf16=True
# Your GB10 Blackwell supports bf16 natively. It's faster and more stable.
# bf16 = 16-bit brain float — better numerical range than fp16 for training.
fp16 = False
bf16 = True              # CORRECTED — use bf16 on Blackwell

# Batch size of 1 is fine for a tiny dataset
per_device_train_batch_size = 4          # CORRECTED: 1 → 4, you have 120GB VRAM use it

per_device_eval_batch_size = 4           # CORRECTED: 1 → 4

# Gradient accumulation: simulates larger batch size without more VRAM
# effective_batch = per_device_batch * gradient_accumulation_steps
# CORRECTED: 1 → 4 — gives effective batch of 16
gradient_accumulation_steps = 4

# Gradient checkpointing: trades compute for VRAM
# Recomputes activations during backward pass instead of storing them
# On 120GB with 1B model, you don't need this — but it's good practice
gradient_checkpointing = True

# Gradient clipping: prevents exploding gradients
# Think of it like a limiter in your CFD solver — caps the update magnitude
max_grad_norm = 0.3

# CORRECTED: 2e-4 → 2e-5
# 2e-4 is aggressive for a 20-row dataset — causes instability
# Lower LR = smaller weight updates = less risk of forgetting pretrained knowledge
learning_rate = 2e-5

weight_decay = 0.01      # CORRECTED: 0.001 → 0.01 — more regularization for tiny data

optim = "paged_adamw_32bit"   # Memory-efficient AdamW optimizer

# Cosine LR schedule: LR starts at learning_rate, decays following a cosine curve
# Better than linear for fine-tuning — smooth decay prevents oscillation
lr_scheduler_type = "cosine"

max_steps = -1           # -1 means use num_train_epochs instead

# Warmup: LR starts at 0, ramps up to learning_rate over warmup_ratio of steps
# Prevents large updates at the start when weights are still "cold"
warmup_ratio = 0.03

# Groups sequences of similar length together in batches
# Reduces padding waste — efficiency optimization
group_by_length = True

# CORRECTED: 0 → 25 — save checkpoint every 25 steps so you don't lose progress
save_steps = 25

logging_steps = 5        # CORRECTED: 25 → 5 — log more frequently for tiny dataset

# =============================================================================
# BLOCK 8: SFT (SUPERVISED FINE-TUNING) PARAMETERS
# =============================================================================

# max_seq_length: maximum number of tokens per training example
# None = use model default. CORRECTED: set explicitly to 512
# Your Q&A pairs are short — 512 is more than enough and saves VRAM
max_seq_length = 512     # CORRECTED: None → 512

# Packing: combines multiple short examples into one sequence to fill context
# CORRECTED: False → True — with 20 rows you want to pack for efficiency
packing = True           # CORRECTED: False → True

# Device map: which GPU to load model on
# "auto" = HuggingFace decides intelligently
# CORRECTED: {"": 0} → "auto" for multi-GPU awareness (even on single GPU)
device_map = "auto"      # CORRECTED: {"": 0} → "auto"

# =============================================================================
# BLOCK 9: BUILD BitsAndBytes CONFIG
# This object tells the model loader HOW to quantize the weights at load time
# =============================================================================

compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
print(f"Compute dtype: {compute_dtype}")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=use_nested_quant,
)

# Check GPU bfloat16 support
if compute_dtype == torch.float16 and use_4bit:
    major, _ = torch.cuda.get_device_capability()
    if major >= 8:
        print("=" * 80)
        print("Your GPU supports bfloat16: accelerate training with bf16=True")
        print("=" * 80)

# =============================================================================
# BLOCK 10: LOAD BASE MODEL + TOKENIZER
# AutoModelForCausalLM: loads the LLM (decoder-only transformer, like GPT)
# "Causal" means each token can only attend to previous tokens — left to right
# AutoTokenizer: loads the tokenizer that converts text → token IDs → text
# =============================================================================

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map=device_map,
    torch_dtype=compute_dtype,         # ADDED: explicit dtype
    attn_implementation="flash_attention_2",  # ADDED: Flash Attention for speed
)

# use_cache=False required when gradient_checkpointing=True
# They conflict — cache stores KV states, checkpointing discards intermediate states
model.config.use_cache = False

# pretraining_tp=1: tensor parallelism during pretraining (keep at 1 for fine-tuning)
model.config.pretraining_tp = 1

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# pad_token: used to pad shorter sequences in a batch to equal length
# Llama has no pad token by default — using eos_token as pad is standard practice
tokenizer.pad_token = tokenizer.eos_token

# padding_side="right": pad on the right side of sequences
# Important for causal LMs — left padding can confuse the attention mask
tokenizer.padding_side = "right"

EOS_TOKEN = tokenizer.eos_token

# =============================================================================
# BLOCK 11: DATASET FORMATTING FUNCTION
# This function takes raw Q&A rows and wraps them in the prompt template.
# The model trains on the FULL text (input + output) in causal LM style.
# EOS_TOKEN is appended so the model knows when to stop generating.
# =============================================================================

def formatting_prompts_func(examples):
    inputs = examples["Questions"]
    outputs = examples["Answers"]
    texts = []
    for input_text, output_text in zip(inputs, outputs):
        # Fill the prompt template with actual Q&A
        text = input_prompt.format(input_text, output_text) + EOS_TOKEN
        texts.append(text)
    return {"text": texts}

# Load CSV as HuggingFace Dataset object
dataset = load_dataset('csv', data_files='output.csv', split="train")

# Apply formatting to every row in the dataset
# batched=True: processes multiple rows at once — more efficient
dataset = dataset.map(formatting_prompts_func, batched=True)

print(f"Dataset size: {len(dataset)} rows")
print(f"Sample: {dataset[0]['text'][:200]}")

# =============================================================================
# BLOCK 12: LoRA CONFIGURATION
# target_modules: which weight matrices to inject LoRA adapters into
# q_proj, k_proj, v_proj, o_proj = attention layers (Query, Key, Value, Output)
# gate_proj, up_proj, down_proj = FFN layers (Feed Forward Network)
# Targeting all 7 is standard for Llama — gives good coverage
# =============================================================================

peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",          # Don't add LoRA to bias terms — standard practice
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
)

# =============================================================================
# BLOCK 13: TRAINING ARGUMENTS
# This is the configuration object for the HuggingFace Trainer
# Controls: epochs, batch size, LR, logging, checkpointing, precision
# =============================================================================

training_arguments = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    save_steps=save_steps,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    fp16=fp16,
    bf16=bf16,
    max_grad_norm=max_grad_norm,
    max_steps=max_steps,
    warmup_ratio=warmup_ratio,
    group_by_length=group_by_length,
    lr_scheduler_type=lr_scheduler_type,
    report_to="tensorboard",
    save_total_limit=2,              # ADDED: only keep last 2 checkpoints
    load_best_model_at_end=False,    # ADDED: explicit — no eval set so False
    dataloader_num_workers=4,        # ADDED: parallel data loading
)

# =============================================================================
# BLOCK 14: SFTTrainer — Supervised Fine-Tuning Trainer
# SFTTrainer = HuggingFace Trainer specialized for instruction fine-tuning
# It handles: tokenization, loss masking, LoRA integration, training loop
# =============================================================================

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="text",   # Which column in dataset to use
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=training_arguments,
    packing=packing,
)

# =============================================================================
# BLOCK 15: TRAIN
# This runs the actual training loop:
# forward pass → compute loss → backward pass → update LoRA weights → repeat
# =============================================================================

print("Starting training...")
trainer.train()

# Save the LoRA adapter weights (NOT the full model — just the deltas)
trainer.model.save_pretrained(new_model)
tokenizer.save_pretrained(new_model)  # ADDED: always save tokenizer with model
print(f"LoRA adapter saved to: {new_model}")

# =============================================================================
# BLOCK 16: INFERENCE TEST (before merging)
# Test the fine-tuned model before merging weights
# =============================================================================

print("\n--- Inference Test ---")
inputs = tokenizer(
    [
        input_prompt.format(
            "who is nandakishor?",   # your test question
            "",                       # leave blank — model fills this
        )
    ],
    return_tensors="pt"
).to("cuda")

outputs = model.generate(
    **inputs,
    max_new_tokens=100,
    use_cache=True,
    temperature=0.1,          # ADDED: low temperature = more deterministic
    do_sample=True,           # ADDED: required when temperature != 1.0
    pad_token_id=tokenizer.eos_token_id,  # ADDED: suppress padding warning
)

generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
first_response = generated_text.split('### Response:')[1].strip()
output_text = first_response.split('###')[0].strip()
print("Response:", output_text)

# =============================================================================
# BLOCK 17: MERGE LoRA WEIGHTS INTO BASE MODEL
# LoRA trains delta weights (ΔW). Merging adds them to base: W_final = W + ΔW
# After merging you get a standalone model — no PEFT library needed at inference
# This is what you serve with vLLM
# =============================================================================

print("\nMerging LoRA weights into base model...")

# Reload base model in bf16 (full precision — no quantization for final model)
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.bfloat16,   # CORRECTED: float16 → bfloat16 for Blackwell
    device_map=device_map,
)

# Load LoRA adapter on top of base model
model_with_lora = PeftModel.from_pretrained(base_model, new_model)

# Merge and unload: adds ΔW into W, removes PEFT overhead
merged_model = model_with_lora.merge_and_unload()

# =============================================================================
# BLOCK 18: SAVE FINAL MERGED MODEL
# Saves in HuggingFace safetensors format — safe, fast, vLLM compatible
# safetensors = safer alternative to pickle (.bin) — no arbitrary code execution
# =============================================================================

output_dir = "final_weights_new"
merged_model.save_pretrained(output_dir, safe_serialization=True)  # ADDED: safe_serialization=True for .safetensors

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
tokenizer.save_pretrained(output_dir)

print(f"Merged model saved to: {output_dir}")
print("Files in output dir:")
for f in os.listdir(output_dir):
    print(f"  {f}")

# =============================================================================
# BLOCK 19: PUSH TO HUGGING FACE HUB
# HfApi: programmatic interface to HuggingFace Hub
# You create a repo and upload the entire model folder
# SECURITY NOTE: Never hardcode HF_TOKEN in code — use environment variable
# =============================================================================

from huggingface_hub import HfApi, login

# CORRECTED: read from environment variable — never hardcode tokens
HF_TOKEN = os.environ.get("HF_TOKEN", "")
if not HF_TOKEN:
    print("WARNING: HF_TOKEN environment variable not set. Skipping upload.")
else:
    login(token=HF_TOKEN)

    readme_content = """# Llama 3.2 1B Fine-tuned with Q-LoRA

Fine-tuned version of `unsloth/Llama-3.2-1B-Instruct` on custom Q&A data using QLoRA.

## Model Details
- **Base Model:** Llama 3.2 1B Instruct
- **Method:** QLoRA (4-bit quantization + LoRA adapters, merged to full precision)
- **Task:** Instruction Following / Company Knowledge Q&A
- **Format:** SafeTensors (bf16)

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_id = "your-username/llama3.2-fine-tuned"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map='auto',
    torch_dtype=torch.bfloat16
)

prompt = \"\"\"Below is a Human Input, write appropriate Response based on the input.

### Input:
who is nandakishor?

### Response:
\"\"\"

inputs = tokenizer(prompt, return_tensors='pt').to('cuda')
outputs = model.generate(**inputs, max_new_tokens=100, temperature=0.1, do_sample=True)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```
"""

    with open(f"{output_dir}/README.md", "w") as f:
        f.write(readme_content)

    api = HfApi()
    user_name = "your-username"                          # REPLACE with your HF username
    repo_id = f"{user_name}/llama3.2-fine-tuned"

    try:
        api.create_repo(repo_id=repo_id, exist_ok=True)
        api.upload_folder(
            folder_path=output_dir,
            repo_id=repo_id,
            repo_type="model"
        )
        print(f"Successfully pushed to https://huggingface.co/{repo_id}")
    except Exception as e:
        print(f"Error uploading: {e}")
