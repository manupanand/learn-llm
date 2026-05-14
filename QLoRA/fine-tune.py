#fine-tune.py
import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer
import pandas as pd

# Read the XLSX file
xlsx_file = pd.read_excel('/content/training.xlsx')

# Convert XLSX to CSV
xlsx_file.to_csv('output.csv', index=False)
input_prompt = """Below is a Human Input, write appropriate Response based on the input.

### Input:
{}

### Response:
{}"""
# The model that you want to train from the Hugging Face hub
#model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
model_name = "unsloth/Llama-3.2-1B-Instruct" #pretrained model
#model_mame = "/content/final_weights_new"

# The instruction dataset to use
#dataset_name = "mlabonne/guanaco-llama2-1k"

# Fine-tuned model name
new_model = "llama3.2-fine-tuned"

################################################################################
# QLoRA parameters
################################################################################

# LoRA attention dimension
lora_r = 64

# Alpha parameter for LoRA scaling
lora_alpha = 16

# Dropout probability for LoRA layers
lora_dropout = 0.1

################################################################################
# bitsandbytes parameters
################################################################################

# Activate 4-bit precision base model loading
use_4bit = True

# Compute dtype for 4-bit base models
bnb_4bit_compute_dtype = "float16"

# Quantization type (fp4 or nf4)
bnb_4bit_quant_type = "nf4"

# Activate nested quantization for 4-bit base models (double quantization)
use_nested_quant = False

################################################################################
# TrainingArguments parameters
################################################################################

# Output directory where the model predictions and checkpoints will be stored
output_dir = "./results"

# Number of training epochs
num_train_epochs = 20

# Enable fp16/bf16 training (set bf16 to True with an A100)
fp16 = False
bf16 = False

# Batch size per GPU for training
per_device_train_batch_size = 1

# Batch size per GPU for evaluation
per_device_eval_batch_size = 1

# Number of update steps to accumulate the gradients for
gradient_accumulation_steps = 1

# Enable gradient checkpointing
gradient_checkpointing = True

# Maximum gradient normal (gradient clipping)
max_grad_norm = 0.3

# Initial learning rate (AdamW optimizer)
learning_rate = 2e-4 #0.0002 2x10-4

# Weight decay to apply to all layers except bias/LayerNorm weights
weight_decay = 0.001

# Optimizer to use
optim = "paged_adamw_32bit"

# Learning rate schedule
lr_scheduler_type = "cosine"

# Number of training steps (overrides num_train_epochs)
max_steps = -1

# Ratio of steps for a linear warmup (from 0 to learning rate)
warmup_ratio = 0.03

# Group sequences into batches with same length
# Saves memory and speeds up training considerably
group_by_length = True

# Save checkpoint every X updates steps
save_steps = 0

# Log every X updates steps
logging_steps = 25

################################################################################
# SFT parameters
################################################################################

# Maximum sequence length to use
max_seq_length = None

# Pack multiple short examples in the same input sequence to increase efficiency
packing = False

# Load the entire model on the GPU 0, 4 gpu=0,1,2,3
device_map = {"": 0} # "auto"
# Load dataset (you can process it here)
#dataset = load_dataset(dataset_name, split="train")
%cd "/content"
# Load tokenizer and model with QLoRA configuration
compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
print(compute_dtype)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=use_nested_quant,
)

# Check GPU compatibility with bfloat16
if compute_dtype == torch.float16 and use_4bit:
    major, _ = torch.cuda.get_device_capability()
    if major >= 8:
        print("=" * 80)
        print("Your GPU supports bfloat16: accelerate training with bf16=True")
        print("=" * 80)

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map=device_map
)
model.config.use_cache = False
model.config.pretraining_tp = 1

# Load LLaMA tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training
EOS_TOKEN = tokenizer.eos_token
def formatting_prompts_func(examples):
    inputs       = examples["Questions"]
    outputs      = examples["Answers"]
    texts = []
    for input, output in zip(inputs, outputs):
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        text = input_prompt.format(input, output) + EOS_TOKEN
        texts.append(text)
    print(texts)
    return { "text" : texts, }
pass
'''
def formatting_prompts_func(examples):
    inputs       = examples["instruction"]
    outputs      = examples["output"]
    texts = []
    for input, output in zip(inputs, outputs):
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        text = input_prompt.format(input, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }
pass'''

from datasets import load_dataset
dataset = load_dataset('csv', data_files='output.csv',split="train")
#dataset = load_dataset("nmdr/Mini-Physics-Instruct-1k", split = "train")
dataset = dataset.map(formatting_prompts_func, batched = True,)
# Load LoRA configuration
peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
    task_type="CAUSAL_LM",
      target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
)

# Set training parameters
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

)

# Set supervised fine-tuning parameters
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=training_arguments,
    packing=packing,

)

# Train model
trainer.train()

# Save trained model
trainer.model.save_pretrained(new_model)
##Inference
inputs = tokenizer(
[
    input_prompt.format(
        "who is nandakishor?", # input
        "",   # leave blank as response generated by AI

    )
], return_tensors = "pt").to("cuda")

outputs = model.generate(**inputs, max_new_tokens = 100, use_cache = True)
generated_text = tokenizer.batch_decode(outputs)[0]
first_response = generated_text.split('### Response:')[1].strip()
output = first_response.split('###')[0].strip()
print("the response is: ",output)
# Reload model in FP16 and merge it with LoRA weights w = w+del(w)
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map=device_map,
)
model = PeftModel.from_pretrained(base_model, new_model)
model = model.merge_and_unload() #W=w+del(w)

# Reload tokenizer to save it
#tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
#tokenizer.pad_token = tokenizer.eos_token
#tokenizer.padding_side = "right"
output_dir = "final_weights_new"
model.save_pretrained(output_dir)

# Reload tokenizer to save it
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
tokenizer.save_pretrained(output_dir)
from huggingface_hub import HfApi, login
import os

# 1. Login to Hugging Face
HF_TOKEN = "HF_TOKEN"  # Replace with your actual token
login(token=HF_TOKEN)

# 2. Create a README.md file with model details and usage instructions
readme_content = """# Llama 3.2 1B Fine-tuned with Q-LoRA

This model is a fine-tuned version of `unsloth/Llama-3.2-1B-Instruct` trained on custom instructional data using Q-LoRA for efficient parameter adaptation.

## Model Details
- **Base Model:** Llama 3.2 1B Instruct
- **Method:** Q-LoRA (4-bit quantization with LoRA adapters)
- **Task:** Causal Language Modeling / Instruction Following

## How to Run

You can load and run this model using the `transformers` library:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_id = "your-username/llama3.2-fine-tuned"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map='auto',
    torch_dtype=torch.float16
)

input_text = "Below is a Human Input, write appropriate Response based on the input.\\n\\n### Input:\\nwho is nandakishor?\\n\\n### Response:\\n"
inputs = tokenizer(input_text, return_tensors='pt').to('cuda')
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```
"""

with open("/content/final_weights_new/README.md", "w") as f:
    f.write(readme_content)

# 3. Initialize the API and upload the folder
api = HfApi()
user_name = "username"
repo_id = "{}/llama3.2-fine-tuned".format(user_name)# Replace with your username/repo-name

try:
    api.create_repo(repo_id=repo_id, exist_ok=True)
    api.upload_folder(
        folder_path="/content/final_weights_new",
        repo_id=repo_id,
        repo_type="model"
    )
    print(f"Successfully pushed to https://huggingface.co/{repo_id}")
except Exception as e:
    print(f"Error uploading: {e}")

# Run text generation pipeline with our next model
# Load model directly </s>
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline

tokenizer = AutoTokenizer.from_pretrained("/content/final_weights_new")
model = AutoModelForCausalLM.from_pretrained("/content/final_weights_new", device_map = "auto")
pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=150)
prompt=input_prompt.format(
        "nandakishor and his role in convai?", # input
        "", # leave blank as response generated by AI

    )
result = pipe(prompt, temperature=0.05)
generated_text  = result[0]['generated_text']

first_response = generated_text.split('### Response:')[1].strip()
first_response = first_response.split("\n")[0]

print(first_response)

prompt=input_prompt.format(
        "tell me about the ceo", # input
        "", # leave blank as response generated by AI

    )
result = pipe(prompt, temperature=0.00000000001)
generated_text  = result[0]['generated_text']

first_response = generated_text.split('### Response:')[1].strip()
first_response = first_response.split("\n")[0]

print(first_response)