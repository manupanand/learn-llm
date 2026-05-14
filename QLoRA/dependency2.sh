#!/bin/bash
# =============================================================================
# dependency.sh — Environment Setup for LLM Fine-tuning + Serving Pipeline
# Hardware: NVIDIA GB10 Blackwell 120GB
# =============================================================================

# =============================================================================
# BLOCK 1: CORE ML LIBRARIES
# accelerate: HuggingFace library for distributed training, mixed precision
# peft: LoRA, QLoRA, adapter methods — Parameter Efficient Fine-Tuning
# bitsandbytes: 4-bit / 8-bit quantization for model loading
# transformers: core HuggingFace library — models, tokenizers, trainers
# =============================================================================

pip install accelerate peft bitsandbytes

# Pin specific versions for compatibility — version mismatches cause silent bugs
# trl 0.12.2: SFTTrainer API is stable at this version
# transformers 4.47.0: CORRECTED from 4.57.3 (does not exist as of training data)
#   Always pin transformers — HF breaks APIs between minor versions
pip install trl==0.12.2
pip install transformers==4.47.0    # CORRECTED: 4.57.3 does not exist

# Flash Attention 2: memory-efficient attention mechanism
# Reduces attention memory from O(n²) to O(n) — critical for long sequences
# ADDED: was missing from original script
pip install flash-attn --no-build-isolation

# datasets: HuggingFace datasets library — load_dataset, map, filter
# ADDED: was missing from original
pip install datasets

# =============================================================================
# BLOCK 2: INFERENCE & SERVING LIBRARIES
# vllm: high-throughput LLM inference server
#   - Uses PagedAttention (efficient KV cache management)
#   - OpenAI-compatible API — drop-in replacement
#   - Much faster than naive HuggingFace generate() for serving
# haystack-ai: LLM orchestration framework — connects LLMs, retrievers, tools
# =============================================================================

pip install vllm
pip install haystack-ai

# =============================================================================
# BLOCK 3: RAG STACK LIBRARIES
# langchain + langchain-community: RAG pipeline framework
# sentence-transformers: embedding models (text → dense vectors for retrieval)
# chromadb: vector database — stores and searches embeddings
# langchain-huggingface: HuggingFace integration for LangChain
# langchain-text-splitters: splits long documents into chunks for embedding
# unstructured: parses PDFs, Word docs, HTMLs into plain text
# =============================================================================

pip install langchain
pip install langchain-community
pip install langchain-huggingface
pip install langchain-text-splitters
pip install sentence-transformers
pip install unstructured
pip install chromadb

# =============================================================================
# BLOCK 4: DATA HANDLING
# openpyxl: required for pandas to read .xlsx files (pd.read_excel)
# ADDED: was missing — your fine-tune.py will crash without this
# =============================================================================

pip install openpyxl    # ADDED: required for pd.read_excel()

# =============================================================================
# BLOCK 5: DATA DIRECTORY SETUP
# Create the data directory that rag.py expects to exist
# =============================================================================

mkdir -p ./data
cp output.csv ./data/output.csv

# =============================================================================
# BLOCK 6: START vLLM SERVER
# vLLM serves your fine-tuned model as an OpenAI-compatible REST API
# --model: path to your merged model (safetensors format)
# --dtype auto: auto-detect best dtype (will use bfloat16 on Blackwell)
# --max-model-len: maximum context window in tokens
# --host / --port: where to serve (localhost:8000 by default)
#
# nohup: runs the process in background even if terminal closes
# > vllm.log: redirects all output to log file
# & : runs in background
#
# CORRECTED: removed /content/ paths — use relative paths for portability
# ADDED: --gpu-memory-utilization 0.5 — only use 50% of VRAM for KV cache
#        (you have 120GB, 1B model only needs ~2GB — leave headroom)
# ADDED: --trust-remote-code — needed for some model configs
# =============================================================================

echo "Starting vLLM server..."

nohup python -m vllm.entrypoints.openai.api_server \
    --model ./final_weights_new \
    --dtype auto \
    --max-model-len 2048 \
    --host 0.0.0.0 \
    --port 8000 \
    --gpu-memory-utilization 0.5 \
    --trust-remote-code \
    > vllm.log 2>&1 &

# =============================================================================
# BLOCK 7: WAIT FOR SERVER READY
# Poll the log file until vLLM prints "Application startup complete"
# sleep 5: wait 5 seconds between checks to avoid busy loop
# tail -n 1: show last line of log so you can see progress
# =============================================================================

echo "Waiting for vLLM server to start..."
while ! grep -q "Application startup complete" vllm.log; do
    tail -n 1 vllm.log
    sleep 5
done

echo "vLLM server is ready at http://localhost:8000"
echo "Test with: curl http://localhost:8000/v1/models"