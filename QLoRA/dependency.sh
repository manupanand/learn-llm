#!/bin/bash
# dependency.sh
pip install  accelerate peft bitsandbytes transformers
pip install trl==0.12.2
pip install transformers==4.57.3
pip install llama-cpp-python
pip install vllm haystack-ai

nohup python -m vllm.entrypoints.openai.api_server \
                  --model /content/final_weights_new \
                  --dtype auto \
                  --max-model-len 2048 \
                  > vllm.log &
while ! grep -q "Application startup complete" vllm.log; do tail -n 1 vllm.log; sleep 5; done


!pip install langchain # library for RAG
!pip install llama-cpp-python
!pip install -U langchain-community
!pip install sentence-transformers
!pip install unstructured
!pip install chromadb #vector db
!pip install -q langchain-huggingface

!mkdir "/content/data"
!cp "/content/output.csv" "/content/data"

!pip install -U langchain-text-splitters
!pip install -U langchain-community gpt4all

