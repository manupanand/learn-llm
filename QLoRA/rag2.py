# =============================================================================
# rag.py — Retrieval Augmented Generation (RAG) Pipeline
# RAG = instead of relying purely on what the model memorized during fine-tuning,
#       you RETRIEVE relevant facts from a document store at query time,
#       then give those facts to the model as context.
#
# Why RAG for a 20-row dataset?
#   Fine-tuning 20 rows teaches the model patterns but facts can be forgotten.
#   RAG guarantees the exact facts are always available at query time.
#   Think of it like giving the model an open-book exam vs closed-book.
#
# Flow: User Query → Embed Query → Search ChromaDB → Retrieve Relevant Chunks
#       → Inject into Prompt → LLM generates answer grounded in retrieved facts
# =============================================================================

# =============================================================================
# BLOCK 1: IMPORTS
# =============================================================================

import os
import torch
from torch import cuda

# LangChain components
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import DirectoryLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# vLLM OpenAI-compatible client for LangChain
# CORRECTED: original used LlamaCpp which requires GGUF format
# Since you're using safetensors + vLLM, use OpenAI client instead
from langchain_community.llms import VLLMOpenAI   # CORRECTED: LlamaCpp → VLLMOpenAI

# =============================================================================
# BLOCK 2: EMBEDDING MODEL SETUP
# Embeddings convert text into dense vectors (lists of numbers)
# Similar text → similar vectors → can find relevant documents by vector distance
#
# all-MiniLM-L6-v2:
#   - Lightweight embedding model (22M parameters)
#   - Produces 384-dimensional vectors
#   - Fast, good quality for English Q&A retrieval
#   - Free, runs locally — no API calls needed
#
# Think of it like CFD: the embedding space is a high-dimensional manifold,
# similar documents cluster together, retrieval = nearest neighbor search
# =============================================================================

embed_model_id = 'sentence-transformers/all-MiniLM-L6-v2'

# Use GPU if available for faster embedding computation
device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'
print(f"Embedding device: {device}")

embed_model = HuggingFaceEmbeddings(
    model_name=embed_model_id,
    model_kwargs={'device': device},
    encode_kwargs={
        'device': device,
        'batch_size': 32    # Process 32 text chunks at once for efficiency
    }
)

# =============================================================================
# BLOCK 3: LOAD DOCUMENTS
# DirectoryLoader: loads all files from a directory
# CSVLoader: specifically handles CSV files — better than raw DirectoryLoader for CSV
#
# CORRECTED: original used DirectoryLoader which doesn't handle CSV well
# CSVLoader maps each row to a Document object with metadata
# =============================================================================

data_dir = './data'

# ADDED: Check data directory exists
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
    print(f"Created {data_dir} — copy your output.csv there")

# Load CSV with CSVLoader — each row becomes a LangChain Document
# CORRECTED: DirectoryLoader → CSVLoader for proper CSV handling
loader = CSVLoader(
    file_path=f'{data_dir}/output.csv',
    source_column='Questions',   # Use Questions column as document source ID
)
data = loader.load()
print(f"Loaded {len(data)} documents from CSV")

# =============================================================================
# BLOCK 4: DOCUMENT SPLITTING
# Why split? Embedding models have token limits (typically 256-512 tokens)
# Long documents must be chunked into smaller pieces for embedding
#
# RecursiveCharacterTextSplitter:
#   - Tries to split on paragraphs first, then sentences, then words
#   - "Recursive" = smart about not splitting mid-sentence
#
# chunk_size=500: max characters per chunk
# chunk_overlap=50: overlap between chunks to preserve context at boundaries
#
# CORRECTED: chunk_overlap=0 → 50
# Zero overlap means context at chunk boundaries is lost — bad for retrieval
# =============================================================================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,    # CORRECTED: 0 → 50 — preserve boundary context
)
all_splits = text_splitter.split_documents(data)
print(f"Split into {len(all_splits)} chunks")

# =============================================================================
# BLOCK 5: VECTOR STORE — ChromaDB
# ChromaDB: local vector database
#   - Stores text chunks + their embedding vectors
#   - At query time: embed the query → find nearest vectors → return those chunks
#
# from_documents: embeds all chunks and stores them in ChromaDB
# This is the "indexing" step — done once, then you query many times
#
# persist_directory: ADDED — saves ChromaDB to disk so you don't re-index every run
# =============================================================================

persist_dir = "./chroma_db"

# ADDED: Load existing DB if available, create new if not
if os.path.exists(persist_dir) and os.listdir(persist_dir):
    print("Loading existing ChromaDB...")
    vectorstore = Chroma(
        persist_directory=persist_dir,
        embedding_function=embed_model
    )
else:
    print("Creating new ChromaDB index...")
    vectorstore = Chroma.from_documents(
        documents=all_splits,
        embedding=embed_model,
        persist_directory=persist_dir,   # ADDED: persist to disk
    )
    print(f"Indexed {len(all_splits)} chunks into ChromaDB")

# =============================================================================
# BLOCK 6: LLM CONNECTION — vLLM via OpenAI Client
# CORRECTED: original used LlamaCpp which requires GGUF format
# Since you serve safetensors via vLLM, use VLLMOpenAI LangChain integration
#
# VLLMOpenAI connects to your running vLLM server
# temperature=0.1: low randomness for factual Q&A
# max_tokens=512: cap response length
# =============================================================================

import random
import string

# Placeholder API key for vLLM (not validated)
fake_api_key = ''.join(random.choices(string.ascii_uppercase + string.digits, k=20))

llm = VLLMOpenAI(
    openai_api_key=fake_api_key,
    openai_api_base="http://localhost:8000/v1",
    model_name="./final_weights_new",
    temperature=0.1,
    max_tokens=512,
)

# =============================================================================
# BLOCK 7: PROMPT TEMPLATE FOR RAG
# This prompt tells the LLM:
#   1. Here are retrieved facts (context)
#   2. Answer ONLY based on these facts
#   3. Here is the user's question
#
# Why constrain to context? Prevents hallucination — model can't make things up
# if you explicitly tell it to only use what's provided.
# =============================================================================

template = """Answer the question based only on the following context.
If the answer is not in the context, say "I don't have information about that."

Context:
{context}

Question: {question}

Answer:"""

prompt = ChatPromptTemplate.from_template(template)

# =============================================================================
# BLOCK 8: RETRIEVER CONFIGURATION
# vectorstore.as_retriever(): converts ChromaDB into a retriever object
# search_kwargs={"k": 3}: retrieve top 3 most similar chunks
# CORRECTED: default k=4 → k=3, more focused for small dataset
# =============================================================================

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}    # CORRECTED: retrieve top 3 relevant chunks
)

# =============================================================================
# BLOCK 9: RAG CHAIN — LCEL (LangChain Expression Language)
# This is a pipeline (like a Unix pipe) that connects components:
#
# Step 1: {"context": retriever, "question": RunnablePassthrough()}
#   - retriever: takes the question → searches ChromaDB → returns relevant chunks
#   - RunnablePassthrough(): passes the question through unchanged
#   - Result: {"context": "retrieved text...", "question": "user question"}
#
# Step 2: prompt
#   - Fills the template with context and question
#   - Result: fully formatted prompt string
#
# Step 3: llm
#   - Sends prompt to your vLLM server
#   - Result: raw LLM response
#
# Step 4: StrOutputParser()
#   - Extracts just the string from the LLM response object
#   - Result: clean answer string
# =============================================================================

def format_docs(docs):
    """Join retrieved document chunks into a single context string."""
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {
        "context": retriever | format_docs,   # CORRECTED: added format_docs to join docs
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

# =============================================================================
# BLOCK 10: RUN RAG QUERIES
# .invoke(): runs the full pipeline once and returns result
# .stream(): runs pipeline and yields chunks as they're generated
#   - Better UX — user sees response being written in real time
#   - Important for production chatbots — reduces perceived latency
# =============================================================================

print("\n--- RAG Query Test ---")

question = "who is Nandakishor?"

# First test: non-streaming
print(f"\nQuestion: {question}")
response = rag_chain.invoke(question)
print(f"Answer: {response}")

# Second test: streaming (shows tokens as they're generated)
print("\n--- Streaming Response ---")
question2 = "what is convai?"
print(f"Question: {question2}")
print("Answer: ", end="", flush=True)
for chunk in rag_chain.stream(question2):
    print(chunk, end="", flush=True)
print()  # newline after streaming

# =============================================================================
# BLOCK 11: INTERACTIVE RAG LOOP (ADDED)
# Lets you ask multiple questions interactively
# =============================================================================

print("\n--- Interactive RAG Mode (type 'Q' to quit) ---")
while True:
    user_question = input("\n🧑 Ask: ").strip()
    if user_question.upper() == "Q":
        break
    if not user_question:
        continue
    print("🤖 Answer: ", end="", flush=True)
    for chunk in rag_chain.stream(user_question):
        print(chunk, end="", flush=True)
    print()
