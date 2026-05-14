#rag.py
from torch import cuda
from langchain_huggingface import HuggingFaceEmbeddings

embed_model_id = 'sentence-transformers/all-MiniLM-L6-v2'

device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

embed_model = HuggingFaceEmbeddings(
    model_name=embed_model_id,
    model_kwargs={'device': device},
    encode_kwargs={'device': device, 'batch_size': 32}
)
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DirectoryLoader
loader = DirectoryLoader('data')
data = loader.load()

from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
all_splits = text_splitter.split_documents(data)
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

#embed_model = GPT4AllEmbeddings()

vectorstore = Chroma.from_documents(
    documents=all_splits,
    embedding=embed_model
)

from langchain_community.embeddings import LlamaCppEmbeddings
from langchain_community.llms import LlamaCpp

from langchain_core.callbacks import CallbackManager
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
n_gpu_layers = 30  # Metal set to 1 is enough.
n_batch = 512  # Should be between 1 and n_ctx, consider the amount of RAM of your Apple Silicon Chip.
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

#llama = LlamaCppEmbeddings(model_path="/data/llama.cpp/models/llama-2-7b-chat/ggml-model-q4_0.bin")
llm = LlamaCpp(
    model_path="/content/final_weights_new/ggml-model-q4_k_m.gguf",
    n_gpu_layers=n_gpu_layers,
    n_batch=n_batch,
    n_ctx=2048,
    f16_kv=True,  # MUST set to True, otherwise you will run into problem after a couple of calls
    callback_manager=callback_manager,
    verbose=False,
)

question = "who is Nandakishor"
docs = vectorstore.similarity_search(question)
#result = llm_chain(docs)
docs
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# 1. Define your prompt
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# 2. Build the chain using LCEL (Pipe syntax)
rag_chain = (
    {"context": vectorstore.as_retriever(), "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 3. Run it
response = rag_chain.invoke(question)
question = "who is nandakishor?"
# .stream() prints the response chunk-by-chunk as it is generated,
# making the application feel much faster to the end user.
for chunk in rag_chain.stream(question):
    print(chunk, end="", flush=True)