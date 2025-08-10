
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.postgres import PGVectorStore
import os

def init_settings():
    Settings.chunk_size = 512
    Settings.chunk_overlap = 50

    Settings.llm = Ollama(
        model="llama3.2:1b",
        request_timeout=120.0,
        context_window=8000,
    )

    Settings.embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-small-en-v1.5",
        backend="onnx",
        device="cpu",
    )
    print("✅ Settings initialized with Ollama and HuggingFace embeddings.")

def init_vector_store():
    return PGVectorStore.from_params(
        host="localhost",
        port=5432,
        database="rag_db",
        user="mac",
        password="",
        table_name="contextual_rag_docs",
        embed_dim=384
    )
    print("✅ Vector store initialized with PGVector.")
