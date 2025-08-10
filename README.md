# Contextual Retrieval - Generation

A FastAPI backend that performs contextual retrieval-augmented generation (RAG). Documents are ingested into a PGVector store, retrieved with a custom tool, and streamed back to the client in OpenAI-compatible chunks—perfect for use behind Open WebUI or other chat UIs.

## Features

- **FastAPI service with streaming responses**  
  The backend exposes an OpenAI-style `/v1/chat/completions` endpoint that streams tokens over Server‑Sent Events and returns a placeholder model via `/v1/models`​:codex-file-citation[codex-file-citation]{line_range_start=67 line_range_end=107 path=main.py git_url="https://github.com/adeel-shahzad/contextual_retrieval_gen/blob/main/main.py#L67-L107"}​
- **Contextual retrieval tool**  
  `ContextualRetrievalTool` fetches top‑k passages and returns cited snippets for the user's question​:codex-file-citation[codex-file-citation]{line_range_start=12 line_range_end=51 path=agent/retrieval_tool.py git_url="https://github.com/adeel-shahzad/contextual_retrieval_gen/blob/main/agent/retrieval_tool.py#L12-L51"}​
- **Document ingestion pipeline**  
  `scripts/ingest.py` chunkifies files, adds contextual headers, stores them in PGVector, and applies optional Cohere re‑ranking​:codex-file-citation[codex-file-citation]{line_range_start=43 line_range_end=90 path=scripts/ingest.py git_url="https://github.com/adeel-shahzad/contextual_retrieval_gen/blob/main/scripts/ingest.py#L43-L90"}​
- **Configurable LlamaIndex settings**  
  Uses Ollama for generation and HuggingFace embeddings, with PGVector configured as the vector store​:codex-file-citation[codex-file-citation]{line_range_start=8 line_range_end=33 path=rag/settings.py git_url="https://github.com/adeel-shahzad/contextual_retrieval_gen/blob/main/rag/settings.py#L8-L33"}​
- **Typed request models**  
  Requests follow a simple schema of chat messages and a `stream` flag​:codex-file-citation[codex-file-citation]{line_range_start=5 line_range_end=12 path=models/body.py git_url="https://github.com/adeel-shahzad/contextual_retrieval_gen/blob/main/models/body.py#L5-L12"}​

## Getting Started

### Prerequisites
- Python 3.10+
- PostgreSQL with the `pgvector` extension
- (Optional) [Cohere API key](https://dashboard.cohere.com/) for re-ranking
- (Optional) [Arize](https://www.arize.com/) credentials for telemetry

### Installation

```bash
git clone https://github.com/your-org/contextual_retrieval_gen.git
cd contextual_retrieval_gen
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

