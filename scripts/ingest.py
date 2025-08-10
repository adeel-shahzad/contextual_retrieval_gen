import os
import psycopg2
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.readers import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.postprocessor.cohere_rerank import CohereRerank
from rag.settings import init_settings, init_vector_store, Settings
from dotenv import load_dotenv

load_dotenv()

DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "rag_db",
    "user": "mac",
    "password": ""
}

def make_contextual_text(node, doc_title="Abu Dhabi Procurement Standards"):
    section_path = " > ".join([str(v) for k, v in node.metadata.items() if "header" in k.lower() or "section" in k.lower()])
    section_path = section_path or node.metadata.get("file_name", "Unknown Section")
    head = (
        f"[DOC_TITLE] {doc_title}\n"
        f"[SECTION_PATH] {section_path}\n"
        f"[INTENT] Assist a procurement practitioner answering policy/process questions\n"
        f"[CHUNK]\n"
    )
    return head + node.text

def is_document_indexed(doc_name):
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute(
        "SELECT COUNT(*) FROM data_contextual_rag_docs WHERE metadata_::TEXT LIKE %s;",
        (f'%{doc_name}%',)
    )
    count = cur.fetchone()[0]
    cur.close()
    conn.close()
    return count > 0

def ingest_document(file_path, cohere_api_key=None):
    doc_name = os.path.basename(file_path)
    if is_document_indexed(doc_name):
        print(f"❌ Skipping '{doc_name}': already indexed.")
        return

    print(f"Processing: {file_path}")
    init_settings()
    vector_store = init_vector_store()
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    documents = SimpleDirectoryReader(input_files=[file_path]).load_data()
    if not documents:
        print(f"⚠️ No documents found in {file_path}")
        return

    splitter = SentenceSplitter(chunk_size=Settings.chunk_size, chunk_overlap=Settings.chunk_overlap)
    nodes = splitter.get_nodes_from_documents(documents)

    for node in nodes:
        node.metadata["orig_text"] = node.text
        node.metadata["doc_name"] = doc_name
        node.text = make_contextual_text(node, doc_name)  # Pass doc_name as doc_title

    index = VectorStoreIndex(nodes, storage_context=storage_context)
    index.storage_context.persist()
    print(f"✅ Ingested {len(nodes)} nodes from '{doc_name}' and stored in DB.")

def ingest_all_documents(rag_docs_dir, cohere_api_key=None):
    files = [
        os.path.join(rag_docs_dir, f)
        for f in os.listdir(rag_docs_dir)
        if os.path.isfile(os.path.join(rag_docs_dir, f))
    ]
    if not files:
        print(f"No files found in {rag_docs_dir}")
        return
    for file_path in files:
        ingest_document(file_path, cohere_api_key=cohere_api_key)

def get_query_engine(cohere_api_key=None):
    init_settings()
    vector_store = init_vector_store()
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex([], storage_context=storage_context)
    return index.as_query_engine(
        similarity_top_k=20,
        node_postprocessors=[CohereRerank(api_key=cohere_api_key)] if cohere_api_key else None,
        llm=Settings.llm,
    )

if __name__ == "__main__":
    cohere_key = os.getenv("COHERE_API_KEY")
    rag_docs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "rag_documents")
    ingest_all_documents(rag_docs_dir, cohere_api_key=cohere_key)

