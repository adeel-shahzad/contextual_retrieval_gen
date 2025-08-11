import os
os.environ["OPENAI_API_KEY"] = ""

import json
from datasets import Dataset
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from ragas.embeddings import LangchainEmbeddingsWrapper
# from langchain_ollama import OllamaLLM
from langchain_openai import ChatOpenAI
from ragas.llms import LangchainLLMWrapper
from ragas.run_config import RunConfig
from ragas import evaluate
from ragas.metrics import context_precision, context_recall, faithfulness, answer_relevancy

from scripts.ingest import get_query_engine

your_eval_questions = []
with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), "eval_dataset/eval_ds_Abu Dhabi Procurement Standards.json"), "r", encoding="utf-8") as f:
    your_eval_questions = json.load(f)

# def get_length():
#     print("Length of your_eval_questions:", len(your_eval_questions))
#     print(your_eval_questions[0])

def create_dataset(query_engine):
    print("in dataset...")
    # Suppose you have eval_data as above (with 'question' and 'answer')
    questions = [d["question"] for d in your_eval_questions]
    answers = []
    ground_truths = [d["answer"] for d in your_eval_questions]
    references = [d["reference"] for d in your_eval_questions]

    # For each question, use your retriever to get context(s)
    for i, q in enumerate(questions):
        print(f"Querying ({i+1}/{len(questions)})")
        # This assumes LlamaIndex returns context nodes; adjust as needed
        resp = query_engine.query(q)
        # resp.source_nodes returns list of nodes; get their text fields
        ctxs = [node.node.text for node in getattr(resp, "source_nodes", [])]
        answers.append(ctxs)

    print("Creating dataset...")
    ds = Dataset.from_dict({
        "question": questions,
        "contexts": answers,
        "answer": ground_truths,
        "reference": references
    })
    return ds

def execute_eval(query_engine=None):
    if query_engine is None:
        query_engine = get_query_engine()
    
    ds = create_dataset(query_engine)
    print("starting_up_evaluation")
    hf_embedder = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5", backend="onnx", device="cpu")
    wrapped_embeddings = LangchainEmbeddingsWrapper(hf_embedder)


    # ollama_model = OllamaLLM(model="llama3.2:1b")
    # wrapped_llm = LangchainLLMWrapper(ollama_model)

    openai_llm = ChatOpenAI(
        model="gpt-4o-mini",  # or "gpt-4o"
        temperature=0
    )

    print("Wrapping LLM...")
    my_run_config = RunConfig(
        timeout=300,         # 5 minutes max per job
        max_retries=3,        # fewer retry attempts
        max_wait=30,          # reduce wait between retries
        max_workers=2         # fewer concurrent jobs
    )

    print("Evaluating...")
    results = evaluate(
        ds,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=openai_llm,
        embeddings=wrapped_embeddings,
        run_config=my_run_config,
        raise_exceptions=False,
    )
    print(results)
    return results

if __name__ == "__main__":
    execute_eval()