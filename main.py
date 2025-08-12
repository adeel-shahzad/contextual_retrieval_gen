import os
import time
import json
import asyncio
import logging

from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from models.body import ChatRequest
from dotenv import load_dotenv

from scripts.ingest import get_query_engine
from agents.retrieval_agent import RetrievalAgent
# from ragas_local.eval_local import execute_eval

from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
from arize.otel import register
from getpass import getpass

logging.basicConfig(level=logging.DEBUG)
logging.getLogger("llama_index").setLevel(logging.DEBUG)

# tracer_provider = register(project_name="rag-app", auto_instrument=True)
# LlamaIndexInstrumentor().instrument(tracer_provider=tracer_provider)

load_dotenv()
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# quick probe: put this somewhere in startup or a test endpoint
# from opentelemetry import trace
# tracer = trace.get_tracer("probe")
# with tracer.start_as_current_span("hello-phoenix"):
#     pass

query_engine = None
retrieval_agent = None

@app.on_event("startup")
async def startup_event():
    # Setup OTEL via our convenience function
    tracer_provider = register(
        space_id=getpass("Enter your Arize Space ID: "),
        api_key=getpass("Enter your Arize API Key: "),
        project_name="llamaindex-contextual-rag-tracing",
    )

    # Finish automatic instrumentation
    LlamaIndexInstrumentor().instrument(
        tracer_provider=tracer_provider, skip_dep_check=True
    )

    global query_engine, retrieval_agent
    query_engine = get_query_engine(cohere_api_key=os.getenv("COHERE_API_KEY"))
    retrieval_agent = RetrievalAgent(query_engine)
    print("✅ FastAPI startup complete: query engine ready.")


@app.post("/v1/chat/completions")
async def ask_question(chat_req: ChatRequest):
    if retrieval_agent is None:
        raise HTTPException(status_code=500, detail="Retrieval agent not initialized.")

    user_message = chat_req.messages[-1].content

    # ✅ Run the agent/crew pipeline
    assistant_reply = retrieval_agent.execute(
        question=user_message,
        top_k=3,
        return_sources=2
    )
    # keep your existing “simulated streaming”
    return StreamingResponse(event_stream(assistant_reply, chat_req), media_type="text/event-stream")

@app.get("/api/ragas")
async def get_ragas():
    result = execute_eval(query_engine)  # get your EvaluationResult

    # Try to get a dict from result
    if hasattr(result, "dict"):
        json_ready = result.dict()
    elif hasattr(result, "__dict__"):
        json_ready = result.__dict__
    else:
        json_ready = result  # fallback, if already a dict

    # Serialize, allowing NaN and Infinity
    json_str = json.dumps(json_ready, allow_nan=True)
    return Response(content=json_str, media_type="application/json")

async def event_stream(assistant_reply: str, chat_req: ChatRequest):
    tokens = assistant_reply.splitlines(keepends=True)
    for token in tokens:
        chunk = {
            "id": "chatcmpl-123",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": chat_req.model,
            "choices": [{"delta": {"content": token + " "}, "index": 0, "finish_reason": None}],
        }
        yield f"data: {json.dumps(chunk)}\n\n"
        await asyncio.sleep(0.05)

    final_chunk = {
        "id": "chatcmpl-123",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": chat_req.model,
        "choices": [{"delta": {}, "index": 0, "finish_reason": "stop"}],
    }
    yield f"data: {json.dumps(final_chunk)}\n\n"
    yield "data: [DONE]\n\n"

@app.get("/v1/models")
async def list_models():
    return {
        "data": [
            {"id": "my-fastapi-model", "object": "model", "owned_by": "fastapi-backend"}
        ]
    }

# @app.post("/v1/chat/completions")
# async def chat_completions(chat_req: ChatRequest):
#     try:
#         if chat_req.model != "my-fastapi-model":
#             raise HTTPException(status_code=400, detail="Unknown model")

#         user_message = chat_req.messages[-1].content
#         assistant_reply = query_engine.query(user_message)
#         assistant_reply = str(assistant_reply.response)

#         if chat_req.stream:
#             return StreamingResponse(event_stream(assistant_reply, chat_req), media_type="text/event-stream")

#         return {
#             "id": "chatcmpl-123",
#             "object": "chat.completion",
#             "choices": [{"message": {"role": "assistant", "content": assistant_reply}}],
#         }

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))
