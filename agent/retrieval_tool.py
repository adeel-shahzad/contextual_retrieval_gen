from typing import Optional, List
from pydantic import BaseModel, Field, ConfigDict, PrivateAttr
from crewai.tools import BaseTool

# --- strict args passed from the agent
class RetrievalInput(BaseModel):
    model_config = ConfigDict(extra='forbid')
    question: str = Field(..., description="The user's natural-language question.")
    top_k: int = Field(20, description="How many candidates to retrieve before re-ranking.")
    return_sources: int = Field(3, description="How many top passages to return after re-ranking.")

class ContextualRetrievalTool(BaseTool):
    name: str = "contextual_rag_retrieval"
    description: str = (
        "Retrieve the most relevant passages from the Abu Dhabi Procurement Standards.\n"
        "USAGE:\n"
        "  contextual_rag_retrieval({\"question\": \"<user question>\", \"top_k\": 20, \"return_sources\": 3})\n"
        "ALWAYS pass a real string for `question`."
    )
    args_schema: type[RetrievalInput] = RetrievalInput

    # mark runtime-only attributes as *private*
    _query_engine: any = PrivateAttr(default=None)
    _return_sources_cap: int = PrivateAttr(default=3)

    def set_query_engine(self, qe, return_sources_cap: int = 3):
        self._query_engine = qe
        self._return_sources_cap = return_sources_cap

    def _run(self, question: str, top_k: int = 20, return_sources: int = 3) -> str:
        print('I am in the _run')
        if self._query_engine is None:
            return "Retrieval tool not initialized: no query engine set."

        # You can pass top_k via as_query_engine when you create qe,
        # or override here if your qe supports it. Simple path:
        resp = self._query_engine.query(question)

        lines = [f"Answer: {str(resp)}"]
        # add sources (bounded by both user request and safety cap)
        k = min(return_sources, self._return_sources_cap)
        if getattr(resp, "source_nodes", None):
            srcs = []
            for i, sn in enumerate(resp.source_nodes[:k], 1):
                meta = (sn.node.metadata or {})
                name = meta.get("file_name") or meta.get("file_path") or "doc"
                score = getattr(sn, "score", None)
                snippet = (sn.node.get_content() or "")[:160].replace("\n", " ")
                srcs.append(f"{i}. {name} | score={score} | “{snippet}…”")
            if srcs:
                lines.append("Sources:\n" + "\n".join(srcs))
        return "\n".join(lines)