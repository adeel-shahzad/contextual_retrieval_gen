from crewai import Agent, Task, Crew, LLM, Process
from llama_index.llms.ollama import Ollama
from .retrieval_tool import ContextualRetrievalTool

class RetrievalAgent:
    def __init__(self, query_engine):
        # Initialize retrieval tool
        self.retrieval_tool = ContextualRetrievalTool()
        self.retrieval_tool.set_query_engine(query_engine, return_sources_cap=3)

        # LLM config
        # self.agent_llm = LLM(
        #     model="ollama/llama3.2:1b",
        #     base_url="http://localhost:11434",
        #     temperature=0.2,
        #     timeout=120,
        # )

        # LLM config
        self.agent_llm = Ollama(
            # model="llama3.1:8b",
            model="llama3.2:1b",
            base_url="http://localhost:11434",
            # base_url="http://13.212.147.243:11434",
            request_timeout=120.0,
            # LlamaIndex passes these to Ollama under "options"
            additional_kwargs={"options": {
                "num_ctx": 1024,       # try 2048–4096
                "num_batch": 4,       # optional: reduce if still tight
                # "num_gpu": 0,        # if CPU-only, make explicit
            }}
        )

        # Agent definition
        self.agent = Agent(
            role="UAE Public-Sector Law & Security Retrieval Agent",
            goal=(
                "Return the fewest but most relevant snippets from the indexed corpus to answer "
                "queries on Abu Dhabi/UAE procurement, HR bylaws, and NESA IA standards; "
                "always include inline citations and source titles."
            ),
            backstory=(
                "Expert on Abu Dhabi Procurement Standards & Manuals (Business Process, SAP Ariba), "
                "HR Bylaws (Decision 10 of 2020 implementing Law 6 of 2016), and NESA UAE IA Standards. "
                "Optimize for precision over verbosity; if confidence is low, ask one clarifying question; "
                "otherwise provide 2–5 sourced snippets plus a 1–2 sentence synthesis. Refuse out-of-scope topics."
            ),
            tools=[self.retrieval_tool],
            llm=self.agent_llm,
            allow_delegation=False,
            verbose=True
        )

    def execute(self, question: str, top_k: int = 3, return_sources: int = 2) -> str:
        """Runs the agent task and returns the result."""
        retrieval_task = Task(
            description=(
                f"Use the `contextual_rag_retrieval` tool to retrieve relevant passages. "
                f"Call the tool with the following: "
                f'{{"question": "{question}", "top_k": {top_k}, "return_sources": {return_sources}}}. '
                f"Ensure the tool is called exactly with this structure."
            ),
            expected_output="A concise answer plus 2–3 source lines.",
            agent=self.agent,
        )

        crew = Crew(
            agents=[self.agent],
            tasks=[retrieval_task],
            process=Process.sequential,
            verbose=True
        )

        return crew.kickoff()