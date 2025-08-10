from crewai import Agent, Task, Crew, LLM, Process
from .retrieval_tool import ContextualRetrievalTool

class RetrievalAgent:
    def __init__(self, query_engine):
        # Initialize retrieval tool
        self.retrieval_tool = ContextualRetrievalTool()
        self.retrieval_tool.set_query_engine(query_engine, return_sources_cap=3)

        # LLM config
        self.agent_llm = LLM(
            model="ollama/llama3.2:1b",
            base_url="http://localhost:11434",
            temperature=0.2,
            timeout=120,
        )

        # Agent definition
        self.agent = Agent(
            role="Procurement Retrieval Specialist",
            goal="Find the most relevant passages and answer with citations.",
            backstory="Expert librarian for Abu Dhabi procurement.",
            tools=[self.retrieval_tool],
            llm=self.agent_llm,
            allow_delegation=False,
            verbose=True
        )

    def execute(self, question: str, top_k: int = 20, return_sources: int = 3) -> str:
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