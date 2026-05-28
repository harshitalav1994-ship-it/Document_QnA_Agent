"""
Agent service.

Builds a LangChain tool-calling agent for a given document. The agent has
exactly one tool, `retrieve_context`, bound to that document's FAISS index.
The system prompt instructs the agent to use the tool for factual questions
and to refuse gracefully when the document doesn't contain the answer.

A couple of non-obvious things this module does:

  1. Short-circuit on empty retrieval. If a pre-flight similarity search
     finds nothing above a minimal relevance floor, we skip the LLM call
     entirely and return the canonical refusal. Saves ~all the cost on
     out-of-scope questions, which are common in real traffic.

  2. Provider abstraction in get_llm(). Swap vendors via env var, not code.

I considered using LangGraph here for explicit state and easier tracing,
but for a one-document / one-tool agent the LangChain AgentExecutor is the
shortest correct path. Revisit when we add multi-doc or chat history.
"""
from __future__ import annotations

from typing import Any

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import Tool

from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.store import DocumentRecord
from app.prompts.registry import get_prompt
from langchain_core.tools import tool
from pydantic import BaseModel, Field

logger = get_logger(__name__)

# The canonical refusal string. Keep in sync with the system prompt's
# rule 3 and with REFUSAL_MARKER in scripts/run_eval.py.
REFUSAL = "I cannot answer this question from the provided document."

# FAISS returns L2 distances; lower is closer. This is a loose floor, not
# a tuned threshold. The right way to set this is to measure on the eval
# set, which I haven't done yet — flagged in the README.
RETRIEVAL_DISTANCE_FLOOR = 1.5


def get_llm() -> BaseChatModel:
    """
    LLM factory. Provider-agnostic so swapping vendors is one config change.

    Reads provider + model + temperature + api_key from settings and passes
    them explicitly to the provider constructor. We don't rely on the SDK
    reading os.environ because pydantic-settings loads from .env into the
    settings object, not into the process environment.
    """
    settings = get_settings()
    provider = settings.llm_provider

    if provider == "groq":
        from langchain_groq import ChatGroq
        if not settings.groq_api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Add it to .env or export it in your shell."
            )
        return ChatGroq(
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            api_key=settings.groq_api_key,
            streaming=False  # Eval is more stable with streaming off; toggle for experimentation.
        )
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        if not settings.anthropic_api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set.")
        return ChatAnthropic(
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            api_key=settings.anthropic_api_key,
        )
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is not set.")
        return ChatOpenAI(
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            api_key=settings.openai_api_key,
        )
    raise ValueError(f"Unsupported LLM provider: {provider}")


def _preflight_retrieval(record: DocumentRecord, query: str) -> list[tuple]:
    """Cheap retrieval check before we pay for the LLM call."""
    settings = get_settings()
    return record.vectorstore.similarity_search_with_score(
        query, k=settings.retrieval_k
    )


def _build_retriever_tool(record: DocumentRecord) -> tuple[BaseTool, list[dict]]:
    """
    Retriever tool bound to one document.

    The captured_chunks list mutates during the agent run so the route can
    read it afterwards to populate `source_chunks` in the response. Not
    pretty but cleaner than parsing intermediate_steps for tool outputs.
    """
    settings = get_settings()
    captured_chunks: list[dict] = []

    def _retrieve(query: str) -> str:
        results = record.vectorstore.similarity_search_with_score(
            query, k=settings.retrieval_k
        )
        rendered_parts: list[str] = []
        for doc, score in results:
            chunk_index = doc.metadata.get("chunk_index")
            captured_chunks.append({
                "content": doc.page_content,
                "score": float(score),
                "chunk_index": chunk_index,
            })
            rendered_parts.append(f"[chunk {chunk_index}]\n{doc.page_content}")
        if not rendered_parts:
            return "No relevant content found in the document."
        return "\n\n---\n\n".join(rendered_parts)

    @tool
    def retrieve_context(query: str) -> str:
        """Retrieve the most relevant passages from the document for a search query. Use this to find facts before answering."""
        return _retrieve(query)
    return retrieve_context, captured_chunks


def build_agent_executor(
    record: DocumentRecord,
    callbacks: list[BaseCallbackHandler] | None = None,
) -> tuple[AgentExecutor, list[dict]]:
    """Construct the AgentExecutor for one document."""
    llm = get_llm()
    tool, captured_chunks = _build_retriever_tool(record)

    system_prompt = get_prompt("doc_qa_agent").template
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    agent = create_tool_calling_agent(llm, [tool], prompt)
    executor = AgentExecutor(
        agent=agent,
        tools=[tool],
        max_iterations=3,
        return_intermediate_steps=True,
        handle_parsing_errors=True,
        callbacks=callbacks or [],
        verbose=False,
    )
    return executor, captured_chunks


def answer_question(
    record: DocumentRecord,
    question: str,
    callbacks: list[BaseCallbackHandler] | None = None,
) -> dict[str, Any]:
    """
    Run the agent against one question.

    Short-circuits the LLM call if pre-flight retrieval finds nothing above
    the relevance floor — we return the canonical refusal directly, saving
    cost and latency on the common "this doc doesn't cover that" case.
    """
    # Pre-flight: cheap retrieval check before we pay for the LLM.
    preflight = _preflight_retrieval(record, question)
    if not preflight or all(score > RETRIEVAL_DISTANCE_FLOOR for _, score in preflight):
        logger.info(
            "short_circuit_refusal",
            extra={
                "doc_id": record.doc_id,
                "preflight_hits": len(preflight),
                "best_score": min((s for _, s in preflight), default=None),
            },
        )
        return {
            "answer": REFUSAL,
            "source_chunks": [],
            "tool_calls": 0,
            "short_circuited": True,
        }

    executor, captured_chunks = build_agent_executor(record, callbacks=callbacks)
    result = executor.invoke({"input": question})
    intermediate = result.get("intermediate_steps", [])
    return {
        "answer": result.get("output", ""),
        "source_chunks": captured_chunks,
        "tool_calls": len(intermediate),
        "short_circuited": False,
    }
