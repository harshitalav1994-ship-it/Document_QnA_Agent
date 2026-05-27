"""
Agent tests with a mocked LLM.

We replace get_llm with a scripted chat model so the agent loop runs end
to end without any network calls. The script is: tool call -> final answer.
"""
from __future__ import annotations

from typing import Any

import pytest
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from app.core.store import DocumentStore
from app.services import agent as agent_module
from app.services.agent import REFUSAL, answer_question
from app.services.ingestion import ingest_document
import logging 

class ScriptedChatModel(BaseChatModel):
    """Chat model that returns a pre-set sequence of AIMessages."""

    script: list[AIMessage]

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):  # type: ignore[override]
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        if not self.script:
            raise RuntimeError("ScriptedChatModel ran out of scripted responses.")
        msg = self.script.pop(0)
        return ChatResult(generations=[ChatGeneration(message=msg)])


def _make_scripted_llm(answer_text: str, search_query: str) -> ScriptedChatModel:
    tool_call_msg = AIMessage(
        content="",
        tool_calls=[{
            "name": "retrieve_context",
            "args": {"__arg1": search_query},
            "id": "call_1",
            "type": "tool_call",
        }],
    )
    final_msg = AIMessage(content=answer_text)
    return ScriptedChatModel(script=[tool_call_msg, final_msg])


def test_agent_calls_retriever_and_returns_answer(fake_embeddings, monkeypatch):
    store = DocumentStore()
    text = (
        "The Eiffel Tower was completed in 1889 for the Paris World's Fair. "
        "It stands 330 metres tall including antennas. "
        "Gustave Eiffel's company designed it."
    )
    record = ingest_document(text=text, store=store)

    scripted = _make_scripted_llm(
        answer_text="The Eiffel Tower was completed in 1889.",
        search_query="Eiffel Tower completion year",
    )
    monkeypatch.setattr(agent_module, "get_llm", lambda: scripted)
    # Disable short-circuit for this test by setting an absurd floor — we
    # specifically want to exercise the full agent loop.
    monkeypatch.setattr(agent_module, "RETRIEVAL_DISTANCE_FLOOR", 999.0)

    result = answer_question(record=record, question="When was the Eiffel Tower completed?")

    assert "1889" in result["answer"]
    assert result["tool_calls"] == 1
    assert result["short_circuited"] is False
    assert len(result["source_chunks"]) >= 1


def test_agent_short_circuits_when_retrieval_empty(fake_embeddings, monkeypatch):
    """A question with no relevant chunks should skip the LLM entirely."""
    store = DocumentStore()
    record = ingest_document(text="Cats purr when they are happy.", store=store)

    # Force the short-circuit by setting an unsatisfiable floor (lower than
    # any FAISS distance can be).
    monkeypatch.setattr(agent_module, "RETRIEVAL_DISTANCE_FLOOR", -1.0)

    def _should_not_be_called():
        raise AssertionError("LLM was called on a short-circuited request")

    monkeypatch.setattr(agent_module, "get_llm", _should_not_be_called)

    result = answer_question(record=record, question="What is quantum chromodynamics?")
    assert result["answer"] == REFUSAL
    assert result["short_circuited"] is True
    assert result["tool_calls"] == 0


def test_agent_with_full_api_request(client, fake_embeddings, monkeypatch):
    ingest_resp = client.post(
        "/documents",
        json={"text": "Photosynthesis converts light energy into chemical energy."},
    )
    doc_id = ingest_resp.json()["doc_id"]

    scripted = _make_scripted_llm(
        answer_text="Photosynthesis converts light into chemical energy.",
        search_query="photosynthesis",
    )
    monkeypatch.setattr(agent_module, "get_llm", lambda: scripted)
    monkeypatch.setattr(agent_module, "RETRIEVAL_DISTANCE_FLOOR", 999.0)

    resp = client.post(
        f"/questions/{doc_id}",
        json={"question": "What does photosynthesis do?"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "chemical energy" in body["answer"].lower()
    assert body["metadata"]["doc_id"] == doc_id
    assert body["metadata"]["short_circuited"] is False
    assert body["metadata"]["tool_calls"] == 1

