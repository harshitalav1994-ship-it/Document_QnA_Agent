"""
Observability callbacks.

A LangChain BaseCallbackHandler that records:
  - input / output / total tokens (when the provider returns usage metadata)
  - estimated USD cost per request, based on a hardcoded pricing table
  - per-call LLM latency
  - tool invocation events

Designed to be cheap and lock-free: each agent run gets its own instance,
so there's no cross-request state to coordinate.

"""
from __future__ import annotations

import time
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

from app.core.logging import get_logger

logger = get_logger(__name__)


# Per-million-token USD prices. Kept rough; refresh from the provider's
# pricing page periodically. The point isn't precision to four decimals —
# it's that every request gets a cost annotation so dashboards work.
# Last reviewed: 2026-05.
MODEL_PRICING: dict[str, dict[str, float]] = {
    # Groq — free tier; we still track notional cost as if paid, so the
    # numbers are comparable when you swap to a paid provider.
    "llama-3.3-70b-versatile": {"input": 0.59, "output": 0.79},
    # Anthropic
    "claude-haiku-4-5": {"input": 1.00, "output": 5.00},
    "claude-sonnet-4-5": {"input": 3.00, "output": 15.00},
    # OpenAI
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
}


def estimate_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float | None:
    """Return estimated USD cost, or None if the model isn't in the pricing table."""
    pricing = MODEL_PRICING.get(model)
    if not pricing:
        return None
    return (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1_000_000


class TokenLatencyCallback(BaseCallbackHandler):
    """Per-request callback that accumulates token + latency stats."""

    def __init__(self, request_id: str) -> None:
        self.request_id = request_id
        self.input_tokens: int = 0
        self.output_tokens: int = 0
        self.total_tokens: int = 0
        self.llm_calls: int = 0
        self.tool_calls: int = 0
        self._llm_start_times: dict[UUID, float] = {}
        self._tool_start_times: dict[UUID, float] = {}

    # --- LLM lifecycle ---

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        self._llm_start_times[run_id] = time.perf_counter()

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, **kwargs: Any) -> None:
        start = self._llm_start_times.pop(run_id, None)
        latency_ms = int((time.perf_counter() - start) * 1000) if start else None
        self.llm_calls += 1

        usage = self._extract_token_usage(response)
        if usage:
            self.input_tokens += usage.get("input_tokens", 0)
            self.output_tokens += usage.get("output_tokens", 0)
            self.total_tokens += usage.get("total_tokens", 0)

        logger.info(
            "llm_call_completed",
            extra={
                "request_id": self.request_id,
                "latency_ms": latency_ms,
                "input_tokens": usage.get("input_tokens") if usage else None,
                "output_tokens": usage.get("output_tokens") if usage else None,
            },
        )

    # --- Tool lifecycle ---

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        self._tool_start_times[run_id] = time.perf_counter()
        logger.info(
            "tool_call_started",
            extra={
                "request_id": self.request_id,
                "tool": serialized.get("name"),
                # Truncate to keep logs sane.
                "input_preview": input_str[:200],
            },
        )

    def on_tool_end(self, output: str, *, run_id: UUID, **kwargs: Any) -> None:
        start = self._tool_start_times.pop(run_id, None)
        latency_ms = int((time.perf_counter() - start) * 1000) if start else None
        self.tool_calls += 1
        logger.info(
            "tool_call_completed",
            extra={"request_id": self.request_id, "latency_ms": latency_ms},
        )

    # --- helpers ---

    @staticmethod
    def _extract_token_usage(response: LLMResult) -> dict[str, int]:
        """
        Extract token usage from an LLMResult. Different providers report this
        differently; we look in the standard places and normalise.
        """
        usage: dict[str, int] = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

        # llm_output is the legacy location (OpenAI-style).
        llm_output = response.llm_output or {}
        token_usage = llm_output.get("token_usage") or llm_output.get("usage") or {}
        if token_usage:
            usage["input_tokens"] = token_usage.get("prompt_tokens", 0) or token_usage.get("input_tokens", 0)
            usage["output_tokens"] = token_usage.get("completion_tokens", 0) or token_usage.get("output_tokens", 0)
            usage["total_tokens"] = token_usage.get("total_tokens", 0)
            return usage

        # Newer LangChain surfaces usage on the message itself (usage_metadata).
        for generations in response.generations:
            for gen in generations:
                msg = getattr(gen, "message", None)
                meta = getattr(msg, "usage_metadata", None) if msg else None
                if meta:
                    usage["input_tokens"] += meta.get("input_tokens", 0)
                    usage["output_tokens"] += meta.get("output_tokens", 0)
                    usage["total_tokens"] += meta.get("total_tokens", 0)
        return usage
