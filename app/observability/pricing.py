"""
Per-million-token pricing for cost estimation.

This is the laziest possible implementation. In production:
  - prices change; this should be data, not code
  - billing should come from the provider's invoice/usage API, not estimated
  - per-tenant cost attribution needs request_id correlation upstream

Useful enough for showing operators what a request cost in the response
metadata and in logs, which is what you actually want during development
and for capacity planning.

Prices as of late 2025; verify before relying on them.
"""

# (model_substring, input_per_million_usd, output_per_million_usd)
# Match by substring so "llama-3.3-70b-versatile" matches "llama-3.3-70b".
_PRICES: list[tuple[str, float, float]] = [
    ("llama-3.3-70b", 0.59, 0.79),       # Groq
    ("llama-3.1-8b", 0.05, 0.08),        # Groq
    ("claude-haiku", 0.80, 4.00),        # Anthropic Haiku 3.5
    ("claude-sonnet", 3.00, 15.00),      # Anthropic Sonnet 4
    ("gpt-4o-mini", 0.15, 0.60),         # OpenAI
    ("gpt-4o", 2.50, 10.00),             # OpenAI
]


def estimate_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float | None:
    """
    Return estimated cost in USD, or None if the model is unknown.

    Returning None (rather than 0.0) is intentional: a missing model is a
    signal to update the price table, not a silent zero in dashboards.
    """
    if not model:
        return None
    model_lower = model.lower()
    for needle, in_price, out_price in _PRICES:
        if needle in model_lower:
            return round(
                (input_tokens / 1_000_000) * in_price
                + (output_tokens / 1_000_000) * out_price,
                6,
            )
    return None
