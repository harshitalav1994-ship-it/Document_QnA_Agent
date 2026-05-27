# 001 — Provider-agnostic LLM factory

Date: during the build.
Status: accepted.

## Context

The brief says "use any LLM of your choice." Easy in the moment, but in a real system the LLM is the single most volatile dependency: pricing changes, models get deprecated, providers have outages, and "best model for this task" shifts every few months. Hardcoding the provider into agent code makes all of those into refactors.

## Decision

A single `get_llm()` factory in `app/services/agent.py` reads `LLM_PROVIDER` from settings and returns a `BaseChatModel`. Currently supports Groq, Anthropic, OpenAI. Swapping is a one-env-var change.

API keys are read by the provider SDKs from environment, not passed explicitly. This keeps them out of stack traces and out of any object `repr`.

## Alternatives considered

- **Direct `ChatGroq(...)` in the agent.** Faster to write, harder to evolve. Rejected.
- **A full strategy pattern with a `LLMProvider` interface.** Overkill for three providers that all already conform to `BaseChatModel`. Would revisit if we needed provider-specific behaviour (streaming differences, structured-output handling, retry policy per provider).

## Consequences

- Each new provider needs an `if provider == "..."` branch and the matching `langchain-*` package. Acceptable.
- The factory doesn't currently do retry/fallback. That's the natural next layer — when the primary provider 5xxs, fall through to a secondary. Out of scope for this iteration.

## What I'd revisit

When we add a second use case (summarisation, classification) on the same service, the factory should probably move up a layer to `app/llm/` and be parameterised by use-case, since different tasks want different models and temperatures.
