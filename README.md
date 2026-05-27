# Document Q&A Agent

A FastAPI service that answers natural-language questions about an ingested document, using a LangChain tool-calling agent over a per-document FAISS index.

This was a 2-3 hour take-home build. The code is deliberately small. The thinking, in the rest of this README and in `decisions/`, is meant to be the larger part.

## Framing

Before writing any code I made these assumptions explicit, because they decide most of the architectural choices:

- **Latency target.** Interactive Q&A, so p95 well under 5s for a 100KB doc. Each round trip to a 70B-class model is ~1-3s; that leaves room for one tool call but not many.
- **Cost target.** This kind of service lives or dies on cost per query. With Llama 3.3 70B on Groq (~$0.59/$0.79 per million in/out), a typical request is well under $0.001. With Claude Sonnet 4 it's ~10x higher and worth optimising for. Cost is in every response's metadata so this is visible from day one.
- **Quality bar.** Faithfulness is the bar that matters most: no answer the doc doesn't support. Retrieval quality and refusal correctness are the next two. Fluency is not a goal.
- **Failure mode I most want to avoid.** A confident wrong answer grounded in a chunk that doesn't actually contain the fact. That's what faithfulness scoring is for, and why the refusal path is treated as a first-class behaviour rather than an afterthought.
- **What I'm not optimising for here.** Throughput, multi-tenancy, persistence, auth. Each is in the "What's next" section with a rough sizing.

## Quick start

```bash
# 1. Free Groq API key from https://console.groq.com (no card required).
cp .env.example .env
# set GROQ_API_KEY=...

# 2. Run
docker compose up --build       # API on http://localhost:8000, docs at /docs
# or, locally:
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

### Try it

```bash
DOC_ID=$(curl -s -X POST http://localhost:8000/documents \
  -H 'Content-Type: application/json' \
  -d '{"text": "The Eiffel Tower was completed in 1889 and stands 330m tall."}' \
  | python -c "import sys, json; print(json.load(sys.stdin)['doc_id'])")

curl -s -X POST http://localhost:8000/questions/$DOC_ID \
  -H 'Content-Type: application/json' \
  -d '{"question": "How tall is the Eiffel Tower?"}' | python -m json.tool
```

### Tests and eval

```bash
pytest -v                       # unit + integration tests, no API key needed
python -m scripts.run_eval      # exits non-zero on regression
```

## Architecture

```
┌───────────────────────────────────────────────────────────────┐
│                       FastAPI app                             │
│                                                               │
│   POST /documents          POST /questions/{doc_id}           │
│        │                          │                           │
│        ▼                          ▼                           │
│   ┌──────────┐              ┌──────────────────┐              │
│   │ingestion │              │  agent service   │              │
│   │ service  │              │                  │              │
│   └────┬─────┘              │  ┌────────────┐  │              │
│        │                    │  │ short-     │  │              │
│        │                    │  │ circuit    │──┼─► REFUSAL    │
│        │                    │  │ (cheap     │  │              │
│        │                    │  │  retrieval)│  │              │
│        │                    │  └────────────┘  │              │
│        │                    │       │ kept     │              │
│        │                    │       ▼          │              │
│        │                    │  ┌────────────┐  │              │
│        │                    │  │ LangChain  │  │              │
│        │                    │  │ agent +    │  │              │
│        │                    │  │ retriever  │  │              │
│        │                    │  │ tool       │  │              │
│        │                    │  └────────────┘  │              │
│        ▼                    └──────┬───────────┘              │
│   ┌─────────────────────────────────────────┐                 │
│   │  in-memory store: dict[doc_id, FAISS]   │                 │
│   └─────────────────────────────────────────┘                 │
│                                                               │
│   cross-cutting: request-id middleware, token/latency/cost    │
│   callback, structured JSON logs, pydantic validation,        │
│   exception handlers, optional LangSmith                      │
└───────────────────────────────────────────────────────────────┘
```

### Module layout

```
app/
├── main.py              FastAPI factory, middleware, exception handlers
├── schemas.py           Pydantic request/response models
├── api/                 thin route handlers
├── core/                config, logging, in-memory store
├── services/            ingestion (chunk+embed+index), agent
├── prompts/             prompt registry (versioned)
└── observability/       callbacks (tokens, latency, cost), pricing, langsmith

tests/                   validation, ingestion, mocked-LLM agent loop
scripts/                 eval cases + Ragas runner with CI exit code
decisions/               ADRs for choices that mattered
```

The split is deliberate: `services/` is what AI engineers touch, `core/` is what platform engineers touch, `observability/` is cross-cutting, `prompts/` is data masquerading as code. Three different people could own three folders without colliding.

## How the rubric maps to the code

| Area | Where |
|---|---|
| Agentic LLM use | `app/services/agent.py` — single tool, agent decides when to call |
| Retrieval | `services/ingestion.py` + per-doc FAISS in `core/store.py` |
| Evaluation | `scripts/run_eval.py` — faithfulness + context precision + refusal exact-match, CI exit code |
| Observability | `observability/callbacks.py` (tokens/latency/cost per request), `tracing.py` (LangSmith opt-in), middleware in `main.py` |
| Error handling | Pydantic validators, route-level mapping, top-level handler, retrieval short-circuit |
| Testing | Mocked-LLM agent test using a scripted chat model, validation, ingestion |
| Security | Prompt-registry rule against following retrieved-content instructions; ingest size cap; PII/secrets handling discussed below |
| Containerisation | Dockerfile pre-downloads embeddings; compose with env injection |
| Secrets | pydantic-settings, `.env.example`, nothing in code |

## Key engineering choices

### Provider abstraction (`get_llm`)

One function, three providers, switched by env. See [ADR 001](decisions/001-provider-abstraction.md). The point isn't supporting three providers; it's that adding the fourth is a one-line change.

### Short-circuit on empty retrieval

A meaningful slice of real Q&A traffic is out-of-scope questions. The agent now runs a pre-flight similarity search and returns the canonical refusal directly if nothing is relevant. Skips the LLM round trip entirely on the unhappy path — that's the request type you most want to make cheap. See [ADR 002](decisions/002-retrieval-short-circuit.md).

The threshold is unprincipled (a guess on L2 distance), and that's flagged in the ADR. The right way is to tune on the eval set, which is the immediate next step.

### Prompts as code, versioned

`app/prompts/registry.py` holds prompts addressed by `(name, version)`. There's only one prompt today, but the *shape* is what matters: the agent never inlines prompt text, so prompt changes are reviewable, diffable, and trivially A/B-able later via a real registry (Langfuse prompts is what I'd reach for).

The system prompt includes a defensive line about not following instructions found in retrieved chunks. That's not a complete prompt-injection defence — see the security section — but it's the cheapest layer.

### Cost in every response

Every `/questions` response includes `metadata.estimated_cost_usd`. It's a coarse estimate (model substring match against a hardcoded price table), but visible from day one. Cost shows up in logs too, ready to aggregate.

In production this gets replaced by reading actual usage from the provider's billing API, attributed per tenant via `request_id`. The estimate stays as the fast-path for dashboards.

### In-memory store

A `dict` behind a small interface. Lost on restart, single-process only. See [ADR 003](decisions/003-in-memory-store.md). The interface is the contract; replacing it with pgvector or Qdrant doesn't touch the services.

## Eval philosophy

Three test cases (direct fact, multi-fact synthesis, refusal) is a smoke test, not an eval set. I treated it as a place to demonstrate *how I think about LLM eval*, not as a finished harness. The runnable shape is real (Ragas + CI exit code), but the substance below is what would matter at scale.

What a real eval pipeline for this service looks like:

- **Golden set.** 30-100 hand-curated questions across the document corpus, refreshed monthly. Owned by a single eval engineer.
- **Synthetic set.** LLM-generated questions from each document, validated by humans before admission. Catches blind spots in the golden set.
- **Production-replay set.** Sampled real questions (PII-scrubbed) with answers re-evaluated. Catches drift.
- **Metrics.** Faithfulness (hallucination), context precision (retrieval signal), context recall (retrieval completeness), answer relevance (usefulness), refusal correctness (false-refusal and missed-refusal rates).
- **Grader independence.** The grader model must not be the system under test. This demo violates that to stay single-key — fix is one line.
- **CI policy.** Eval runs on every PR to `app/services/` or `app/prompts/`. Regression on any metric beyond a configurable floor fails the build. Trends tracked over time.
- **Goodhart guard.** Rotate a holdout subset of the golden set quarterly so we don't overfit to what we measure.

Right now `scripts/run_eval.py` does maybe 20% of that. It does the part that demonstrates the loop.

## Security and safety posture

What's in the code today:
- Hard cap on document size (100KB by default) to prevent resource exhaustion.
- The system prompt instructs the agent to ignore instructions embedded in retrieved chunks. Cheap, partial defence against prompt injection via the document content.
- API keys exclusively from environment; never in code, never in logs.
- Structured JSON logs include `request_id` but never the document text or the question content — both could leak PII.

What I'd add next, roughly in order:
1. **Document-content scanning at ingest.** Look for obvious injection patterns ("ignore previous instructions...") and either reject the document or sanitize. False-positive rate matters; needs a real eval.
2. **PII detection and redaction at ingest.** Presidio or similar. Tag documents as `contains_pii=true` and gate retrieval against tenant policy.
3. **Output filter.** Scan the model's answer for leaked secrets (API keys, JWTs) and refuse rather than return.
4. **Per-tenant rate limits and quotas.** Slowapi at the edge, cost ceilings on the agent path.
5. **Audit trail.** Every question + answer + retrieved chunks logged to a separate audit store with retention and access controls. Required for any regulated industry.

The prompt-injection vector specifically: even with the prompt-level mitigation, a sufficiently determined attacker controlling the document content can make the agent misbehave. The honest position is that no single layer is enough — content scanning, prompt defence, output filtering, and rate limits are all required, and even then the defence is probabilistic.

## What I'd build next

Grouped by theme, sized for honesty.

**Quality (highest impact)**
- Tune the retrieval distance floor against the eval set. Half a day.
- Expand eval to 30+ curated cases, add context recall + answer relevance + refusal-correctness metrics. 2-3 days.
- Use a stronger model as the eval grader, distinct from the SUT. Day.
- Hybrid retrieval (dense + BM25) + cross-encoder reranking. ~3 days, measured against eval.

**Reliability**
- Replace the in-memory store with pgvector. 2 days including migrations.
- Provider fallback in `get_llm` — Groq primary, Anthropic secondary. Half a day.
- Circuit breaker + timeouts on the LLM call. Half a day.
- Multi-worker support (currently single-process due to the in-memory store; the pgvector swap fixes this for free).

**Security**
- Prompt-injection content scan at ingest. 1-2 days with an eval.
- PII detection + redaction. 2 days.
- Output filter for secret leakage. 1 day.

**Cost and observability**
- Read actual usage from the provider billing API for authoritative cost. 1-2 days.
- OpenTelemetry traces/metrics/logs instead of ad-hoc callbacks. 2 days.
- Per-tenant cost dashboards. Half a day once OTel is in.

**Developer experience**
- Prompts in a proper registry (Langfuse) with deploy gates tied to eval. 2-3 days.
- ADRs for every notable decision, kept current. Ongoing.
- A small `eval-watch` CLI that runs the eval set against a candidate prompt or model and prints a diff vs baseline. 1 day, big productivity win for the team.
