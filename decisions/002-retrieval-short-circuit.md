# 002 — Short-circuit when retrieval finds nothing relevant

Date: during the build.
Status: accepted, with caveats.

## Context

A meaningful fraction of real production traffic to a doc-QA service is out-of-scope questions — users asking things the doc doesn't cover. Naively, every one of those goes through the full agent loop: tool call → LLM round trip → refusal. That's ~all the cost and latency of a normal request, for an answer we could have predicted before calling the model.

## Decision

Before invoking the agent, run a cheap similarity search. If zero chunks come back, or if every chunk's distance is above a relevance floor, skip the LLM entirely and return the canonical refusal string directly.

The response includes `metadata.short_circuited: true` so callers (and our own dashboards) can tell the answer came from the fast path.

## Alternatives considered

- **Let the agent decide via the prompt alone.** Works, but pays full token cost on every out-of-scope question. Rejected on cost.
- **Train a tiny classifier for in-scope/out-of-scope.** Better, but a 2-3 hour build can't justify a model. Could be a real improvement later — call it future work.

## Consequences

- We trust the embedding model's distance metric as a relevance signal. The floor (currently 1.5 for L2 on MiniLM) is a guess, not a tuned value. If it's too tight we miss valid questions; too loose and we pay for nothing.
- The threshold is hardcoded. It should be in settings, and ideally per-document (some docs are denser than others).

## What I'd revisit

Tune the floor against the eval set. Add a `false_refusal_rate` metric. Once tuned, expose the floor as a per-tenant setting and surface "we were unsure, here's our best guess anyway" as an alternative behaviour for accounts that prefer recall over precision.
