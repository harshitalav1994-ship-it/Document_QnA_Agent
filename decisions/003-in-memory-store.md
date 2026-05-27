# 003 — In-memory store, deferred persistence

Date: during the build.
Status: accepted for this iteration, must be replaced before any real use.

## Context

The service needs to remember an ingested document between the `/documents` call and the subsequent `/questions/{doc_id}` calls. Options range from "Python dict" to "Postgres + pgvector with proper migrations." Picking right is mostly about how much we want to commit to in 2-3 hours.

## Decision

A module-level `dict[doc_id, DocumentRecord]` guarded by an `RLock`, behind a small `DocumentStore` interface. Lost on process restart, no cross-process sharing.

The interface is intentionally tiny (`put`, `get`, `exists`, `delete`) so a Postgres-backed implementation can drop in without touching the services.

## Alternatives considered

- **SQLite + FAISS files on disk.** Persists across restarts, still no infra. Tempting, but adds file management, embedding-version handling, and an init path. Not worth the time for an MVP.
- **Postgres + pgvector.** What I'd actually use in production. Out of scope.
- **A managed vector DB (Pinecone, Qdrant Cloud).** Same — production answer, demo overkill.

## Consequences

- Single-process only. Two uvicorn workers will give you "doc not found" errors on half your requests because they don't share state. The Dockerfile launches a single worker; that's the constraint, not an accident.
- No tenancy. `doc_id` is a UUID with no namespace.
- The lock is fine-grained enough for the current workload but `_records.clear()` from tests reaches into the private attribute. Mild wart.

## What I'd revisit

This is the first thing I'd swap. The shape of the swap is clear: implement `PgVectorDocumentStore` against the same interface, ship migrations, add a tenant_id namespace at the same time. Two-day job done properly.
