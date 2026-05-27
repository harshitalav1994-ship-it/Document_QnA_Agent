"""
In-memory document store.

Maps doc_id -> (vectorstore, metadata). Lost on process restart; this
trade-off is documented in the README. A lock guards concurrent writes
since FastAPI runs handlers on a thread pool by default for sync code
and we want safe inserts under load.

For production, replace with a persistent vector DB (pgvector, Qdrant,
Pinecone). The interface here is intentionally small so swapping is easy.
"""
from dataclasses import dataclass, field
from datetime import datetime
from threading import RLock
from typing import Any

from langchain_core.vectorstores import VectorStore


@dataclass
class DocumentRecord:
    """Everything we keep about an ingested document."""

    doc_id: str
    vectorstore: VectorStore
    chunk_count: int
    char_count: int
    created_at: datetime
    metadata: dict[str, Any] = field(default_factory=dict)


class DocumentStore:
    """Tiny thread-safe in-memory key-value store for DocumentRecords."""

    def __init__(self) -> None:
        self._records: dict[str, DocumentRecord] = {}
        self._lock = RLock()

    def put(self, record: DocumentRecord) -> None:
        with self._lock:
            self._records[record.doc_id] = record

    def get(self, doc_id: str) -> DocumentRecord | None:
        with self._lock:
            return self._records.get(doc_id)

    def exists(self, doc_id: str) -> bool:
        with self._lock:
            return doc_id in self._records

    def delete(self, doc_id: str) -> bool:
        with self._lock:
            return self._records.pop(doc_id, None) is not None

    def __len__(self) -> int:
        with self._lock:
            return len(self._records)



# _store = DocumentStore() — at module import time, Python creates one DocumentStore instance and assigns it to a module-level variable. The leading underscore is a Python convention meaning "this is private; don't import it directly from outside this module."
# def get_store() -> DocumentStore: return _store — a function that returns that same single instance every time it's called.
_store = DocumentStore()


def get_store() -> DocumentStore:
    """Dependency-injectable accessor."""
    return _store
