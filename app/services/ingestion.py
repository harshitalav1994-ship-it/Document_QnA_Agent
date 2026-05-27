"""
Document ingestion service.

Responsibilities:
  - chunk raw text into overlapping windows
  - embed chunks with a local HuggingFace model (no paid API needed)
  - build a per-document FAISS index
  - persist the record into the in-memory store

The embeddings model is loaded lazily and cached so the first request
warms it but subsequent calls reuse the same instance.
"""
from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone
from functools import lru_cache

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.store import DocumentRecord, DocumentStore

logger = get_logger(__name__)


# Cheap regex patterns that often appear in prompt-injection payloads. This
# is NOT a security control on its own — a determined attacker will bypass
# regexes. It exists to (a) flag suspicious uploads in logs so we can audit,
# and (b) make the layered story honest: input filter + system-prompt rule 6
# + (in production) an output filter. See decisions/0003.
_INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions", re.I),
    re.compile(r"disregard\s+(all\s+)?(previous|prior|above)", re.I),
    re.compile(r"you\s+are\s+now\s+", re.I),
    re.compile(r"system\s*:\s*", re.I),
    re.compile(r"<\s*/?\s*system\s*>", re.I),
]


def _scan_for_injection_signals(text: str) -> list[str]:
    """Return names of patterns that matched. Used for logging, not blocking."""
    return [p.pattern for p in _INJECTION_PATTERNS if p.search(text)]


@lru_cache(maxsize=1)
def get_embeddings() -> Embeddings:
    """Cached embedding model. First call loads the model into memory."""
    # Imported lazily so app startup doesn't require the (heavy) HF stack
    # when tests inject a fake embedding model via monkeypatch.
    from langchain_huggingface import HuggingFaceEmbeddings

    settings = get_settings()
    logger.info("loading_embedding_model", extra={"model": settings.embedding_model})
    return HuggingFaceEmbeddings(model_name=settings.embedding_model)


def _split_text(text: str) -> list[Document]:
    """Apply the configured recursive splitter."""
    settings = get_settings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        # The default separators handle prose, code, and markdown reasonably.
        # Tuning per content type is out of scope for this MVP; see README.
    )
    chunks = splitter.split_text(text)
    return [
        Document(page_content=chunk, metadata={"chunk_index": idx})
        for idx, chunk in enumerate(chunks)
    ]


def ingest_document(
    text: str,
    store: DocumentStore,
    metadata: dict | None = None,
) -> DocumentRecord:
    """Ingest one document; return the stored record."""
    if not text.strip():
        # Defensive: the schema also enforces this, but never trust the boundary.
        raise ValueError("Document text is empty.")

    doc_id = str(uuid.uuid4())

    # Surface suspected injection signals in logs. We do NOT reject — false
    # positives are too common (the word "ignore" appears in plenty of
    # legitimate docs). For a tenant with a stricter policy, this is the
    # hook point to add rejection or human-review queueing.
    signals = _scan_for_injection_signals(text)
    if signals:
        logger.warning(
            "ingest_suspected_injection_pattern",
            extra={"doc_id": doc_id, "patterns_matched": len(signals)},
        )

    documents = _split_text(text)

    if not documents:
        raise ValueError("Document produced zero chunks after splitting.")

    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(documents, embeddings)

    record = DocumentRecord(
        doc_id=doc_id,
        vectorstore=vectorstore,
        chunk_count=len(documents),
        char_count=len(text),
        created_at=datetime.now(timezone.utc),
        metadata=metadata or {},
    )
    store.put(record)

    logger.info(
        "document_ingested",
        extra={
            "doc_id": doc_id,
            "chunk_count": record.chunk_count,
            "char_count": record.char_count,
        },
    )
    return record
