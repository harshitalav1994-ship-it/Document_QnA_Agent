"""Unit tests for the ingestion service."""
from __future__ import annotations

import pytest

from app.core.store import DocumentStore
from app.services.ingestion import ingest_document


def test_ingest_chunks_long_document(fake_embeddings):
    store = DocumentStore()
    long_text = "The mitochondria is the powerhouse of the cell. " * 200
    record = ingest_document(text=long_text, store=store)
    assert record.chunk_count > 1
    assert store.get(record.doc_id) is record


def test_ingest_rejects_whitespace_only(fake_embeddings):
    store = DocumentStore()
    with pytest.raises(ValueError):
        ingest_document(text="   \n\t  ", store=store)


def test_ingest_assigns_unique_doc_ids(fake_embeddings):
    store = DocumentStore()
    r1 = ingest_document(text="content one " * 10, store=store)
    r2 = ingest_document(text="content two " * 10, store=store)
    assert r1.doc_id != r2.doc_id
