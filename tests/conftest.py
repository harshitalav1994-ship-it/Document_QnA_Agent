"""Shared pytest fixtures."""
from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from app.core.store import DocumentStore, get_store
from app.main import app


class FakeEmbeddings(Embeddings):
    """
    Deterministic, dependency-free embeddings for tests.

    We don't want to download the real HuggingFace model in unit tests, so we
    use a simple bag-of-words vector. Good enough to make FAISS
    similarity rank chunks predictably for the tests we care about.
    """

    DIMS = 64

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.DIMS
        for token in text.lower().split():
            vec[hash(token) % self.DIMS] += 1.0
        # L2-normalise for stable distances.
        norm = sum(v * v for v in vec) ** 0.5 or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


@pytest.fixture(autouse=True)
def isolate_store():
    """Each test starts with a fresh in-memory store."""
    store = get_store()
    # Clear any state from previous tests.
    store._records.clear()  # type: ignore[attr-defined]
    yield store
    store._records.clear()  # type: ignore[attr-defined]


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture
def fake_embeddings(monkeypatch) -> FakeEmbeddings:
    """Replace HuggingFace embeddings with a fake everywhere they're used."""
    fake = FakeEmbeddings()
    from app.services import ingestion

    monkeypatch.setattr(ingestion, "get_embeddings", lambda: fake)
    return fake


@pytest.fixture
def make_record(fake_embeddings):
    """Factory for an ingested DocumentRecord without going through the API."""
    from app.services.ingestion import ingest_document

    def _make(text: str, store: DocumentStore) -> Any:
        return ingest_document(text=text, store=store)

    return _make
