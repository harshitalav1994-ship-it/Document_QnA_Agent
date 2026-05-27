"""Input validation tests — exercise the API surface without calling an LLM."""
from __future__ import annotations

import pytest

from app.core.config import get_settings


def test_ingest_rejects_empty_text(client, fake_embeddings):
    resp = client.post("/documents", json={"text": ""})
    assert resp.status_code == 422  # pydantic min_length


def test_ingest_rejects_oversized_text(client, fake_embeddings):
    limit = get_settings().max_document_chars
    resp = client.post("/documents", json={"text": "a" * (limit + 1)})
    assert resp.status_code == 422
    body = resp.json()
    assert "exceeds maximum size" in str(body).lower()


def test_ingest_happy_path(client, fake_embeddings):
    resp = client.post(
        "/documents",
        json={"text": "The quick brown fox jumps over the lazy dog. " * 20},
    )
    assert resp.status_code == 201
    body = resp.json()
    assert "doc_id" in body
    assert body["chunk_count"] >= 1
    assert body["char_count"] > 0


def test_question_missing_document_returns_404(client, fake_embeddings):
    resp = client.post(
        "/questions/does-not-exist",
        json={"question": "anything?"},
    )
    assert resp.status_code == 404


def test_question_rejects_empty_question(client, fake_embeddings):
    # First ingest something so we hit the validation, not the 404.
    ingest_resp = client.post("/documents", json={"text": "hello world"})
    doc_id = ingest_resp.json()["doc_id"]

    resp = client.post(f"/questions/{doc_id}", json={"question": ""})
    assert resp.status_code == 422


def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}
