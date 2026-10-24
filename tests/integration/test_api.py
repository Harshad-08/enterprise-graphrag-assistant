"""API-level user flows with an injected deterministic container (no external services)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.config import PROJECT_ROOT
from app.services.container import set_container
from tests.conftest import make_container

TXT = b"""---
document_id: API-TEST-1
title: Badge Access Procedure
doc_type: procedure
department: Facilities
status: active
authority: department
effective_date: 2024-01-01
publication_date: 2024-01-01
version: 1.0
---
# 1. Badges
Visitors must be escorted at all times. Lost badges must be reported within 24 hours.
"""


@pytest.fixture
def client(tmp_path):
    container = make_container(tmp_path, max_upload_mb=1)
    from app.services.document_service import DocumentService

    DocumentService(container).seed_directory(PROJECT_ROOT / "data" / "synthetic_documents")
    set_container(container)
    from app.main import app

    with TestClient(app) as test_client:
        yield test_client
    set_container(None)


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["graph_connected"] and body["mock_mode"]
    assert body["documents_indexed"] == 13


def test_upload_list_get_chat_delete_flow(client):
    response = client.post("/documents/upload", files={"file": ("../../etc/badge procedure.txt", TXT, "text/plain")})
    assert response.status_code == 201, response.text
    result = response.json()
    assert result["document_id"] == "API-TEST-1" and result["chunk_count"] >= 1 and result["claim_count"] >= 1

    ids = [d["metadata"]["document_id"] for d in client.get("/documents").json()]
    assert "API-TEST-1" in ids
    record = client.get("/documents/API-TEST-1").json()
    assert record["metadata"]["filename"] == "badge_procedure.txt"  # path components and spaces sanitised

    chat = client.post("/chat", json={"question": "How quickly must lost badges be reported?"}).json()
    assert not chat["abstained"]
    assert "24 hours" in chat["answer"]
    assert chat["citations"][0]["document_id"] == "API-TEST-1"
    for key in ("supporting_chunk_ids", "recovery_performed", "conflict_detected", "verified", "abstained", "reliability_notes"):
        assert key in chat

    assert client.delete("/documents/API-TEST-1").json()["deleted"]
    assert client.get("/documents/API-TEST-1").status_code == 404
    assert client.delete("/documents/API-TEST-1").status_code == 404
    after = client.post("/chat", json={"question": "How quickly must lost badges be reported?"}).json()
    assert "API-TEST-1" not in {c["document_id"] for c in after["citations"]}
    assert not any(cid.startswith("API-TEST-1") for cid in after["retrieved_chunk_ids"])  # vector + BM25 entries gone


def test_upload_validation(client):
    assert client.post("/documents/upload", files={"file": ("malware.exe", b"MZ...", "application/octet-stream")}).status_code == 400
    assert client.post("/documents/upload", files={"file": ("fake.pdf", b"hello", "application/pdf")}).status_code == 400
    assert client.post("/documents/upload", files={"file": ("empty.txt", b"", "text/plain")}).status_code == 400
    big = b"a" * (1024 * 1024 + 10)
    assert client.post("/documents/upload", files={"file": ("big.txt", big, "text/plain")}).status_code == 413


def test_chat_request_validation(client):
    assert client.post("/chat", json={"question": "x"}).status_code == 422
    assert client.post("/chat", json={"question": "Valid question?", "mode": "magic"}).status_code == 422


def test_challenge_flows_through_api(client):
    absent = client.post("/chat", json={"question": "What is the company's maternity leave policy?"}).json()
    assert absent["abstained"] and absent["retrieval_outcome"] == "not_in_corpus"
    conflict = client.post("/chat", json={"question": "What is the inactivity timeout for VPN sessions?"}).json()
    assert conflict["conflict_detected"] and any(not c["resolved"] for c in conflict["conflicts"])
    multi_hop = client.post("/chat", json={"question": "Which procedure depends on the API authentication policy?"}).json()
    assert "PROC-PART-003" in {c["document_id"] for c in multi_hop["citations"]}
    baseline = client.post("/chat", json={"question": "What is the company's maternity leave policy?", "mode": "baseline"}).json()
    assert baseline["mode"] == "baseline" and not baseline["verified"]


def test_evaluation_endpoint(client):
    report = client.post("/evaluation/run", json={"limit": 3}).json()
    assert set(report["custom_metrics"]) == {"baseline", "improved"}
    assert report["item_count"] == 3
    assert report["ragas"]["status"] == "not_requested"
    assert client.get("/evaluation/latest").status_code == 200
