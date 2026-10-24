"""Rank fusion, BM25, hybrid retrieval, filters and reranking integration."""

from __future__ import annotations

import math

import numpy as np
import pytest

from app.models.schemas import Chunk, DocumentMetadata, RetrievedChunk
from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.keyword_search import BM25Index
from app.retrieval.reranking import CrossEncoderReranker, LexicalReranker, rerank
from tests.conftest import ingest_texts, make_container


def _chunk(cid: str, text: str) -> Chunk:
    return Chunk(chunk_id=cid, document_id=cid.split("::")[0], text=text, chunk_index=0, content_hash="h")


class TestFusion:
    def test_rrf_scores_and_order(self):
        fused = reciprocal_rank_fusion([["a", "b", "c"], ["b", "d"]], k=60)
        scores = dict(fused)
        assert fused[0][0] == "b"  # appears high in both lists
        assert math.isclose(scores["b"], 1 / 62 + 1 / 61)
        assert math.isclose(scores["d"], 1 / 62)
        assert set(scores) == {"a", "b", "c", "d"}

    def test_duplicates_within_a_list_count_once(self):
        assert dict(reciprocal_rank_fusion([["a", "a", "b"]], k=10))["a"] == pytest.approx(1 / 11)

    def test_empty_rankings(self):
        assert reciprocal_rank_fusion([[], []]) == []


class TestBM25:
    def test_keyword_search_and_document_frequency(self):
        index = BM25Index()
        index.build([_chunk("A::c0", "Passwords must contain 14 characters."), _chunk("B::c0", "VPN sessions time out."),
                     _chunk("C::c0", "Backups are kept for 35 days.")])
        results = index.search("password length", k=5)
        assert results[0][0].chunk_id == "A::c0"
        assert index.document_frequency("passwords") == 1
        assert index.document_frequency("maternity") == 0
        assert index.search("password", k=5, allowed_document_ids={"B"}) == []


DOCS = {
    "a.txt": "---\ndocument_id: PWD-1\ntitle: Password Standard\ndoc_type: standard\ndepartment: Information Security\nstatus: active\n---\n"
             "# 1. Rules\nPasswords must contain at least 14 characters.",
    "b.txt": "---\ndocument_id: BKP-1\ntitle: Backup Guide\ndoc_type: technical\ndepartment: Platform Engineering\nstatus: superseded\n---\n"
             "# 1. Backups\nDatabase backups are kept for 35 days.",
    # A third document: with only two documents BM25Okapi gives a term found in exactly one
    # of them an IDF of log(1.5/1.5) = 0, so keyword search would return nothing.
    "c.txt": "---\ndocument_id: VPN-1\ntitle: VPN Guide\ndoc_type: technical\nstatus: active\n---\n"
             "# 1. Sessions\nVPN sessions are disconnected after 15 minutes of inactivity.",
}


class TestHybridRetrieval:
    def test_hybrid_merges_vector_and_keyword_hits(self, tmp_path):
        c = make_container(tmp_path)
        ingest_texts(c, DOCS)
        candidates = c.hybrid.candidates("How long are database backups kept?")
        top = candidates[0]
        assert top.chunk.document_id == "BKP-1"
        assert {"vector", "keyword"} <= set(top.sources)
        assert top.rrf_score and top.vector_score is not None and top.keyword_score is not None

    def test_metadata_filters(self, tmp_path):
        c = make_container(tmp_path)
        ingest_texts(c, DOCS)
        only_security = c.hybrid.retrieve("backups passwords", filters={"department": "Information Security"})
        assert {r.chunk.document_id for r in only_security} == {"PWD-1"}
        c2 = make_container(tmp_path, include_superseded_in_retrieval=False)
        ingest_texts(c2, DOCS)
        assert "BKP-1" not in {r.chunk.document_id for r in c2.hybrid.retrieve("database backups kept")}

    def test_vector_only_baseline(self, tmp_path):
        c = make_container(tmp_path)
        ingest_texts(c, DOCS)
        results = c.hybrid.vector_only("password characters", k=1)
        assert len(results) == 1 and results[0].sources == ["vector"] and results[0].rerank_score is None


class FakeCrossEncoder:
    """Stands in for sentence_transformers.CrossEncoder: high logit when 'backup' is in the passage."""

    def predict(self, pairs, show_progress_bar=False):
        return np.array([4.0 if "backup" in passage.lower() else -4.0 for _, passage in pairs])


class TestReranking:
    def _items(self):
        meta = DocumentMetadata(document_id="X", title="T")
        return [RetrievedChunk(chunk=_chunk("X::c0", "Passwords rules."), document=meta),
                RetrievedChunk(chunk=_chunk("X::c1", "Backup retention."), document=meta)]

    def test_cross_encoder_scores_are_sigmoid_probabilities(self):
        reranker = CrossEncoderReranker("fake", model=FakeCrossEncoder())
        ranked = rerank(reranker, "how long are backups kept", self._items())
        assert ranked[0].chunk_id == "X::c1"
        assert ranked[0].rerank_score == pytest.approx(1 / (1 + math.exp(-4)))
        assert 0 < ranked[1].rerank_score < 0.05

    def test_alternative_queries_take_max_score(self):
        reranker = LexicalReranker()
        items = self._items()
        plain = rerank(reranker, "zebra", [i.model_copy(deep=True) for i in items])
        assert all(i.rerank_score == 0 for i in plain)
        boosted = rerank(reranker, "zebra", items, alt_queries=["backup retention"])
        assert boosted[0].chunk_id == "X::c1" and boosted[0].rerank_score == 1.0
