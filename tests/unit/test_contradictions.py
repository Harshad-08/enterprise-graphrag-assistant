"""Challenge 2 — documents contradict each other."""

from __future__ import annotations

from datetime import date

from app.config import PROJECT_ROOT
from app.graph.entity_extraction import Gazetteer
from app.models.schemas import (
    ChatRequest,
    Chunk,
    Claim,
    ClaimRelationType,
    DocumentMetadata,
    DocumentStatus,
    Polarity,
)
from app.reliability.claim_extraction import (
    detect_polarity,
    extract_claims,
    extract_conditions,
    parse_quantity,
    validate_claim_provenance,
)
from app.reliability.conflict_analysis import analyze_conflicts
from app.reliability.conflict_resolution import ConflictResolver
from app.reliability.contradiction_detection import ContradictionDetector
from app.graph.entity_extraction import analyze_query
from app.services.chat_service import ChatService
from tests.conftest import make_settings

GAZ = Gazetteer.from_ontology(PROJECT_ROOT / "data" / "ontology.json")
SETTINGS = make_settings()
TODAY = date(2026, 1, 1)


def claim(cid: str, doc: str, text: str, *, subject="system:vpn", prop="session_timeout", polarity=Polarity.REQUIRE,
          quantity=None, unit=None, conditions=None, status=DocumentStatus.ACTIVE, authority="department",
          effective=date(2024, 1, 1), version="1.0", title=None) -> Claim:
    return Claim(
        claim_id=cid, document_id=doc, chunk_id=f"{doc}::c000", subject_id=subject, subject=subject, property=prop,
        text=text, polarity=polarity, quantity=quantity, unit=unit, conditions=conditions or [], status=status,
        authority=authority, effective_date=effective, version=version, document_title=title or doc,
    )


class TestClaimExtraction:
    def test_polarity_quantity_and_conditions(self):
        assert detect_polarity("Personal devices are prohibited for remote access.") == Polarity.PROHIBIT
        assert detect_polarity("Employees must use a company laptop.") == Polarity.REQUIRE
        assert detect_polarity("Employees may work from home.") == Polarity.PERMIT
        assert parse_quantity("Logs are retained for 12 months.") == (12 * 43200, "minutes")
        assert parse_quantity("retained for 90 days") == (90 * 1440, "minutes")
        assert parse_quantity("up to 3 days per week") == (3, "days_per_week")
        assert parse_quantity("at least 14 characters") == (14, "characters")
        assert extract_conditions("Devices are permitted only when accessing webmail through a browser.") == [
            "only when accessing webmail through a browser"
        ]

    def test_extracts_structured_claims_with_provenance(self):
        meta = DocumentMetadata(document_id="SEC-X", title="Network Guidelines", doc_type="guideline",
                                status=DocumentStatus.ACTIVE, authority="department", version="1.0")
        chunk = Chunk(chunk_id="SEC-X::c000", document_id="SEC-X", text="VPN sessions must be disconnected after 15 minutes of inactivity. This is background text.",
                      chunk_index=0, content_hash="h", section="Remote")
        claims = extract_claims(meta, [chunk], GAZ.copy())
        assert len(claims) == 1
        c = claims[0]
        assert (c.subject, c.property, c.polarity, c.quantity, c.unit) == ("VPN Sessions", "session_timeout", Polarity.REQUIRE, 15, "minutes")
        assert (c.document_id, c.chunk_id, c.section, c.status, c.authority, c.version) == (
            "SEC-X", "SEC-X::c000", "Remote", DocumentStatus.ACTIVE, "department", "1.0")


class TestContradictionDetection:
    detector = ContradictionDetector(SETTINGS)

    def test_numeric_conflict_is_contradiction(self):
        a = claim("a", "D1", "VPN sessions disconnect after 60 minutes.", quantity=60, unit="minutes")
        b = claim("b", "D2", "VPN sessions disconnect after 15 minutes.", quantity=15, unit="minutes")
        assert self.detector.compare(a, b).relation == ClaimRelationType.CONTRADICTS

    def test_opposing_polarity_is_contradiction(self):
        a = claim("a", "D1", "Personal devices may be used.", prop="remote_access", polarity=Polarity.PERMIT)
        b = claim("b", "D2", "Personal devices are prohibited.", prop="remote_access", polarity=Polarity.PROHIBIT)
        assert self.detector.compare(a, b).relation == ClaimRelationType.CONTRADICTS

    def test_different_conditions_are_compatible(self):
        general = claim("a", "D1", "Personal devices are prohibited for remote access.", prop="remote_access", polarity=Polarity.PROHIBIT)
        exception = claim("b", "D2", "Personal devices are permitted only when accessing webmail.", prop="remote_access",
                          polarity=Polarity.PERMIT, conditions=["only when accessing webmail"])
        assert self.detector.compare(general, exception).relation == ClaimRelationType.COMPATIBLE_CONDITIONAL

    def test_different_wording_same_rule_supports(self):
        a = claim("a", "D1", "Incidents must be reported within 1 hour.", prop="incident_reporting", quantity=60, unit="minutes")
        b = claim("b", "D2", "All suspected incidents must be reported within one hour of discovery.", prop="incident_reporting", quantity=60, unit="minutes")
        assert self.detector.compare(a, b).relation == ClaimRelationType.SUPPORTS

    def test_incorrectly_extracted_claims_do_not_create_false_conflicts(self):
        # Wrong subject assigned by extraction -> different subjects -> never compared as conflicting.
        a = claim("a", "D1", "VPN sessions disconnect after 60 minutes.", quantity=60, unit="minutes", subject="system:vpn")
        b = claim("b", "D2", "Passwords expire after 15 minutes.", quantity=15, unit="minutes", subject="concept:passwords")
        assert self.detector.compare(a, b).relation == ClaimRelationType.UNRELATED
        # Different property -> unrelated.
        c = claim("c", "D2", "VPN client must update within 14 days.", quantity=20160, unit="minutes", prop="client_update")
        assert self.detector.compare(a, c).relation == ClaimRelationType.UNRELATED
        # Unstructured claims are not guessed without an NLI model.
        d = claim("d", "D1", "Split tunnelling is not permitted.", prop=None, polarity=Polarity.PROHIBIT)
        e = claim("e", "D2", "Remote connections must use the VPN.", prop=None)
        assert self.detector.compare(d, e).relation == ClaimRelationType.UNRELATED
        # Hallucinated claim text that is not in the source chunk fails provenance validation.
        chunk = Chunk(chunk_id="D1::c000", document_id="D1", text="VPN sessions disconnect after 60 minutes.", chunk_index=0, content_hash="h")
        assert validate_claim_provenance(a, chunk)
        assert not validate_claim_provenance(a.model_copy(update={"text": "VPN sessions never disconnect."}), chunk)

    def test_same_document_claims_are_not_compared(self):
        a = claim("a", "D1", "x after 60 minutes", quantity=60, unit="minutes")
        b = claim("b", "D1", "x after 15 minutes", quantity=15, unit="minutes")
        assert self.detector.detect([a, b]) == []


class TestConflictResolution:
    def _pair(self, **b_overrides):
        a = claim("a", "NEW", "Personal devices are prohibited.", prop="remote_access", polarity=Polarity.PROHIBIT,
                  effective=date(2024, 3, 1), version="2.0", title="Remote Work Policy", authority="executive")
        b_args = dict(prop="remote_access", polarity=Polarity.PERMIT, effective=date(2021, 2, 1), version="1.0",
                      title="Remote Work Policy", authority="department", status=DocumentStatus.SUPERSEDED)
        b_args.update(b_overrides)
        b = claim("b", "OLD", "Personal devices may be used.", **b_args)
        return a, b

    def test_authoritative_new_policy_supersedes_old(self):
        a, b = self._pair()
        result = ConflictResolver(SETTINGS, [("NEW", "OLD")], TODAY).resolve(a, b)
        assert result.resolved and result.winner_document_id == "NEW" and result.rule == "supersession"

    def test_insufficient_precedence_metadata_stays_unresolved(self):
        a = claim("a", "IT", "VPN 60 minutes", quantity=60, unit="minutes", title="Remote Access Standard")
        b = claim("b", "SEC", "VPN 15 minutes", quantity=15, unit="minutes", effective=None, title="Network Guidelines")
        result = ConflictResolver(SETTINGS, [], TODAY).resolve(a, b)
        assert not result.resolved and result.winner_claim_id is None
        assert any("Effective date missing" in r for r in result.rules_considered)

    def test_newest_document_not_trusted_when_authority_unclear(self):
        newer = claim("a", "NEW", "60 minutes", quantity=60, unit="minutes", effective=date(2025, 1, 1), authority=None)
        older = claim("b", "OLD", "15 minutes", quantity=15, unit="minutes", effective=date(2023, 1, 1))
        result = ConflictResolver(SETTINGS, [], TODAY).resolve(newer, older)
        assert not result.resolved

    def test_equal_authority_later_effective_date_wins(self):
        newer = claim("a", "NEW", "60 minutes", quantity=60, unit="minutes", effective=date(2025, 1, 1))
        older = claim("b", "OLD", "15 minutes", quantity=15, unit="minutes", effective=date(2023, 1, 1))
        result = ConflictResolver(SETTINGS, [], TODAY).resolve(newer, older)
        assert result.resolved and result.winner_document_id == "NEW" and result.rule == "effective_date"

    def test_missing_or_incorrect_metadata(self):
        # Circular supersession metadata is ignored rather than trusted.
        a, b = self._pair(status=DocumentStatus.ACTIVE, authority="executive")
        circular = ConflictResolver(SETTINGS, [("NEW", "OLD"), ("OLD", "NEW")], TODAY).resolve(a, b)
        assert circular.rule != "supersession"
        assert any("circular" in r for r in circular.rules_considered)
        # Unknown status and authority: those rules are skipped, not guessed.
        unknown = claim("u", "UNK", "x", prop="remote_access", polarity=Polarity.PERMIT, status=DocumentStatus.UNKNOWN,
                        authority=None, effective=None, version=None, title="Other")
        result = ConflictResolver(SETTINGS, [], TODAY).resolve(a, unknown)
        assert not result.resolved
        # A document that is not yet effective loses to the one currently in force.
        future = claim("f", "FUT", "15 minutes", quantity=15, unit="minutes", effective=date(2027, 1, 1))
        current = claim("c", "CUR", "60 minutes", quantity=60, unit="minutes", effective=date(2024, 1, 1))
        assert ConflictResolver(SETTINGS, [], TODAY).resolve(future, current).winner_document_id == "CUR"

    def test_precedence_order_is_configurable(self):
        a, b = self._pair()
        authority_first = SETTINGS.model_copy(update={"conflict_precedence": ["authority", "supersession"]})
        assert ConflictResolver(authority_first, [("NEW", "OLD")], TODAY).resolve(a, b).rule == "authority"


class TestConflictsInCorpus:
    def test_conflicts_detected_across_graph_connected_documents(self, seeded_container):
        """Evidence contains only one side; the counterpart is found through shared subject/property."""
        c = seeded_container
        evidence = [e for e in c.hybrid.retrieve("VPN sessions disconnected after 60 minutes of inactivity")
                    if e.chunk.document_id == "IT-STD-015"][:1]
        for e in evidence:
            e.rerank_score = 1.0
        analysis = analyze_query("What is the inactivity timeout for VPN sessions?", c.gazetteer)
        result = analyze_conflicts(analysis, evidence, c.store, c.detector, c.settings)
        assert len(result.conflicts) == 1
        report = result.conflicts[0]
        assert {report.comparison.claim_a.document_id, report.comparison.claim_b.document_id} == {"IT-STD-015", "SEC-GDL-009"}
        assert not report.resolution.resolved
        assert any(cid.startswith("SEC-GDL-009") for cid in result.extra_chunk_ids)

    def test_stored_claim_relationships(self, seeded_container):
        rels = seeded_container.store.claim_relations()
        contradictions = {(frozenset((r["document_a"], r["document_b"])), r["subject"], r.get("winner_document_id"))
                          for r in rels if r["type"] == "CONTRADICTS"}
        assert (frozenset(("HR-POL-007", "HR-POL-007-V1")), "Personal Devices", "HR-POL-007") in contradictions
        assert (frozenset(("LEG-POL-003", "SRE-STD-010")), "Application Logs", "LEG-POL-003") in contradictions
        assert (frozenset(("IT-STD-015", "SEC-GDL-009")), "VPN Sessions", None) in contradictions
        assert any(r["type"] == "SUPPORTS" for r in rels)

    def test_unresolved_conflict_is_disclosed_with_both_sources(self, seeded_container):
        response = ChatService(seeded_container).answer(ChatRequest(question="What is the inactivity timeout for VPN sessions?"))
        assert response.conflict_detected and not response.abstained
        unresolved = [c for c in response.conflicts if not c.resolved]
        assert unresolved and unresolved[0].applicable_document is None
        assert "60 minutes" in response.answer and "15 minutes" in response.answer
        assert "Unresolved conflict" in response.answer
        assert {"IT-STD-015", "SEC-GDL-009"} <= {c.document_id for c in response.citations}

    def test_resolved_conflict_answers_with_applicable_rule(self, seeded_container):
        response = ChatService(seeded_container).answer(ChatRequest(question="How long must application logs be retained?"))
        resolved = [c for c in response.conflicts if c.resolved]
        assert resolved and resolved[0].applicable_document == "LEG-POL-003" and resolved[0].rule == "authority"
        assert "12 months" in response.answer
        assert "retained for 90 days" not in response.answer.split("Note:")[0]
