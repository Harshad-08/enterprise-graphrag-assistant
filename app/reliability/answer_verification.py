"""Claim-level grounding verification of generated answers.

For every generated claim:

1. Citation labels are checked against the evidence actually supplied to the
   generator (``E1`` ... ``En``). Unknown labels are invalid citations.
2. Each validly-cited passage is checked for support with the configured
   checker (NLI, LLM judge or lexical heuristic), plus a hard rule: every number
   in the claim must appear in the supporting passage.
3. If the cited passage does not support the claim but another supplied passage
   does, the claim is marked *miscited* (``supporting_ids`` set) so the citation
   can be repaired deterministically instead of trusting the model.

The LLM's own confidence is never used as an acceptance criterion.
"""

from __future__ import annotations

import logging
from typing import Protocol

from app.config import Settings
from app.models.schemas import (
    ClaimVerification,
    GeneratedAnswer,
    RetrievedChunk,
    VerificationReport,
    VerificationStatus,
)
from app.reliability.nli import NLIModel
from app.utils.text import NEGATIONS, content_terms, numbers_in, split_sentences, tokenize

logger = logging.getLogger(__name__)


class SupportChecker(Protocol):
    name: str

    def check(self, claim: str, passage: str, title: str | None = None) -> tuple[str, float]:
        """Return (label, score) with label in {"supported", "contradicted", "unsupported"}.

        ``title`` is the source document's title. Passages often refer to their own document
        ("This procedure depends on ..."), so without it a claim naming the document cannot be
        matched to the passage that states it.
        """
        ...


def _numbers_consistent(claim: str, passage: str) -> bool:
    return numbers_in(claim) <= numbers_in(passage)


def _windows(passage: str, title: str | None) -> list[str]:
    """Single sentences and adjacent pairs, each prefixed with the document title so that
    self-references ("this procedure", "this standard") resolve to the document's name."""
    sentences = split_sentences(passage) or [passage]
    windows = sentences + [" ".join(sentences[i:i + 2]) for i in range(len(sentences) - 1)]
    return [f"{title}: {w}" for w in windows] if title else windows


class LexicalSupportChecker:
    """Heuristic: share of the claim's content terms found in one passage sentence window.

    Clearly a heuristic — it cannot detect paraphrase or subtle contradiction. It is
    the offline/mock fallback, and the UI labels verification method accordingly.
    """

    name = "lexical"

    def __init__(self, threshold: float = 0.6) -> None:
        self.threshold = threshold

    def check(self, claim: str, passage: str, title: str | None = None) -> tuple[str, float]:
        claim_terms = content_terms(claim)
        if not claim_terms:
            return "unsupported", 0.0
        windows = _windows(passage, title)
        best_score, best_window = 0.0, ""
        for window in windows:
            score = len(claim_terms & content_terms(window)) / len(claim_terms)
            if score > best_score:
                best_score, best_window = score, window
        if best_score < self.threshold or not _numbers_consistent(claim, passage):
            return "unsupported", best_score
        claim_neg = bool(NEGATIONS & set(tokenize(claim, remove_stopwords=False, stemmed=False)))
        window_neg = bool(NEGATIONS & set(tokenize(best_window, remove_stopwords=False, stemmed=False)))
        if claim_neg != window_neg:
            return "contradicted", best_score
        return "supported", best_score


class NLISupportChecker:
    """Entailment of the claim by the best sentence window of the passage."""

    name = "nli"

    def __init__(self, nli: NLIModel, entailment_threshold: float, contradiction_threshold: float) -> None:
        self.nli = nli
        self.entailment_threshold = entailment_threshold
        self.contradiction_threshold = contradiction_threshold

    def check(self, claim: str, passage: str, title: str | None = None) -> tuple[str, float]:
        # Short premises work far better with small NLI models than whole chunks.
        windows = _windows(passage, title)
        scores = self.nli.predict([(w, claim) for w in windows])
        best_entail = max(s.entailment for s in scores)
        best_contra = max(s.contradiction for s in scores)
        if best_entail >= self.entailment_threshold and _numbers_consistent(claim, passage):
            return "supported", best_entail
        if best_contra >= self.contradiction_threshold:
            return "contradicted", best_contra
        return "unsupported", best_entail


class LLMSupportChecker:
    name = "llm"

    def __init__(self, llm) -> None:
        self.llm = llm

    def check(self, claim: str, passage: str, title: str | None = None) -> tuple[str, float]:
        from app.generation.prompts import VERIFY_PROMPT, SupportJudgement

        premise = f"(From the document '{title}') {passage}" if title else passage
        judgement: SupportJudgement = self.llm.generate_structured(
            VERIFY_PROMPT.format(claim=claim, passage=premise), SupportJudgement
        )
        label = judgement.label if judgement.label in ("supported", "contradicted", "unsupported") else "unsupported"
        if label == "supported" and not _numbers_consistent(claim, passage):
            label = "unsupported"
        return label, 1.0 if label == "supported" else 0.0


def build_support_checker(settings: Settings, nli: NLIModel | None, llm=None) -> SupportChecker:
    if settings.verifier_backend == "nli" and nli is not None:
        return NLISupportChecker(nli, settings.nli_entailment_threshold, settings.nli_contradiction_threshold)
    if settings.verifier_backend == "llm" and llm is not None and not getattr(llm, "is_mock", False):
        return LLMSupportChecker(llm)
    return LexicalSupportChecker(settings.lexical_support_threshold)


def verify_answer(
    answer: GeneratedAnswer,
    evidence: dict[str, RetrievedChunk],
    checker: SupportChecker,
    repair_threshold: float = 0.8,
) -> VerificationReport:
    """``repair_threshold``: moving a citation to a passage the model did *not* cite needs a
    clearly higher support score than accepting the cited one; borderline verifier scores on
    unrelated passages would otherwise attach claims to the wrong source."""
    verifications: list[ClaimVerification] = []
    cache: dict[tuple[str, str], tuple[str, float]] = {}

    def check(claim_text: str, label: str) -> tuple[str, float]:
        key = (claim_text, label)
        if key not in cache:
            item = evidence[label]
            cache[key] = checker.check(claim_text, item.chunk.text, item.document.title)
        return cache[key]

    for claim in answer.claims:
        cited = list(dict.fromkeys(claim.citation_ids))
        valid = [c for c in cited if c in evidence]
        invalid = [c for c in cited if c not in evidence]
        supported_by, contradicted_by, best = [], [], 0.0
        for label in valid:
            verdict, score = check(claim.text, label)
            best = max(best, score)
            if verdict == "supported":
                supported_by.append(label)
            elif verdict == "contradicted":
                contradicted_by.append(label)

        if supported_by:
            status, note = VerificationStatus.SUPPORTED, None
            if invalid:
                note = f"Ignored invalid citation(s): {', '.join(invalid)}."
        else:
            # Look for support elsewhere in the supplied evidence (miscitation).
            alternatives = []
            for label in evidence:
                if label in valid:
                    continue
                verdict, score = check(claim.text, label)
                if verdict == "supported" and score >= repair_threshold:
                    alternatives.append(label)
                    best = max(best, score)
            if contradicted_by:
                status, note = VerificationStatus.CONTRADICTED, f"Cited passage(s) {', '.join(contradicted_by)} contradict the claim."
            elif invalid and not valid:
                status, note = VerificationStatus.INVALID_CITATION, f"Citation(s) {', '.join(invalid)} do not refer to retrieved passages."
            else:
                status = VerificationStatus.UNSUPPORTED
                note = "No cited passage supports this claim." if valid else "Claim has no citation."
            if alternatives:
                note += f" Supported by {', '.join(alternatives)} instead."
            supported_by = alternatives

        verifications.append(
            ClaimVerification(
                claim=claim.text,
                status=status,
                cited_ids=cited,
                valid_citation_ids=valid,
                supporting_ids=supported_by,
                support_score=round(best, 4),
                method=checker.name,
                note=note,
            )
        )
    return VerificationReport(verifications=verifications, method=checker.name)


def repair_citations(report: VerificationReport) -> tuple[VerificationReport, int]:
    """Re-attribute miscited claims to the passage that verifiably supports them."""
    repaired = 0
    fixed: list[ClaimVerification] = []
    for v in report.verifications:
        if v.status != VerificationStatus.SUPPORTED and v.supporting_ids and v.status != VerificationStatus.CONTRADICTED:
            v = v.model_copy(
                update={
                    "status": VerificationStatus.SUPPORTED,
                    "note": f"Citation corrected from {v.cited_ids or 'none'} to {v.supporting_ids}.",
                }
            )
            repaired += 1
        fixed.append(v)
    return VerificationReport(verifications=fixed, method=report.method), repaired
