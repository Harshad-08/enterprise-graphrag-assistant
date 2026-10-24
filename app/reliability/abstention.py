"""Abstention policy and user-facing abstention messages."""

from __future__ import annotations

from app.graph.entity_extraction import QUESTION_STOPWORDS
from app.models.schemas import QueryAnalysis, RetrievalOutcome, VerificationReport
from app.utils.text import stem, tokenize


def _topic(analysis: QueryAnalysis) -> str:
    if analysis.entities:
        return ", ".join(e.name for e in analysis.entities)
    words = [
        w for w in tokenize(analysis.query, stemmed=False)
        if stem(w) not in QUESTION_STOPWORDS and len(w) > 2 and not w.isdigit()
    ]
    return " ".join(words[:4]) or "this question"


def abstention_message(outcome: RetrievalOutcome | None, analysis: QueryAnalysis, reason: str) -> str:
    topic = _topic(analysis)
    if outcome == RetrievalOutcome.NOT_IN_CORPUS:
        return (
            f"I couldn't verify information about {topic} from the available documents. After retrieval "
            "recovery, no indexed document appears to cover this topic. Please consult the appropriate "
            "policy owner (for example HR, Legal or Information Security) or the authoritative source."
        )
    if outcome == RetrievalOutcome.DOCUMENTS_WITHOUT_ANSWER:
        return (
            f"I couldn't verify an answer. Indexed documents mention {topic}, but none of the passages I "
            "found establishes what you asked, even after graph-based recovery. Please consult the document "
            "owner or rephrase the question."
        )
    if reason == "verification":
        return (
            "I couldn't produce an answer whose statements are all supported by the retrieved documents, "
            "so I am not answering rather than risk an unsupported response. Please review the cited sources "
            "directly or consult the document owner."
        )
    return (
        f"I couldn't find sufficient evidence about {topic} in the indexed documents, even after retrieval "
        "recovery. Please consult the appropriate source."
    )


def should_abstain(evidence_sufficient: bool, report: VerificationReport | None) -> tuple[bool, str]:
    """Abstain when evidence is insufficient or no generated claim survives verification."""
    if not evidence_sufficient:
        return True, "evidence"
    if report is None or not report.supported:
        return True, "verification"
    return False, ""
