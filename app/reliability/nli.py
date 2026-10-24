"""Natural-language-inference model wrapper (cross-encoder NLI, e.g. nli-deberta-v3-xsmall).

Used for (a) semantic contradiction checks between claims whose properties could
not be structured and (b) claim-level grounding verification of answers.
Small NLI models are imperfect: they struggle with long premises, numbers and
domain jargon, which is why verification also applies explicit number checks.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class NLIScores:
    entailment: float
    contradiction: float
    neutral: float


class NLIModel:
    def __init__(self, model_name: str, model=None) -> None:
        self.name = model_name
        if model is None:
            from sentence_transformers import CrossEncoder

            model = CrossEncoder(model_name)
        self._model = model
        id2label = getattr(getattr(getattr(model, "model", None), "config", None), "id2label", None) or {
            0: "contradiction", 1: "entailment", 2: "neutral"
        }
        self._labels = {int(k): str(v).lower() for k, v in id2label.items()}

    def predict(self, pairs: list[tuple[str, str]]) -> list[NLIScores]:
        """pairs are (premise, hypothesis)."""
        if not pairs:
            return []
        probabilities = self._model.predict(pairs, apply_softmax=True, show_progress_bar=False)
        results = []
        for row in probabilities:
            scores = {self._labels[i]: float(p) for i, p in enumerate(row)}
            results.append(
                NLIScores(
                    entailment=scores.get("entailment", 0.0),
                    contradiction=scores.get("contradiction", 0.0),
                    neutral=scores.get("neutral", 0.0),
                )
            )
        return results


def build_nli(model_name: str) -> NLIModel | None:
    try:
        return NLIModel(model_name)
    except Exception as exc:  # download/load failure
        logger.warning("NLI model %s unavailable (%s); falling back to lexical verification.", model_name, exc)
        return None
