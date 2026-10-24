"""Calibrate evidence-sufficiency thresholds on the labelled evaluation set.

For every question two evidence states are collected: the initial hybrid
retrieval and the evidence after the maximum number of recovery rounds. Each
state is labelled *sufficient* iff the question is answerable AND the evidence
contains at least one labelled relevant document. A grid search then picks the
(sufficiency_threshold, min_chunk_relevance, min_concept_coverage) combination
with the best F1 for predicting that label, and leave-one-out cross-validation
estimates how well the choice generalises (the dataset is small, so expect noise).

Writes data/evaluation/calibration.json; config.py loads its "recommended" values
unless the corresponding environment variables are set.

Usage: python scripts/calibrate_thresholds.py [--seed]
"""

from __future__ import annotations

import argparse
import itertools
import json
from datetime import datetime, timezone

import numpy as np
from _bootstrap import make_container

from app.evaluation.datasets import load_eval_items
from app.graph.entity_extraction import analyze_query
from app.retrieval.evidence_assessment import assess_evidence
from app.services.document_service import DocumentService

THRESHOLDS = [round(x, 2) for x in np.arange(0.30, 0.91, 0.05)]
MIN_RELEVANCE = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
MIN_CONCEPT = [0.34, 0.5, 0.67]


def _f1(pred: np.ndarray, gold: np.ndarray) -> float:
    tp = float(np.sum(pred & gold))
    fp = float(np.sum(pred & ~gold))
    fn = float(np.sum(~pred & gold))
    return 0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn)


def _balanced_accuracy(pred: np.ndarray, gold: np.ndarray) -> float:
    pos, neg = gold, ~gold
    tpr = float(np.sum(pred & pos) / max(np.sum(pos), 1))
    tnr = float(np.sum(~pred & neg) / max(np.sum(neg), 1))
    return (tpr + tnr) / 2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", action="store_true")
    args = parser.parse_args()
    # Force every recovery round to run so the post-recovery evidence state is collected.
    container = make_container()
    if args.seed and not container.store.list_documents():
        DocumentService(container).seed_directory(container.settings.synthetic_docs_dir)
    base_settings = container.settings
    forcing = base_settings.model_copy(update={"sufficiency_threshold": 2.0})
    container.orchestrator.settings = forcing

    samples = []  # (analysis, evidence, claims, equivalents, label, description)
    for item in load_eval_items(base_settings.eval_dataset_path):
        result = container.orchestrator.retrieve(item.question)
        analysis = analyze_query(item.question, container.gazetteer)
        seeds = container.graph.link_entities(analysis)
        equivalents = container.graph.equivalents(seeds)
        # Re-run plain hybrid retrieval to get scored objects for the initial state
        # (RetrievalResult keeps only the initial chunk IDs).
        initial = container.hybrid.rerank_pool(
            item.question, container.hybrid.candidates(item.question)[: base_settings.rerank_candidates], base_settings.final_top_k
        )
        for state, evidence in (("initial", initial), ("recovered", result.evidence)):
            docs = {e.chunk.document_id for e in evidence}
            label = (not item.should_abstain) and bool(docs & set(item.relevant_doc_ids))
            claims = container.store.claims_for_chunks([e.chunk_id for e in evidence])
            samples.append((analysis, evidence, claims, equivalents, label, f"{item.id}:{state}"))
    container.orchestrator.settings = base_settings

    gold = np.array([s[4] for s in samples])
    grid = list(itertools.product(THRESHOLDS, MIN_RELEVANCE, MIN_CONCEPT))
    predictions = np.zeros((len(grid), len(samples)), dtype=bool)
    for g, (threshold, min_rel, min_concept) in enumerate(grid):
        settings = base_settings.model_copy(
            update={"sufficiency_threshold": threshold, "min_chunk_relevance": min_rel, "min_concept_coverage": min_concept}
        )
        for s, (analysis, evidence, claims, equivalents, _, _) in enumerate(samples):
            predictions[g, s] = assess_evidence(analysis, evidence, claims, settings, container.gazetteer, equivalents).sufficient

    def score(g: int, mask: np.ndarray) -> tuple[float, float]:
        return _f1(predictions[g, mask], gold[mask]), _balanced_accuracy(predictions[g, mask], gold[mask])

    everything = np.ones(len(samples), dtype=bool)

    def rank_key(g: int, mask: np.ndarray) -> tuple:
        # Ties are common on a small dataset; prefer the most conservative parameters,
        # because wrongly declaring evidence sufficient is costlier than abstaining.
        return (*score(g, mask), grid[g][1], grid[g][0], grid[g][2])

    ranked = sorted(range(len(grid)), key=lambda g: rank_key(g, everything), reverse=True)
    best = ranked[0]

    # Leave-one-out: choose parameters without sample i, then predict sample i.
    loo_correct = 0
    for i in range(len(samples)):
        mask = everything.copy()
        mask[i] = False
        g_i = max(range(len(grid)), key=lambda g: rank_key(g, mask))
        loo_correct += predictions[g_i, i] == gold[i]

    f1, bacc = score(best, everything)
    default_index = grid.index((0.55, 0.25, 0.5)) if (0.55, 0.25, 0.5) in grid else None
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "samples": len(samples),
        "positive_samples": int(gold.sum()),
        "label_definition": "answerable AND evidence contains >=1 labelled relevant document",
        "recommended": {
            "sufficiency_threshold": grid[best][0],
            "min_chunk_relevance": grid[best][1],
            "min_concept_coverage": grid[best][2],
        },
        "in_sample": {"f1": round(f1, 4), "balanced_accuracy": round(bacc, 4)},
        "leave_one_out_accuracy": round(loo_correct / len(samples), 4),
        "default_parameters_in_sample": (
            dict(zip(("f1", "balanced_accuracy"), [round(v, 4) for v in score(default_index, everything)]))
            if default_index is not None else None
        ),
        "top_5": [
            {"params": dict(zip(("sufficiency_threshold", "min_chunk_relevance", "min_concept_coverage"), grid[g])),
             "f1": round(score(g, everything)[0], 4), "balanced_accuracy": round(score(g, everything)[1], 4)}
            for g in ranked[:5]
        ],
        "misclassified_with_recommended": [
            samples[s][5] for s in range(len(samples)) if predictions[best, s] != gold[s]
        ],
        "caveat": "Small synthetic dataset; thresholds may not transfer to a different corpus. Re-run after changing documents or models.",
    }
    base_settings.calibration_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
