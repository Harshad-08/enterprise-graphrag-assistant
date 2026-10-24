"""Run baseline vs improved evaluation and write JSON/CSV reports to data/evaluation/results/.

Usage: python scripts/run_evaluation.py [--ragas] [--limit N] [--categories a,b] [--seed]
"""

from __future__ import annotations

import argparse
import json

from _bootstrap import make_container

from app.evaluation.runner import run_evaluation
from app.services.document_service import DocumentService


def _fmt(value) -> str:
    return "n/a" if value is None else (f"{value:.3f}" if isinstance(value, float) else str(value))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ragas", action="store_true", help="Also run Ragas metrics (needs GROQ_API_KEY or GEMINI_API_KEY).")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--categories", type=str, default=None)
    parser.add_argument("--seed", action="store_true", help="Seed the synthetic corpus first if the store is empty.")
    args = parser.parse_args()

    container = make_container()
    if args.seed and not container.store.list_documents():
        DocumentService(container).seed_directory(container.settings.synthetic_docs_dir)
    report = run_evaluation(
        container,
        include_ragas=args.ragas,
        limit=args.limit,
        categories=args.categories.split(",") if args.categories else None,
    )
    configs = report["configurations"]
    print(f"\nEnvironment: {json.dumps(report['environment'])}")
    print(f"\n{'metric':36}" + "".join(f"{c:>12}" for c in configs))
    for metric in report["custom_metrics"][configs[0]]:
        print(f"{metric:36}" + "".join(f"{_fmt(report['custom_metrics'][c][metric]):>12}" for c in configs))
    corpus = report["corpus_conflict_metrics"]
    print("\nCorpus-level contradiction detection:", corpus["detection"])
    print("Conflict resolution accuracy:", _fmt(corpus["resolution_accuracy"]))
    print("Compatible pairs flagged as contradictions:", corpus["compatible_pairs_flagged_as_contradiction"])
    ragas = report["ragas"]
    print(f"\nRagas: {ragas['status']}" + (f" ({ragas['reason']})" if ragas.get("reason") else ""))
    if ragas["status"] == "completed":
        for config, metrics in ragas["aggregate"].items():
            print(f"  {config}: " + ", ".join(f"{k}={_fmt(v['mean'])} (n={v['scored_items']})" for k, v in metrics.items()))
    for note in report["notes"]:
        print("Note:", note)
    print("\nReports written:", report.get("files"))


if __name__ == "__main__":
    main()
