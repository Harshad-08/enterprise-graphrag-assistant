"""Score a saved evaluation report with Ragas (resumable).

Generation and Ragas judging are separate steps so a rate-limit stall in judging never
loses the generated answers. Scores are checkpointed per item in
data/evaluation/results/ragas_checkpoint.jsonl; re-running resumes where it stopped.

Usage: python scripts/run_ragas.py [--report data/evaluation/results/latest.json] [--limit N]
"""

from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path

from _bootstrap import ROOT

from app.config import configure_logging, load_settings
from app.evaluation.custom_metrics import ItemRecord
from app.evaluation.ragas_evaluation import run_ragas
from app.evaluation.reporting import write_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=ROOT / "data" / "evaluation" / "results" / "latest.json")
    parser.add_argument("--limit", type=int, default=None, help="Score only the first N items per configuration.")
    parser.add_argument("--checkpoint-only", action="store_true",
                        help="Aggregate already-checkpointed scores without making API calls.")
    parser.add_argument("--referenced-only", action="store_true",
                        help="Only questions with a reference answer (where all four metrics apply).")
    args = parser.parse_args()
    settings = load_settings()
    configure_logging(settings.log_level)

    report = json.loads(args.report.read_text())
    names = {f.name for f in fields(ItemRecord)}
    records = [ItemRecord(**{k: v for k, v in item.items() if k in names}) for item in report.pop("items")]
    if args.referenced_only:
        records = [r for r in records if r.reference]
    if args.limit:
        kept, counts = [], {}
        for r in records:
            counts[r.configuration] = counts.get(r.configuration, 0) + 1
            if counts[r.configuration] <= args.limit:
                kept.append(r)
        records = kept
    report.pop("files", None)
    report["ragas"] = run_ragas(
        records, settings, checkpoint=settings.eval_results_dir / "ragas_checkpoint.jsonl", run_id=report["generated_at"],
        score_missing=not args.checkpoint_only,
    )
    report["files"] = write_report(report, records, settings.eval_results_dir)
    ragas = report["ragas"]
    print(f"Ragas: {ragas['status']}" + (f" ({ragas['reason']})" if ragas.get("reason") else ""))
    for config, metrics in ragas.get("aggregate", {}).items():
        print(f"  {config}: " + ", ".join(
            f"{k}={v['mean'] if v['mean'] is not None else 'n/a'} (n={v['scored_items']}, failed={v['failed_items']})"
            for k, v in metrics.items()))
    for name, m in ragas.get("matched", {}).items():
        print(f"  matched {name}: n={m['items']} " + ", ".join(f"{k}={v}" for k, v in m.items() if k != "items"))
    print("Judges:", ragas.get("judge_models"))
    print("Reports written:", report["files"])


if __name__ == "__main__":
    main()
