"""Write evaluation reports (JSON with full detail, CSV for comparison and per-item results)."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.evaluation.custom_metrics import ItemRecord


def comparison_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    """One row per metric with a column per configuration."""
    configs = list(report["custom_metrics"].keys())
    metric_names: list[str] = []
    for config in configs:
        for name in report["custom_metrics"][config]:
            if name not in metric_names:
                metric_names.append(name)
    rows = []
    for name in metric_names:
        row = {"metric": name, "source": "custom"}
        for config in configs:
            row[config] = report["custom_metrics"][config].get(name)
        rows.append(row)
    ragas = report.get("ragas", {})
    if ragas.get("status") in ("completed", "partial"):
        for name in ("context_recall", "context_precision", "faithfulness", "answer_correctness"):
            matched = ragas.get("matched", {}).get(name, {})
            row = {"metric": f"ragas_{name}",
                   "source": f"ragas matched n={matched.get('items', 0)} ({ragas.get('judge_models', {}).get(name, 'judge')})"}
            for config in configs:
                row[config] = matched.get(config)
            rows.append(row)
    return rows


def write_report(report: dict[str, Any], records: list[ItemRecord], output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    paths = {
        "json": output_dir / f"evaluation_{stamp}.json",
        "comparison_csv": output_dir / f"comparison_{stamp}.csv",
        "items_csv": output_dir / f"items_{stamp}.csv",
    }
    full = {**report, "items": [asdict(r) for r in records]}
    paths["json"].write_text(json.dumps(full, indent=2, default=str))
    rows = comparison_rows(report)
    if rows:
        with paths["comparison_csv"].open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    item_fields = [
        "configuration", "item_id", "category", "abstained", "should_abstain", "retrieval_outcome",
        "recovery_triggered", "retrieved_doc_ids", "relevant_doc_ids", "citations_valid", "citations_total",
        "claims_supported", "claims_total", "expected_conflict", "conflicts", "latency_ms", "error", "answer",
    ]
    with paths["items_csv"].open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=item_fields)
        writer.writeheader()
        for record in records:
            data = asdict(record)
            writer.writerow({k: json.dumps(data[k]) if isinstance(data[k], (list, dict)) else data[k] for k in item_fields})
    # Stable name for the UI / README to pick up the latest run.
    (output_dir / "latest.json").write_text(json.dumps(full, indent=2, default=str))
    return {k: str(v) for k, v in paths.items()}
