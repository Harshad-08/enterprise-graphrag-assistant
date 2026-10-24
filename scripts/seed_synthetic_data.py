"""Index the synthetic enterprise corpus (data/synthetic_documents).

Usage: python scripts/seed_synthetic_data.py [--reset]
"""

from __future__ import annotations

import argparse

from _bootstrap import make_container

from app.services.document_service import DocumentService


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reset", action="store_true", help="Clear the graph before seeding.")
    args = parser.parse_args()
    container = make_container()
    if not container.store.verify_connectivity():
        raise SystemExit("Graph store is not reachable.")
    if args.reset:
        container.store.clear()
        container.store.setup_schema(container.embedder.dimension)
    results = DocumentService(container).seed_directory(container.settings.synthetic_docs_dir)
    for r in results:
        print(f"{r.document_id:15} chunks={r.chunk_count:3} entities={r.entity_count:3} "
              f"relations={r.relation_count:3} claims={r.claim_count:3} conflicts={r.conflicts_detected}")
        for warning in r.warnings:
            if "not indexed yet" not in warning:  # cross-references are re-checked below
                print(f"{'':15} warning: {warning}")
    unresolved = container.ingestion.relink_documents()
    print("Unresolved cross-document references:", unresolved or "none")
    print("Graph stats:", container.store.stats())


if __name__ == "__main__":
    main()
