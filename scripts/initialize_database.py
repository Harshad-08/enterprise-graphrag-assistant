"""Create Neo4j constraints, indexes and the vector index (idempotent).

Usage: python scripts/initialize_database.py [--reset]
"""

from __future__ import annotations

import argparse

from _bootstrap import make_container

from app.graph.schema import schema_statements


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reset", action="store_true", help="Delete ALL nodes before creating the schema.")
    args = parser.parse_args()
    container = make_container()
    store = container.store
    if not store.verify_connectivity():
        raise SystemExit(f"Cannot connect to {store.backend_name}. Is Neo4j running (docker compose up -d neo4j)?")
    if args.reset:
        store.clear()
        print("All nodes deleted.")
    store.setup_schema(container.embedder.dimension)
    for statement in schema_statements(container.embedder.dimension):
        print("OK:", statement.split(" IF NOT EXISTS")[0])
    print("Stats:", store.stats())


if __name__ == "__main__":
    main()
