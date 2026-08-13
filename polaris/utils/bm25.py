"""BM25 / Elasticsearch helpers: client, index schema, indexing, search.

Used by ``polaris.generate_preference_pairs`` and
``evaluation/evaluate.py``.  Needs a running Elasticsearch
(``docker compose up -d``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from elasticsearch import Elasticsearch, helpers

# bm25_schema.json lives at the package root (polaris/)
_RESOURCE_DIR = Path(__file__).resolve().parent.parent


def load_schema() -> dict[str, Any]:
    """Return the BM25 index schema (settings + mappings) shipped with the
    package."""
    return json.loads(
        (_RESOURCE_DIR / "bm25_schema.json").read_text(encoding="utf-8")
    )


def connect_or_die(url: str) -> Elasticsearch:
    """Connect to Elasticsearch, or exit with a hint when the cluster is
    unreachable (instead of a raw ConnectionError traceback)."""
    client = Elasticsearch(url)
    try:
        reachable = bool(client.ping())
    except Exception:  # transport errors vary by client version
        reachable = False
    if not reachable:
        raise SystemExit(
            f"Cannot reach Elasticsearch at {url} - start it with: "
            f"docker compose up -d"
        )
    return client


def create_index(
    client: Elasticsearch, name: str, schema: dict[str, Any] | None = None
) -> None:
    """Create index *name* from *schema* (default: the shipped BM25
    schema), deleting any existing index of that name first."""
    if schema is None:
        schema = load_schema()
    if client.indices.exists(index=name):
        client.indices.delete(index=name)
    client.indices.create(
        index=name,
        settings=schema.get("settings"),
        mappings=schema.get("mappings"),
    )


def index_tables(
    client: Elasticsearch, name: str, docs: Iterable[dict[str, Any]]
) -> int:
    """Bulk-index documents into index *name* and refresh it.

    Each element of *docs* is a plain dict (``table_id``, ``description``,
    ...); ``table_id`` doubles as the document id, so duplicate ids
    overwrite instead of piling up.  Returns the indexed document count.
    """
    success, _errors = helpers.bulk(
        client,
        ({"_index": name, "_id": doc["table_id"], "_source": doc} for doc in docs),
    )
    client.indices.refresh(index=name)
    return int(success)


def search(
    client: Elasticsearch, name: str, query_text: str, size: int = 100
) -> list[str]:
    """BM25 ``match`` on ``description``; returns the ranked ``table_id``
    list.  The default ``size=100`` covers every evaluation cutoff."""
    query = {"match": {"description": {"query": query_text}}}
    response = client.search(index=name, query=query, size=size)
    return [hit["_source"]["table_id"] for hit in response["hits"]["hits"]]
