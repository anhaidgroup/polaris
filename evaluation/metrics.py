"""Retrieval metrics: Recall@K and NDCG@K, computed the way the paper does.

A table counts as relevant when its relevance score is above zero. NDCG uses
the raw (possibly fractional) score as the gain, never rounded.
"""

from __future__ import annotations

from math import log2
from typing import Iterable, Sequence

#: The paper reports metrics at these cutoffs.
DEFAULT_KS: tuple[int, ...] = (5, 10, 20, 50, 100)

def recall_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Recall@k: ``|top-k ∩ relevant| / |relevant|``, rounded to 4 decimals.

    ``relevant`` is the gold set (table_ids with a positive relevance
    score). Returns 0.0 if it is empty.
    """
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    hits = set(retrieved[:k]) & relevant_set
    return round(len(hits) / len(relevant_set), 4)


def dcg_at_k(retrieved: Sequence[str], relevance_map: dict[str, float], k: int) -> float:
    """DCG@k with linear gain ``rel / log2(i + 2)``; unjudged ids gain 0."""
    return sum(
        relevance_map.get(doc_id, 0) / log2(i + 2)
        for i, doc_id in enumerate(retrieved[:k])
    )


def ndcg_at_k(retrieved: Sequence[str], relevance_map: dict[str, float], k: int) -> float:
    """NDCG@k over raw float relevance scores, rounded to 4 decimals.

    ``relevance_map`` is ``{table_id: relevance_score}`` for one query (all
    judged pairs, zeros included). IDCG takes the k highest scores with the
    same linear gain. Returns 0.0 when IDCG is not positive.
    """
    dcg = dcg_at_k(retrieved, relevance_map, k)
    ideal_rels = sorted(relevance_map.values(), reverse=True)
    idcg = sum(rel / log2(i + 2) for i, rel in enumerate(ideal_rels[:k]))
    return round(dcg / idcg, 4) if idcg > 0 else 0.0


def evaluate_run(
    run: dict[str, list[str]],
    dataset,  # Dataset (polaris.load_data)
    ks: Sequence[int] = DEFAULT_KS,
) -> dict:
    """Score a retrieval run against a dataset's qrels.

    ``run`` maps ``query_id`` (str) to the ranked list of retrieved
    table_ids, best first. Queries without any positive gold table are
    skipped; a query missing from ``run`` counts as an empty ranking.

    Returns::

        {
          "per_query": {query_id: {"R@5": ..., "NDCG@5": ..., ...}},
          "mean": {"R@5": ..., "NDCG@5": ..., ...},
          "num_queries": <number of evaluated queries>,
        }

    Per-query metrics are rounded to 4 decimals; means are plain averages
    of those rounded values.
    """
    relevance_maps = dataset.relevance_map
    per_query: dict[str, dict[str, float]] = {}
    for query_id in dataset.queries:
        rmap = relevance_maps.get(query_id, {})
        golds = [tid for tid, score in rmap.items() if score > 0]
        if not golds:
            continue
        # dedupe, first occurrence wins: a repeated id must not earn
        # NDCG gain twice
        retrieved = list(dict.fromkeys(run.get(query_id, [])))
        metrics: dict[str, float] = {}
        for k in ks:
            metrics[f"R@{k}"] = recall_at_k(retrieved, golds, k)
        for k in ks:
            metrics[f"NDCG@{k}"] = ndcg_at_k(retrieved, rmap, k)
        per_query[query_id] = metrics

    mean: dict[str, float] = {}
    if per_query:
        for k in ks:
            for prefix in ("R", "NDCG"):
                key = f"{prefix}@{k}"
                mean[key] = sum(m[key] for m in per_query.values()) / len(per_query)
    return {"per_query": per_query, "mean": mean, "num_queries": len(per_query)}
