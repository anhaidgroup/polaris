"""Generate the DPO preference pairs by ranking each table's candidate
descriptions with BM25 retrieval (the paper's Sec 3.3 and Algorithm 1).

For every training table, the candidate descriptions from
``polaris.generate_candidates`` are ranked by how well BM25 retrieves
the table with each candidate (needs a running Elasticsearch); the
best-ranked description becomes ``chosen`` and the worst-ranked becomes
``rejected``, forming the preference pairs that ``polaris.dpo``
consumes.  Tables with fewer than two ranked candidates are dropped.

Run:  python -m polaris.generate_preference_pairs --datasets arctic ecir lter wikitables wtr
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # only type names; no runtime import of heavy deps
    from elasticsearch import Elasticsearch

    from .load_data import Dataset, TableMeta

_MARK = "<rank>"

#: Parallel per-table ranking cycles against Elasticsearch.
RANKING_WORKERS = 4


def rank_candidates(
    client: Elasticsearch,
    dataset: Dataset,
    candidates: dict[str, list[str]],
    index_prefix: str = "polaris_rank",
) -> dict[str, list[tuple[int, float]]]:
    """Rank each table's candidate descriptions by BM25 retrievability.

    For each table with >= 2 candidates and >= 1 relevant query, builds a
    temporary index per table (in parallel, RANKING_WORKERS at a time)
    holding one copy of the table per candidate (under
    marker ids, so all candidates compete in the same corpus) plus every
    other table once, then scores each candidate by its mean BM25 score
    over the table's relevant queries.

    candidates is {table_id: [candidate descriptions]}. Returns
    {table_id: [(candidate_index, score), ...]} sorted best-first;
    never-retrieved candidates stay at 0.0, at the bottom.
    """
    from .utils import progress
    from .utils.bm25 import create_index, index_tables

    # relevant queries per table: qrels rows with relevance_score > 0
    relevant_queries: dict[str, list[str]] = {}
    for query_id, table_id, score in dataset.qrels:
        if score > 0:
            query_text = dataset.queries.get(query_id)
            if query_text is None:
                raise ValueError(
                    f"qrels row references unknown query_id {query_id!r} "
                    f"(not present in the queries file)"
                )
            relevant_queries.setdefault(table_id, []).append(query_text)

    metas: dict[str, TableMeta] = {t.table_id: t for t in dataset.tables}
    # ES index names must be lowercase alphanumerics
    safe_name = re.sub(r"[^a-z0-9_-]+", "_", dataset.name.lower())

    def rank_one(item: tuple[int, tuple[str, list[str]]]):
        seq, (table_id, cands) = item
        queries = relevant_queries.get(table_id)
        if len(cands) < 2 or not queries:
            return None
        meta = metas.get(table_id)
        if meta is None:
            raise ValueError(
                f"Unknown table_id {table_id!r} in candidates: not present "
                f"in dataset {dataset.name!r} metadata. The candidates file "
                f"and the dataset probably refer to different corpora — "
                f"check that both come from the same tables."
            )

        index_name = f"{index_prefix}_{safe_name}_{seq}"
        create_index(client, index_name)  # deletes any stale leftover first

        copy_ids = [copy_id(i) for i in range(len(cands))]

        def _corpus_docs():
            # K copies of the target table, one per candidate description.
            for i, desc in enumerate(cands):
                doc = _table_doc(meta, desc)
                doc["table_id"] = copy_ids[i]
                yield doc
            # Every other table once, with its first candidate description.
            for other in dataset.tables:
                if other.table_id == table_id:
                    continue
                other_cands = candidates.get(other.table_id)
                desc = other_cands[0] if other_cands else ""
                yield _table_doc(other, desc)

        # Every candidate enters the ranking; ones never retrieved by any
        # relevant query keep score 0 and sort to the bottom (= rejected).
        total_score: dict[str, float] = {sid: 0.0 for sid in copy_ids}
        try:
            index_tables(client, index_name, _corpus_docs())
            for query_text in queries:
                # score only the K copies; the rest of the corpus is there
                # to shape the collection statistics
                wrapped = {
                    "bool": {
                        "must": {"match": {"description": {"query": query_text}}},
                        "filter": [{"terms": {"table_id": copy_ids}}],
                    }
                }
                response = client.search(
                    index=index_name, query=wrapped, size=len(copy_ids)
                )
                for hit in response["hits"]["hits"]:
                    sid = hit["_source"]["table_id"]
                    total_score[sid] += float(hit["_score"])
        finally:
            client.indices.delete(index=index_name)

        n_queries = len(queries)
        return table_id, sorted(
            (
                (copy_index(sid), score / n_queries)
                for sid, score in total_score.items()
            ),
            key=lambda pair: pair[1],
            reverse=True,
        )

    # Parallel over tables; pool.map keeps the input order.
    rankings: dict[str, list[tuple[int, float]]] = {}
    items = list(enumerate(candidates.items()))
    with ThreadPoolExecutor(max_workers=RANKING_WORKERS) as pool:
        for result in progress(
            pool.map(rank_one, items), total=len(items),
            desc=f"ranking {dataset.name}",
        ):
            if result is not None:
                rankings[result[0]] = result[1]
    return rankings


def build_pairs(
    prompts: dict[str, Any],
    rankings: dict[str, list[tuple[int, float]]],
    candidates: dict[str, list[str]],
) -> list[dict[str, Any]]:
    """Build DPO preference pairs from the rankings.

    Returns one {"prompt": ..., "chosen": best description, "rejected":
    worst description} per ranked table. Tables whose best and worst
    scores are tied are dropped (no preference signal).
    """
    pairs: list[dict[str, Any]] = []
    degenerate = 0
    for table_id, ranking in rankings.items():
        if not ranking or len(ranking) < 2:
            continue
        if table_id not in prompts or table_id not in candidates:
            continue
        if ranking[0][1] == ranking[-1][1]:
            # tied best/worst (nothing retrieved, or identical candidate
            # texts): no preference signal
            degenerate += 1
            continue
        cands = candidates[table_id]
        best_idx = int(ranking[0][0])
        worst_idx = int(ranking[-1][0])
        pairs.append(
            {
                "prompt": prompts[table_id],
                "chosen": cands[best_idx],
                "rejected": cands[worst_idx],
            }
        )
    if degenerate:
        print(f"Dropped {degenerate} degenerate pair(s) with tied best/worst scores")
    return pairs


def copy_id(i: int) -> str:
    """table_id used for the i-th copy of a table ('<rank>i<rank>')."""
    return f"{_MARK}{i}{_MARK}"


def copy_index(cid: str) -> int:
    """Recover the candidate index from a copy id."""
    return int(cid[len(_MARK) : -len(_MARK)])


def _table_doc(meta: TableMeta, description: str) -> dict:
    """One corpus document: the table id and its candidate description."""
    return {"table_id": meta.table_id, "description": description}


def to_jsonl(pairs: list[dict[str, Any]], path: str | Path) -> None:
    """Write pairs to *path* as UTF-8 JSON Lines, one pair per line."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")


def _read_candidates(path: str | Path) -> dict[str, list[str]]:
    """Read a candidates jsonl (from ``polaris.generate_candidates``)
    into ``{table_id: [candidate descriptions]}``, stripping the leading
    ```` ```json ```` fence from each candidate."""
    candidates: dict[str, list[str]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            cands = row.get("candidates") or (
                [row["description"]] if row.get("description") else []
            )
            candidates[str(row["table_id"])] = [
                c.split("```json")[-1] for c in cands
            ]
    return candidates


def _dataset_pairs(
    client: Elasticsearch,
    dataset: Dataset,
    candidates: dict[str, list[str]],
    expansions: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Rank one dataset's candidates and build its preference pairs."""
    from .prompts import build_table_info, prepare_tbl_desc

    rankings = rank_candidates(client, dataset, candidates)
    prompts = {}
    for meta in dataset.tables:
        if meta.table_id not in rankings:
            continue
        exp = expansions.get(meta.table_id, {})
        prompts[meta.table_id] = prepare_tbl_desc(
            build_table_info(
                meta,
                column_name_expansion=exp.get("column_name_expansion"),
                table_name_expansion=exp.get("table_name_expansion"),
            )
        )
    return build_pairs(prompts, rankings, candidates)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m polaris.generate_preference_pairs",
        description="Generate DPO preference pairs by ranking candidate "
        "descriptions with BM25 retrieval (paper Sec 3.3; needs a running "
        "Elasticsearch).",
    )
    parser.add_argument(
        "--datasets", metavar="NAME", nargs="+", required=True,
        help="datasets to rank and pair, reading candidates_{NAME}.jsonl "
        "and expansions_{NAME}.jsonl; writes all pairs to pairs.jsonl",
    )
    parser.add_argument(
        "--es-url",
        default="http://localhost:9200",
        help="Elasticsearch URL (default: http://localhost:9200)",
    )
    args = parser.parse_args(argv)

    from .utils import line_buffered

    line_buffered()

    from .expand_names import resolve_expansions
    from .load_data import load_dataset_or_exit
    from .utils.bm25 import connect_or_die

    names = list(dict.fromkeys(args.datasets))
    for name in names:
        if not Path(f"candidates_{name}.jsonl").is_file():
            raise SystemExit(
                f"candidates_{name}.jsonl not found - generate it first: "
                f"python -m polaris.generate_candidates --datasets {name}"
            )
    client = connect_or_die(args.es_url)
    pairs: list[dict[str, Any]] = []
    for name in names:
        candidates = _read_candidates(f"candidates_{name}.jsonl")
        expansions = resolve_expansions(name)
        dataset = load_dataset_or_exit(name)
        pairs.extend(_dataset_pairs(client, dataset, candidates, expansions))
    to_jsonl(pairs, "pairs.jsonl")
    print(f"Wrote {len(pairs)} preference pairs to pairs.jsonl")
    return 0


if __name__ == "__main__":
    sys.exit(main())
