"""Evaluate generated table descriptions with BM25 keyword search.

Evaluates table retrieval the way the paper does: each table becomes one
Elasticsearch document carrying its description, the dataset's queries
are run as BM25 keyword searches, and the rankings are scored with
NDCG@K and Recall@K (K = 5, 10, 20, 50, 100).  Needs a running
Elasticsearch (``docker compose up -d``).

Run:  python evaluation/evaluate.py --descriptions descriptions_aw.jsonl --dataset aw
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# Standalone script: make `metrics` and `import polaris` importable no
# matter where it is run from.
_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE), str(_HERE.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from metrics import DEFAULT_KS, evaluate_run  # noqa: E402


def run_evaluation(args: argparse.Namespace) -> int:
    """Index the descriptions, run every query, and print the scores."""
    from polaris.load_data import load_dataset_or_exit
    from polaris.utils.bm25 import connect_or_die, search

    dataset = load_dataset_or_exit(args.dataset)
    if not Path(args.descriptions).is_file():
        raise SystemExit(
            f"{args.descriptions} not found - generate it first: python -m "
            f"polaris.generate_descriptions --datasets {args.dataset}"
        )
    client = connect_or_die(args.es_url)
    index_name = re.sub(r"[^a-z0-9_-]+", "_", args.dataset.lower())
    _index_descriptions(client, args.descriptions, index_name)
    ks = list(DEFAULT_KS)
    run = {
        qid: search(client, index_name, text, size=max(ks))
        for qid, text in dataset.queries.items()
    }
    report = evaluate_run(run, dataset, ks=ks)

    print(f"dataset={dataset.name}  queries={report['num_queries']}")
    print(f"{'K':>5}  {'Recall@K':>10}  {'NDCG@K':>10}")
    for k in ks:
        recall = report["mean"].get(f"R@{k}", 0.0)
        ndcg = report["mean"].get(f"NDCG@{k}", 0.0)
        print(f"{k:>5}  {recall:>10.4f}  {ndcg:>10.4f}")
    return 0


def _index_descriptions(client, path: str, index: str) -> None:
    """(Re)build one Elasticsearch index from a descriptions jsonl."""
    from polaris.utils.bm25 import create_index, index_tables

    docs = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "table_id" not in row:
                raise SystemExit(f"{path}:{line_no}: missing table_id")
            # candidates/adapter are bookkeeping, not search text
            doc = {k: v for k, v in row.items() if k not in ("candidates", "adapter")}
            if "description" not in doc and row.get("candidates"):
                doc["description"] = row["candidates"][0]
            docs.append(doc)
    create_index(client, index)
    n = index_tables(client, index, docs)
    print(f"Indexed {n}/{len(docs)} documents into '{index}'")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python evaluation/evaluate.py",
        description="The paper's evaluation: BM25 keyword search over the "
        "generated descriptions, scored with NDCG@K and Recall@K.",
    )
    parser.add_argument(
        "--descriptions", metavar="FILE", required=True,
        help="descriptions jsonl from polaris.generate_descriptions",
    )
    parser.add_argument(
        "--dataset", metavar="NAME", required=True,
        help="the dataset to evaluate on (queries and relevance labels "
        "are read from data/{NAME}/)",
    )
    parser.add_argument(
        "--es-url",
        default="http://localhost:9200",
        help="Elasticsearch URL (default: http://localhost:9200)",
    )
    args = parser.parse_args(argv)
    return run_evaluation(args)


if __name__ == "__main__":
    sys.exit(main())
