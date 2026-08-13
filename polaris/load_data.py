"""Load the Polaris datasets: table metadata, queries, and relevance labels.

    from polaris.load_data import load_dataset
    ds = load_dataset("aw")   # reads data/aw/ relative to the working directory
    ds.tables[0], ds.queries["q1"], ds.qrels[:3]

Datasets live under data/{name}/ as metadata.csv, queries.csv, and
qrels.csv (install them with python data/download.py). query_id is always
a string, relevance scores stay raw floats (never rounded), and wtr
table ids contain slashes.

Run:  python -m polaris.load_data --dataset aw   # sanity-check a local dataset
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

# The six released dataset names.
DATASETS: tuple[str, ...] = ("aw", "arctic", "lter", "ecir", "wikitables", "wtr")

# Some cells hold large JSON blobs; raise the csv limit once.
csv.field_size_limit(10**9)


@dataclass
class TableMeta:
    """One table: id, column headers, optional name and context dict."""

    table_id: str
    column_names: list[str]
    table_name: str | None = None
    table_context: dict = field(default_factory=dict)


@dataclass
class Dataset:
    """One loaded dataset: tables, queries, and relevance judgments."""

    name: str
    tables: list[TableMeta]
    queries: dict[str, str]
    qrels: list[tuple[str, str, float]]  # (query_id, table_id, raw score)

    @property
    def gold_tables(self) -> set[str]:
        """Table ids with any relevance_score > 0."""
        return {tid for _qid, tid, score in self.qrels if score > 0}

    @property
    def relevance_map(self) -> dict[str, dict[str, float]]:
        """{query_id: {table_id: relevance_score}}, zeros included."""
        rmap: dict[str, dict[str, float]] = {}
        for qid, tid, score in self.qrels:
            rmap.setdefault(qid, {})[tid] = score
        return rmap


def load_dataset(name: str, root: str | Path = "data") -> Dataset:
    """Load one dataset from root/{name}/ — a released dataset, or your
    own in the same three-file layout (see data/README.md)."""
    directory = Path(root) / name
    for filename in ("metadata.csv", "queries.csv", "qrels.csv"):
        if not (directory / filename).is_file():
            if name in DATASETS:
                raise FileNotFoundError(
                    f"{directory / filename} not found. Install the dataset "
                    f"with: python data/download.py --dataset {name}"
                )
            raise FileNotFoundError(
                f"{directory / filename} not found. Released datasets: "
                f"{', '.join(DATASETS)}; for your own dataset's layout see "
                f"data/README.md"
            )
    return Dataset(
        name=name,
        tables=load_tables_csv(directory / "metadata.csv"),
        queries=load_queries_csv(directory / "queries.csv"),
        qrels=load_qrels_csv(directory / "qrels.csv"),
    )


def load_dataset_or_exit(name: str, root: str | Path = "data") -> Dataset:
    """load_dataset for command-line use: data problems exit with the
    error message instead of a traceback."""
    try:
        return load_dataset(name, root=root)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc


def load_tables_csv(path: str | Path) -> list[TableMeta]:
    """Parse metadata.csv into TableMeta records.

    Needs table_id and column_names (a JSON list) columns; table_name and
    table_context (a JSON object) are optional. Bad cells raise ValueError
    naming the file and line.
    """
    path = Path(path)
    idx, rows = _read_csv(path, required=("table_id", "column_names"))
    tables: list[TableMeta] = []
    for line_no, row in rows:
        try:
            column_names = _parse_json_list(row[idx["column_names"]])
            context = (
                _parse_json_dict(row[idx["table_context"]])
                if "table_context" in idx
                else {}
            )
        except ValueError as exc:
            raise ValueError(f"{path}: line {line_no}: {exc}") from None
        table_name = row[idx["table_name"]] if "table_name" in idx else ""
        tables.append(
            TableMeta(
                table_id=row[idx["table_id"]],
                column_names=column_names,
                table_name=table_name or None,
                table_context=context,
            )
        )
    _check_unique(path, [t.table_id for t in tables], "table_id")
    return tables


def load_queries_csv(path: str | Path) -> dict[str, str]:
    """Parse queries.csv (query_id,query) into {query_id: query}.

    query_id is a string, even when the file stores bare integers.
    """
    path = Path(path)
    idx, rows = _read_csv(path, required=("query_id", "query"))
    _check_unique(path, [row[idx["query_id"]] for _n, row in rows], "query_id")
    return {row[idx["query_id"]]: row[idx["query"]] for _n, row in rows}


def load_qrels_csv(path: str | Path) -> list[tuple[str, str, float]]:
    """Parse qrels.csv into (query_id, table_id, relevance_score) tuples.

    Scores stay raw floats, never rounded.
    """
    path = Path(path)
    idx, rows = _read_csv(path, required=("query_id", "table_id", "relevance_score"))
    qrels: list[tuple[str, str, float]] = []
    for line_no, row in rows:
        raw = row[idx["relevance_score"]]
        try:
            score = float(raw)
        except ValueError:
            raise ValueError(
                f"{path}: line {line_no}: relevance_score {raw!r} is not a number"
            ) from None
        qrels.append((row[idx["query_id"]], row[idx["table_id"]], score))
    return qrels


def _read_csv(
    path: Path, required: tuple[str, ...]
) -> tuple[dict[str, int], list[tuple[int, list[str]]]]:
    """Read a CSV as ({column: cell index}, [(file line number, row), ...]).

    Checks that the required columns exist and every row has as many
    cells as the header. Blank lines are skipped; cells come back as
    strings (utf-8-sig tolerates a leading BOM).
    """
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            raise ValueError(f"{path}: empty file") from None
        idx = {col: i for i, col in enumerate(header)}
        for col in required:
            if col not in idx:
                raise ValueError(
                    f"{path}: missing required column {col!r} (header: {header})"
                )
        rows: list[tuple[int, list[str]]] = []
        for row in reader:
            if not any(row):
                continue
            if len(row) != len(header):
                raise ValueError(
                    f"{path}: line {reader.line_num} has {len(row)} cells, "
                    f"expected {len(header)}"
                )
            rows.append((reader.line_num, row))
    return idx, rows


def _check_unique(path: Path, values: list[str], column: str) -> None:
    seen: set[str] = set()
    dupes: set[str] = set()
    for v in values:
        if v in seen:
            dupes.add(v)
        seen.add(v)
    if dupes:
        raise ValueError(f"{path}: duplicate {column} values: {sorted(dupes)[:5]}")


def _parse_json_list(text: str) -> list[str]:
    """Parse a JSON-encoded list of column names; empty cell -> []."""
    text = text.strip()
    if not text:
        return []
    value = json.loads(text)
    if not isinstance(value, list):
        raise ValueError(f"column_names did not parse to a list: {text[:80]!r}")
    return [str(c) for c in value]


def _parse_json_dict(text: str) -> dict:
    """Parse a JSON-encoded table_context object; empty cell -> {}."""
    text = text.strip()
    if not text:
        return {}
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError(f"table_context did not parse to a dict: {text[:80]!r}")
    return value


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m polaris.load_data",
        description="Sanity-check locally installed datasets "
        "(install them with data/download.py).",
    )
    parser.add_argument("--dataset", default="all", help="dataset name or 'all'")
    parser.add_argument("--root", default="data", help="datasets root (default: data)")
    args = parser.parse_args(argv)

    names = list(DATASETS) if args.dataset == "all" else [args.dataset]
    failures = 0
    for name in names:
        try:
            dataset = load_dataset(name, root=args.root)
        except (FileNotFoundError, ValueError) as exc:
            print(f"{name}: {exc}")
            failures += 1
            continue
        print(
            f"{dataset.name}: tables={len(dataset.tables)} "
            f"queries={len(dataset.queries)} qrels={len(dataset.qrels)} "
            f"gold_tables={len(dataset.gold_tables)}"
        )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
