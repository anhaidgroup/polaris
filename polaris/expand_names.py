"""Expand abbreviated table and column names into full-form phrases with an LLM.

Cryptic names (``BusEntId``, ``DTPh``) make poor input for description
generation, so this first pipeline stage rewrites them (paper Sec 3.1).
Column names go to the model in batches of 10, and the ``### Final
Answer`` section of each completion is parsed into token → expansion
rules.  Tables that have a name get a second pass that expands the table
name, using the expanded column names as context.  The model is Llama 4
Maverick on AWS Bedrock; any
``(messages: list[dict]) -> str`` callable works as a backend.

Run:  python -m polaris.expand_names --datasets arctic ecir lter wikitables wtr
      python -m polaris.expand_names --datasets aw
"""

from __future__ import annotations

import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

#: Table and column name expansion model.
BEDROCK_MODEL_ID = "us.meta.llama4-maverick-17b-instruct-v1:0"

#: Decoding settings for ``converse()``.
BEDROCK_INFERENCE_CONFIG: dict[str, Any] = {
    "maxTokens": 4196,  # 4196, not 4096 - do not "fix"
    "temperature": 0.2,
    "topP": 0.9,
}

#: Environment variable holding the Bedrock API bearer token; boto3
#: picks it up when creating the client.
BEDROCK_TOKEN_ENV = "AWS_BEARER_TOKEN_BEDROCK"

#: Bedrock failures worth retrying: the service is busy, the model timed
#: out, or the connection dropped - none of them a bad request.  Matched
#: by class name so botocore need not be imported here.
BEDROCK_TRANSIENT_ERRORS = frozenset({
    "ThrottlingException",
    "ModelTimeoutException",
    "ServiceUnavailableException",
    "InternalServerException",
    "ReadTimeoutError",
    "ConnectTimeoutError",
    "EndpointConnectionError",
})

#: Attempts per Bedrock call, with jittered exponential backoff between
#: them (jitter keeps the worker threads from retrying in lockstep).
BEDROCK_MAX_ATTEMPTS = 6

#: A backend takes a chat (list of {"role", "content"} dicts) and
#: returns the raw model completion text.
ExpansionBackend = Callable[[list[dict]], str]


# Paper prompt builders.
#
# The two functions below are byte-for-byte copies of the paper's prompt
# builders (Tables 4 and 5) — do not edit them, trailing spaces included.
# Each takes a mapping ``row`` plus the keys to
# read the table name and column names from; the ``build_*_prompt``
# wrappers underneath adapt them to plain arguments.

from .utils import line_buffered, progress
from .prompts.column_expansion import (
    COLUMNS_PER_PROMPT,
    batch_columns,
    build_column_prompt,
    parse_final_answer,
)
from .prompts.description_generation import TableMeta
from .prompts.table_expansion import _parse_table_final_answer, build_table_prompt


#: Parallel LLM calls during expansion.
EXPANSION_WORKERS = 8


def expand_tables(
    tables: Sequence[TableMeta],
    backend: ExpansionBackend,
    expand_table_names: bool = True,
    workers: int = EXPANSION_WORKERS,
) -> Iterator[dict]:
    """Expand every table's column names (and, optionally, table name).

    Tables are expanded in parallel (workers threads); results keep the
    input order.  Per table, columns are prompted in batches of 10 and the parsed
    expansions are joined in original column order; a column whose
    expansion is missing or empty keeps its raw name.  Tables that have a
    ``table_name`` then get a table name expansion pass that uses the expanded
    column names as context; an unparsable completion falls back to the
    raw table name.

    Args:
        tables: metadata records (``table_id``, ``column_names``,
            optional ``table_name``).
        backend: any ``(messages) -> str`` callable, e.g.
            :class:`BedrockBackend`.
        expand_table_names: set ``False`` to skip the table name expansion pass.

    Yields:
        One dict per table, in input order, so callers can write results
        to disk as they arrive.  Table *k* is yielded once it and every
        earlier table have finished::

            {
                "table_id": str,
                "column_name_expansion": list[str],  # parallel to column_names
                "table_name_expansion": str,  # absent when no table_name
            }

    """
    def expand_one(meta: TableMeta) -> dict:
        table_name = getattr(meta, "table_name", None)
        columns = list(getattr(meta, "column_names", None) or [])

        mapping: dict[str, str] = {}
        for batch in batch_columns(columns, COLUMNS_PER_PROMPT):
            completion = backend(build_column_prompt(table_name, batch))
            mapping.update(parse_final_answer(completion))
        expanded = [mapping.get(col) or col for col in columns]

        table_exp: str | None = None
        if expand_table_names and table_name is not None and str(table_name).strip():
            completion = backend(build_table_prompt(table_name, expanded))
            table_exp = _parse_table_final_answer(completion) or table_name

        result = {"table_id": meta.table_id, "column_name_expansion": expanded}
        if table_exp is not None:
            result["table_name_expansion"] = table_exp
        return result

    with ThreadPoolExecutor(max_workers=workers) as pool:
        yield from progress(
            pool.map(expand_one, tables), total=len(tables), desc="expanding names"
        )


def read_expansions(path: str | Path) -> dict[str, dict[str, str | None]]:
    """Read an expansions jsonl (this module's output).

    Returns {table_id: {"column_name_expansion": ..., "table_name_expansion": ...}}.
    Empty values become None so absent expansions fall back to the raw
    metadata.
    """
    expansions: dict[str, dict[str, str | None]] = {}
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "table_id" not in row:
                raise ValueError(f"{path}:{line_no}: missing table_id")
            if "column_name_expansion" not in row:
                raise ValueError(
                    f"{path}:{line_no}: missing column_name_expansion - "
                    f"old-format expansions file? re-run expand_names"
                )
            expansions[str(row["table_id"])] = {
                "column_name_expansion": row.get("column_name_expansion") or None,
                "table_name_expansion": row.get("table_name_expansion") or None,
            }
    return expansions


def resolve_expansions(dataset: str) -> dict[str, dict[str, str | None]]:
    """Read expansions_{dataset}.jsonl from the working directory, or
    return {} when it does not exist (raw names are used then)."""
    path = Path(f"expansions_{dataset}.jsonl")
    if not path.is_file():
        return {}
    print(f"using {path}")
    try:
        return read_expansions(path)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


class BedrockBackend:
    """Amazon Bedrock ``converse()`` backend.

    The chat's first message is sent as the system text and its last
    message as the single user turn.  Authentication uses a Bedrock API
    bearer token exported as ``AWS_BEARER_TOKEN_BEDROCK``.  API errors
    propagate as exceptions.
    """

    def __init__(
        self,
        region: str | None = None,  # default: AWS_REGION, else us-east-1
        model_id: str = BEDROCK_MODEL_ID,
        inference_config: dict[str, Any] | None = None,
    ) -> None:
        try:
            import boto3  # lazy: keep `import polaris` light
        except ImportError as exc:
            raise ImportError(
                "boto3 is required for the Bedrock expansion backend but is "
                "not installed. Install it with: pip install boto3"
            ) from exc
        if not os.environ.get(BEDROCK_TOKEN_ENV):
            raise RuntimeError(
                f"{BEDROCK_TOKEN_ENV} is not set. Name expansion "
                f"authenticates to Amazon Bedrock with an API bearer token; "
                f"create one in the AWS console and export "
                f"{BEDROCK_TOKEN_ENV}=<token> before running."
            )
        self.model_id = model_id
        self.inference_config = dict(inference_config or BEDROCK_INFERENCE_CONFIG)
        region = region or os.environ.get("AWS_REGION", "us-east-1")
        self.client = boto3.client("bedrock-runtime", region_name=region)

    def __call__(self, messages: list[dict]) -> str:
        system_text = messages[0]["content"]
        user_text = messages[-1]["content"]
        for attempt in range(BEDROCK_MAX_ATTEMPTS):
            try:
                response = self.client.converse(
                    modelId=self.model_id,
                    system=[{"text": system_text}],
                    messages=[{"role": "user", "content": [{"text": user_text}]}],
                    inferenceConfig=self.inference_config,
                )
                break
            except Exception as exc:
                # Throttling and model timeouts are routine on long runs.
                if (type(exc).__name__ not in BEDROCK_TRANSIENT_ERRORS
                        or attempt == BEDROCK_MAX_ATTEMPTS - 1):
                    raise
                time.sleep(2 ** attempt * (0.5 + random.random()))
        return response["output"]["message"]["content"][0]["text"]


def expanded_ids(path: str | Path) -> set[str]:
    """The table_ids already expanded in an expansions jsonl.

    Empty when *path* is absent or written by an older version, so the
    caller rewrites it from scratch.  A truncated final line - the tail of
    a run that was killed mid-write - ends the scan and is overwritten.
    """
    ids: set[str] = set()
    if not Path(path).is_file():
        return ids
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                break  # truncated tail; resume from here
            if "column_name_expansion" not in row:
                return set()  # old format: expand the dataset again
            ids.add(str(row["table_id"]))
    return ids


def expand_missing(names: list[str]) -> int:
    """Run the expansion for datasets whose expansions are missing or
    incomplete (a partial file resumes rather than counting as done)."""
    from .load_data import load_dataset_or_exit

    todo = []
    for name in names:
        tables = load_dataset_or_exit(name).tables
        if len(expanded_ids(f"expansions_{name}.jsonl")) >= len(tables):
            print(f"using existing expansions_{name}.jsonl")
        else:
            todo.append(name)
    if not todo:
        return 0
    return main(["--datasets", *todo])


def main(argv=None) -> int:
    import argparse

    from .utils.env import load_env

    load_env()
    line_buffered()
    parser = argparse.ArgumentParser(
        prog="python -m polaris.expand_names",
        description="Expand abbreviated table/column names with an LLM "
        "(paper Sec 3.1; needs AWS_BEARER_TOKEN_BEDROCK)",
    )
    parser.add_argument(
        "--datasets", metavar="NAME", nargs="+", required=True,
        help="datasets under data/ to expand; writes expansions_{NAME}.jsonl each",
    )
    args = parser.parse_args(argv)

    from .load_data import load_dataset_or_exit

    # Resolve all inputs before creating the Bedrock backend.
    jobs = [
        (load_dataset_or_exit(name).tables, f"expansions_{name}.jsonl")
        for name in dict.fromkeys(args.datasets)
    ]
    try:
        backend = BedrockBackend()
    except (ImportError, RuntimeError) as exc:
        raise SystemExit(str(exc)) from None
    for tables, output in jobs:
        out = Path(output)
        # Rows are appended as they finish, so an interrupted run keeps
        # what it expanded and a rerun skips those tables.
        done = expanded_ids(out)
        todo = [t for t in tables if t.table_id not in done]
        if done:
            print(f"{out}: {len(done)} already expanded, {len(todo)} to go")
        n = len(done)
        with open(out, "a" if done else "w", encoding="utf-8") as f:
            try:
                for row in expand_tables(todo, backend):
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    f.flush()
                    n += 1
            except Exception as exc:
                raise SystemExit(
                    f"Bedrock call failed ({type(exc).__name__}: {exc}); the "
                    f"{n} rows already in {out} are kept, rerunning resumes "
                    f"after them"
                ) from None
        print(f"{out}: {n} rows")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
