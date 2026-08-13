"""Prompt construction for POLARIS table description generation.

Builds the Table 6 description-generation prompt: :func:`build_table_info`
turns a table's metadata (name, column names, context) into a short
``table_info`` block, and :func:`prepare_tbl_desc` wraps that block in a
4-message chat (system text with a JSON schema, a one-shot demo, and the
final request).  The prompt strings are frozen -- do not edit them.
Stdlib-only and safe to import anywhere.
"""

from __future__ import annotations

import json
from typing import Protocol, Sequence, runtime_checkable


@runtime_checkable
class TableMeta(Protocol):
    """Structural type for one row of a dataset's ``metadata.csv``.

    Every table has ``table_id`` and ``column_names``.  The aw/arctic/lter
    datasets additionally have ``table_name`` (``table_context`` is
    ``None``); the ecir/wikitables/wtr datasets have ``table_context``
    instead (``table_name`` is ``None``) -- see data/README.md.
    ``table_context`` may be the parsed JSON object (a dict) or the raw
    JSON string as it appears in ``metadata.csv``.

    A Protocol so any concrete dataclass with this shape (e.g. the data
    loader's metadata record) is accepted via duck typing.
    """

    table_id: str
    table_name: str | None
    column_names: list[str]
    table_context: dict | str | None


def build_table_info(
    meta: TableMeta,
    column_name_expansion: Sequence[str] | str | None = None,
    table_name_expansion: str | None = None,
) -> str:
    """Assemble the ``table_info`` block fed to :func:`prepare_tbl_desc`.

    Emits up to three newline-joined lines, each only when its value is
    non-empty: ``Table Name: ...``, ``Column Names: ...`` (the first 25
    names, comma-joined), and ``Table Context: <START>...<END>`` (the
    context string truncated to 1000 characters).

    Args:
        meta: metadata record; provides the fallback ``table_name``,
            ``column_names``, and ``table_context`` fields.
        column_name_expansion: LLM-expanded column names, a list parallel to
            ``column_names`` (the first 25 are used, like raw names).  A
            plain string is used verbatim as the already-joined column
            line.  When ``None``, falls back to ``meta.column_names``.
        table_name_expansion: LLM-expanded table name.  When ``None``, falls
            back to ``meta.table_name``.

    Returns:
        The newline-joined ``table_info`` block.
    """
    table_name: str | None
    if table_name_expansion is not None:
        table_name = table_name_expansion
    else:
        table_name = getattr(meta, "table_name", None)

    columns: Sequence[str] | str | None
    if column_name_expansion is not None:
        columns = column_name_expansion
    else:
        columns = getattr(meta, "column_names", None)

    context = getattr(meta, "table_context", None)
    if isinstance(context, dict):
        # A dict context is re-serialized (json.dumps, default
        # ensure_ascii=True, matching the escaped style of the released
        # CSVs) before the 1000-char slice; an empty dict counts as absent.
        context = json.dumps(context) if context else None

    table_info: list[str] = []
    if table_name is not None and str(table_name).strip():
        table_info.append(f"Table Name: {table_name}")
    if columns:
        if isinstance(columns, str):
            col_str = columns
        else:
            col_str = ", ".join(list(columns)[:25])
        table_info.append(f"Column Names: {col_str}")
    if context is not None and str(context).strip():
        table_info.append(f"Table Context: <START>{str(context)[:1000]}<END>")
    return "\n".join(table_info)


def prepare_tbl_desc(table_info: str) -> list[dict[str, str]]:
    """Build the Table 6 description-generation chat for one table.

    Returns the 4-message chat (system + one-shot demo user/assistant +
    final user message) ready for ``tokenizer.apply_chat_template``.
    The strings below are frozen -- do not edit them.
    """
    sys = (
        "You are a careful, concise data documentation assistant. "
        "You ONLY use the provided table information, which may include table name, column names and table context, to infer high-level information. "
        "If something is not obvious from the schema, say 'unknown' instead of guessing. "
        "You do NOT need to restate the table name, column names, or table context in the JSON."
        "Your output MUST be valid JSON, wrapped in triple backticks with the language tag json. "
        "Do NOT include any text outside the JSON code fence. "
        "Follow the schema exactly—no extra keys."
    )

    # Instructional schema: shown to the model as text, not used for
    # validation.
    schema = {
        "type": "object",
        "properties": {
            "dataset_overview": {"type": "string"},  # 2–5 sentences max
            "key_themes_or_topics": {"type": "array", "items": {"type": "string"}},  # 5–10 items
            "applications_and_use_cases": {"type": "array", "items": {"type": "string"}},  # 4–8 items
            "concepts_and_synonyms": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "concept": {"type": "string"},
                        "synonyms": {"type": "array", "items": {"type": "string"}}
                    },
                    "required": ["concept", "synonyms"]
                }
            },
            "keywords_and_themes": {"type": "array", "items": {"type": "string"}},  # 6–12 items
            "additional_context": {"type": "array", "items": {"type": "string"}}
        },
        "required": [
            "dataset_overview",
            "key_themes_or_topics", "applications_and_use_cases",
            "concepts_and_synonyms", "keywords_and_themes",
            "additional_context"
        ],
        "additionalProperties": False
    }

    demo_q = (
        "Given the table information below:\nTable Name: insurers\nColumn Names:company_name, year, insurer_type, total_assets, liabilities, premiums_written\n"
        "Produce documentation following the JSON schema in the system message. "
        "Keep it concise and truthful. If unsure, use 'unknown'."
    )
    demo_a = """```json
{
  "dataset_overview": "The dataset captures financial attributes for health insurance organizations, including Health Maintenance Organizations (HMO), Managed Care Health (MCH), and Accident & Health (A&H) providers. It summarizes assets, liabilities, and premiums written by company and year, enabling analyses of financial condition and market behavior across time.",
  "key_themes_or_topics": [
    "Health insurance finance",
    "Insurer types (HMO, MCH, A&H)",
    "Financial statement metrics",
    "Risk and solvency indicators",
    "Premium structures",
    "Temporal trend analysis"
  ],
  "applications_and_use_cases": [
    "Evaluate financial health of insurers",
    "Support actuarial modeling and forecasting",
    "Benchmark performance across insurer types",
    "Policy analysis on insurance markets",
    "Research trends in premiums and liabilities"
  ],
  "concepts_and_synonyms": [
    {"concept": "Health Insurance", "synonyms": ["Medical Insurance", "Healthcare Coverage"]},
    {"concept": "Premiums Written", "synonyms": ["Premium Income", "Gross Written Premium"]},
    {"concept": "Liabilities", "synonyms": ["Financial Obligations", "Reserves"]},
    {"concept": "Assets", "synonyms": ["Total Assets", "Asset Valuation"]}
  ],
  "keywords_and_themes": [
    "health insurance dataset",
    "insurer finance",
    "premiums",
    "assets",
    "liabilities",
    "yearly financials",
    "HMO",
    "MCH",
    "A&H",
    "market analysis"
  ],
  "additional_context": [
    "Values are organized by company and year.",
    "Useful for financial stability and solvency assessment.",
    "Facilitates comparisons among insurer types.",
    "Temporal coverage inferred from the 'year' column only."
  ]
}
```"""

    query = (
        "Using the table information provided below, generate JSON that follows the schema in the system message. "
        "Write 2–5 sentences for 'dataset_overview'. Provide specific, schema-derived items; avoid speculation. "
        "If information is unclear, use 'unknown'. Output JSON ONLY, wrapped in ```json fences.\n\n"
        f"{table_info}"
    )

    schema_text = (
        "JSON_SCHEMA (instructions, not to be echoed verbatim):\n"
        f"{schema}\n\n"
        "Formatting rules:\n"
        "- Return ONLY a JSON object inside a ```json code fence.\n"
        "- No explanations, no headers, no markdown outside the fence.\n"
        "- Maintain key order as in the schema if possible.\n"
        "- For 'concepts_and_synonyms', include 3–6 items; each item must have 'concept' and 'synonyms' (1–4)."
    )

    message = [
        {"role": "system", "content": sys + "\n\n" + schema_text},
        {"role": "user", "content": demo_q},
        {"role": "assistant", "content": demo_a},
        {"role": "user", "content": query}
    ]
    return message
