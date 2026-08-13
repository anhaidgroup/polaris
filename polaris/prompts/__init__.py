"""The paper's frozen prompts (Tables 4, 5, and 6) and their parsers.

One module per prompt:

- ``polaris.prompts.column_expansion``       column name expansion (Table 4)
- ``polaris.prompts.table_expansion``        table name expansion (Table 5)
- ``polaris.prompts.description_generation`` description generation (Table 6)

The embedded paper prompt builders (``column_expansion``,
``table_expansion``) are byte-for-byte copies of the paper's prompt code —
do not edit them, trailing spaces included. The public names are
re-exported here so callers can simply ``from .prompts import ...``.

The ``prompt_examples/`` folder shows one real rendered prompt of each
type, built from an aw table.
"""

from .column_expansion import (
    COLUMNS_PER_PROMPT,
    batch_columns,
    build_column_prompt,
    column_expansion,
    parse_final_answer,
)
from .description_generation import build_table_info, prepare_tbl_desc
from .table_expansion import build_table_prompt, table_expansion

__all__ = [
    "COLUMNS_PER_PROMPT",
    "batch_columns",
    "build_column_prompt",
    "column_expansion",
    "parse_final_answer",
    "build_table_info",
    "prepare_tbl_desc",
    "build_table_prompt",
    "table_expansion",
]
