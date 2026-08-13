"""Table name expansion prompt (the paper's Table 5) and reply parsing.

The embedded function is a byte-for-byte copy of the paper's original
prompt builder — do not edit it, trailing spaces included.
"""

from __future__ import annotations

import re
from typing import Sequence

from .column_expansion import _parse_rules

# --- begin unchanged paper code: table name expansion prompt (Table 5) ---
def table_expansion(row,TABLE_NAME, COLUMN_NAME):
    system_prompt = "You are a helpful assistant, answer the question from the user and reply in the same format."
    
    c_names = " | ".join(row[COLUMN_NAME][:25])
    t_name = row[TABLE_NAME]

    query_2 = f"""
Your task is to expand abbreviated table names into full-form phrases. 
You should return the tokens of each table name and their associated expansions.
First reason step by step and then return your final answer. 
Follow the guidelines below when you expand:
1. Expand all abbreviations in the table names. 
2. Expand chemical symbols and units of measure to their full names.
3. Do not expand or mutate numbers. 
4. Do not add extra words or explanations.
5. Maintain the original order of tokens in the expansion.
6. The tokens should be as concise and simple as possible.
7. Only provide 1 expansion for each token, even if you are uncertain, output the most possible one.
Do not put ambiguous or need context after your expansion.
8. If the token is not abbreviated, the expansion should be itself, do not paraphrase or add other words around it.


[Question]
Given the following table information:
Table name: Prchs_info
Column names: c_name | pCd | dt

What is the full form of the table name Prchs_info? Think step by step
[Answer]
### Reasoning

1. **Understand the context from the table name:**

   * The table name is **"Prchs_info"**.
   * The suffix `_info` clearly suggests **"Information"**.
   * The prefix **"Prchs"** is a shortened form, likely representing **"Purchase"**.

2. **Tokenize the table name:**

   * Tokens: `Prchs`, `info`
   * `Prchs` → common abbreviation for **Purchase**.
   * `info` → abbreviation for **Information**.

### Final Answer
Prchs_info: Prchs → Purchase, info → Information


[Question]
Given the following table information:
Table name: Emp_info
Column names: emp_id | emp_nm | dept_cd | doj | sal_amt | mgr_id

What is the full form of the table name Emp_info? Think step by step
[Answer]
### Reasoning
1. **Understand the context from the table name:**

   * The table name is **"Emp_info"**.
   * Based on standard naming conventions, it appears to contain employee-related data.
   * The suffix `_info` suggests **Information**.
   * The prefix `Emp` is a common abbreviation for **Employee**.

2. **Tokenize the table name:**

   * Tokens: `Emp`, `info`
   * `Emp` → abbreviation for **Employee**
   * `info` → abbreviation for **Information**

### Final Answer
Emp_info: Emp → Employee, info → Information

[Question]
Given the following table information:
Table name: AirQ_data
Column names: stn_id | pm25_lvl | pm10_lvl | no2_lvl | co_lvl | temp_C | rec_dt

What is the full form of the table name AirQ_data? Think step by step
[Answer]
### Reasoning
1. **Understand the context from the table name:**

   * The table name is **"AirQ_data"**.
   * The suffix `_data` suggests the content is a dataset.
   * The prefix `AirQ` appears to be a common abbreviation for **Air Quality**, especially in environmental datasets.

2. **Tokenize the table name:**

   * Tokens: `AirQ`, `data`
   * `AirQ` → likely stands for **Air Quality**
   * `data` → stands for **Data**

### Final Answer
AirQ_data: AirQ → Air Quality, data → Data


[Question]
Given the following table information:
Table name: Pop_census
Column names: hh_id | res_cty | age_grp | gender_cd | edu_lvl | occ_cd

What is the full form of the table name Pop_census? Think step by step
[Answer]
### Reasoning
1. **Understand the context from the table name:**

   * The table name is **"Pop_census"**.
   * The suffix `census` refers to a systematic population count.
   * The prefix `Pop` is a standard abbreviation for **Population**.

2. **Tokenize the table name:**

   * Tokens: `Pop`, `census`
   * `Pop` → abbreviation for **Population**
   * `census` → not abbreviated, remains **Census**

### Final Answer
Pop_census: Pop → Population, census → Census


[Question]
Given the following table information:
Table name: {t_name}
Column names: {c_names}

What is the full form of the table name {t_name}? Think step by step
[Answer]
"""
    messages = [
            {"role": "developer", "content": system_prompt},
            {"role": "user", "content": query_2}
    ]
    return messages
# --- end unchanged paper code ---

def build_table_prompt(
    table_name: str, expanded_columns: Sequence[str]
) -> list[dict]:
    """Build the table name expansion chat (paper Table 5) for one table.

    ``expanded_columns`` should be the already-expanded column names; at
    most the first 25 are used as context.  Returns the same two-message
    chat shape as :func:`build_column_prompt`.
    """
    row = {"table_name": table_name, "column_expansion": list(expanded_columns)}
    return table_expansion(row, "table_name", "column_expansion")


def _parse_table_final_answer(text: str) -> str:
    """Parse a table name expansion completion into the expanded name.

    Only the first line of the ``### Final Answer`` block is parsed, as
    one ``name: rules`` line (models may append chatter after it).
    Returns ``""`` when nothing parses.
    """
    if "### Final Answer" not in text:
        return ""
    clean = text.split("### Final Answer")[1].strip().split("\n")[0]
    rules = _parse_rules(clean.split(": ")[-1])
    return " ".join(rule[-1] for rule in rules)
