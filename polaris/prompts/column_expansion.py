"""Column name expansion prompt (the paper's Table 4) and reply parsing.

The embedded function is a byte-for-byte copy of the paper's original
prompt builder — do not edit it, trailing spaces included.
"""

from __future__ import annotations

import re
from typing import Sequence

#: Column names sent per prompt.
COLUMNS_PER_PROMPT = 10


# --- begin unchanged paper code: column name expansion prompt (Table 4) ---
def column_expansion(row,TABLE_NAME, COLUMN_NAME):
    system_prompt = "You are a helpful assistant, answer the question from the user and reply in the same format."
    
    # new_query = f"Given a list of tables and their topics in the same dataset:\n{df[df['topic_group']==topic_group][[TABLE_NAME, 'topics']].drop_duplicates().values.tolist()}\n\n"
    c_names = " | ".join(row[COLUMN_NAME])
    if TABLE_NAME is not None:
      t_name = row[TABLE_NAME]
      info = f"named {t_name}, {c_names}"
    else:
      info = f", {c_names}"
    

    query_2 = f"""
Your task is to expand abbreviated column names into full-form phrases. 
You should return the tokens of each column name and their associated expansions.
First reason step by step and then return your final answer. 
Follow the guidelines below when you expand:
1. Expand all abbreviations in the column names. 
2. Expand chemical symbols and units of measure to their full names.
3. Do not expand or mutate numbers. 
4. Do not add extra words or explanations.
5. Maintain the original order of tokens in the expansion.
6. The tokens should be as concise and simple as possible.
7. Only provide 1 expansion for each token, even if you are uncertain, output the most possible one.
Do not put ambiguous or need context after your expansion.
8. If the token is not abbreviated, the expansion should be itself, do not paraphrase or add other words around it.


[Question]
As abbreviations of column names from a table named Prchs_info, c_name | pCd | dt stand for? Think step by step
[Answer]
### Reasoning 
1. **Identify the table name context:**  
   - The table is named **"Prchs_info"**, which likely stands for **"Purchase Information"**.  
   - This suggests that the column names are related to purchase details.  

2. **Break down each column name into tokens and infer meanings:**  
   - **c_name** → `c`, `name`  
     - `c` is commonly used for **Customer** in business contexts.  
     - `name` clearly refers to **Name**.  
     - Together, `c_name` most likely means **Customer Name**.  

   - **pCd** → `p`, `Cd`  
     - `p` is frequently used for **Product** in sales or inventory tables.  
     - `Cd` is a common abbreviation for **Code**.  
     - Together, `pCd` most likely means **Product Code**.  

   - **dt** → `dt`  
     - `dt` is a standard abbreviation for **Date**.  
     - In the context of a purchase table, it likely refers to **Purchase Date**.  

### Final Answer  
c_name: c → Customer, name → Name
pCd: p → Product, Cd → Code
dt: dt → Date


[Question]
As abbreviations of column names from a table named Emp_info, emp_id | emp_nm | dept_cd | doj | sal_amt | mgr_id stand for
[Answer]
### Reasoning  
1. **Identify the table name context:**  
   - The table is named **"Emp_info"**, which likely stands for **"Employee Information"**.  
   - This suggests that the column names are related to employee records.  

2. **Break down each column name into tokens and infer meanings:**  
   - **emp_id** → `emp`, `id`  
     - `emp` is commonly used for **Employee**.  
     - `id` stands for **Identifier**.  
     - Together, `emp_id` means **Employee Identifier**.  

   - **emp_nm** → `emp`, `nm`  
     - `emp` stands for **Employee**.  
     - `nm` is a common abbreviation for **Name**.  
     - Together, `emp_nm` means **Employee Name**.  

   - **dept_cd** → `dept`, `cd`  
     - `dept` is short for **Department**.  
     - `cd` stands for **Code**.  
     - Together, `dept_cd` means **Department Code**.  

   - **doj** → `doj`  
     - `doj` is commonly used for **Date of Joining**.  

   - **sal_amt** → `sal`, `amt`  
     - `sal` stands for **Salary**.  
     - `amt` stands for **Amount**.  
     - Together, `sal_amt` means **Salary Amount**.  

   - **mgr_id** → `mgr`, `id`  
     - `mgr` is short for **Manager**.  
     - `id` stands for **Identifier**.  
     - Together, `mgr_id` means **Manager Identifier**.  

### Final Answer
emp_id: emp → Employee, id → Identifier
emp_nm: emp → Employee, nm → Name
dept_cd: dept → Department, cd → Code
doj: doj → Date of Joining
sal_amt: sal → Salary, amt → Amount
mgr_id: mgr → Manager, id → Identifier

[Question]
As abbreviations of column names from a table named AirQ_data, stn_id | pm25_lvl | pm10_lvl | no2_lvl | co_lvl | temp_C | rec_dt stand for
[Answer]
### Reasoning  
1. **Identify the table name context:**  
   - The table is named **"AirQ_data"**, which likely stands for **"Air Quality Data"**.  
   - This suggests that the column names are related to air quality measurements.  

2. **Break down each column name into tokens and infer meanings:**  
   - **stn_id** → `stn`, `id`  
     - `stn` stands for **Station**.  
     - `id` stands for **Identifier**.  
     - Together, `stn_id` means **Station Identifier**.  

   - **pm25_lvl** → `pm25`, `lvl`  
     - `pm25` stands for **Particulate Matter 2.5**.  
     - `lvl` stands for **Level**.  
     - Together, `pm25_lvl` means **Particulate Matter 2.5 Level**.  

   - **pm10_lvl** → `pm10`, `lvl`  
     - `pm10` stands for **Particulate Matter 10**.  
     - `lvl` stands for **Level**.  
     - Together, `pm10_lvl` means **Particulate Matter 10 Level**.  

   - **no2_lvl** → `no2`, `lvl`  
     - `no2` stands for **Nitrogen Dioxide**.  
     - `lvl` stands for **Level**.  
     - Together, `no2_lvl` means **Nitrogen Dioxide Level**.  

   - **co_lvl** → `co`, `lvl`  
     - `co` stands for **Carbon Monoxide**.  
     - `lvl` stands for **Level**.  
     - Together, `co_lvl` means **Carbon Monoxide Level**.  

   - **temp_C** → `temp`, `C`  
     - `temp` stands for **Temperature**.  
     - `C` stands for **Celsius**.  
     - Together, `temp_C` means **Temperature Celsius**.  

   - **rec_dt** → `rec`, `dt`  
     - `rec` stands for **Recorded**.  
     - `dt` stands for **Date**.  
     - Together, `rec_dt` means **Recorded Date**.  

### Final Answer  
stn_id: stn → Station, id → Identifier
pm25_lvl: pm25 → Particulate Matter 2.5, lvl → Level
pm10_lvl: pm10 → Particulate Matter 10, lvl →Level
no2_lvl: no2 → Nitrogen Dioxide, lvl → Level
co_lvl: co → Carbon Monoxide, lvl → Level
temp_C: temp → Temperature, C → Celsius
rec_dt: rec → Recorded, dt → Date  


[Question]
As abbreviations of column names from a table named Pop_census, hh_id | res_cty | age_grp | gender_cd | edu_lvl | occ_cd stand for
[Answer]
### Reasoning  
1. **Identify the table name context:**  
   - The table is named **"Pop_census"**, which likely stands for **"Population Census"**.  
   - This suggests that the column names are related to demographic and census data.  

2. **Break down each column name into tokens and infer meanings:**  
   - **hh_id** → `hh`, `id`  
     - `hh` stands for **Household**.  
     - `id` stands for **Identifier**.  
     - Together, `hh_id` means **Household Identifier**.  

   - **res_cty** → `res`, `cty`  
     - `res` stands for **Residential**.  
     - `cty` stands for **County**.  
     - Together, `res_cty` means **Residential County**.  

   - **age_grp** → `age`, `grp`  
     - `age` stands for **Age**.  
     - `grp` stands for **Group**.  
     - Together, `age_grp` means **Age Group**.  

   - **gender_cd** → `gender`, `cd`  
     - `gender` stands for **Gender**.  
     - `cd` stands for **Code**.  
     - Together, `gender_cd` means **Gender Code**.  

   - **edu_lvl** → `edu`, `lvl`  
     - `edu` stands for **Education**.  
     - `lvl` stands for **Level**.  
     - Together, `edu_lvl` means **Education Level**.  

   - **occ_cd** → `occ`, `cd`  
     - `occ` stands for **Occupation**.  
     - `cd` stands for **Code**.  
     - Together, `occ_cd` means **Occupation Code**.  

### Final Answer  
hh_id: hh → Household, id → Identifier
res_cty: res → Residential, cty → County
age_grp: age → Age, grp → Group
gender_cd: gender → Gender, cd → Code
edu_lvl: edu → Education, lvl → Level
occ_cd: occ → Occupation, cd → Code  

[Question]
As abbreviations of column names from a table {info} stand for
[Answer]
"""
    messages = [
            {"role": "developer", "content": system_prompt},
            {"role": "user", "content": query_2}
    ]
    return messages
# --- end unchanged paper code ---

def build_column_prompt(
    table_name: str | None, columns: Sequence[str]
) -> list[dict]:
    """Build the column-expansion chat (paper Table 4) for one batch of columns.

    Pass ``table_name=None`` for tables that have no name; the prompt then
    omits it.  Returns a two-message chat whose first entry is the system
    text and last entry is the user question.
    """
    row = {"table_name": table_name, "column_names": list(columns)}
    key = None if table_name is None else "table_name"
    return column_expansion(row, key, "column_names")


def _is_numeric_token(s: str) -> bool:
    """True when ``s`` contains digits but no letters.

    Such "tokens" are numbers the model mangled; prompt guideline 3 says
    never to mutate numbers, so their expansion is forced back to the
    token itself.
    """
    contains_digit = bool(re.search(r"\d", s))
    no_letters = not bool(re.search(r"[a-zA-Z]", s))
    return contains_digit and no_letters


_RULE_PATTERN = re.compile(
    r"([\w/^%():#/\.\-\s]+?)\s*→\s*([^→]+?)(?=(?:,\s*[\w/^%():#/\.\-\s]+?\s*→|$))"
)


def _parse_rules(rule_text: str) -> list[list[str]]:
    """Parse an ``a → b, c → d`` rule list into ``[[token, expansion], ...]``.

    Both sides are stripped; numeric-only tokens keep themselves.
    """
    rules = [
        [key.strip(), value.strip()]
        for key, value in _RULE_PATTERN.findall(rule_text)
    ]
    rules = [r for r in rules if r[0]]  # a rule needs a non-empty token
    return [[k, k] if _is_numeric_token(k) else [k, v] for k, v in rules]


def parse_final_answer(text: str) -> dict[str, str]:
    """Parse a completion's ``### Final Answer`` block into ``{name: expansion}``.

    Each non-blank line is expected to look like ``name: tok → Expansion,
    tok → Expansion``; the expansion values are space-joined into the
    full-form name.  Text without a ``### Final Answer`` section yields
    ``{}``, and lines with no ``name:`` prefix or no parsable ``→`` rules
    are skipped — callers fall back to the raw name for anything missing.
    """
    if "### Final Answer" not in text:
        return {}
    lines = [
        line
        for line in text.split("### Final Answer")[1].split("\n")
        if line.strip() != ""
    ]
    mapping: dict[str, str] = {}
    for line in lines:
        parts = line.split(": ")
        if len(parts) < 2:
            continue  # no "name: " prefix to key the expansion by
        rules = _parse_rules(parts[-1])
        if not rules:
            continue
        name = ": ".join(parts[:-1]).strip()
        mapping[name] = " ".join(rule[-1] for rule in rules)
    return mapping


def batch_columns(columns: Sequence[str], k: int = COLUMNS_PER_PROMPT) -> list[list[str]]:
    """Split ``columns`` into consecutive batches of at most ``k``."""
    lst = list(columns)
    return [lst[i:i + k] for i in range(0, len(lst), k)]
