# data/

The six Polaris datasets are downloaded from the Hugging Face Hub into this directory:

```bash
python data/download.py                  # all six (~14 MB)
python data/download.py --dataset aw     # just one
python data/download.py --version v2     # adds full table contents (~4.8 GB)
```

Each dataset (`anhaidgroup/polaris-{name}-{v1,v2}` on the Hub) has three files:

- `metadata.csv` — table id, name, column names, and context
- `queries.csv` — the queries
- `qrels.csv` — relevance labels

Per-dataset licenses are inside each Hub repo.

## Using your own dataset

Create a folder `data/{name}/` with the same three files:

| File | Columns |
|---|---|
| `metadata.csv` | `table_id`, `table_name`, `column_names` (a JSON list, e.g. `["c_name", "pCd"]`), and optionally `table_context` |
| `queries.csv` | `query_id`, `query` |
| `qrels.csv` | `query_id`, `table_id`, `relevance_score` (above zero means relevant) |

Pipeline commands then accept the folder name via `--datasets`, exactly like the
released datasets (see the main README's "Using your own dataset").

If you write your own loading code instead, note three things
(`polaris/load_data.py` already handles all of them):

- Query ids must be read as strings.
- `ecir` relevance scores are fractional and must never be rounded; a table is
  relevant when its score is above zero.
- `wtr` table ids contain slashes.
