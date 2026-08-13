"""Polaris: retrieval-optimized table descriptions.

Polaris generates natural-language table descriptions optimized for BM25
keyword search, trained directly from retrieval feedback via DPO.  From the
paper "Polaris: Learning to Generate Table Descriptions from Retrieval
Feedback" (Cai, Phan, Doan).

Two commands run the whole pipeline:

- ``polaris.train``                 train the description model
- ``polaris.generate_descriptions``  generate descriptions with the trained model

Pipeline stages (the paper's Algorithm 1; each runnable as
``python -m polaris.<module>``):

- ``polaris.expand_names``               expand abbreviated table/column names (Sec 3.1)
- ``polaris.generate_candidates``        generate candidate descriptions (GPU)
- ``polaris.generate_preference_pairs``  build DPO pairs via BM25 candidate ranking (Sec 3.3)
- ``polaris.dpo``                        DPO-finetune the description model (GPU; Sec 3.4)

Supporting modules:

- ``polaris.load_data``    dataset loading
- ``polaris.utils.bm25``   Elasticsearch/BM25 helpers
- ``polaris.prompts``      the paper's prompts (Tables 4, 5, 6)
- ``polaris.utils.env``    ``.env`` credential loading

Evaluation lives outside the package in ``evaluation/evaluate.py``; the
dataset installer is ``data/download.py``.
"""

__version__ = "0.1.0"
