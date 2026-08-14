"""Train the description model on training datasets.

Runs the four stages in order: expand_names, generate_candidates,
generate_preference_pairs, dpo.  Each
stage is also runnable on its own.  Datasets whose
expansions_{dataset}.jsonl already exists are not re-expanded.

Run:  python -m polaris.train --datasets arctic ecir lter wikitables wtr
"""

from __future__ import annotations

import argparse
import sys

from . import dpo, expand_names, generate_candidates, generate_preference_pairs
from .utils import line_buffered


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m polaris.train",
        description="Train the description model: expand table/column "
        "names, generate candidate descriptions, build preference pairs, "
        "and train with DPO.",
    )
    parser.add_argument(
        "--datasets", metavar="NAME", nargs="+", required=True,
        help="training datasets under data/",
    )
    parser.add_argument(
        "--es-url",
        default="http://localhost:9200",
        help="Elasticsearch URL (default: http://localhost:9200)",
    )
    args = parser.parse_args(argv)
    line_buffered()
    names = list(dict.fromkeys(args.datasets))

    print("[1/4] expand_names: expanding table and column names")
    code = expand_names.expand_missing(names)
    if code:
        return code
    print("[2/4] generate_candidates: generating candidate descriptions")
    code = generate_candidates.main(["--datasets", *names])
    if code:
        return code
    print("[3/4] generate_preference_pairs: building preference pairs")
    code = generate_preference_pairs.main(
        ["--datasets", *names, "--es-url", args.es_url]
    )
    if code:
        return code
    print("[4/4] dpo: training")
    return dpo.main([])


if __name__ == "__main__":
    sys.exit(main())
