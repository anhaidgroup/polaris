"""Generate descriptions for datasets with the trained model.

Runs expand_names (skipped for datasets whose expansions_{dataset}.jsonl
already exists), then generates one description per table with the
trained adapter. Writes descriptions_{dataset}.jsonl per dataset.

Run:  python -m polaris.generate_descriptions --datasets aw
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .expand_names import expand_missing
from .generate_candidates import run_generation
from .utils import line_buffered


def main(argv=None) -> int:
    from .utils.env import load_env

    load_env()
    parser = argparse.ArgumentParser(
        prog="python -m polaris.generate_descriptions",
        description="Generate a description for each table of the named "
        "datasets with the trained model.",
    )
    parser.add_argument(
        "--datasets", metavar="NAME", nargs="+", required=True,
        help="datasets under data/; writes descriptions_{NAME}.jsonl each",
    )
    parser.add_argument(
        "--adapter", default="out/adapter",
        help="LoRA adapter from train (default: out/adapter)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=8,
        help="tables per GPU batch (lower it on GPU out-of-memory)",
    )
    args = parser.parse_args(argv)
    if args.adapter == "out/adapter" and not Path(args.adapter).exists():
        raise SystemExit(
            "out/adapter not found - train the model first: "
            "python -m polaris.train --datasets <training datasets>"
        )

    line_buffered()
    print("[1/2] expand_names: expanding table and column names")
    code = expand_missing(list(dict.fromkeys(args.datasets)))
    if code:
        return code
    print("[2/2] generating descriptions")
    return run_generation(
        args.datasets, 1, args.adapter, args.batch_size, "descriptions"
    )


if __name__ == "__main__":
    sys.exit(main())
