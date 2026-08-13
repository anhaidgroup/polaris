"""Install the Polaris datasets from the Hugging Face Hub.

Standalone installer, deliberately separate from the ``polaris`` package:
it has no polaris imports and needs only ``huggingface_hub`` (installed by
``requirements.txt``), so the datasets can be fetched before anything
else is set up. Files land in ``{root}/{name}/`` in the HF release layout
(``metadata.csv``/``queries.csv``/``qrels.csv``, plus ``tuples.zip`` in
v2) that ``polaris.load_data`` loads.

Run:  python data/download.py                      # all six datasets (v1, ~14 MB)
      python data/download.py --dataset aw --version v2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

#: Canonical dataset names (HF naming).
DATASETS: tuple[str, ...] = ("aw", "arctic", "lter", "ecir", "wikitables", "wtr")


#: One Hugging Face dataset repo per dataset and version (v1 = CSVs only;
#: v2 adds tuples.zip with the table contents).
HF_REPO_PATTERN = "anhaidgroup/polaris-{name}-{version}"



def download(
    name_or_all: str,
    root: str | Path = "data",
    version: str = "v1",
) -> Path:
    """Download dataset files from the Hugging Face Hub into ``root/{name}/``.

    ``name_or_all`` is a dataset name from ``DATASETS``
    or ``"all"``. ``version`` selects ``"v1"`` (CSVs only, ~14 MB total) or
    ``"v2"`` (CSVs plus ``tuples.zip`` with full table contents; ~4.8 GB
    zipped for all six — see data/README.md).
    """
    from huggingface_hub import hf_hub_download

    if version not in ("v1", "v2"):
        raise ValueError(f"version must be 'v1' or 'v2', got {version!r}")
    if name_or_all != "all" and name_or_all not in DATASETS:
        raise SystemExit(f"Unknown dataset {name_or_all!r}; expected one of {DATASETS} or 'all'")
    names = list(DATASETS) if name_or_all == "all" else [name_or_all]
    files = ["metadata.csv", "queries.csv", "qrels.csv"]
    if version == "v2":
        files.append("tuples.zip")
    root = Path(root)
    for name in names:
        repo_id = HF_REPO_PATTERN.format(name=name, version=version)
        dest = root / name
        dest.mkdir(parents=True, exist_ok=True)
        for filename in files:
            hf_hub_download(
                repo_id=repo_id, filename=filename, repo_type="dataset",
                local_dir=dest,
            )
        print(f"{name}: downloaded {', '.join(files)} from {repo_id} -> {dest}")
    return root


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python data/download.py",
        description="Download the Polaris dataset(s) from the "
        "Hugging Face Hub.",
    )
    parser.add_argument(
        "--dataset", default="all", help="dataset name or 'all' (default)"
    )
    parser.add_argument(
        "--root", default="data", help="destination root (default: data)"
    )
    parser.add_argument(
        "--version", default="v1", choices=("v1", "v2"),
        help="v1: CSVs only (~14 MB total); v2: + table contents (~4.8 GB)",
    )
    args = parser.parse_args(argv)
    download(args.dataset, root=args.root, version=args.version)
    return 0


if __name__ == "__main__":
    sys.exit(main())
