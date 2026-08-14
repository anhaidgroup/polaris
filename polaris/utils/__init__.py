"""Supporting utilities: Elasticsearch helpers, .env loading, console output."""

from __future__ import annotations

import sys
from typing import Iterable


def line_buffered() -> None:
    """Flush prints per line, so logs fill in real time when stdout is
    redirected to a file."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(line_buffering=True)
        except AttributeError:
            pass


def progress(iterable: Iterable, total: int, desc: str) -> Iterable:
    """tqdm progress bar if tqdm is installed, otherwise the iterable unchanged."""
    try:
        from tqdm import tqdm
    except ImportError:
        return iterable
    return tqdm(iterable, total=total, desc=desc)
