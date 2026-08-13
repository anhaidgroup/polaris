"""Minimal ``.env`` credential loader (stdlib only, no python-dotenv).

Commands that need credentials (``expand_names``,
``generate_candidates``, ``dpo``) call load_env at
startup so tokens can live in a local ``.env`` file (copy
``.env.example`` and fill it in; ``.env`` is gitignored). Values are
applied with ``os.environ.setdefault``, so variables already exported in
the real environment always win.
"""

from __future__ import annotations

import os
from pathlib import Path


def load_env(path: str | Path = ".env") -> None:
    """Load ``KEY=VALUE`` lines from *path* into ``os.environ``.

    A missing file is a no-op. Blank lines, ``#`` comments, and lines
    without ``=`` are skipped; surrounding quotes on values are stripped.
    Existing environment variables are never overridden.
    """
    path = Path(path)
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _sep, value = line.partition("=")
        key = key.strip().removeprefix("export ").strip()
        value = value.strip().strip("'\"")
        if key:
            os.environ.setdefault(key, value)
