"""Load local ``.env`` configuration (e.g. ``PYESS_USER_ID``) for development.

Mirrors the lightweight loader used by ``py-ess``'s own test suite: no extra
runtime dependency (e.g. ``python-dotenv``) is required just to read a
handful of ``KEY=value`` lines. Existing environment variables always take
precedence over ``.env`` contents.
"""

from __future__ import annotations

import os
from pathlib import Path

_ENV_FILENAME = ".env"


def load_dotenv(start_dir: Path | None = None) -> None:
    """Populate ``os.environ`` from a ``.env`` file, if one is found.

    Searches ``start_dir`` (default: current working directory) and its
    parents for a ``.env`` file, so the CLI works whether it's invoked from
    the repository root or a subdirectory. Does nothing if no ``.env`` file
    is found.
    """
    directory = start_dir or Path.cwd()
    for candidate in (directory, *directory.parents):
        env_path = candidate / _ENV_FILENAME
        if env_path.exists():
            _apply_env_file(env_path)
            return


def _apply_env_file(env_path: Path) -> None:
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        name = name.strip()
        value = value.strip().strip("\"'")
        if name and value and name not in os.environ:
            os.environ[name] = value
