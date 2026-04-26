"""Pytest fixtures for the Digital Human suite.

Why a temp DB per session
-------------------------
The default `memory.db` carries developer state from manual smoke runs. We
want tests to be deterministic — a fresh DB at session start, populated by
seed fixtures, torn down at teardown.

How
---
We point `main.DB_PATH` (read by `app.db._db_path()`) at a tmp file BEFORE
importing anything that uses it. After import, every `db()` context goes to
the temp file. No subprocess, no fork — same process, just a redirected path.
"""
from __future__ import annotations
import os
import sys
import tempfile
from pathlib import Path

import pytest

# Project root must be on sys.path so `import main` works regardless of cwd.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def temp_db_path():
    """Session-scoped path to a fresh sqlite file."""
    fd, path = tempfile.mkstemp(prefix="dh_test_", suffix=".db")
    os.close(fd)
    # Remove the empty file so SQLite creates a fresh one with our schema
    # rather than treating the zero-byte stub as a corrupt DB.
    try: os.remove(path)
    except FileNotFoundError: pass
    yield path
    # Best-effort cleanup. Windows may hold WAL locks for a moment after close.
    for ext in ("", "-journal", "-wal", "-shm"):
        try: os.remove(path + ext)
        except FileNotFoundError: pass
        except PermissionError: pass


@pytest.fixture(scope="session")
def app(temp_db_path):
    """Initialized FastAPI app with DB redirected to a temp file.
    `app.db._db_path()` reads `main.DB_PATH` lazily, so swapping it here
    redirects every subsequent `db()` context manager."""
    import main as _main
    _main.DB_PATH = temp_db_path
    _main.init_db()
    return _main.app


@pytest.fixture
def client(app):
    """TestClient with a localhost client tuple so `_is_remote_request`
    treats requests as trusted (otherwise localhost-only endpoints 403)."""
    from starlette.testclient import TestClient
    with TestClient(app, client=("127.0.0.1", 12345)) as c:
        yield c


@pytest.fixture
def app_token():
    """Read the project's app token (auto-generated on first start)."""
    import main as _main
    return open(_main.TOKEN_PATH, encoding="utf-8").read().strip()


@pytest.fixture
def headers(app_token):
    return {"X-App-Token": app_token, "X-User-Id": "master"}
