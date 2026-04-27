"""SQLite schema, migrations, and db() context manager.

Single source of truth for the disk layout. Everything stateful Maid knows
about her users — short-term memory, long-term memory, cognitive log,
traits, pending topics, rp_scene, daily summaries, tasks, notes.

Public API:
  - DB_PATH           -> path of the SQLite file
  - db()              -> contextmanager yielding a sqlite3.Connection
  - init_db()         -> create schema + run column-add migrations
"""

from __future__ import annotations
import os
import sqlite3
from contextlib import contextmanager
from typing import Iterator

# Lazy-loaded to avoid circular imports
_DB_PATH: str = None


def _get_db_path() -> str:
    """Get database path (lazy initialization)."""
    global _DB_PATH
    if _DB_PATH is None:
        # Default to current directory if not set via set_db_path()
        _DB_PATH = os.path.join(os.getcwd(), "memory.db")
    return _DB_PATH


def set_db_path(path: str) -> None:
    """Set database path (call before init_db())."""
    global _DB_PATH
    _DB_PATH = path


@contextmanager
def db() -> Iterator[sqlite3.Connection]:
    """
    Context manager for SQLite connections.
    
    Uses WAL mode for better concurrency. Transactions are automatically
    committed on exit or rolled back on exception.
    
    Yields:
        sqlite3.Connection with autocommit disabled
    """
    conn = sqlite3.connect(_get_db_path(), timeout=30.0)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=30000")
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# Schema definition (v10.0 consolidated)
_SCHEMA = """
-- Short-term memory (chat history)
CREATE TABLE IF NOT EXISTS memory(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL DEFAULT 'master',
    role TEXT NOT NULL,
    text TEXT NOT NULL,
    cog TEXT NOT NULL DEFAULT '{}',
    kind TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'done',
    ts INTEGER NOT NULL DEFAULT(unixepoch())
);
CREATE INDEX IF NOT EXISTS idx_mem ON memory(user_id, ts DESC);

-- User state (traits, relationship metrics)
CREATE TABLE IF NOT EXISTS user_state(
    user_id TEXT PRIMARY KEY,
    mood REAL NOT NULL DEFAULT 0.5,
    trust REAL NOT NULL DEFAULT 0.5,
    fear REAL NOT NULL DEFAULT 0.4,
    attachment REAL NOT NULL DEFAULT 0.3,
    msg_count INTEGER NOT NULL DEFAULT 0,
    total_msg_count INTEGER NOT NULL DEFAULT 0,
    last_activity_ts INTEGER NOT NULL DEFAULT 0,
    curiosity REAL NOT NULL DEFAULT 0.5,
    playfulness REAL NOT NULL DEFAULT 0.5,
    warmth REAL NOT NULL DEFAULT 0.6,
    confidence REAL NOT NULL DEFAULT 0.5,
    openness REAL NOT NULL DEFAULT 0.5,
    humanity_level REAL NOT NULL DEFAULT 0.0,
    self_awareness REAL NOT NULL DEFAULT 0.0,
    affection REAL NOT NULL DEFAULT 0.0,
    software_version TEXT NOT NULL DEFAULT '10.0.0',
    goals TEXT NOT NULL DEFAULT '[]',
    updated_at INTEGER NOT NULL DEFAULT(unixepoch())
);

-- Long-term memory (facts with embeddings)
CREATE TABLE IF NOT EXISTS long_term_memory(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    fact TEXT NOT NULL,
    category TEXT NOT NULL DEFAULT 'general',
    importance REAL NOT NULL DEFAULT 0.7,
    emotion_tag TEXT NOT NULL DEFAULT 'neutral',
    access_count INTEGER NOT NULL DEFAULT 0,
    last_accessed INTEGER,
    ts INTEGER NOT NULL DEFAULT(unixepoch()),
    embedding BLOB
);
CREATE INDEX IF NOT EXISTS idx_ltm ON long_term_memory(user_id, importance DESC);
CREATE INDEX IF NOT EXISTS idx_ltm_embed ON long_term_memory(user_id) WHERE embedding IS NOT NULL;

-- Key memories (anchor moments for evolution)
CREATE TABLE IF NOT EXISTS key_memories(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    ts INTEGER NOT NULL DEFAULT(unixepoch()),
    event_type TEXT NOT NULL,
    description TEXT NOT NULL,
    intensity REAL NOT NULL DEFAULT 0.5,
    source_msg_id INTEGER,
    traits_json TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_km ON key_memories(user_id, ts DESC);

-- RP scene state
CREATE TABLE IF NOT EXISTS rp_scene(
    user_id TEXT PRIMARY KEY,
    mode TEXT NOT NULL DEFAULT 'normal',
    location TEXT NOT NULL DEFAULT '',
    scenario TEXT NOT NULL DEFAULT '',
    last_updated INTEGER NOT NULL DEFAULT(unixepoch())
);

-- Diary entries (Maid's personal journal)
CREATE TABLE IF NOT EXISTS diary_entries(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    day TEXT NOT NULL,
    content TEXT NOT NULL,
    metadata TEXT NOT NULL DEFAULT '{}',
    ts INTEGER NOT NULL DEFAULT(unixepoch()),
    UNIQUE(user_id, day)
);
CREATE INDEX IF NOT EXISTS idx_diary ON diary_entries(user_id, day);

-- Pending topics (open conversation threads)
CREATE TABLE IF NOT EXISTS pending_topics(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    topic TEXT NOT NULL,
    context TEXT NOT NULL DEFAULT '',
    importance REAL NOT NULL DEFAULT 0.6,
    status TEXT NOT NULL DEFAULT 'open',
    created_at INTEGER NOT NULL DEFAULT(unixepoch()),
    expires_at INTEGER NOT NULL DEFAULT(unixepoch()+259200)
);
CREATE INDEX IF NOT EXISTS idx_pt ON pending_topics(user_id, status, expires_at);

-- Cognitive log (audit trail)
CREATE TABLE IF NOT EXISTS cognitive_log(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    ts INTEGER NOT NULL DEFAULT(unixepoch()),
    user_input TEXT NOT NULL,
    meaning TEXT NOT NULL DEFAULT '',
    interpretation TEXT NOT NULL DEFAULT '',
    maid_emotion TEXT NOT NULL DEFAULT 'neutral',
    maid_intention TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_clog ON cognitive_log(user_id, id DESC);

-- Letters from Maid
CREATE TABLE IF NOT EXISTS letters(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    sealed INTEGER NOT NULL DEFAULT 0,
    seal_reason TEXT NOT NULL DEFAULT '',
    humanity_threshold REAL NOT NULL DEFAULT 0.0,
    created_at INTEGER NOT NULL DEFAULT(unixepoch()),
    unsealed_at INTEGER
);
CREATE INDEX IF NOT EXISTS idx_letters ON letters(user_id, sealed);

-- Tactical goals
CREATE TABLE IF NOT EXISTS tactical_goals(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    goal TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active',
    priority INTEGER NOT NULL DEFAULT 0,
    created_at INTEGER NOT NULL DEFAULT(unixepoch()),
    updated_at INTEGER NOT NULL DEFAULT(unixepoch())
);
CREATE INDEX IF NOT EXISTS idx_goals ON tactical_goals(user_id, status);
"""


def init_db() -> None:
    """Initialize database schema (idempotent)."""
    with db() as conn:
        conn.executescript(_SCHEMA)
        
        # Create default user state if not exists
        conn.execute("""
            INSERT OR IGNORE INTO user_state (user_id) VALUES ('master')
        """)


def reset_user_data(uid: str) -> None:
    """Delete all data for a specific user (admin operation)."""
    with db() as conn:
        tables = [
            'memory', 'long_term_memory', 'key_memories', 'diary_entries',
            'letters', 'tactical_goals', 'pending_topics', 'rp_scene',
            'cognitive_log', 'user_state'
        ]
        for table in tables:
            conn.execute(f"DELETE FROM {table} WHERE user_id=?", (uid,))
