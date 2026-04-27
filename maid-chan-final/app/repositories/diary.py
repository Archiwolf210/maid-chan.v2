"""Diary repository - access layer for diary_entries."""
from __future__ import annotations
import json
import time
from typing import Optional, Dict, Any, List
from app.db import db


class DiaryRepository:
    """Repository for diary entries (Maid's personal journal)."""
    
    @staticmethod
    def get_entry(uid: str, day: str) -> Optional[Dict[str, Any]]:
        """Get diary entry for a specific day."""
        with db() as conn:
            row = conn.execute(
                "SELECT id, day, content, metadata, ts FROM diary_entries "
                "WHERE user_id=? AND day=?", (uid, day)
            ).fetchone()
            if row:
                return {
                    "id": row[0], 
                    "day": row[1], 
                    "content": row[2],
                    "metadata": json.loads(row[3] or "{}"), 
                    "ts": row[4]
                }
        return None
    
    @staticmethod
    def save_entry(uid: str, day: str, content: str, metadata: Dict[str, Any]) -> int:
        """Save or update diary entry for a day. Returns row id."""
        with db() as conn:
            cur = conn.execute(
                "INSERT INTO diary_entries(user_id, day, content, metadata, ts) "
                "VALUES(?,?,?,?,unixepoch()) "
                "ON CONFLICT(user_id, day) DO UPDATE SET content=excluded.content, metadata=excluded.metadata",
                (uid, day, content, json.dumps(metadata, ensure_ascii=False))
            )
            return cur.lastrowid
    
    @staticmethod
    def list_days(uid: str, limit: int = 30) -> List[Dict[str, Any]]:
        """List recent diary days (newest first)."""
        with db() as conn:
            rows = conn.execute(
                "SELECT day, ts, substr(content, 1, 100) as preview FROM diary_entries "
                "WHERE user_id=? ORDER BY day DESC LIMIT ?", (uid, limit)
            ).fetchall()
            return [{"day": r[0], "ts": r[1], "preview": r[2]} for r in rows]
    
    @staticmethod
    def has_entry(uid: str, day: str) -> bool:
        """Check if entry exists for a day."""
        with db() as conn:
            row = conn.execute(
                "SELECT 1 FROM diary_entries WHERE user_id=? AND day=?", (uid, day)
            ).fetchone()
            return row is not None
