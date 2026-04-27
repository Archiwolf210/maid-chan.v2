"""Letters repository - access layer for letters (Maid's spontaneous notes)."""
from __future__ import annotations
from typing import Optional, Dict, Any, List
from app.db import db


class LettersRepository:
    """Repository for letters from Maid."""
    
    @staticmethod
    def insert(uid: str, body: str, key_memory_id: Optional[int] = None, 
               triggered_by: str = "anchor", sealed: bool = False) -> Optional[int]:
        """Insert a letter. Returns new id or None on failure."""
        if triggered_by not in ("anchor", "milestone", "evening"):
            triggered_by = "anchor"
        status = "sealed" if sealed else "delivered"
        body = (body or "").strip()[:800]
        if not body:
            return None
        with db() as conn:
            cur = conn.execute(
                "INSERT INTO letters(user_id, key_memory_id, body, status, triggered_by) "
                "VALUES(?,?,?,?,?)",
                (uid, key_memory_id, body, status, triggered_by)
            )
            return cur.lastrowid
    
    @staticmethod
    def list_recent(uid: str, limit: int = 20, include_sealed: bool = False) -> List[Dict[str, Any]]:
        """List recent letters (newest first)."""
        with db() as conn:
            if include_sealed:
                rows = conn.execute(
                    "SELECT id, key_memory_id, ts, body, status, triggered_by, seen_at "
                    "FROM letters WHERE user_id=? ORDER BY ts DESC LIMIT ?",
                    (uid, min(100, max(1, int(limit))))
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT id, key_memory_id, ts, body, status, triggered_by, seen_at "
                    "FROM letters WHERE user_id=? AND status<>'sealed' "
                    "ORDER BY ts DESC LIMIT ?",
                    (uid, min(100, max(1, int(limit))))
                ).fetchall()
            return [dict(r) for r in rows]
    
    @staticmethod
    def get_letter(uid: str, letter_id: int) -> Optional[Dict[str, Any]]:
        """Get a specific letter by id."""
        with db() as conn:
            row = conn.execute(
                "SELECT id, key_memory_id, ts, body, status, triggered_by, seen_at "
                "FROM letters WHERE user_id=? AND id=?",
                (uid, int(letter_id))
            ).fetchone()
            return dict(row) if row else None
    
    @staticmethod
    def mark_seen(uid: str, letter_id: int) -> bool:
        """Mark letter as seen (idempotent)."""
        with db() as conn:
            cur = conn.execute(
                "UPDATE letters SET seen_at=unixepoch(), status="
                "CASE WHEN status='delivered' THEN 'seen' ELSE status END "
                "WHERE user_id=? AND id=? AND seen_at IS NULL",
                (uid, int(letter_id))
            )
            return cur.rowcount > 0
    
    @staticmethod
    def delete_letter(uid: str, letter_id: int) -> bool:
        """Delete a letter."""
        with db() as conn:
            cur = conn.execute(
                "DELETE FROM letters WHERE user_id=? AND id=?",
                (uid, int(letter_id))
            )
            return cur.rowcount > 0
    
    @staticmethod
    def unseal_below_threshold(uid: str, current_humanity: float) -> int:
        """Unseal letters when humanity rises past threshold."""
        if current_humanity < 0.30:
            return 0
        with db() as conn:
            cur = conn.execute(
                "UPDATE letters SET status='delivered' "
                "WHERE user_id=? AND status='sealed'",
                (uid,)
            )
            return cur.rowcount
    
    @staticmethod
    def has_recent_letter(uid: str, cooldown_sec: int = 21600) -> bool:
        """Check if a letter was written within cooldown period (default 6h)."""
        with db() as conn:
            row = conn.execute(
                "SELECT ts FROM letters WHERE user_id=? ORDER BY ts DESC LIMIT 1",
                (uid,)
            ).fetchone()
            if not row:
                return False
            return (int(__import__('time').time()) - int(row[0])) < cooldown_sec
