"""Tactical goals repository - access layer for tactical_goals."""
from __future__ import annotations
import time
from typing import Optional, Dict, Any, List
from app.db import db


class TacticalGoalsRepository:
    """Repository for tactical goals (Maid's short-term self-directed goals)."""
    
    @staticmethod
    def list_active(uid: str) -> List[Dict[str, Any]]:
        """Get all active (non-expired) goals for user."""
        with db() as conn:
            rows = conn.execute(
                "SELECT id, horizon, text, reasoning, status, created_at, expires_at "
                "FROM tactical_goals WHERE user_id=? AND status='active' "
                "AND expires_at > unixepoch() ORDER BY horizon ASC, id DESC",
                (uid,)
            ).fetchall()
            return [dict(r) for r in rows]
    
    @staticmethod
    def create_goal(uid: str, horizon: str, text: str, reasoning: str = "") -> Optional[int]:
        """Create a new goal. Closes any existing active goal of same horizon."""
        if horizon not in ("day", "week"):
            return None
        ttl = 24 * 3600 if horizon == "day" else 7 * 86400
        with db() as conn:
            # Close existing active goal of same horizon
            conn.execute(
                "UPDATE tactical_goals SET status='expired', completed_at=unixepoch() "
                "WHERE user_id=? AND horizon=? AND status='active'",
                (uid, horizon)
            )
            cur = conn.execute(
                "INSERT INTO tactical_goals(user_id, horizon, text, reasoning, expires_at) "
                "VALUES(?,?,?,?, unixepoch()+?)",
                (uid, horizon, text[:200], reasoning[:280], ttl)
            )
            return cur.lastrowid
    
    @staticmethod
    def mark_done(uid: str, goal_id: int) -> bool:
        """Mark goal as done. Returns True if updated."""
        with db() as conn:
            cur = conn.execute(
                "UPDATE tactical_goals SET status='done', completed_at=unixepoch() "
                "WHERE user_id=? AND id=? AND status='active'",
                (uid, int(goal_id))
            )
            return cur.rowcount > 0
    
    @staticmethod
    def mark_abandoned(uid: str, goal_id: int) -> bool:
        """Mark goal as abandoned. Returns True if updated."""
        with db() as conn:
            cur = conn.execute(
                "UPDATE tactical_goals SET status='abandoned', completed_at=unixepoch() "
                "WHERE user_id=? AND id=? AND status='active'",
                (uid, int(goal_id))
            )
            return cur.rowcount > 0
    
    @staticmethod
    def expire_old(uid: Optional[str] = None) -> int:
        """Expire past-due active goals. Pass uid=None to sweep all users."""
        with db() as conn:
            if uid is None:
                cur = conn.execute(
                    "UPDATE tactical_goals SET status='expired', completed_at=unixepoch() "
                    "WHERE status='active' AND expires_at <= unixepoch()"
                )
            else:
                cur = conn.execute(
                    "UPDATE tactical_goals SET status='expired', completed_at=unixepoch() "
                    "WHERE user_id=? AND status='active' AND expires_at <= unixepoch()",
                    (uid,)
                )
            return cur.rowcount
    
    @staticmethod
    def has_active_goal(uid: str, horizon: str) -> bool:
        """Check if user has an active goal for given horizon."""
        with db() as conn:
            row = conn.execute(
                "SELECT 1 FROM tactical_goals "
                "WHERE user_id=? AND horizon=? AND status='active' AND expires_at > unixepoch() "
                "LIMIT 1",
                (uid, horizon)
            ).fetchone()
            return row is not None
