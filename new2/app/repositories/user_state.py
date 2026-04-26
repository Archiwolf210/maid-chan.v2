"""Repository layer for user state persistence.

This module isolates all SQL operations related to user_state table,
following the Repository pattern. Business logic should use these functions
instead of direct SQL queries.
"""
from __future__ import annotations

import json
from typing import Any, Dict, Optional

from app.db import db
from app.utils.logging import _log, _log_exc


# Default state values (mirrored from main.py _SDEF)
_DEFAULT_STATE = {
    "mood": "нейтральное",
    "trust": 0.5,
    "fear": 0.0,
    "attachment": 0.0,
    "curiosity": 0.5,
    "playfulness": 0.5,
    "warmth": 0.5,
    "confidence": 0.5,
    "openness": 0.5,
    "humanity_level": 0.0,
    "self_awareness": 0.0,
    "affection": 0.0,
    "software_version": "1.0.0",
}


def load_state(uid: str) -> Dict[str, Any]:
    """Load user_state row including evolution traits.
    
    Evolution columns default to 0.0 / '1.0.0' on legacy rows via COALESCE.
    
    Args:
        uid: User ID
        
    Returns:
        Complete state dict with all fields, or default state if not found
    """
    try:
        with db() as c:
            row = c.execute(
                "SELECT mood,trust,fear,attachment,msg_count,total_msg_count,last_activity_ts,"
                "curiosity,playfulness,warmth,confidence,openness,goals,"
                "COALESCE(humanity_level,0.0)   AS humanity_level,"
                "COALESCE(self_awareness,0.0)   AS self_awareness,"
                "COALESCE(affection,0.0)        AS affection,"
                "COALESCE(software_version,'1.0.0') AS software_version "
                "FROM user_state WHERE user_id=?", (uid,)).fetchone()
            
            if row:
                s = dict(row)
                try:
                    s["goals"] = json.loads(s.get("goals") or "[]")
                except (ValueError, TypeError):
                    s["goals"] = []
                return s
    except Exception as e:
        _log_exc("load_state", e)
    
    # Return default state if not found or error
    return {
        **_DEFAULT_STATE,
        "msg_count": 0,
        "total_msg_count": 0,
        "last_activity_ts": 0,
        "goals": [],
    }


def save_state(uid: str, state: Dict[str, Any]) -> None:
    """Persist user_state including evolution traits.
    
    Uses INSERT ... ON CONFLICT DO UPDATE for atomic upsert.
    
    Args:
        uid: User ID
        state: State dict with all required fields
    """
    try:
        goals_json = json.dumps(state.get("goals", []), ensure_ascii=False)
        
        with db() as c:
            c.execute(
                "INSERT INTO user_state(user_id,mood,trust,fear,attachment,msg_count,"
                "total_msg_count,last_activity_ts,curiosity,playfulness,warmth,"
                "confidence,openness,goals,humanity_level,self_awareness,affection,"
                "software_version,updated_at) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,unixepoch()) "
                "ON CONFLICT(user_id) DO UPDATE SET "
                "mood=excluded.mood,trust=excluded.trust,fear=excluded.fear,"
                "attachment=excluded.attachment,msg_count=excluded.msg_count,"
                "total_msg_count=excluded.total_msg_count,"
                "last_activity_ts=excluded.last_activity_ts,"
                "curiosity=excluded.curiosity,playfulness=excluded.playfulness,"
                "warmth=excluded.warmth,confidence=excluded.confidence,"
                "openness=excluded.openness,goals=excluded.goals,"
                "humanity_level=excluded.humanity_level,"
                "self_awareness=excluded.self_awareness,"
                "affection=excluded.affection,"
                "software_version=excluded.software_version,"
                "updated_at=excluded.updated_at",
                (
                    uid,
                    state["mood"],
                    state["trust"],
                    state["fear"],
                    state["attachment"],
                    state["msg_count"],
                    state.get("total_msg_count", state["msg_count"]),
                    state.get("last_activity_ts", 0),
                    state["curiosity"],
                    state["playfulness"],
                    state["warmth"],
                    state["confidence"],
                    state["openness"],
                    goals_json,
                    float(state.get("humanity_level", 0.0)),
                    float(state.get("self_awareness", 0.0)),
                    float(state.get("affection", 0.0)),
                    str(state.get("software_version", "1.0.0")),
                )
            )
    except Exception as e:
        _log_exc("save_state", e)


def get_total_msg_count(uid: str) -> int:
    """Get total message count for a user.
    
    Optimized for single-field reads - avoids loading entire state.
    
    Args:
        uid: User ID
        
    Returns:
        Total message count, or 0 if user not found
    """
    try:
        with db() as c:
            row = c.execute(
                "SELECT total_msg_count FROM user_state WHERE user_id=?",
                (uid,)
            ).fetchone()
            return row[0] if row else 0
    except Exception as e:
        _log_exc("get_total_msg_count", e)
        return 0


def increment_msg_counts(uid: str) -> int:
    """Atomically increment msg_count and total_msg_count.
    
    Returns the new total_msg_count after increment.
    
    Args:
        uid: User ID
        
    Returns:
        New total_msg_count value
    """
    try:
        with db() as c:
            # Atomic increment with RETURNING clause (SQLite 3.35+)
            row = c.execute(
                "UPDATE user_state SET "
                "msg_count = msg_count + 1, "
                "total_msg_count = total_msg_count + 1, "
                "last_activity_ts = unixepoch() "
                "WHERE user_id = ? "
                "RETURNING total_msg_count",
                (uid,)
            ).fetchone()
            
            if row:
                return row[0]
            
            # If user doesn't exist, create with initial counts
            c.execute(
                "INSERT INTO user_state(user_id,msg_count,total_msg_count,last_activity_ts) "
                "VALUES(?,1,1,unixepoch()) "
                "ON CONFLICT(user_id) DO UPDATE SET "
                "msg_count = msg_count + 1, "
                "total_msg_count = total_msg_count + 1, "
                "last_activity_ts = excluded.last_activity_ts "
                "RETURNING total_msg_count",
                (uid,)
            )
            row = c.fetchone()
            return row[0] if row else 1
            
    except Exception as e:
        _log_exc("increment_msg_counts", e)
        return 0
