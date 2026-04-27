"""Repository layer for database operations.

This module provides a clean separation between business logic and data access,
following the Repository pattern. All SQL queries are contained here.

Security: All state keys are validated against an allowlist to prevent SQL injection.
"""

import sqlite3
import json
import time
from typing import Optional, Dict, Any, List
from contextlib import contextmanager

from app.db import db

# Import new repositories
from .diary import DiaryRepository as NewDiaryRepository
from .tactical_goals import TacticalGoalsRepository
from .letters import LettersRepository


# Whitelist of allowed user_state keys (security)
ALLOWED_STATE_KEYS = frozenset({
    'mood', 'trust', 'fear', 'attachment', 'curiosity', 
    'playfulness', 'warmth', 'confidence', 'openness',
    'humanity_level', 'self_awareness', 'affection',
    'software_version', 'goals', 'msg_count', 'total_msg_count',
    'last_activity_ts'
})


class UserStateRepository:
    """Repository for user_state table operations."""
    
    @staticmethod
    def get(uid: str) -> Optional[Dict[str, Any]]:
        """Load user state by ID."""
        with db() as c:
            row = c.execute(
                "SELECT * FROM user_state WHERE user_id=?", (uid,)
            ).fetchone()
            if not row:
                return None
            
            # Convert sqlite3.Row to dict
            return dict(row)
    
    @staticmethod
    def save(uid: str, state: Dict[str, Any]) -> None:
        """Save user state (validated keys only)."""
        with db() as c:
            # Build dynamic update based on state keys (validated)
            sets = []
            vals = []
            for k, v in state.items():
                if k == 'user_id':
                    continue
                if k not in ALLOWED_STATE_KEYS:
                    # Log warning but don't fail silently
                    import sys
                    print(f"[WARNING] Disallowed state key ignored: {k}", file=sys.stderr)
                    continue
                sets.append(f"{k}=?")
                vals.append(v)
            
            if not sets:
                return
            
            vals.append(uid)
            c.execute(
                f"UPDATE user_state SET {', '.join(sets)}, updated_at=unixepoch() WHERE user_id=?",
                vals
            )
            
            # Insert if not exists (rowcount==0 means no rows updated)
            if c.rowcount == 0:
                cols = ['user_id'] + [k for k in state.keys() if k != 'user_id' and k in ALLOWED_STATE_KEYS]
                placeholders = ['?'] * len(cols)
                vals = [uid] + [state[k] for k in cols if k != 'user_id']
                c.execute(
                    f"INSERT OR IGNORE INTO user_state ({', '.join(cols)}) VALUES ({', '.join(placeholders)})",
                    vals
                )
    
    @staticmethod
    def create_if_not_exists(uid: str) -> None:
        """Create default user state if not exists."""
        with db() as c:
            c.execute("""
                INSERT OR IGNORE INTO user_state (user_id) VALUES (?)
            """, (uid,))
    
    @staticmethod
    def increment_msg_count(uid: str, session_increment: int = 1, total_increment: int = 1) -> Dict[str, Any]:
        """Atomically increment message counters and return updated state."""
        with db() as c:
            c.execute("""
                UPDATE user_state 
                SET msg_count = msg_count + ?, 
                    total_msg_count = total_msg_count + ?,
                    last_activity_ts = unixepoch(),
                    updated_at = unixepoch()
                WHERE user_id = ?
            """, (session_increment, total_increment, uid))
            
            # Return updated state
            row = c.execute(
                "SELECT * FROM user_state WHERE user_id=?", (uid,)
            ).fetchone()
            return dict(row) if row else {}


class MemoryRepository:
    """Repository for memory table operations (short-term memory)."""
    
    @staticmethod
    def add(uid: str, role: str, text: str, cog: Dict, kind: str, status: str = "done") -> int:
        """Add a message to memory. Returns message ID."""
        ts = int(time.time())
        cog_json = json.dumps(cog) if cog else "{}"
        
        with db() as c:
            c.execute("""
                INSERT INTO memory (user_id, ts, role, text, cog, kind, status)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (uid, ts, role, text, cog_json, kind, status))
            return c.lastrowid
    
    @staticmethod
    def discard_pending(uid: str, msg_id: int) -> None:
        """Discard a pending message."""
        with db() as c:
            c.execute(
                "DELETE FROM memory WHERE user_id=? AND id=? AND status='pending'",
                (uid, msg_id)
            )
    
    @staticmethod
    def get_recent(uid: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent messages for context (chronological order)."""
        with db() as c:
            rows = c.execute("""
                SELECT id, ts, role, text, cog, kind, status
                FROM memory
                WHERE user_id=? AND status='done'
                ORDER BY ts DESC
                LIMIT ?
            """, (uid, limit)).fetchall()
            
            result = []
            for row in reversed(rows):  # Reverse to get chronological order
                item = dict(row)
                try:
                    item['cog'] = json.loads(row['cog']) if row['cog'] else {}
                except:
                    item['cog'] = {}
                result.append(item)
            return result
    
    @staticmethod
    def count_total(uid: str) -> int:
        """Count total messages for user."""
        with db() as c:
            row = c.execute(
                "SELECT COUNT(*) FROM memory WHERE user_id=?", (uid,)
            ).fetchone()
            return row[0] if row else 0
    
    @staticmethod
    def cleanup_stale_pending(max_age_seconds: int = 3600) -> int:
        """Remove pending messages older than max_age_seconds. Returns count deleted."""
        with db() as c:
            c.execute("""
                DELETE FROM memory 
                WHERE status='pending' AND ts < unixepoch() - ?
            """, (max_age_seconds,))
            return c.rowcount


class LongTermMemoryRepository:
    """Repository for long_term_memory table operations."""
    
    @staticmethod
    def get_facts(uid: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get LTM facts for user (ordered by importance)."""
        with db() as c:
            rows = c.execute("""
                SELECT fact, importance, category, emotion_tag, created_at
                FROM long_term_memory
                WHERE user_id=?
                ORDER BY importance DESC, created_at DESC
                LIMIT ?
            """, (uid, limit)).fetchall()
            
            return [dict(row) for row in rows]
    
    @staticmethod
    def add_fact(uid: str, fact: str, importance: float = 0.5, 
                 category: str = 'general', emotion_tag: str = 'neutral',
                 embedding: Optional[bytes] = None) -> int:
        """Add a fact to long-term memory."""
        ts = int(time.time())
        
        with db() as c:
            c.execute("""
                INSERT INTO long_term_memory (user_id, fact, importance, category, emotion_tag, ts, embedding)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (uid, fact, importance, category, emotion_tag, ts, embedding))
            return c.lastrowid
    
    @staticmethod
    def get_unembedded_batch(uid: str, batch_size: int = 200) -> List[Dict[str, Any]]:
        """Get facts without embeddings for background processing."""
        with db() as c:
            rows = c.execute("""
                SELECT id, fact FROM long_term_memory
                WHERE user_id=? AND embedding IS NULL
                LIMIT ?
            """, (uid, batch_size)).fetchall()
            return [dict(row) for row in rows]
    
    @staticmethod
    def update_embedding(fid: int, embedding: bytes) -> None:
        """Update embedding for a fact."""
        with db() as c:
            c.execute("""
                UPDATE long_term_memory SET embedding=?, last_accessed=unixepoch()
                WHERE id=?
            """, (embedding, fid))


class KeyMemoryRepository:
    """Repository for key_memories table operations."""
    
    @staticmethod
    def get_recent(uid: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Get recent key memories for user (newest first)."""
        with db() as c:
            rows = c.execute("""
                SELECT id, ts, event_type, description, intensity, traits_json
                FROM key_memories
                WHERE user_id=?
                ORDER BY ts DESC, id DESC
                LIMIT ?
            """, (uid, limit)).fetchall()
            
            result = []
            for row in rows:
                item = dict(row)
                try:
                    item["traits_delta"] = json.loads(row["traits_json"]) if row["traits_json"] else {}
                except:
                    item["traits_delta"] = {}
                result.append(item)
            return result
    
    @staticmethod
    def add(uid: str, event_type: str, description: str, intensity: float,
            source_msg_id: Optional[int] = None, traits_delta: Optional[Dict] = None) -> int:
        """Add a key memory."""
        ts = int(time.time())
        delta_json = json.dumps(traits_delta) if traits_delta else "{}"
        
        with db() as c:
            c.execute("""
                INSERT INTO key_memories (user_id, ts, event_type, description, intensity, source_msg_id, traits_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (uid, ts, event_type, description[:200], intensity, source_msg_id, delta_json))
            return c.lastrowid
    
    @staticmethod
    def get_timeline(uid: str, limit: int = 200) -> List[Dict[str, Any]]:
        """Get evolution timeline (oldest first for reconstruction)."""
        with db() as c:
            rows = c.execute("""
                SELECT id, ts, event_type, description, intensity, traits_json
                FROM key_memories
                WHERE user_id=?
                ORDER BY ts ASC, id ASC
                LIMIT ?
            """, (uid, limit)).fetchall()
            
            result = []
            for row in rows:
                item = dict(row)
                try:
                    item["traits_delta"] = json.loads(row["traits_json"]) if row["traits_json"] else {}
                except:
                    item["traits_delta"] = {}
                result.append(item)
            return result


class RPSceneRepository:
    """Repository for rp_scene table operations."""
    
    @staticmethod
    def get(uid: str) -> Dict[str, Any]:
        """Get current RP scene for user."""
        with db() as c:
            row = c.execute(
                "SELECT mode, location, scenario, last_updated FROM rp_scene WHERE user_id=?",
                (uid,)
            ).fetchone()
            
            if not row:
                return {"mode": "normal", "location": "", "scenario": "", "last_updated": 0}
            
            return dict(row)
    
    @staticmethod
    def save(uid: str, mode: str, location: str, scenario: str) -> None:
        """Save RP scene."""
        ts = int(time.time())
        
        with db() as c:
            c.execute("""
                INSERT OR REPLACE INTO rp_scene (user_id, mode, location, scenario, last_updated)
                VALUES (?, ?, ?, ?, ?)
            """, (uid, mode, location, scenario, ts))


class DiaryRepository:
    """Repository for diary_entries table operations."""
    
    @staticmethod
    def get_latest(uid: str) -> Optional[Dict[str, Any]]:
        """Get latest diary entry."""
        with db() as c:
            row = c.execute("""
                SELECT id, day, content, metadata, ts
                FROM diary_entries
                WHERE user_id=?
                ORDER BY ts DESC
                LIMIT 1
            """, (uid,)).fetchone()
            
            if not row:
                return None
            
            item = dict(row)
            try:
                item["metadata"] = json.loads(row["metadata"]) if row["metadata"] else {}
            except:
                item["metadata"] = {}
            return item
    
    @staticmethod
    def add(uid: str, content: str, day: Optional[str] = None, 
            metadata: Optional[Dict] = None) -> int:
        """Add diary entry."""
        import datetime
        if day is None:
            day = datetime.datetime.now().strftime("%Y-%m-%d")
        ts = int(time.time())
        meta_json = json.dumps(metadata or {})
        
        with db() as c:
            c.execute("""
                INSERT OR REPLACE INTO diary_entries (user_id, day, content, metadata, ts)
                VALUES (?, ?, ?, ?, ?)
            """, (uid, day, content, meta_json, ts))
            return c.lastrowid
    
    @staticmethod
    def exists_for_day(uid: str, day: str) -> bool:
        """Check if diary entry exists for a specific day."""
        with db() as c:
            row = c.execute("""
                SELECT 1 FROM diary_entries WHERE user_id=? AND day=?
            """, (uid, day)).fetchone()
            return row is not None


class PendingTopicsRepository:
    """Repository for pending_topics table operations."""
    
    @staticmethod
    def get_open_topics(uid: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Get open topics for proactive check-ins."""
        with db() as c:
            rows = c.execute("""
                SELECT id, topic, context, importance, created_at, expires_at
                FROM pending_topics
                WHERE user_id=? AND status='open' AND expires_at > unixepoch()
                ORDER BY importance DESC, created_at DESC
                LIMIT ?
            """, (uid, limit)).fetchall()
            return [dict(row) for row in rows]
    
    @staticmethod
    def add_topic(uid: str, topic: str, context: str = "", 
                  importance: float = 0.6, expires_in_sec: int = 259200) -> int:
        """Add a pending topic."""
        with db() as c:
            c.execute("""
                INSERT INTO pending_topics (user_id, topic, context, importance, expires_at)
                VALUES (?, ?, ?, ?, unixepoch() + ?)
            """, (uid, topic[:300], context[:500], importance, expires_in_sec))
            return c.lastrowid
    
    @staticmethod
    def close_topic(topic_id: int) -> bool:
        """Mark topic as closed."""
        with db() as c:
            cur = c.execute(
                "UPDATE pending_topics SET status='closed' WHERE id=?",
                (topic_id,)
            )
            return cur.rowcount > 0


# Convenience functions for backward compatibility
def load_state(uid: str) -> Optional[Dict[str, Any]]:
    """Legacy wrapper for UserStateRepository.get()"""
    return UserStateRepository.get(uid)


def save_state(uid: str, state: Dict[str, Any]) -> None:
    """Legacy wrapper for UserStateRepository.save()"""
    return UserStateRepository.save(uid, state)


def get_memory_for_prompt(uid: str, limit: int = 20) -> List[Dict[str, Any]]:
    """Legacy wrapper for MemoryRepository.get_recent()"""
    return MemoryRepository.get_recent(uid, limit)


def save_message(uid: str, role: str, text: str, cog: Dict, kind: str, status: str = "done") -> int:
    """Legacy wrapper for MemoryRepository.add()"""
    return MemoryRepository.add(uid, role, text, cog, kind, status)


def _discard_pending(uid: str, msg_id: int) -> None:
    """Legacy wrapper for MemoryRepository.discard_pending()"""
    return MemoryRepository.discard_pending(uid, msg_id)


def get_ltm_facts(uid: str, limit: int = 50) -> List[Dict[str, Any]]:
    """Legacy wrapper for LongTermMemoryRepository.get_facts()"""
    return LongTermMemoryRepository.get_facts(uid, limit)


def get_recent_key_memories(uid: str, limit: int = 5) -> List[Dict[str, Any]]:
    """Legacy wrapper for KeyMemoryRepository.get_recent()"""
    return KeyMemoryRepository.get_recent(uid, limit)


def load_rp_scene(uid: str) -> Dict[str, Any]:
    """Legacy wrapper for RPSceneRepository.get()"""
    return RPSceneRepository.get(uid)


def save_rp_scene(uid: str, mode: str, location: str, scenario: str) -> None:
    """Legacy wrapper for RPSceneRepository.save()"""
    return RPSceneRepository.save(uid, mode, location, scenario)


def get_diary(uid: str) -> Optional[Dict[str, Any]]:
    """Legacy wrapper for DiaryRepository.get_latest()"""
    return DiaryRepository.get_latest(uid)


def add_diary_entry(uid: str, content: str, metadata: Dict) -> int:
    """Legacy wrapper for DiaryRepository.add()"""
    return DiaryRepository.add(uid, content, metadata=metadata)


def get_open_topics(uid: str, limit: int = 5) -> List[Dict[str, Any]]:
    """Legacy wrapper for PendingTopicsRepository.get_open_topics()"""
    return PendingTopicsRepository.get_open_topics(uid, limit)
