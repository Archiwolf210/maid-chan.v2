"""Repository layer for database operations.

This module provides a clean separation between business logic and data access,
following the Repository pattern. All SQL queries should be contained here.
"""

import sqlite3
from typing import Optional, Dict, Any, List
from contextlib import contextmanager

# Import db context manager from db module
from app.db import db


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
            
            # Get column names
            cols = [d[0] for d in c.description]
            return dict(zip(cols, row))
    
    @staticmethod
    def save(uid: str, state: Dict[str, Any]) -> None:
        """Save user state."""
        with db() as c:
            # Build dynamic update based on state keys
            sets = []
            vals = []
            for k, v in state.items():
                if k == 'user_id':
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
            
            # Insert if not exists
            if c.rowcount == 0:
                cols = ['user_id'] + [k for k in state.keys() if k != 'user_id']
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


class MemoryRepository:
    """Repository for memory table operations (short-term memory)."""
    
    @staticmethod
    def add(uid: str, role: str, text: str, cog: Dict, kind: str, status: str = "done") -> int:
        """Add a message to memory. Returns message ID."""
        import json
        import time
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
        """Get recent messages for context."""
        import json
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
                cols = ['id', 'ts', 'role', 'text', 'kind', 'status']
                item = dict(zip(cols, row[:6]))
                try:
                    item['cog'] = json.loads(row[4]) if row[4] else {}
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


class LongTermMemoryRepository:
    """Repository for long_term_memory table operations."""
    
    @staticmethod
    def get_facts(uid: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get LTM facts for user."""
        with db() as c:
            rows = c.execute("""
                SELECT fact, importance, created_at
                FROM long_term_memory
                WHERE user_id=?
                ORDER BY importance DESC, created_at DESC
                LIMIT ?
            """, (uid, limit)).fetchall()
            
            return [
                {"fact": row[0], "importance": row[1], "created_at": row[2]}
                for row in rows
            ]
    
    @staticmethod
    def add_fact(uid: str, fact: str, importance: float = 0.5, embedding: Optional[bytes] = None) -> int:
        """Add a fact to long-term memory."""
        import time
        ts = int(time.time())
        
        with db() as c:
            c.execute("""
                INSERT INTO long_term_memory (user_id, fact, importance, created_at, embedding)
                VALUES (?, ?, ?, ?, ?)
            """, (uid, fact, importance, ts, embedding))
            return c.lastrowid


class KeyMemoryRepository:
    """Repository for key_memories table operations."""
    
    @staticmethod
    def get_recent(uid: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Get recent key memories for user."""
        import json
        with db() as c:
            rows = c.execute("""
                SELECT id, summary, intensity, traits_delta, created_at
                FROM key_memories
                WHERE user_id=?
                ORDER BY created_at DESC
                LIMIT ?
            """, (uid, limit)).fetchall()
            
            result = []
            for row in rows:
                item = {
                    "id": row[0],
                    "summary": row[1],
                    "intensity": row[2],
                    "created_at": row[4]
                }
                try:
                    item["traits_delta"] = json.loads(row[3]) if row[3] else {}
                except:
                    item["traits_delta"] = {}
                result.append(item)
            return result
    
    @staticmethod
    def add(uid: str, summary: str, intensity: float, traits_delta: Optional[Dict] = None) -> int:
        """Add a key memory."""
        import json
        import time
        ts = int(time.time())
        delta_json = json.dumps(traits_delta) if traits_delta else None
        
        with db() as c:
            c.execute("""
                INSERT INTO key_memories (user_id, summary, intensity, traits_delta, created_at)
                VALUES (?, ?, ?, ?, ?)
            """, (uid, summary, intensity, delta_json, ts))
            return c.lastrowid


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
            
            return {
                "mode": row[0],
                "location": row[1],
                "scenario": row[2],
                "last_updated": row[3]
            }
    
    @staticmethod
    def save(uid: str, mode: str, location: str, scenario: str) -> None:
        """Save RP scene."""
        import time
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
                SELECT id, content, metadata, created_at
                FROM diary_entries
                WHERE user_id=?
                ORDER BY created_at DESC
                LIMIT 1
            """, (uid,)).fetchone()
            
            if not row:
                return None
            
            import json
            return {
                "id": row[0],
                "content": row[1],
                "metadata": json.loads(row[2]) if row[2] else {},
                "created_at": row[3]
            }
    
    @staticmethod
    def add(uid: str, content: str, metadata: Dict) -> int:
        """Add diary entry."""
        import json
        import time
        ts = int(time.time())
        meta_json = json.dumps(metadata)
        
        with db() as c:
            c.execute("""
                INSERT INTO diary_entries (user_id, content, metadata, created_at)
                VALUES (?, ?, ?, ?)
            """, (uid, content, meta_json, ts))
            return c.lastrowid


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
    return DiaryRepository.add(uid, content, metadata)
