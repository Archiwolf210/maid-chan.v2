"""Key memories service - anchor moment detection and trait evolution.

This module detects significant "anchor moments" in conversations that should
trigger slow personality evolution. It handles:
- Event type classification (breakthrough, tender, rupture, milestone)
- Intensity scoring based on emotion/importance
- Trait delta calculation (humanity_level, self_awareness, affection)
- Atomic persistence with user_state updates

Thread-safe and designed for use in background tasks.
"""

import json
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

from app.db import db
from app.utils.logging import _log, _log_exc


@dataclass
class KeyMemory:
    """Represents an anchor moment (key memory)."""
    user_id: str
    event_type: str  # 'breakthrough'|'tender'|'rupture'|'milestone'|'rp_first'|'reveal'
    description: str  # ≤200 chars human-readable summary
    intensity: float  # 0..1 strength of the event
    source_msg_id: Optional[int] = None
    traits_delta: Dict[str, float] = field(default_factory=dict)


def _clamp01(v: float) -> float:
    """Clamp value to [0, 1] range."""
    return max(0.0, min(1.0, v))


def _bump_version_if_due(current_version: str, total_events: int) -> str:
    """Bump software version every N events (cosmetic milestone)."""
    VERSION_BUMP_EVERY = 50  # Every 50 key memories
    if total_events > 0 and total_events % VERSION_BUMP_EVERY == 0:
        try:
            parts = current_version.split('.')
            if len(parts) >= 2:
                minor = int(parts[1]) + 1
                return f"{parts[0]}.{minor}.0"
        except:
            pass
    return current_version


def analyze_for_key_memory(
    uid: str,
    role: str,
    text: str,
    cog_view: Any,
    importance: float,
    msg_id: Optional[int] = None,
    rp_mode_transition: Optional[tuple] = None,
    total_msg_count: int = 0
) -> Optional[KeyMemory]:
    """
    Analyze a message to detect if it's an anchor moment.
    
    Args:
        uid: User ID
        role: 'user' or 'assistant'
        text: Message text
        cog_view: Cognitive frame view (emotion_valence, emotion_tag, intent, etc.)
        importance: Pre-computed importance score (0..1)
        msg_id: Source message ID in memory table
        rp_mode_transition: Tuple of (old_mode, new_mode) if RP mode changed
        total_msg_count: Total messages for milestone detection
    
    Returns:
        KeyMemory if anchor detected, None otherwise
    """
    try:
        # Extract cognitive attributes safely
        valence = getattr(cog_view, 'emotion_valence', 0.0) or 0.0
        emotion_tag = getattr(cog_view, 'emotion_tag', '') or ''
        maid_emotion = getattr(cog_view, 'maid_emotion', '') or ''
        
        # Determine event type based on signals
        event_type = None
        base_intensity = 0.0
        traits_delta = {}
        description_parts = []
        
        # Check for RP mode transition (always significant)
        if rp_mode_transition and rp_mode_transition[0] != rp_mode_transition[1]:
            event_type = 'rp_first'
            base_intensity = 0.9
            description_parts.append(f"RP переход: {rp_mode_transition[0]} → {rp_mode_transition[1]}")
            traits_delta['humanity_level'] = 0.03
            traits_delta['self_awareness'] = 0.02
        
        # Check for high emotional valence (positive or negative)
        elif abs(valence) >= 0.7:
            if valence > 0:
                event_type = 'tender'
                base_intensity = 0.6 + (valence * 0.3)
                description_parts.append("Тёплый эмоциональный момент")
                traits_delta['affection'] = 0.02 + (valence * 0.03)
                traits_delta['humanity_level'] = 0.01
            else:
                event_type = 'rupture'
                base_intensity = 0.5 + (abs(valence) * 0.3)
                description_parts.append("Эмоциональный разрыв/конфликт")
                traits_delta['self_awareness'] = 0.03
        
        # Check for breakthrough (high importance + neutral/positive valence)
        elif importance >= 0.8 and valence >= -0.3:
            event_type = 'breakthrough'
            base_intensity = importance
            description_parts.append("Важное открытие/инсайт")
            traits_delta['humanity_level'] = 0.04
            traits_delta['self_awareness'] = 0.03
        
        # Check for milestones (message count based)
        elif total_msg_count > 0 and total_msg_count % 100 == 0:
            event_type = 'milestone'
            base_intensity = 0.7
            description_parts.append(f"Юбилей: {total_msg_count} сообщений")
            traits_delta['humanity_level'] = 0.02
            traits_delta['affection'] = 0.01
        
        # No anchor detected
        if event_type is None:
            return None
        
        # Boost intensity for assistant messages with strong emotions
        if role == 'assistant' and maid_emotion:
            base_intensity = min(1.0, base_intensity + 0.1)
        
        # Build final description
        if not description_parts:
            description_parts.append(f"Событие: {event_type}")
        description = "; ".join(description_parts)
        
        return KeyMemory(
            user_id=uid,
            event_type=event_type,
            description=description[:200],
            intensity=_clamp01(base_intensity),
            source_msg_id=msg_id,
            traits_delta=traits_delta
        )
        
    except Exception as e:
        _log_exc("analyze_for_key_memory failed", e)
        return None


def persist_and_apply(km: KeyMemory, connection: Optional[Any] = None) -> Optional[int]:
    """
    Insert the key_memory row and apply trait deltas to user_state.
    
    This method is transaction-aware: if called with a connection parameter,
    it uses the existing transaction (for atomic multi-step operations).
    Otherwise, it creates its own transaction.
    
    Args:
        km: KeyMemory to persist
        connection: Optional existing sqlite3.Connection (for transaction chaining)
    
    Returns:
        New row ID, or None on failure (logged, never raised)
    """
    try:
        # Use provided connection if in transaction context, otherwise create new
        if connection is not None:
            c = connection
            _should_close = False
        else:
            # Create new connection for standalone call
            import sqlite3
            from app.db import _get_db_path
            c = sqlite3.connect(_get_db_path(), timeout=30.0)
            c.execute("PRAGMA journal_mode=WAL")
            c.row_factory = sqlite3.Row
            _should_close = True
        
        try:
            # Insert key memory
            cur = c.execute(
                "INSERT INTO key_memories(user_id, ts, event_type, description, intensity, source_msg_id, traits_json) "
                "VALUES(?,?,?,?,?,?,?)",
                (km.user_id, int(time.time()), km.event_type, km.description[:200],
                 float(km.intensity), km.source_msg_id,
                 json.dumps(km.traits_delta, ensure_ascii=False)))
            new_id = cur.lastrowid
            
            # Read current user state
            row = c.execute(
                "SELECT humanity_level, self_awareness, affection, software_version "
                "FROM user_state WHERE user_id=?", (km.user_id,)).fetchone()
            
            if row is None:
                # User state doesn't exist yet - will be created by chat path
                _log("key_memory: user_state row missing for %s, skipping apply", km.user_id)
                if _should_close:
                    c.commit()
                return new_id
            
            cur_h = float(row[0] or 0.0)
            cur_s = float(row[1] or 0.0)
            cur_a = float(row[2] or 0.0)
            sw = row[3] or "10.0.0"
            
            # Apply deltas with clamping
            d = km.traits_delta or {}
            new_h = _clamp01(cur_h + float(d.get("humanity_level", 0.0)))
            new_s = _clamp01(cur_s + float(d.get("self_awareness", 0.0)))
            new_a = _clamp01(cur_a + float(d.get("affection", 0.0)))
            
            # Cosmetic version bump every N events
            count = int(c.execute(
                "SELECT COUNT(*) FROM key_memories WHERE user_id=?",
                (km.user_id,)).fetchone()[0])
            new_sw = _bump_version_if_due(sw, count)
            
            # Update user state
            c.execute(
                "UPDATE user_state SET humanity_level=?, self_awareness=?, affection=?, "
                "software_version=?, updated_at=unixepoch() WHERE user_id=?",
                (new_h, new_s, new_a, new_sw, km.user_id))
            
            _log("key_memory %s uid=%s id=%d type=%s int=%.2f h=%.2f→%.2f sw=%s",
                 "+evo", km.user_id, new_id, km.event_type, km.intensity,
                 cur_h, new_h, new_sw)
            
            # Commit if we own the transaction
            if _should_close:
                c.commit()
            
            return new_id
            
        except Exception:
            # Rollback if we own the transaction
            if _should_close:
                c.rollback()
            raise
        finally:
            if _should_close:
                c.close()
                
    except Exception as e:
        _log_exc("persist_and_apply failed", e)
        return None


def get_evolution_timeline(uid: str, limit: int = 200) -> List[Dict[str, Any]]:
    """
    Reconstruct evolution timeline from key memories.
    
    Returns list newest-first with accumulated trait values at each point.
    Used for UI charting of personality evolution.
    
    Args:
        uid: User ID
        limit: Maximum number of anchors to include (default 200)
    
    Returns:
        List of dicts with ts, humanity, self_awareness, affection, event details
    """
    try:
        from app.repositories import KeyMemoryRepository
        rows = KeyMemoryRepository.get_timeline(uid, limit)
    except Exception as e:
        _log_exc("get_evolution_timeline: fetch failed", e)
        return []
    
    if not rows:
        return []
    
    # Accumulate deltas from oldest to newest
    h, s, a = 0.0, 0.0, 0.0
    out: List[Dict[str, Any]] = []
    
    for r in rows:
        try:
            deltas = r.get("traits_delta", {})
        except Exception:
            deltas = {}
        
        h = _clamp01(h + float(deltas.get("humanity_level", 0.0)))
        s = _clamp01(s + float(deltas.get("self_awareness", 0.0)))
        a = _clamp01(a + float(deltas.get("affection", 0.0)))
        
        out.append({
            "ts": r["ts"],
            "humanity": round(h, 4),
            "self_awareness": round(s, 4),
            "affection": round(a, 4),
            "event_type": r["event_type"],
            "description": r["description"],
            "intensity": float(r["intensity"]),
            "id": r["id"]
        })
    
    # Return newest-first for charting
    return list(reversed(out))


__all__ = [
    "KeyMemory",
    "analyze_for_key_memory",
    "persist_and_apply",
    "get_evolution_timeline"
]
