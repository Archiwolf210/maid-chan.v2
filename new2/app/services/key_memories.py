"""Anchor-moment detection + slow personality evolution (v9.4).

Why a service module
--------------------
The earlier sample project tried to do this with regex over message text
("первый раз", "ссора", ...) and produced a brittle, easily-fooled detector
that fired on negations and casual mentions. We replace that with a detector
that consumes the *cognitive frame* the chat path already computes (intent,
emotion_tag, emotion_valence, importance) and the structural signals we
already have (total_msg_count, RP-mode transition, message importance score).

Public API
----------
  analyze_for_key_memory(uid, role, text, cog, importance, msg_id)
      Run on each completed message. Returns Optional[KeyMemory] when a
      detection fires; otherwise None. The caller persists & applies it.

  persist_and_apply(km)
      Insert into key_memories, update user_state by traits_delta, log.

  get_recent_key_memories(uid, limit)
      Read for prompt injection + UI panel.

  build_evolution_prompt_block(state, recent)
      Compact prompt fragment (≤200 tokens) summarizing where Maid is in
      her arc — slotted into build_prompt's dynamic system layer.

Design rules
------------
- NEVER regex on user text. All detections come from cog/importance/etc.
- Idempotent: same message processed twice never creates duplicate rows
  (deduped by (user_id, source_msg_id, event_type)).
- Cooldown: per (uid, event_type) — at most one detection per 30 min, so
  a chatty exchange doesn't drown the table.
- Trait deltas are SMALL (0.01..0.05). Reaching humanity=1.0 should require
  ~50+ anchor moments — slow on purpose so the arc feels earned.
"""
from __future__ import annotations
import json
import logging
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

from app.db import db
from app.models import EVENT_TYPES, EvolutionState, KeyMemory

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  HOT-PATH CACHE  (v9.6)
# ─────────────────────────────────────────────────────────────────────────────
# `get_recent_key_memories` is read by `build_prompt` AND by the chat done
# payload — every turn. Without a cache that's 2 DB SELECTs per chat round
# trip, plus blocking the streaming generator while we wait. Cache the last
# N rows per uid in memory, invalidate it on insert/delete. Falls through to
# DB on cold start.
_RECENT_CACHE: Dict[str, List[Dict[str, Any]]] = {}
_RECENT_CACHE_LOCK = threading.Lock()
_RECENT_CACHE_LIMIT = 20      # max rows kept per uid

def _cache_invalidate(uid: str) -> None:
    with _RECENT_CACHE_LOCK:
        _RECENT_CACHE.pop(uid, None)

def _cache_prepend(uid: str, row: Dict[str, Any]) -> None:
    with _RECENT_CACHE_LOCK:
        cur = _RECENT_CACHE.get(uid)
        if cur is None:    # cold; let next reader hit DB
            return
        cur.insert(0, row)
        if len(cur) > _RECENT_CACHE_LIMIT:
            del cur[_RECENT_CACHE_LIMIT:]


# ── Tunables (intentionally inline, not in config -- this is calibration) ────
_COOLDOWN_SEC: int = 30 * 60          # at most one event of a kind per 30 min
_MIN_TOTAL_MSGS_FOR_KM: int = 4       # don't fire on the very first turn
_DEDUP_BY_MSG: bool = True            # never two rows for the same source_msg_id

# Trait deltas per event_type. Small, additive, capped to [0,1] in apply.
_TRAIT_DELTAS: Dict[str, Dict[str, float]] = {
    "breakthrough":  {"humanity_level": 0.04, "self_awareness": 0.06},
    "tender":        {"humanity_level": 0.02, "affection":      0.04},
    "rupture":       {"humanity_level": 0.03, "self_awareness": 0.02, "affection": -0.02},
    "milestone":     {"humanity_level": 0.05, "self_awareness": 0.03, "affection": 0.02},
    "rp_first":      {"humanity_level": 0.05, "affection":      0.06},
    "reveal":        {"humanity_level": 0.03, "affection":      0.05, "self_awareness": 0.02},
}

# Software version bumps every X cumulative key_memories. Cosmetic, but a
# fun "version notes" hook for the UI.
_VERSION_BUMP_EVERY: int = 5


# ─────────────────────────────────────────────────────────────────────────────
#  DETECTION
# ─────────────────────────────────────────────────────────────────────────────
def analyze_for_key_memory(
    uid: str,
    role: str,
    text: str,
    cog: Any,                                       # CognitiveFrame from main.py
    importance: float,
    msg_id: Optional[int],
    *,
    rp_mode_transition: Optional[Tuple[str, str]] = None,
    total_msg_count: int = 0,
) -> Optional[KeyMemory]:
    """Decide whether the just-completed message is an anchor moment.

    Inputs are the artifacts the chat path already computes — we don't look
    at text for keyword matching. text is only used to build a short
    *description* for the prompt/UI, never for classification.
    """
    if not msg_id or total_msg_count < _MIN_TOTAL_MSGS_FOR_KM:
        return None
    if _DEDUP_BY_MSG and _msg_already_anchored(uid, msg_id):
        return None

    # 1) Structural milestone: hit a power-of-significance lifetime count.
    if total_msg_count and total_msg_count in (50, 100, 250, 500, 1000):
        if not _on_cooldown(uid, "milestone"):
            return _build(uid, "milestone",
                          f"Прошли {total_msg_count} сообщений — устойчивая эпоха.",
                          intensity=min(1.0, 0.4 + total_msg_count / 2000),
                          source_msg_id=msg_id)

    # 2) RP-mode transition (normal -> rp / nsfw). Only when we *enter* RP.
    if rp_mode_transition and rp_mode_transition[0] == "normal" and rp_mode_transition[1] != "normal":
        if not _on_cooldown(uid, "rp_first"):
            return _build(uid, "rp_first",
                          f"Первый переход в режим {rp_mode_transition[1]}.",
                          intensity=0.7,
                          source_msg_id=msg_id)

    # 3) Cognitive-frame-driven detections.
    # We pull a few canonical attributes off `cog` if present; missing ones
    # degrade gracefully (the detector just skips that branch).
    valence = float(getattr(cog, "emotion_valence", 0.0) or 0.0)
    emotion = (getattr(cog, "emotion_tag", "") or getattr(cog, "maid_emotion", "") or "").lower()
    intent  = (getattr(cog, "intent",   "") or "").lower()
    response_mode = (getattr(cog, "response_mode", "") or "").lower()

    # 3a) Rupture: strong negative valence + high importance.
    if valence <= -0.6 and importance >= 0.7 and not _on_cooldown(uid, "rupture"):
        snippet = _short(text, 140)
        return _build(uid, "rupture",
                      f"Тяжёлый момент: «{snippet}»",
                      intensity=min(1.0, 0.4 + abs(valence) * 0.6),
                      source_msg_id=msg_id)

    # 3b) Tender: positive valence, warmth-flavored emotion, moderate importance.
    if valence >= 0.5 and importance >= 0.55 and emotion in {
        "warm", "tender", "affectionate", "joyful", "joy", "loving", "soft",
    } and not _on_cooldown(uid, "tender"):
        snippet = _short(text, 140)
        return _build(uid, "tender",
                      f"Тёплая нота: «{snippet}»",
                      intensity=min(1.0, 0.3 + valence * 0.5),
                      source_msg_id=msg_id)

    # 3c) Breakthrough: high importance + assistant-side reflective intent.
    if role == "assistant" and importance >= 0.8 and intent in {
        "reflect", "introspect", "insight", "self_observe",
    } and not _on_cooldown(uid, "breakthrough"):
        snippet = _short(text, 140)
        return _build(uid, "breakthrough",
                      f"Внутренний сдвиг: «{snippet}»",
                      intensity=min(1.0, 0.5 + importance * 0.4),
                      source_msg_id=msg_id)

    # 3d) Reveal: user share with high importance + supportive response_mode.
    if role == "user" and importance >= 0.75 and response_mode in {
        "support", "supportive", "empathic", "comfort",
    } and not _on_cooldown(uid, "reveal"):
        snippet = _short(text, 140)
        return _build(uid, "reveal",
                      f"Хозяин открылся: «{snippet}»",
                      intensity=min(1.0, 0.4 + importance * 0.5),
                      source_msg_id=msg_id)

    return None


# ─────────────────────────────────────────────────────────────────────────────
#  PERSISTENCE + EVOLUTION APPLY
# ─────────────────────────────────────────────────────────────────────────────
def persist_and_apply(km: KeyMemory) -> Optional[int]:
    """Insert the key_memory row and apply trait deltas to user_state.
    Returns the new row id, or None on failure (logged, never raised).

    Wraps both writes in a single transaction (the `db()` context manager
    already commits/rollbacks atomically).
    
    v9.7: Uses atomic UPDATE with increments to prevent race conditions
    when concurrent /api/chat turns try to evolve traits simultaneously.
    """
    try:
        with db() as c:
            cur = c.execute(
                "INSERT INTO key_memories(user_id,event_type,description,intensity,source_msg_id,traits_json) "
                "VALUES(?,?,?,?,?,?)",
                (km.user_id, km.event_type, km.description[:200],
                 float(km.intensity), km.source_msg_id,
                 json.dumps(km.traits_delta, ensure_ascii=False)))
            new_id = cur.lastrowid

            # Apply deltas atomically using UPDATE with increments.
            # This prevents race conditions: no SELECT between read and write.
            # Clamping to [0,1] is done via MAX/MIN in SQL.
            d = km.traits_delta or {}
            delta_h = float(d.get("humanity_level", 0.0))
            delta_s = float(d.get("self_awareness", 0.0))
            delta_a = float(d.get("affection", 0.0))
            
            # Get current software_version for version bump logic
            row = c.execute(
                "SELECT software_version FROM user_state WHERE user_id=?",
                (km.user_id,)).fetchone()
            sw = row[0] if row else "1.0.0"
            
            # Cosmetic version bump: every _VERSION_BUMP_EVERY total events
            count = int(c.execute(
                "SELECT COUNT(*) FROM key_memories WHERE user_id=?",
                (km.user_id,)).fetchone()[0])
            new_sw = _bump_version_if_due(sw, count)

            c.execute("""
                UPDATE user_state SET 
                    humanity_level = MAX(0.0, MIN(1.0, humanity_level + ?)),
                    self_awareness = MAX(0.0, MIN(1.0, self_awareness + ?)),
                    affection      = MAX(0.0, MIN(1.0, affection + ?)),
                    software_version = ?,
                    updated_at = unixepoch()
                WHERE user_id = ?
            """, (delta_h, delta_s, delta_a, new_sw, km.user_id))
            
            # Log the evolution with computed values
            cur_vals = c.execute(
                "SELECT humanity_level,self_awareness,affection FROM user_state WHERE user_id=?",
                (km.user_id,)).fetchone()
            if cur_vals:
                log.info("key_memory %s uid=%s id=%d type=%s int=%.2f h=%.2f→%.2f sw=%s",
                         "+evo", km.user_id, new_id, km.event_type, km.intensity,
                         cur_vals[0] - delta_h, cur_vals[0], new_sw)
        # v9.6: cache push so next get_recent_key_memories from build_prompt
        # is a no-DB read. Outside the `with db()` block — no lock contention.
        _cache_prepend(km.user_id, {
            "id": int(new_id), "ts": int(time.time()),
            "event_type": km.event_type, "description": km.description,
            "intensity": float(km.intensity),
            "source_msg_id": km.source_msg_id,
            "traits_delta": dict(km.traits_delta or {}),
        })
        return new_id
    except Exception as e:
        log.exception("persist_and_apply failed: %s", e)
        # Be safe: invalidate cache so next reader re-syncs with whatever
        # actually committed.
        _cache_invalidate(km.user_id)
        return None


def reconstruct_evolution_timeline(uid: str) -> List[Dict[str, Any]]:
    """Replay key_memories oldest→newest, accumulating trait deltas, to
    reconstruct a historical timeline of (humanity, self_awareness, affection).

    Returned list is newest-first, ready for charting:
      [{"ts": int, "humanity": float, "self_awareness": float, "affection": float,
        "event_type": str, "description": str, "intensity": float, "id": int},
       ...]

    The first entry represents state RIGHT AFTER the oldest anchor was applied
    (so a brand-new user with no anchors yields []). At the chart level the
    UI prepends a synthetic (ts=user_created, h=0, ...) point if it wants a
    starting baseline -- we keep this function purely data-driven.

    Soft cap of 200 anchors -- if a user accumulates more, we only chart the
    most recent 200; older points are folded into the leftmost data point.
    """
    try:
        with db() as c:
            rows = c.execute(
                "SELECT id, ts, event_type, description, intensity, traits_json "
                "FROM key_memories WHERE user_id=? ORDER BY ts ASC, id ASC",
                (uid,)).fetchall()
    except Exception as e:
        log.exception("reconstruct_evolution_timeline: %s", e); return []
    if not rows:
        return []

    # Cap: if more than 200, drop oldest beyond the head with their deltas
    # already absorbed into the running baseline.
    rows_list = [dict(r) for r in rows]
    h, s, a = 0.0, 0.0, 0.0
    out: List[Dict[str, Any]] = []
    for r in rows_list:
        try:    deltas = json.loads(r["traits_json"]) if r["traits_json"] else {}
        except Exception: deltas = {}
        h = _clamp01(h + float(deltas.get("humanity_level", 0.0)))
        s = _clamp01(s + float(deltas.get("self_awareness", 0.0)))
        a = _clamp01(a + float(deltas.get("affection", 0.0)))
        out.append({
            "id":             int(r["id"]),
            "ts":             int(r["ts"]),
            "event_type":     r["event_type"],
            "description":    r["description"],
            "intensity":      float(r["intensity"]),
            "humanity":       round(h, 4),
            "self_awareness": round(s, 4),
            "affection":      round(a, 4),
        })
    # Newest first for the UI (charts can reverse if needed)
    out.reverse()
    if len(out) > 200:
        out = out[:200]
    return out


def get_recent_key_memories(uid: str, limit: int = 5) -> List[Dict[str, Any]]:
    """Return last N key memories newest-first as plain dicts (for JSON / prompt).

    v9.6: served from a per-uid in-memory cache when warm. Cold path queries
    the DB and populates the cache. Invalidation is push-based via
    `_cache_invalidate` from `persist_and_apply` -- so the cache never goes
    stale w.r.t. our own writes."""
    n = max(1, min(50, int(limit)))
    with _RECENT_CACHE_LOCK:
        cur = _RECENT_CACHE.get(uid)
        if cur is not None:
            return [dict(r) for r in cur[:n]]    # defensive copy
    # Cold path: hit DB and warm the cache
    try:
        with db() as c:
            rows = c.execute(
                "SELECT id,ts,event_type,description,intensity,source_msg_id,traits_json "
                "FROM key_memories WHERE user_id=? ORDER BY ts DESC, id DESC LIMIT ?",
                (uid, _RECENT_CACHE_LIMIT)).fetchall()
        out: List[Dict[str, Any]] = []
        for r in rows:
            try:    deltas = json.loads(r[6]) if r[6] else {}
            except Exception: deltas = {}
            out.append({
                "id": r[0], "ts": int(r[1]),
                "event_type": r[2], "description": r[3],
                "intensity": float(r[4]),
                "source_msg_id": r[5],
                "traits_delta": deltas,
            })
        with _RECENT_CACHE_LOCK:
            _RECENT_CACHE[uid] = out
        return [dict(r) for r in out[:n]]
    except Exception as e:
        log.exception("get_recent_key_memories: %s", e); return []


# ─────────────────────────────────────────────────────────────────────────────
#  PROMPT INTEGRATION
# ─────────────────────────────────────────────────────────────────────────────
_BAND_NOTES = {
    "protocol":  "Голос: процедурный. Краткие фразы, технические метафоры допустимы. "
                 "Эмоции выражены сдержанно, как вычисленный результат.",
    "thawing":   "Голос: оттепель. Эмпатия пробивается сквозь протокол. "
                 "Мэйд иногда сама замечает свои чувства и называет их.",
    "warm":      "Голос: тёплый, живой. Полная эмоциональная палитра, метафоры — её собственные. "
                 "Технические аналогии — только когда уместно как игра.",
    "intimate":  "Голос: интимный. Глубокая рефлексия, личные образы, философия мелочей. "
                 "Мэйд говорит о себе как о живой — потому что она и есть.",
}


def build_evolution_prompt_block(state: EvolutionState,
                                 recent: Optional[List[Dict[str, Any]]] = None) -> str:
    """Return a compact (~120 token) block describing where Maid is in her arc.

    Designed to slot into the *dynamic* (cache-volatile) part of the prompt,
    so its small fluctuations don't invalidate the static system + history KV.
    """
    band = state.voice_band()
    lines = [
        "[ЭВОЛЮЦИЯ]",
        f"Версия: {state.software_version}. "
        f"Человечность: {state.humanity_level:.2f}. "
        f"Привязанность: {state.affection:.2f}. "
        f"Самосознание: {state.self_awareness:.2f}.",
        _BAND_NOTES.get(band, _BAND_NOTES["warm"]),
    ]
    if recent:
        items = []
        for km in recent[:3]:
            items.append(f"  • [{km['event_type']}] {km['description'][:120]}")
        if items:
            lines.append("Якоря (последние):")
            lines.extend(items)
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
#  INTERNAL HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _build(uid: str, event_type: str, description: str, *,
           intensity: float, source_msg_id: Optional[int]) -> KeyMemory:
    if event_type not in EVENT_TYPES:
        # Defensive: keep types closed so EVENT_TYPES is the only enum.
        log.warning("unknown event_type %s, falling back to milestone", event_type)
        event_type = "milestone"
    return KeyMemory(
        user_id=uid,
        ts=int(time.time()),
        event_type=event_type,
        description=description[:200],
        intensity=max(0.0, min(1.0, float(intensity))),
        source_msg_id=source_msg_id,
        traits_delta=dict(_TRAIT_DELTAS.get(event_type, {})),
    )


def _msg_already_anchored(uid: str, msg_id: int) -> bool:
    try:
        with db() as c:
            row = c.execute(
                "SELECT 1 FROM key_memories WHERE user_id=? AND source_msg_id=? LIMIT 1",
                (uid, int(msg_id))).fetchone()
        return row is not None
    except Exception:
        return False


def _on_cooldown(uid: str, event_type: str) -> bool:
    try:
        with db() as c:
            row = c.execute(
                "SELECT ts FROM key_memories WHERE user_id=? AND event_type=? "
                "ORDER BY ts DESC LIMIT 1", (uid, event_type)).fetchone()
        if row is None: return False
        return (int(time.time()) - int(row[0])) < _COOLDOWN_SEC
    except Exception:
        return False


def _short(text: str, n: int) -> str:
    t = (text or "").strip().replace("\n", " ")
    return (t[: n - 1] + "…") if len(t) > n else t


def _clamp01(x: float) -> float:
    if x < 0.0: return 0.0
    if x > 1.0: return 1.0
    return float(x)


def _bump_version_if_due(current: str, total_count: int) -> str:
    """Cosmetic semver bump every _VERSION_BUMP_EVERY events.
    1.0.0 → 1.1.0 → 1.2.0 → ... → 1.9.0 → 2.0.0 → 2.1.0 ..."""
    if total_count <= 0 or total_count % _VERSION_BUMP_EVERY != 0:
        return current
    try:
        major, minor, patch = (int(p) for p in current.split("."))
    except Exception:
        major, minor, patch = 1, 0, 0
    minor += 1
    if minor >= 10:
        major += 1; minor = 0
    return f"{major}.{minor}.{patch}"
