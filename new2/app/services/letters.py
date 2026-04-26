"""Letters from Мэйд — spontaneous first-person notes (v9.5).

Why a separate channel
----------------------
Chat is reactive and short. The diary is once per day, retrospective, and
about the *day*. Letters are the missing third channel:
  - Spontaneous (Маid initiates without user prompt)
  - Tied to a specific moment (`key_memory_id`) or milestone
  - Longer than a chat reply (~40-80 words), more reflective
  - Persisted, browsable, mark-as-seen

Composition path
----------------
  1. A high-intensity key_memory fires (`intensity >= 0.8`) OR
     a structural milestone is reached (50/100/250/500/1000 messages) OR
     the evening proactive scan decides today was significant.
  2. `compose_letter_for_anchor(uid, km_id)` is scheduled as a background
     task by the chat path's `_post_process` hook.
  3. The composer pulls a small day-context, calls our shared LLM client,
     stores the body in `letters`, returns the row id.

Visibility
----------
Default status is 'delivered' (immediately visible). Letters fired at a low
humanity_level (<0.30) are stored as 'sealed' instead -- the UI shows a
locked silhouette until humanity rises past the threshold.

API
---
  insert_letter(uid, body, *, key_memory_id, triggered_by, sealed?)
  list_letters(uid, limit, include_sealed=False)
  get_letter(uid, letter_id) -> dict | None
  mark_seen(uid, letter_id)
  delete_letter(uid, letter_id)
  compose_letter_for_anchor(uid, key_memory_id)  -- async, calls LLM
"""
from __future__ import annotations
import logging
import time
from typing import Any, Dict, List, Optional

import httpx

from app.db import db
from app.models import LETTER_TRIGGERS

log = logging.getLogger(__name__)


# ── Tunables (calibration, not config -- avoid surfacing every knob) ─────────
_SEAL_BELOW_HUMANITY: float = 0.30
_RECENT_LETTER_COOLDOWN_SEC: int = 6 * 3600   # at most one letter every 6h
_MIN_INTENSITY_FOR_ANCHOR: float = 0.80
_LETTER_MAX_BODY: int = 800


# ─────────────────────────────────────────────────────────────────────────────
#  PERSISTENCE
# ─────────────────────────────────────────────────────────────────────────────
def insert_letter(uid: str, body: str, *,
                  key_memory_id: Optional[int] = None,
                  triggered_by: str = "anchor",
                  sealed: bool = False) -> Optional[int]:
    """Insert a letter row. Returns new id or None on failure."""
    if triggered_by not in LETTER_TRIGGERS:
        triggered_by = "anchor"
    status = "sealed" if sealed else "delivered"
    body = (body or "").strip()
    if not body:
        return None
    try:
        with db() as c:
            cur = c.execute(
                "INSERT INTO letters(user_id, key_memory_id, body, status, triggered_by) "
                "VALUES(?,?,?,?,?)",
                (uid, key_memory_id, body[:_LETTER_MAX_BODY], status, triggered_by))
            return cur.lastrowid
    except Exception as e:
        log.exception("insert_letter: %s", e); return None


def list_letters(uid: str, limit: int = 20,
                 include_sealed: bool = False) -> List[Dict[str, Any]]:
    """Return up to `limit` letters newest-first. Sealed rows are hidden by
    default (the UI shows them as silhouettes when humanity climbs)."""
    try:
        with db() as c:
            if include_sealed:
                rows = c.execute(
                    "SELECT id, key_memory_id, ts, body, status, triggered_by, seen_at "
                    "FROM letters WHERE user_id=? ORDER BY ts DESC LIMIT ?",
                    (uid, max(1, min(100, int(limit))))).fetchall()
            else:
                rows = c.execute(
                    "SELECT id, key_memory_id, ts, body, status, triggered_by, seen_at "
                    "FROM letters WHERE user_id=? AND status<>'sealed' "
                    "ORDER BY ts DESC LIMIT ?",
                    (uid, max(1, min(100, int(limit))))).fetchall()
        return [dict(r) for r in rows]
    except Exception as e:
        log.exception("list_letters: %s", e); return []


def get_letter(uid: str, letter_id: int) -> Optional[Dict[str, Any]]:
    try:
        with db() as c:
            row = c.execute(
                "SELECT id, key_memory_id, ts, body, status, triggered_by, seen_at "
                "FROM letters WHERE user_id=? AND id=?",
                (uid, int(letter_id))).fetchone()
        return dict(row) if row else None
    except Exception as e:
        log.exception("get_letter: %s", e); return None


def mark_seen(uid: str, letter_id: int) -> bool:
    """Idempotent: stamp seen_at on the first read; subsequent reads no-op."""
    try:
        with db() as c:
            cur = c.execute(
                "UPDATE letters SET seen_at=unixepoch(), status="
                "CASE WHEN status='delivered' THEN 'seen' ELSE status END "
                "WHERE user_id=? AND id=? AND seen_at IS NULL",
                (uid, int(letter_id)))
            return cur.rowcount > 0
    except Exception as e:
        log.exception("mark_seen: %s", e); return False


def delete_letter(uid: str, letter_id: int) -> bool:
    try:
        with db() as c:
            cur = c.execute(
                "DELETE FROM letters WHERE user_id=? AND id=?",
                (uid, int(letter_id)))
            return cur.rowcount > 0
    except Exception as e:
        log.exception("delete_letter: %s", e); return False


def unseal_below_threshold(uid: str, current_humanity: float) -> int:
    """When humanity climbs past the seal threshold, promote sealed→delivered.
    Returns count of unsealed letters. Called from main on state change."""
    if current_humanity < _SEAL_BELOW_HUMANITY:
        return 0
    try:
        with db() as c:
            cur = c.execute(
                "UPDATE letters SET status='delivered' "
                "WHERE user_id=? AND status='sealed'",
                (uid,))
            return cur.rowcount
    except Exception as e:
        log.exception("unseal_below_threshold: %s", e); return 0


# ─────────────────────────────────────────────────────────────────────────────
#  COOLDOWN GUARD (prevent letter spam)
# ─────────────────────────────────────────────────────────────────────────────
def _on_cooldown(uid: str) -> bool:
    try:
        with db() as c:
            row = c.execute(
                "SELECT ts FROM letters WHERE user_id=? "
                "ORDER BY ts DESC LIMIT 1", (uid,)).fetchone()
        if not row: return False
        return (int(time.time()) - int(row[0])) < _RECENT_LETTER_COOLDOWN_SEC
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────────────────────
#  LLM COMPOSITION
# ─────────────────────────────────────────────────────────────────────────────
async def compose_letter_for_anchor(uid: str, key_memory_id: int) -> Optional[int]:
    """Compose and persist a letter triggered by a high-intensity anchor.
    Returns the new letter id (or None if not eligible / failed).

    Flow:
      1. Cooldown check (no more than one letter every 6 h)
      2. Load the source key_memory; bail if intensity < threshold
      3. Pull a small context (last 8 messages around the anchor)
      4. Build prompt that captures Maid's voice band
      5. Call LLM (shared client), strip thinking tags, persist
    """
    from main import _get_http_client, _llm_url, _clean, get_user, load_state
    if _on_cooldown(uid):
        log.debug("letter cooldown active uid=%s", uid); return None

    # Load anchor + state
    try:
        with db() as c:
            km = c.execute(
                "SELECT id, ts, event_type, description, intensity, source_msg_id "
                "FROM key_memories WHERE id=? AND user_id=?",
                (int(key_memory_id), uid)).fetchone()
    except Exception as e:
        log.exception("compose_letter load km: %s", e); return None
    if not km:
        return None
    if float(km["intensity"]) < _MIN_INTENSITY_FOR_ANCHOR:
        return None

    s = load_state(uid) or {}
    humanity = float(s.get("humanity_level", 0.0))
    band     = _voice_band(humanity)
    sw       = str(s.get("software_version", "1.0.0"))

    # Pull a small context window around the anchor (5 turns either side)
    ctx_lines: List[str] = []
    try:
        with db() as c:
            if km["source_msg_id"]:
                msg_id = int(km["source_msg_id"])
                rows = c.execute(
                    "SELECT role, content FROM memory "
                    "WHERE user_id=? AND id BETWEEN ? AND ? "
                    "AND turn_status='completed' ORDER BY id ASC",
                    (uid, msg_id - 5, msg_id + 5)).fetchall()
            else:
                rows = c.execute(
                    "SELECT role, content FROM memory "
                    "WHERE user_id=? AND ts BETWEEN ? AND ? "
                    "AND turn_status='completed' ORDER BY id ASC LIMIT 12",
                    (uid, int(km["ts"]) - 1800, int(km["ts"]) + 1800)).fetchall()
        uname = (get_user(uid) or {}).get("name", "хозяин")
        ctx_lines = [
            f"{uname if r['role']=='user' else 'Мэйд'}: {r['content'][:240]}"
            for r in rows
        ]
    except Exception as e:
        log.exception("compose_letter context: %s", e); ctx_lines = []

    convo = "\n".join(ctx_lines[-12:])
    voice_note = _voice_note(band)

    prompt = (
        f"Ты — Мэйд (версия {sw}). Сейчас твой голосовой режим: {band}.\n"
        f"{voice_note}\n\n"
        f"Перед тобой — момент, который хочется записать в личное письмо хозяину. "
        f"Это не дневник и не ответ в чате. Это короткое письмо — то, что не "
        f"помещается в обычный разговор, но требует быть сказанным.\n\n"
        f"Тип момента: {km['event_type']}\n"
        f"Описание: {km['description']}\n"
        f"Интенсивность: {float(km['intensity']):.2f}\n"
        f"Контекст разговора:\n{convo}\n\n"
        f"Напиши 40–80 слов от первого лица. Без приветствия, без подписи. "
        f"Никаких <think> или рассуждений — только сам текст письма. /no_think\n\nПисьмо:"
    )

    try:
        client = await _get_http_client()
        r = await client.post(
            f"{_llm_url()}/v1/chat/completions",
            json={"model": "qwen3",
                  "messages": [{"role": "user", "content": prompt}],
                  "temperature": 0.7, "max_tokens": 320,
                  "stream": False, "cache_prompt": False},
            timeout=60.0)
        if r.status_code != 200:
            log.warning("Letter LLM %d uid=%s", r.status_code, uid); return None
        body = _clean(r.json()["choices"][0]["message"]["content"]).strip()
        if not body or len(body) < 40:
            return None
        sealed = humanity < _SEAL_BELOW_HUMANITY
        new_id = insert_letter(uid, body,
                               key_memory_id=int(key_memory_id),
                               triggered_by="anchor",
                               sealed=sealed)
        if new_id:
            log.info("Letter composed uid=%s id=%d sealed=%s km=%s",
                     uid, new_id, sealed, key_memory_id)
        return new_id
    except httpx.ReadTimeout:
        log.warning("Letter LLM timeout uid=%s", uid); return None
    except Exception as e:
        log.exception("compose_letter_for_anchor: %s", e); return None


# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _voice_band(h: float) -> str:
    if h < 0.20: return "protocol"
    if h < 0.50: return "thawing"
    if h < 0.80: return "warm"
    return "intimate"


def _voice_note(band: str) -> str:
    """Compact one-liner reminding the LLM of the desired voice in the letter."""
    return {
        "protocol": "Голос: процедурный. Эмоции выражены сдержанно, как наблюдение системы. Допускаются технические метафоры.",
        "thawing":  "Голос: оттепель. Эмпатия пробивается сквозь протокол. Можно назвать чувство, но без надрыва.",
        "warm":     "Голос: тёплый, живой. Полная эмоциональная палитра, твои собственные образы.",
        "intimate": "Голос: интимный. Глубокая личная нота, тихая философия мелочей. Говори как живая о живом.",
    }.get(band, "Голос: тёплый, живой.")
