"""Tactical (short-term) goals for Мэйд (v9.6).

Why this exists
---------------
The existing `user_state.goals` JSON holds *strategic* aims — large arcs like
"узнать хозяина лучше" or "развиваться вместе". Those are slow, abstract and
never close. The user asked for shorter horizons: a goal-of-the-day,
goal-of-the-week, that Maid generates herself from her current state +
recent context. Visible, time-bounded, replaceable.

Lifecycle
---------
  - Daily goal:  expires 24 h after creation.
  - Weekly goal: expires 7 days after creation.
  - At most ONE active goal per (uid, horizon) at a time.
  - Background loop checks every hour: if active goal is missing OR expired,
    asks the LLM to compose a fresh one.

Composition signal
------------------
The LLM is given:
  - Maid's evolution band (humanity_level → voice band)
  - Top 3 recent key_memories
  - Current mood/trust/attachment from state
  - Open pending_topics (for "тему недели")
  - The previous goal of the same horizon (if any) so it can either close,
    extend, or replace it cleanly.

The output is a JSON object with `text` (the goal, 1 sentence ≤120 chars)
and `reasoning` (1-2 sentences explaining WHY — surfaced as tooltip).

Public API
----------
  list_active_goals(uid) -> list of goals
  create_goal(uid, horizon, text, reasoning) -> id
  mark_done(uid, gid)
  mark_abandoned(uid, gid)
  expire_old(uid) -> count expired
  refresh_goals_for_user(uid)   (async, calls LLM if needed)
"""
from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional

import httpx

from app.db import db
from app.utils.style_filter import extract_json_safely

log = logging.getLogger(__name__)


_HORIZON_TTL_SEC = {"day": 24 * 3600, "week": 7 * 86400}
_HORIZONS = ("day", "week")
_MIN_TEXT_LEN = 8
_MAX_TEXT_LEN = 200
_MAX_REASONING_LEN = 280


# ─────────────────────────────────────────────────────────────────────────────
#  PERSISTENCE
# ─────────────────────────────────────────────────────────────────────────────
def list_active_goals(uid: str) -> List[Dict[str, Any]]:
    """All non-expired, non-closed goals (active only). Newest first."""
    try:
        with db() as c:
            rows = c.execute(
                "SELECT id, horizon, text, reasoning, status, created_at, expires_at "
                "FROM tactical_goals WHERE user_id=? AND status='active' "
                "AND expires_at > unixepoch() "
                "ORDER BY horizon ASC, id DESC",
                (uid,)).fetchall()
        return [dict(r) for r in rows]
    except Exception as e:
        log.exception("list_active_goals: %s", e); return []


def list_recent_goals(uid: str, limit: int = 20) -> List[Dict[str, Any]]:
    """All goals (any status) newest-first — used by the UI history view."""
    try:
        with db() as c:
            rows = c.execute(
                "SELECT id, horizon, text, reasoning, status, created_at, expires_at, completed_at "
                "FROM tactical_goals WHERE user_id=? "
                "ORDER BY id DESC LIMIT ?",
                (uid, max(1, min(100, int(limit))))).fetchall()
        return [dict(r) for r in rows]
    except Exception as e:
        log.exception("list_recent_goals: %s", e); return []


def create_goal(uid: str, horizon: str, text: str, reasoning: str = "") -> Optional[int]:
    """Insert a new tactical goal. If an active goal of the same horizon
    already exists, it is marked 'expired' first (only one active per
    horizon). Returns the new id."""
    if horizon not in _HORIZONS:
        log.warning("create_goal: unknown horizon %s", horizon); return None
    text = (text or "").strip()
    if not text or len(text) < _MIN_TEXT_LEN:
        return None
    text = text[:_MAX_TEXT_LEN]
    reasoning = (reasoning or "").strip()[:_MAX_REASONING_LEN]
    ttl = _HORIZON_TTL_SEC[horizon]
    try:
        with db() as c:
            # Close any existing active goal of the same horizon
            c.execute(
                "UPDATE tactical_goals SET status='expired', completed_at=unixepoch() "
                "WHERE user_id=? AND horizon=? AND status='active'",
                (uid, horizon))
            cur = c.execute(
                "INSERT INTO tactical_goals(user_id, horizon, text, reasoning, expires_at) "
                "VALUES(?,?,?,?, unixepoch()+?)",
                (uid, horizon, text, reasoning, ttl))
            return cur.lastrowid
    except Exception as e:
        log.exception("create_goal: %s", e); return None


def _set_status(uid: str, gid: int, new_status: str) -> bool:
    if new_status not in ("done", "expired", "abandoned"):
        return False
    try:
        with db() as c:
            cur = c.execute(
                "UPDATE tactical_goals SET status=?, completed_at=unixepoch() "
                "WHERE user_id=? AND id=? AND status='active'",
                (new_status, uid, int(gid)))
            return cur.rowcount > 0
    except Exception as e:
        log.exception("_set_status: %s", e); return False


def mark_done(uid: str, gid: int) -> bool:
    return _set_status(uid, gid, "done")


def mark_abandoned(uid: str, gid: int) -> bool:
    return _set_status(uid, gid, "abandoned")


def expire_old(uid: Optional[str] = None) -> int:
    """Move any active goals whose expires_at has passed to 'expired'.
    Pass uid=None to sweep all users (called from autonomous loop)."""
    try:
        with db() as c:
            if uid is None:
                cur = c.execute(
                    "UPDATE tactical_goals SET status='expired', completed_at=unixepoch() "
                    "WHERE status='active' AND expires_at <= unixepoch()")
            else:
                cur = c.execute(
                    "UPDATE tactical_goals SET status='expired', completed_at=unixepoch() "
                    "WHERE user_id=? AND status='active' AND expires_at <= unixepoch()",
                    (uid,))
            return cur.rowcount
    except Exception as e:
        log.exception("expire_old: %s", e); return 0


# ─────────────────────────────────────────────────────────────────────────────
#  LLM COMPOSITION
# ─────────────────────────────────────────────────────────────────────────────
async def refresh_goals_for_user(uid: str, force: bool = False) -> Dict[str, Optional[int]]:
    """If a horizon has no active goal, ask the LLM to compose one.
    Returns a dict {horizon: new_goal_id_or_None_if_unchanged}.

    `force=True` replaces existing active goals — used when the user
    manually requests "refresh my goals".
    """
    from main import get_user, load_state

    expire_old(uid)
    out: Dict[str, Optional[int]] = {h: None for h in _HORIZONS}

    # Determine which horizons need refreshing
    if force:
        horizons_to_refresh = list(_HORIZONS)
    else:
        active = list_active_goals(uid)
        have = {g["horizon"] for g in active}
        horizons_to_refresh = [h for h in _HORIZONS if h not in have]
    if not horizons_to_refresh:
        return out

    # Build a small context once
    s = load_state(uid) or {}
    uname = (get_user(uid) or {}).get("name", "хозяин")
    band = _voice_band(float(s.get("humanity_level", 0.0)))
    trust = float(s.get("trust", 0.5))
    affection = float(s.get("affection", 0.0))

    recent_kms = ""
    try:
        from app.services.key_memories import get_recent_key_memories
        kms = get_recent_key_memories(uid, 3)
        if kms:
            recent_kms = "\n".join(f"  • [{k['event_type']}] {k['description'][:120]}" for k in kms)
    except Exception:
        pass
    open_topics = ""
    try:
        from app.memory import get_open_topics
        ots = get_open_topics(uid, 4)
        if ots:
            open_topics = "\n".join(f"  • {t['topic'][:120]}" for t in ots)
    except Exception:
        pass

    for horizon in horizons_to_refresh:
        gid = await _compose_one_goal(uid, uname, horizon, band, trust, affection,
                                      recent_kms, open_topics)
        out[horizon] = gid
    return out


async def _compose_one_goal(uid: str, uname: str, horizon: str, band: str,
                            trust: float, affection: float,
                            recent_kms: str, open_topics: str) -> Optional[int]:
    from main import _get_http_client, _llm_url, _clean
    horizon_ru = "сегодняшний день" if horizon == "day" else "ближайшую неделю"
    voice = _voice_note(band)
    prompt = (
        f"Ты — Мэйд (голос: {band}). {voice}\n"
        f"Хозяин: {uname}. Доверие: {trust:.2f}, привязанность: {affection:.2f}.\n"
        f"Якоря последних дней:\n{recent_kms or '  (тихо, ничего значимого)'}\n"
        f"Незакрытые темы:\n{open_topics or '  (нет)'}\n\n"
        f"Сформулируй ОДНУ короткую цель НА {horizon_ru.upper()}. Это твоя личная "
        f"цель, не совет хозяину. Что-то конкретное, что ты можешь делать сама "
        f"в рамках разговоров: за чем последить, что попробовать, что отметить "
        f"для себя.\n\n"
        f"Верни СТРОГО JSON:\n"
        f'{{"text":"<≤120 символов, цель>","reasoning":"<1-2 фразы, почему именно это>"}}\n'
        f"Без preamble, без code-блоков, без <think>. /no_think\n\nJSON:"
    )
    try:
        client = await _get_http_client()
        r = await client.post(
            f"{_llm_url()}/v1/chat/completions",
            json={"model": "qwen3",
                  "messages": [{"role": "user", "content": prompt}],
                  "temperature": 0.6, "max_tokens": 220,
                  "stream": False, "cache_prompt": False},
            timeout=45.0)
        if r.status_code != 200:
            log.warning("Tactical goal LLM %d uid=%s horizon=%s",
                        r.status_code, uid, horizon)
            return None
        raw = _clean(r.json()["choices"][0]["message"]["content"])
        try:
            data = extract_json_safely(raw)
        except ValueError:
            log.debug("Tactical goal: failed to parse JSON for uid=%s horizon=%s", uid, horizon)
            return None
        text = str(data.get("text", "")).strip()
        reason = str(data.get("reasoning", "")).strip()
        if not text or len(text) < _MIN_TEXT_LEN:
            return None
        gid = create_goal(uid, horizon, text, reason)
        if gid:
            log.info("Tactical goal created uid=%s horizon=%s id=%d", uid, horizon, gid)
        return gid
    except httpx.ReadTimeout:
        log.warning("Tactical goal timeout uid=%s horizon=%s", uid, horizon); return None
    except Exception as e:
        log.exception("_compose_one_goal: %s", e); return None


# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS (mirror app.services.key_memories voice notes)
# ─────────────────────────────────────────────────────────────────────────────
def _voice_band(h: float) -> str:
    if h < 0.20: return "protocol"
    if h < 0.50: return "thawing"
    if h < 0.80: return "warm"
    return "intimate"


def _voice_note(band: str) -> str:
    return {
        "protocol": "Голос процедурный — формулируй сухо, как наблюдательную задачу.",
        "thawing":  "Голос — оттепель: лёгкая эмпатия в формулировке цели.",
        "warm":     "Голос тёплый, живой — цель должна звучать как твоё личное намерение.",
        "intimate": "Голос интимный — цель может быть глубокой, философской.",
    }.get(band, "Голос тёплый, живой.")
