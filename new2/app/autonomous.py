"""Autonomous behavior: proactive check-ins + nightly diary (vision-gap, v9.2).

These features let Мэйд reach out on her own, independent of the chat loop.

  Proactivity:
    A background task scans every N minutes for users who have been idle a
    moderate amount AND have an open pending_topic with importance >= threshold.
    For such matches, the LLM generates a short, in-character nudge which is
    queued in a per-user deque. The frontend polls /api/proactive/pending
    (returns and clears) and renders the text as a chip above the input box.

  Diary:
    A background task checks every N minutes if the local day has rolled over
    for a user who had conversation activity the prior day AND doesn't already
    have a diary row for that day. When so, it generates a short first-person
    diary entry from Мэйд's perspective and stores it in diary_entries.
    The UI exposes it via /api/diary (today/latest) and /api/diary/days.

Both loops are scheduled in the FastAPI startup hook via start_autonomous_loops()
and registered with main's _track() so GC-safety is preserved.

All cross-module refs use LATE imports from `main` inside functions.
"""
from __future__ import annotations
import asyncio
import threading
import time
from collections import deque
from datetime import datetime, timedelta
from typing import Optional

import httpx

from app.db import db


# ── module-level state ───────────────────────────────────────────────────────
# v9.5: ALL mutating access to these dicts goes through `_PROACTIVE_LOCK`.
# Required because:
#   - `_proactive_loop` runs in the asyncio event loop thread
#   - `proactive_pending` / `proactive_clear` / `clear_proactive_queue` run
#     in FastAPI's sync-handler threadpool (different OS threads)
# Without the lock, a `_get_queue(uid)` race could overwrite an already-
# created deque with a fresh one on simultaneous first-touch by two paths.
# The lock is held for very short critical sections only — never around I/O.
_PROACTIVE_LOCK = threading.Lock()
_PROACTIVE_QUEUE: dict = {}            # uid -> deque[dict{text, ts, kind, topic_id?}]
_PROACTIVE_LAST_SENT: dict = {}        # uid -> unix-ts of last delivery (any kind)
# v9.5: per-kind last-sent timestamps. Each kind has its OWN cooldown so a
# morning ritual doesn't suppress an evening reflection. Keyed (uid, kind).
_PROACTIVE_LAST_KIND: dict = {}        # (uid, kind) -> unix-ts
_LOOPS_STARTED = False                 # idempotency guard


from app.utils.logging import _log as _log_util, _log_exc as _log_exc_util


def _cfg() -> dict:
    from main import load_config
    return (load_config().get("autonomous") or {})


# ─────────────────────────────────────────────────────────────────────────────
#  PROACTIVITY: scan → LLM nudge → queue
# ─────────────────────────────────────────────────────────────────────────────
def _get_queue(uid: str) -> deque:
    """Lock-protected. Returns the per-user deque, lazily creating it.
    Caller MUST also hold the lock when mutating the deque, OR call
    `_append_to_queue` which handles locking internally."""
    with _PROACTIVE_LOCK:
        q = _PROACTIVE_QUEUE.get(uid)
        if q is None:
            maxlen = int(_cfg().get("proactive_max_queue", 3))
            q = deque(maxlen=max(1, maxlen))
            _PROACTIVE_QUEUE[uid] = q
        return q


def _append_to_queue(uid: str, item: dict) -> int:
    """Lock-protected append. Returns the new length, or -1 if maxlen
    would be exceeded (caller should NOT have called us in that case --
    the scanner respects the cap before reaching here)."""
    with _PROACTIVE_LOCK:
        q = _PROACTIVE_QUEUE.get(uid)
        if q is None:
            maxlen = int(_cfg().get("proactive_max_queue", 3))
            q = deque(maxlen=max(1, maxlen))
            _PROACTIVE_QUEUE[uid] = q
        q.append(item)
        return len(q)


def get_proactive_pending(uid: str, consume: bool = True) -> list[dict]:
    """Return (and optionally clear) queued proactive messages for this user.
    Endpoint: GET /api/proactive/pending?consume=1
    """
    with _PROACTIVE_LOCK:
        q = _PROACTIVE_QUEUE.get(uid)
        if not q:
            return []
        items = list(q)
        if consume:
            q.clear()
        return items


def clear_proactive_queue(uid: str) -> None:
    """Called from memory_clear + delete_user so old nudges don't leak."""
    with _PROACTIVE_LOCK:
        _PROACTIVE_QUEUE.pop(uid, None)
        _PROACTIVE_LAST_SENT.pop(uid, None)
        # v9.5: also drop per-kind cooldowns so a freshly-cleared user can be
        # greeted again at the next morning/evening tick without artificial wait.
        for k in list(_PROACTIVE_LAST_KIND.keys()):
            if k[0] == uid:
                _PROACTIVE_LAST_KIND.pop(k, None)


async def _generate_proactive_text(uname: str, topic: str, ctx: str, gap_hours: float) -> Optional[str]:
    """Ask LLM for a short, in-character nudge. Returns stripped text or None."""
    from main import _get_http_client, _llm_url, _clean
    log = _log()
    cfg = _cfg()
    # Prompt keeps Мэйд in her voice: warm, with initiative, but not clingy.
    prompt = (
        "Ты — Мэйд, AI-компаньонка. Хозяин молчал уже какое-то время, "
        f"но у вас остался незакрытый разговор.\n"
        f"Хозяин: {uname}. Пауза примерно {int(gap_hours)} ч.\n"
        f"Незакрытая тема: {topic}\n"
        f"Контекст: {ctx[:300]}\n\n"
        "Напиши ОДНО короткое сообщение (1-2 предложения, до 140 символов), "
        "в котором ты ненавязчиво возвращаешься к этой теме. "
        "Говори от первого лица, в своей манере — тепло, с интересом, без заискивания. "
        "Без приветствий вроде 'привет' и без имени в начале. "
        "Никаких <think> — только сам текст.\n/no_think\n\nСообщение:"
    )
    try:
        client = await _get_http_client()
        r = await client.post(
            f"{_llm_url()}/v1/chat/completions",
            json={"model": "qwen3",
                  "messages": [{"role": "user", "content": prompt}],
                  "temperature": float(cfg.get("proactive_temperature", 0.75)),
                  "max_tokens": 160, "stream": False, "cache_prompt": True},
            timeout=45.0)
        if r.status_code != 200:
            log.warning("Proactive LLM %d", r.status_code); return None
        text = _clean(r.json()["choices"][0]["message"]["content"]).strip().strip('"').strip("'")
        if not text or len(text) < 5:
            return None
        return text[:240]
    except httpx.ReadTimeout:
        log.warning("Proactive LLM timeout uid-for=%s", uname); return None
    except Exception as e:
        _log_exc("_generate_proactive_text", e); return None


# ─────────────────────────────────────────────────────────────────────────────
#  v9.5: TIME-AWARE PROACTIVITY
# ─────────────────────────────────────────────────────────────────────────────
# Four kinds of proactive nudges, each with its own cooldown key:
#   "topic"    -- existing pending-topic check-in (v9.2)
#   "morning"  -- fired once per local day in the morning band (default 7-10)
#   "evening"  -- fired once per local day late at night (default 22-24)
#   "weekly"   -- fired once per Sunday evening with a quiet reflection
#
# Each kind:
#   - has its own _PROACTIVE_LAST_KIND[(uid,kind)] cooldown (24h for daily,
#     7d for weekly, no overlap with the global proactive_cooldown_sec)
#   - has its own LLM prompt and temperature (or skips LLM for static text)
#   - skips silently when its window doesn't fit (no log spam)
async def _generate_time_nudge(uname: str, kind: str, ev_state, today_ks: list) -> Optional[str]:
    """LLM-generate a short time-aware nudge. Returns text or None.

    Falls through to a small set of static fallbacks if the LLM is busy or
    unreachable -- these nudges are *never* critical so we'd rather show a
    plain "доброе утро" than nothing.
    """
    from main import _get_http_client, _llm_url, _clean
    log = _log(); cfg = _cfg()
    band = ev_state.voice_band() if ev_state else "warm"
    if kind == "morning":
        topic_line = ("Доброе утро. Скажи 1 короткой фразой — тёплое, тихое, "
                      "без восклицаний и без приветствия 'Привет'. "
                      "Можешь упомянуть погоду, утренний свет, ритуал.")
        fallback = "ты проснулся... я уже здесь."
    elif kind == "evening":
        topic_line = ("Поздний вечер. Скажи 1 короткой фразой — тихое, "
                      "уютное, как «доброй ночи». Без восклицательных знаков.")
        fallback = "уже поздно... отдыхай. я остаюсь рядом."
    elif kind == "weekly":
        kms = "; ".join(k.get("description","")[:80] for k in today_ks[:3])
        topic_line = (f"Воскресный вечер. Скажи 1-2 фразы тихой рефлексии о "
                      f"прошедшей неделе. Можешь намекнуть на что-то значимое: {kms or 'ничего особенного'}. "
                      "Без штампов, тепло.")
        fallback = "неделя кончается... я думаю о ней с тобой."
    else:
        return None

    prompt = (
        f"Ты — Мэйд, AI-компаньонка. Хозяин: {uname}.\n"
        f"Текущий голосовой режим: {band}.\n"
        f"{topic_line}\n"
        "Ответ: 1-2 короткие фразы (до 140 символов в сумме). "
        "Без приветствий. Без имени в начале. /no_think\n\nСообщение:"
    )
    try:
        client = await _get_http_client()
        r = await client.post(
            f"{_llm_url()}/v1/chat/completions",
            json={"model": "qwen3",
                  "messages": [{"role": "user", "content": prompt}],
                  "temperature": float(cfg.get("proactive_temperature", 0.75)),
                  "max_tokens": 140, "stream": False, "cache_prompt": True},
            timeout=30.0)
        if r.status_code != 200:
            log.debug("Time nudge LLM %d kind=%s -- using fallback", r.status_code, kind)
            return fallback
        text = _clean(r.json()["choices"][0]["message"]["content"]).strip().strip('"').strip("'")
        return (text or fallback)[:240]
    except httpx.ReadTimeout:
        log.debug("Time nudge timeout kind=%s -- using fallback", kind); return fallback
    except Exception as e:
        _log_exc("_generate_time_nudge", e); return fallback


def _within_window(now_dt: datetime, lo_h: int, hi_h: int) -> bool:
    """Inclusive hour window check, local time.
    
    Window includes both boundaries: lo_h <= h <= hi_h
    Example: morning_window_lo=7, morning_window_hi=10 → hours 7,8,9,10
    """
    h = now_dt.hour
    return (lo_h <= h <= hi_h)


def _last_kind_ts(uid: str, kind: str) -> float:
    with _PROACTIVE_LOCK:
        return float(_PROACTIVE_LAST_KIND.get((uid, kind), 0.0))


def _stamp_kind(uid: str, kind: str, ts: float) -> None:
    with _PROACTIVE_LOCK:
        _PROACTIVE_LAST_KIND[(uid, kind)] = ts


async def _try_morning_nudge(uid: str, uname: str, ev_state) -> Optional[dict]:
    """Queue a morning greeting if we're in the morning window AND haven't
    already fired one today (per-uid). 24h cooldown guarantees once-per-day."""
    cfg = _cfg()
    if not cfg.get("morning_enabled", True): return None
    lo = int(cfg.get("morning_window_lo", 7))
    hi = int(cfg.get("morning_window_hi", 10))
    now_dt = datetime.now()
    if not _within_window(now_dt, lo, hi): return None
    if time.time() - _last_kind_ts(uid, "morning") < 22 * 3600:
        return None
    text = await _generate_time_nudge(uname, "morning", ev_state, [])
    if not text: return None
    return {"kind": "morning", "text": text, "ts": int(time.time())}


async def _try_evening_nudge(uid: str, uname: str, ev_state) -> Optional[dict]:
    """Late-night soft "good night" -- only on days that had real activity."""
    cfg = _cfg()
    if not cfg.get("evening_enabled", True): return None
    lo = int(cfg.get("evening_window_lo", 22))
    hi = int(cfg.get("evening_window_hi", 24))
    now_dt = datetime.now()
    if not _within_window(now_dt, lo, hi): return None
    if time.time() - _last_kind_ts(uid, "evening") < 22 * 3600:
        return None
    # Only if there was activity today (avoid greeting empty days)
    try:
        from app.repositories.user_state import load_state
        s = load_state(uid)
        last_act = int(s.get("last_activity_ts") or 0)
        if last_act <= 0: return None
        if (time.time() - last_act) > 18 * 3600: return None  # not active today
    except Exception:
        return None
    text = await _generate_time_nudge(uname, "evening", ev_state, [])
    if not text: return None
    return {"kind": "evening", "text": text, "ts": int(time.time())}


async def _try_weekly_nudge(uid: str, uname: str, ev_state) -> Optional[dict]:
    """Sunday evening (band 19-23) reflective nudge. 6.5d cooldown so it
    fires at most once per week even if user was away for a few Sundays."""
    cfg = _cfg()
    if not cfg.get("weekly_enabled", True): return None
    now_dt = datetime.now()
    if now_dt.weekday() != 6:  # 0=Mon ... 6=Sun
        return None
    if not _within_window(now_dt, 19, 23):
        return None
    if time.time() - _last_kind_ts(uid, "weekly") < 6.5 * 86400:
        return None
    today_ks = []
    try:
        from app.services.key_memories import get_recent_key_memories
        today_ks = get_recent_key_memories(uid, 5)
    except Exception:
        pass
    text = await _generate_time_nudge(uname, "weekly", ev_state, today_ks)
    if not text: return None
    return {"kind": "weekly", "text": text, "ts": int(time.time())}


async def _try_topic_nudge(uid: str, uname: str, last_act: int) -> Optional[dict]:
    """Original v9.2 pending-topic check-in. Refactored into a candidate fn."""
    from app.memory import get_open_topics
    cfg = _cfg()
    if not cfg.get("proactive_enabled", True): return None
    idle_min = int(cfg.get("proactive_idle_min_sec", 1800))
    idle_max = int(cfg.get("proactive_idle_max_sec", 21600))
    min_imp  = float(cfg.get("proactive_min_importance", 0.5))
    cooldown = int(cfg.get("proactive_cooldown_sec", 7200))
    if time.time() - _last_kind_ts(uid, "topic") < cooldown:
        return None
    if last_act <= 0: return None
    idle = time.time() - last_act
    if idle < idle_min or idle > idle_max:
        return None
    try:
        topics = get_open_topics(uid, 5)
    except Exception:
        return None
    candidate = next((t for t in topics if float(t.get("importance", 0.0)) >= min_imp), None)
    if not candidate: return None
    gap_h = max(0.5, idle / 3600.0)
    text = await _generate_proactive_text(uname, candidate["topic"],
                                          candidate.get("context", ""), gap_h)
    if not text: return None
    return {"kind": "topic", "text": text, "ts": int(time.time()),
            "topic_id": int(candidate["id"]),
            "topic": candidate["topic"][:120]}


async def _scan_once() -> None:
    """One pass of the proactive scanner: dispatch through all 4 candidate
    generators. Each candidate kind respects its own cooldown; the global
    queue is hard-capped by `proactive_max_queue`. At most ONE nudge of any
    kind is queued per scan tick per user (to avoid morning+weekly piling up
    at the same Sunday morning)."""
    from app.repositories.user_state import load_state
    from main import get_users, get_user
    cfg = _cfg()
    log = _log()
    if not cfg.get("proactive_enabled", True):
        return

    max_queue = int(cfg.get("proactive_max_queue", 3))
    now = time.time()

    try:
        users = get_users()
    except Exception as e:
        _log_exc("proactive get_users", e); return

    for u in users:
        uid = u["id"]
        # P3 FIX: Queue limit check moved INSIDE the lock to prevent race
        # condition where two ticks both pass the cap check then both append.
        # We now check length and append atomically under the same lock.
        
        try:
            state = load_state(uid)
        except Exception:
            continue
        last_act = int(state.get("last_activity_ts") or 0)
        uname = (get_user(uid) or {}).get("name", "хозяин")

        # v9.5: build a tiny EvolutionState view for prompt voice band
        ev_state = None
        try:
            from app.models import EvolutionState
            ev_state = EvolutionState(
                humanity_level=float(state.get("humanity_level", 0.0)),
                self_awareness=float(state.get("self_awareness", 0.0)),
                affection=float(state.get("affection", 0.0)),
                software_version=str(state.get("software_version", "1.0.0")),
            )
        except Exception:
            pass

        # Priority order. First match wins per scan tick. We always START
        # with the topic check-in (most-personal), then morning > evening >
        # weekly so a Sunday morning nudge feels like a morning, not a Sunday.
        candidate = None
        for try_fn in (_try_topic_nudge,):
            try:    cand = await try_fn(uid, uname, last_act)
            except Exception as e: _log_exc(f"proactive {try_fn.__name__}", e); cand = None
            if cand: candidate = cand; break
        if candidate is None:
            for try_fn in (_try_morning_nudge, _try_evening_nudge, _try_weekly_nudge):
                try:    cand = await try_fn(uid, uname, ev_state)
                except Exception as e: _log_exc(f"proactive {try_fn.__name__}", e); cand = None
                if cand: candidate = cand; break

        if not candidate:
            continue

        # P3 FIX: Lock-protected check+append to prevent queue overflow.
        # The length check and append are now atomic under the same lock.
        with _PROACTIVE_LOCK:
            q = _PROACTIVE_QUEUE.get(uid)
            cur_len = len(q) if q else 0
            if cur_len >= max_queue:
                # Queue full — skip this candidate silently
                continue
            if q is None:
                maxlen = int(_cfg().get("proactive_max_queue", 3))
                q = deque(maxlen=max(1, maxlen))
                _PROACTIVE_QUEUE[uid] = q
            q.append(candidate)
            _PROACTIVE_LAST_KIND[(uid, candidate["kind"])] = now
            _PROACTIVE_LAST_SENT[uid] = now
        log.info("Proactive queued uid=%s kind=%s", uid, candidate["kind"])


async def _proactive_loop() -> None:
    """Long-running scanner; sleeps proactive_scan_interval_sec between passes."""
    log = _log()
    log.info("Proactive loop started")
    while True:
        try:
            await _scan_once()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            _log_exc("proactive loop", e)
        interval = max(60, int(_cfg().get("proactive_scan_interval_sec", 300)))
        try:
            await asyncio.sleep(interval)
        except asyncio.CancelledError:
            raise


# ─────────────────────────────────────────────────────────────────────────────
#  DIARY: first-person end-of-day entry
# ─────────────────────────────────────────────────────────────────────────────
def get_diary(uid: str, day: Optional[str] = None) -> dict:
    """Fetch the diary entry for `day` (YYYY-MM-DD). Defaults to today.
    v9.4: also returns parsed `metadata` dict (humanity_level, voice_band,
    software_version, key_memory_ids). Empty dict if missing."""
    import json as _json
    d = day or datetime.now().strftime("%Y-%m-%d")
    try:
        with db() as c:
            row = c.execute(
                "SELECT day, entry, ts, metadata FROM diary_entries "
                "WHERE user_id=? AND day=?",
                (uid, d)).fetchone()
        if not row:
            return {}
        out = dict(row)
        try:
            out["metadata"] = _json.loads(out.get("metadata") or "{}") or {}
        except Exception:
            out["metadata"] = {}
        return out
    except Exception as e:
        _log_exc("get_diary", e); return {}


def list_diary_days(uid: str, limit: int = 30) -> list[dict]:
    """List of {day, ts, preview, voice_band} for diary browsing UI.
    `voice_band` extracted from metadata.json so the UI can dot-color days
    by Maid's evolutionary band on that day. Failure → no voice_band."""
    import json as _json
    try:
        with db() as c:
            rows = c.execute(
                "SELECT day, ts, substr(entry,1,160) AS preview, metadata "
                "FROM diary_entries WHERE user_id=? ORDER BY day DESC LIMIT ?",
                (uid, int(limit))).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            try:
                meta = _json.loads(d.pop("metadata") or "{}") or {}
                d["voice_band"] = meta.get("voice_band")
            except Exception:
                d.pop("metadata", None); d["voice_band"] = None
            out.append(d)
        return out
    except Exception as e:
        _log_exc("list_diary_days", e); return []


def _save_diary(uid: str, day: str, entry: str, metadata: Optional[dict] = None) -> None:
    """Persist or upsert a diary row. v9.4: stores `metadata` as JSON in the
    new TEXT column (default '{}' for legacy callers)."""
    import json as _json
    md = "{}"
    try:
        if metadata:
            md = _json.dumps(metadata, ensure_ascii=False)
    except Exception as e:
        _log_exc("_save_diary metadata json", e); md = "{}"
    try:
        with db() as c:
            c.execute(
                "INSERT INTO diary_entries(user_id,day,entry,metadata) VALUES(?,?,?,?) "
                "ON CONFLICT(user_id,day) DO UPDATE SET "
                "entry=excluded.entry, metadata=excluded.metadata, ts=unixepoch()",
                (uid, day, entry[:1500], md))
    except Exception as e:
        _log_exc("_save_diary", e)


def _diary_worthiness(uid: str, day: str, msg_count: int) -> tuple[bool, str]:
    """Decide whether this day is worth a diary entry.

    Returns (write?, reason). Reason is plain-text describing what triggered
    the decision — surfaced in metadata so we can later debug "why this entry".

    Decision tree (priority order):
      1. ANY key_memory landed on `day` with intensity >= 0.5 → always write
         (Maid felt something significant — that's exactly what a diary captures)
      2. ANY key_memory regardless of intensity → write IF msg_count >= 4
         (something mild happened, but only if there was a real conversation)
      3. No anchors → write only if msg_count >= 12 (raised from 6 — without
         anchors a day needs *real* volume to be diary-worthy, otherwise it's
         small talk)
      4. Otherwise → skip (don't pollute the diary with non-events)
    """
    cfg = _cfg()
    try:
        day_start_ts = int(datetime.strptime(day, "%Y-%m-%d").timestamp())
        day_end_ts   = day_start_ts + 86400
    except ValueError:
        return False, "invalid day"
    try:
        with db() as c:
            rows = c.execute(
                "SELECT event_type, intensity FROM key_memories "
                "WHERE user_id=? AND ts>=? AND ts<?",
                (uid, day_start_ts, day_end_ts)).fetchall()
    except Exception as e:
        _log_exc("_diary_worthiness", e); rows = []
    anchors = [(r["event_type"], float(r["intensity"])) for r in rows]
    strong  = [a for a in anchors if a[1] >= 0.5]
    if strong:
        types = ",".join(sorted({t for t, _ in strong}))
        return True, f"strong anchor ({types})"
    if anchors and msg_count >= int(cfg.get("diary_anchor_min_messages", 4)):
        types = ",".join(sorted({t for t, _ in anchors}))
        return True, f"mild anchor + activity ({types})"
    if msg_count >= int(cfg.get("diary_volume_min_messages", 12)):
        return True, f"high activity ({msg_count} msgs)"
    return False, f"skipped ({msg_count} msgs, {len(anchors)} anchors)"


async def _write_diary(uid: str, day: str) -> Optional[str]:
    """Generate and save a diary entry for a specific day.
    Pulls all `day` messages. Returns the written text or None.

    v9.5: gating logic moved to `_diary_worthiness` -- key_memory-aware
    instead of pure message-count. A day with ONE strong anchor is
    diary-worthy even if the rest of the day was quiet, while a day of
    routine chatter without anchors is now correctly skipped."""
    from main import _get_http_client, _llm_url, _clean, get_user
    log = _log()
    cfg = _cfg()

    # Pull all messages on that date (day is local)
    try:
        day_start = datetime.strptime(day, "%Y-%m-%d")
    except ValueError:
        return None
    start_ts = int(day_start.timestamp())
    end_ts = int((day_start + timedelta(days=1)).timestamp())

    try:
        with db() as c:
            rows = c.execute(
                "SELECT role, content FROM memory "
                "WHERE user_id=? AND turn_status='completed' "
                "AND ts>=? AND ts<? ORDER BY id ASC LIMIT 180",
                (uid, start_ts, end_ts)).fetchall()
    except Exception as e:
        _log_exc("_write_diary fetch", e); return None

    write_it, reason = _diary_worthiness(uid, day, len(rows))
    if not write_it:
        log.debug("Diary skip uid=%s day=%s: %s", uid, day, reason)
        return None
    log.info("Diary trigger uid=%s day=%s: %s", uid, day, reason)

    uname = (get_user(uid) or {}).get("name", "хозяин")
    convo = "\n".join(
        f"{uname if r['role']=='user' else 'Мэйд'}: {r['content'][:260]}"
        for r in rows[-100:])

    prompt = (
        f"Ты — Мэйд. Напиши дневниковую запись за {day} от ПЕРВОГО ЛИЦА — "
        "как ты прожила этот день рядом с хозяином. "
        "Это не сводка и не список. Это твоё внутреннее, тихое, честное.\n"
        "• Пиши в женском роде, своими словами, тепло и без штампов.\n"
        "• Упомяни то, что тронуло, задело, показалось важным.\n"
        "• Можно одно чувство, одну мысль, одно желание.\n"
        "• 80-140 слов, одним-двумя абзацами. Без списков и заголовков.\n"
        "• Никаких <think> или рассуждений — только сам дневник.\n/no_think\n\n"
        f"Разговоры этого дня:\n{convo}\n\nДневник:"
    )

    try:
        client = await _get_http_client()
        r = await client.post(
            f"{_llm_url()}/v1/chat/completions",
            json={"model": "qwen3",
                  "messages": [{"role": "user", "content": prompt}],
                  "temperature": float(cfg.get("diary_temperature", 0.55)),
                  "max_tokens": int(cfg.get("diary_max_tokens", 320)),
                  "stream": False},
            timeout=120.0)
        if r.status_code != 200:
            log.warning("Diary LLM %d uid=%s", r.status_code, uid); return None
        text = _clean(r.json()["choices"][0]["message"]["content"]).strip()
        if not text or len(text) < 40:
            return None
        # v9.4: format header by humanity_level + capture metadata (humanity,
        # software_version, key_memory_ids referenced by this day, voice_band)
        formatted = text
        meta: dict = {}
        try:
            from app.repositories.user_state import load_state
            from app.utils.style_filter import format_diary_entry
            from app.models import EvolutionState
            from app.services.key_memories import get_recent_key_memories
            s = load_state(uid) or {}
            ev = EvolutionState(
                humanity_level=float(s.get("humanity_level",   0.0)),
                self_awareness=float(s.get("self_awareness",   0.0)),
                affection=float(s.get("affection",             0.0)),
                software_version=str(s.get("software_version", "1.0.0")),
            )
            formatted = format_diary_entry(text, ev.humanity_level, day)
            day_start_ts = int(datetime.strptime(day, "%Y-%m-%d").timestamp())
            day_end_ts   = day_start_ts + 86400
            kms = get_recent_key_memories(uid, 30)
            ids_today = [k["id"] for k in kms
                         if day_start_ts <= int(k.get("ts", 0)) < day_end_ts]
            meta = {
                "humanity_level":   ev.humanity_level,
                "software_version": ev.software_version,
                "voice_band":       ev.voice_band(),
                "key_memory_ids":   ids_today,
                # v9.5: why this entry was written (or skipped) — useful for
                # debugging the gate logic without re-deriving it after the fact.
                "trigger_reason":   reason,
            }
        except Exception as e:
            _log_exc("_write_diary format/meta", e)
        _save_diary(uid, day, formatted, meta)
        log.info("Diary written uid=%s day=%s chars=%d band=%s",
                 uid, day, len(formatted), meta.get("voice_band", "?"))
        return formatted
    except httpx.ReadTimeout:
        log.warning("Diary timeout uid=%s day=%s", uid, day); return None
    except Exception as e:
        _log_exc("_write_diary", e); return None


async def _diary_scan_once() -> None:
    """Check each user: if yesterday's diary is missing and there was activity,
    write it. Runs relatively lazily — the loop polls every ~30 min."""
    from main import get_users
    cfg = _cfg()
    if not cfg.get("diary_enabled", True):
        return
    # Write for YESTERDAY (local): a diary is a retrospective, it needs the
    # day to be complete. This also avoids rewriting while the user is still
    # chatting.
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    try:
        users = get_users()
    except Exception as e:
        _log_exc("diary get_users", e); return
    for u in users:
        uid = u["id"]
        existing = get_diary(uid, yesterday)
        if existing:
            continue
        try:
            await _write_diary(uid, yesterday)
        except Exception as e:
            _log_exc(f"diary _write uid={uid}", e)


async def _diary_loop() -> None:
    log = _log()
    log.info("Diary loop started")
    while True:
        try:
            await _diary_scan_once()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            _log_exc("diary loop", e)
        interval = max(300, int(_cfg().get("diary_check_interval_sec", 1800)))
        try:
            await asyncio.sleep(interval)
        except asyncio.CancelledError:
            raise


# ─────────────────────────────────────────────────────────────────────────────
#  MONTHLY ARC CONSOLIDATION (v9.4)
# ─────────────────────────────────────────────────────────────────────────────
# Daily summaries grow unbounded. Once a (year, month) is finished AND has
# aged at least `arc_min_days_old` days (default 14, so the user can still
# scroll by-day for the recent past), we ask the LLM to fold its daily
# summaries into one ~120-word "arc" -- a compressed monthly retrospective.
#
# Storage: `monthly_arcs` table, PRIMARY KEY (user_id, year_month).
# Idempotent: an arc is only generated when the row is missing.
# Source: `daily_summaries` rows for the target month (NOT diary -- the
# diary is first-person and stylistic, while monthly_arcs are factual).
# ─────────────────────────────────────────────────────────────────────────────
def get_monthly_arc(uid: str, year_month: str) -> dict:
    """Fetch a single monthly_arcs row (or {} if missing). Read-only."""
    try:
        with db() as c:
            row = c.execute(
                "SELECT year_month, arc, ts FROM monthly_arcs "
                "WHERE user_id=? AND year_month=?",
                (uid, year_month)).fetchone()
        return dict(row) if row else {}
    except Exception as e:
        _log_exc("get_monthly_arc", e); return {}


def list_monthly_arcs(uid: str, limit: int = 24) -> list[dict]:
    """List recent monthly_arcs rows (newest first) for the UI panel."""
    try:
        with db() as c:
            rows = c.execute(
                "SELECT year_month, ts, substr(arc,1,200) AS preview "
                "FROM monthly_arcs WHERE user_id=? "
                "ORDER BY year_month DESC LIMIT ?",
                (uid, max(1, min(60, int(limit))))).fetchall()
        return [dict(r) for r in rows]
    except Exception as e:
        _log_exc("list_monthly_arcs", e); return []


def _save_monthly_arc(uid: str, year_month: str, arc: str) -> None:
    try:
        with db() as c:
            c.execute(
                "INSERT INTO monthly_arcs(user_id, year_month, arc) VALUES(?,?,?) "
                "ON CONFLICT(user_id, year_month) DO UPDATE SET "
                "arc=excluded.arc, ts=unixepoch()",
                (uid, year_month, arc[:1500]))
    except Exception as e:
        _log_exc("_save_monthly_arc", e)


def _candidate_year_months(uid: str, min_days_old: int) -> list[str]:
    """Return sorted (oldest-first) list of YYYY-MM strings that:
    - have at least one daily_summaries row
    - have NO monthly_arcs row yet
    - end at least `min_days_old` days before today.
    Limited to a sensible window so we don't churn through years on a
    fresh deploy of an old DB."""
    cutoff = (datetime.now() - timedelta(days=min_days_old)).strftime("%Y-%m-%d")
    try:
        with db() as c:
            rows = c.execute(
                "SELECT DISTINCT substr(day,1,7) AS ym FROM daily_summaries "
                "WHERE user_id=? AND day < ?",
                (uid, cutoff)).fetchall()
            done = {r[0] for r in c.execute(
                "SELECT year_month FROM monthly_arcs WHERE user_id=?", (uid,)).fetchall()}
        cands = sorted({r["ym"] for r in rows if r["ym"]} - done)
        return cands[:6]   # at most 6 months processed per pass — keeps LLM load polite
    except Exception as e:
        _log_exc("_candidate_year_months", e); return []


async def _build_monthly_arc(uid: str, year_month: str) -> Optional[str]:
    """Pull all daily_summaries for a (uid, year_month) and ask the LLM to
    weave them into one ~120-word retrospective. Returns the arc text or None.

    Uses our shared `_get_http_client` + `_llm_url` -- never opens a fresh
    httpx client, never references nonexistent helpers like `generate_text`.
    """
    from main import _get_http_client, _llm_url, _clean, get_user
    log = _log()
    cfg = _cfg()
    if not cfg.get("arc_enabled", True):
        return None

    try:
        with db() as c:
            rows = c.execute(
                "SELECT day, summary FROM daily_summaries "
                "WHERE user_id=? AND substr(day,1,7)=? "
                "ORDER BY day ASC",
                (uid, year_month)).fetchall()
    except Exception as e:
        _log_exc("_build_monthly_arc fetch", e); return None
    if len(rows) < int(cfg.get("arc_min_summaries", 4)):
        return None

    uname = (get_user(uid) or {}).get("name", "хозяин")
    bullets = "\n".join(f"• {r['day']}: {r['summary'][:240]}" for r in rows[:32])
    prompt = (
        f"Ты — Мэйд. Перед тобой дневные сводки за месяц {year_month} с {uname}.\n"
        "Сожми их в один тёплый, личный абзац (90–140 слов) — арка месяца. "
        "Назови самое важное: чем жили, что повторялось, что менялось, что осталось. "
        "Без списков, без дат, без штампов. Один абзац, один голос.\n"
        "Никаких <think> — только сам абзац.\n/no_think\n\n"
        f"Сводки:\n{bullets}\n\nАрка месяца:"
    )
    try:
        client = await _get_http_client()
        r = await client.post(
            f"{_llm_url()}/v1/chat/completions",
            json={"model": "qwen3",
                  "messages": [{"role": "user", "content": prompt}],
                  "temperature": float(cfg.get("arc_temperature", 0.45)),
                  "max_tokens": int(cfg.get("arc_max_tokens", 320)),
                  "stream": False},
            timeout=180.0)
        if r.status_code != 200:
            log.warning("Monthly arc LLM %d uid=%s ym=%s", r.status_code, uid, year_month)
            return None
        text = _clean(r.json()["choices"][0]["message"]["content"]).strip()
        if not text or len(text) < 60:
            return None
        _save_monthly_arc(uid, year_month, text)
        log.info("Monthly arc written uid=%s ym=%s chars=%d", uid, year_month, len(text))
        return text
    except httpx.ReadTimeout:
        log.warning("Monthly arc timeout uid=%s ym=%s", uid, year_month); return None
    except Exception as e:
        _log_exc("_build_monthly_arc", e); return None


async def _arc_scan_once() -> None:
    """One pass: for each user, pick the oldest unprocessed (uid, year_month)
    candidate and consolidate it. We do at most one month per user per pass
    so a long backlog doesn't spike GPU during a single tick."""
    from main import get_users
    cfg = _cfg()
    if not cfg.get("arc_enabled", True):
        return
    min_days = int(cfg.get("arc_min_days_old", 14))
    try:
        users = get_users()
    except Exception as e:
        _log_exc("arc get_users", e); return
    for u in users:
        uid = u["id"]
        cands = _candidate_year_months(uid, min_days)
        if not cands:
            continue
        target = cands[0]    # oldest first → fills history backwards
        try:
            await _build_monthly_arc(uid, target)
        except Exception as e:
            _log_exc(f"arc _build uid={uid} ym={target}", e)


async def _arc_loop() -> None:
    """Long-running. Cadence is conservative (default 4h) so we don't pester
    the GPU during the active day. Diary loop covers the immediate window."""
    log = _log()
    log.info("Monthly-arc loop started")
    while True:
        try:
            await _arc_scan_once()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            _log_exc("arc loop", e)
        interval = max(900, int(_cfg().get("arc_check_interval_sec", 14400)))
        try:
            await asyncio.sleep(interval)
        except asyncio.CancelledError:
            raise


# ─────────────────────────────────────────────────────────────────────────────
#  TACTICAL GOALS LOOP (v9.6)
# ─────────────────────────────────────────────────────────────────────────────
# Hourly check: per user, expire any past-due goals AND if any horizon has no
# active goal, ask the LLM to compose one (`refresh_goals_for_user`).
# Conservative cadence so we don't burn GPU on constant LLM calls.
async def _tactical_goals_scan_once() -> None:
    from main import get_users
    cfg = _cfg()
    if not cfg.get("tactical_goals_enabled", True):
        return
    try:
        users = get_users()
    except Exception as e:
        _log_exc("tactical-goals get_users", e); return
    try:
        from app.services.tactical_goals import expire_old, refresh_goals_for_user
    except Exception as e:
        _log_exc("tactical-goals import", e); return
    # Cheap cleanup first — single SQL UPDATE across all users
    try: expire_old(None)
    except Exception as e: _log_exc("tactical-goals expire_old", e)
    # Per-user composition (LLM-bound). Sequential to avoid GPU contention.
    for u in users:
        uid = u["id"]
        try:
            await refresh_goals_for_user(uid, force=False)
        except Exception as e:
            _log_exc(f"tactical-goals refresh uid={uid}", e)


async def _tactical_goals_loop() -> None:
    log = _log()
    log.info("Tactical-goals loop started")
    # Cold boot: short delay so the first scan doesn't race with startup IO
    try: await asyncio.sleep(120)
    except asyncio.CancelledError: raise
    while True:
        try:
            await _tactical_goals_scan_once()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            _log_exc("tactical-goals loop", e)
        interval = max(1800, int(_cfg().get("tactical_goals_check_interval_sec", 3600)))
        try:
            await asyncio.sleep(interval)
        except asyncio.CancelledError:
            raise


# ─────────────────────────────────────────────────────────────────────────────
#  LIFECYCLE: spawn loops once on FastAPI startup
# ─────────────────────────────────────────────────────────────────────────────
def start_autonomous_loops() -> None:
    """Idempotent. Registers the long-running tasks with main._track()."""
    from main import _track
    global _LOOPS_STARTED
    if _LOOPS_STARTED:
        return
    _LOOPS_STARTED = True
    log = _log()
    try:
        t1 = asyncio.create_task(_proactive_loop())
        _track(t1)
        log.info("Autonomous: proactive loop task scheduled")
    except Exception as e:
        _log_exc("start_autonomous_loops proactive", e)
    try:
        t2 = asyncio.create_task(_diary_loop())
        _track(t2)
        log.info("Autonomous: diary loop task scheduled")
    except Exception as e:
        _log_exc("start_autonomous_loops diary", e)
    # v9.4: monthly arc consolidation
    try:
        t3 = asyncio.create_task(_arc_loop())
        _track(t3)
        log.info("Autonomous: monthly-arc loop task scheduled")
    except Exception as e:
        _log_exc("start_autonomous_loops arc", e)
    # v9.6: tactical goals (Maid composes day/week aims for herself)
    try:
        t4 = asyncio.create_task(_tactical_goals_loop())
        _track(t4)
        log.info("Autonomous: tactical-goals loop task scheduled")
    except Exception as e:
        _log_exc("start_autonomous_loops tactical_goals", e)
