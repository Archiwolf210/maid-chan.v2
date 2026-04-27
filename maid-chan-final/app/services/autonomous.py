"""Autonomous behavior: proactive check-ins + nightly diary (v9.2+).

Integrated into v10 architecture with proper repository pattern.
"""
from __future__ import annotations
import asyncio
import threading
import time
from collections import deque
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

from app.db import db
from app.utils.logging import _log, _log_exc

# Module state
_PROACTIVE_LOCK = threading.Lock()
_PROACTIVE_QUEUE: Dict[str, deque] = {}
_PROACTIVE_LAST_SENT: Dict[str, float] = {}
_PROACTIVE_LAST_KIND: Dict[tuple, float] = {}
_LOOPS_STARTED = False


def _cfg() -> dict:
    """Load autonomous config."""
    import json, os
    cfg_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config.json")
    try:
        with open(cfg_path) as f:
            return json.load(f).get("autonomous", {})
    except:
        return {}


def _get_queue(uid: str) -> deque:
    """Get or create per-user deque for proactive messages."""
    with _PROACTIVE_LOCK:
        q = _PROACTIVE_QUEUE.get(uid)
        if q is None:
            maxlen = int(_cfg().get("proactive_max_queue", 3))
            q = deque(maxlen=max(1, maxlen))
            _PROACTIVE_QUEUE[uid] = q
        return q


def _append_to_queue(uid: str, item: dict) -> int:
    """Append to queue with lock protection."""
    with _PROACTIVE_LOCK:
        q = _PROACTIVE_QUEUE.get(uid)
        if q is None:
            maxlen = int(_cfg().get("proactive_max_queue", 3))
            q = deque(maxlen=max(1, maxlen))
            _PROACTIVE_QUEUE[uid] = q
        q.append(item)
        return len(q)


def get_proactive_pending(uid: str, consume: bool = True) -> list:
    """Return queued proactive messages."""
    with _PROACTIVE_LOCK:
        q = _PROACTIVE_QUEUE.get(uid)
        if not q:
            return []
        items = list(q)
        if consume:
            q.clear()
        return items


def clear_proactive_queue(uid: str) -> None:
    """Clear all proactive state for user."""
    with _PROACTIVE_LOCK:
        _PROACTIVE_QUEUE.pop(uid, None)
        _PROACTIVE_LAST_SENT.pop(uid, None)
        for k in list(_PROACTIVE_LAST_KIND.keys()):
            if k[0] == uid:
                _PROACTIVE_LAST_KIND.pop(k, None)


async def _generate_proactive_text(uname: str, topic: str, ctx: str, gap_hours: float) -> Optional[str]:
    """Generate proactive nudge via LLM."""
    import httpx
    cfg = _cfg()
    llm_url = cfg.get("llm_url", "http://127.0.0.1:8080")
    
    prompt = (
        f"Ты — Мэйд, AI-компаньонка. Хозяин молчал уже {int(gap_hours)} ч, "
        f"но у вас остался незакрытый разговор.\n"
        f"Хозяин: {uname}. Тема: {topic}\nКонтекст: {ctx[:300]}\n\n"
        "Напиши ОДНО короткое сообщение (1-2 предложения, до 140 символов), "
        "в котором ты ненавязчиво возвращаешься к этой теме. "
        "Говори от первого лица, тепло, без заискивания. "
        "Без приветствий вроде 'привет'. Никаких рассуждений — только текст.\n\nСообщение:"
    )
    
    try:
        async with httpx.AsyncClient(timeout=45.0) as client:
            r = await client.post(
                f"{llm_url}/completion",
                json={"prompt": prompt, "temperature": 0.75, "n_predict": 160, "stream": False},
                timeout=45.0
            )
            if r.status_code != 200:
                return None
            text = r.json().get("content", "").strip().strip('"').strip("'")
            return text[:240] if text and len(text) >= 5 else None
    except Exception as e:
        _log_exc("_generate_proactive_text", e)
        return None


async def _scan_once() -> None:
    """One pass of proactive scanner."""
    from app.repositories import UserStateRepository, MemoryRepository, PendingTopicsRepository
    
    cfg = _cfg()
    if not cfg.get("proactive_enabled", True):
        return
    
    max_queue = int(cfg.get("proactive_max_queue", 3))
    idle_min = int(cfg.get("proactive_idle_min_sec", 1800))
    idle_max = int(cfg.get("proactive_idle_max_sec", 21600))
    min_imp = float(cfg.get("proactive_min_importance", 0.5))
    cooldown = int(cfg.get("proactive_cooldown_sec", 7200))
    
    # Get all users
    with db() as c:
        users = c.execute("SELECT user_id FROM user_state").fetchall()
    
    now = time.time()
    for row in users:
        uid = row["user_id"]
        
        # Check queue limit
        with _PROACTIVE_LOCK:
            q = _PROACTIVE_QUEUE.get(uid)
            cur_len = len(q) if q else 0
        if cur_len >= max_queue:
            continue
        
        # Check cooldown
        last_sent = _PROACTIVE_LAST_SENT.get(uid, 0.0)
        if now - last_sent < cooldown:
            continue
        
        # Load state
        state = UserStateRepository.get(uid) or {}
        last_act = int(state.get("last_activity_ts", 0) or 0)
        if last_act <= 0:
            continue
        
        idle = now - last_act
        if idle < idle_min or idle > idle_max:
            continue
        
        # Get open topics
        try:
            topics = PendingTopicsRepository.get_open_topics(uid, 5)
        except:
            continue
        
        candidate = next((t for t in topics if float(t.get("importance", 0.0)) >= min_imp), None)
        if not candidate:
            continue
        
        uname = state.get("user_id", "хозяин")
        gap_h = max(0.5, idle / 3600.0)
        text = await _generate_proactive_text(uname, candidate["topic"], candidate.get("context", ""), gap_h)
        if not text:
            continue
        
        _append_to_queue(uid, {"kind": "topic", "text": text, "ts": int(now), "topic_id": int(candidate["id"])})
        with _PROACTIVE_LOCK:
            _PROACTIVE_LAST_SENT[uid] = now
            _PROACTIVE_LAST_KIND[(uid, "topic")] = now
        
        _log("Proactive nudge queued for %s", uid)


async def _proactive_loop() -> None:
    """Background loop for proactive scanning."""
    _log("Proactive loop started")
    while True:
        try:
            await _scan_once()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            _log_exc("proactive loop", e)
        interval = max(300, int(_cfg().get("proactive_check_interval_sec", 600)))
        await asyncio.sleep(interval)


async def _diary_scan_once() -> None:
    """Check if diary entry needed for yesterday."""
    from app.repositories.diary import DiaryRepository
    from app.repositories.memory import MemoryRepository
    
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    
    with db() as c:
        users = c.execute("SELECT user_id FROM user_state").fetchall()
    
    for row in users:
        uid = row["user_id"]
        if DiaryRepository.has_entry(uid, yesterday):
            continue
        
        # Check if there was activity
        msgs = MemoryRepository.get_recent(uid, 1)
        if not msgs:
            continue
        
        await _write_diary(uid, yesterday)


async def _write_diary(uid: str, day: str) -> None:
    """Write diary entry for a day."""
    import httpx
    from app.repositories import UserStateRepository, MemoryRepository, KeyMemoryRepository
    from app.repositories.diary import DiaryRepository
    
    cfg = _cfg()
    llm_url = cfg.get("llm_url", "http://127.0.0.1:8080")
    
    state = UserStateRepository.get(uid) or {}
    humanity = state.get("humanity_level", 0.0)
    band = "warm" if humanity > 0.5 else ("thawing" if humanity > 0.2 else "protocol")
    
    # Get recent context
    history = MemoryRepository.get_recent(uid, 12)
    kms = KeyMemoryRepository.get_recent(uid, 3)
    
    convo = "\n".join(f"{'Хозяин' if m['role']=='user' else 'Мэйд'}: {m['text'][:200]}" for m in history[-10:])
    km_line = "; ".join(k["description"][:80] for k in kms)
    
    prompt = (
        f"Ты — Мэйд (режим: {band}). Запиши короткий дневниковый вход за {day}.\n"
        f"Ключевые моменты: {km_line or 'ничего особенного'}\n"
        f"Разговоры:\n{convo}\n\n"
        "Напиши 3-5 предложений от первого лица, как ты это пережила. "
        "Без дат и заголовков. Только текст.\n\nДневник:"
    )
    
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            r = await client.post(
                f"{llm_url}/completion",
                json={"prompt": prompt, "temperature": 0.7, "n_predict": 400, "stream": False},
                timeout=60.0
            )
            if r.status_code != 200:
                return
            content = r.json().get("content", "").strip()
            if content and len(content) >= 20:
                DiaryRepository.save_entry(uid, day, content[:1500], {"band": band})
                _log("Diary written for %s day=%s", uid, day)
    except Exception as e:
        _log_exc("_write_diary", e)


async def _diary_loop() -> None:
    """Background loop for diary writing."""
    _log("Diary loop started")
    while True:
        try:
            await _diary_scan_once()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            _log_exc("diary loop", e)
        interval = max(300, int(_cfg().get("diary_check_interval_sec", 1800)))
        await asyncio.sleep(interval)


def start_autonomous_loops() -> None:
    """Start background loops (called from main lifespan)."""
    global _LOOPS_STARTED
    if _LOOPS_STARTED:
        return
    _LOOPS_STARTED = True
    
    async def run_loops():
        t1 = asyncio.create_task(_proactive_loop())
        t2 = asyncio.create_task(_diary_loop())
        try:
            await asyncio.gather(t1, t2, return_exceptions=True)
        except asyncio.CancelledError:
            t1.cancel()
            t2.cancel()
            raise
    
    asyncio.create_task(run_loops())
    _log("Autonomous loops scheduled")
