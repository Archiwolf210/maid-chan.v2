"""Letters service - spontaneous notes from Maid to user.

This module handles generation and delivery of "letters" - short spontaneous
notes that Maid writes to the user during significant moments (anchor events,
milestones, evening reflections).

Features:
- Anchor-triggered letters (after key memories with intensity >= 0.75)
- Milestone letters (every N messages)
- Evening reflection letters
- Cooldown management to prevent spam
- Humanity-based gating (sealed letters for low humanity)
"""
from __future__ import annotations
import asyncio
import time
from typing import Optional, Dict, Any
from datetime import datetime

import httpx

from app.db import db
from app.utils.logging import _log, _log_exc, _log_warn
from app.repositories.letters import LettersRepository
from app.repositories import KeyMemoryRepository, UserStateRepository


def _cfg() -> dict:
    """Load immersive/letters config."""
    import json, os
    cfg_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
        "config.json"
    )
    try:
        with open(cfg_path) as f:
            cfg = json.load(f)
            return cfg.get("immersive", {}).get("letters", {})
    except:
        return {}


async def _generate_letter_text(
    uid: str, 
    trigger: str, 
    context: str,
    humanity: float
) -> Optional[str]:
    """Generate letter text via LLM."""
    cfg = _cfg()
    llm_url = cfg.get("llm_url", "http://127.0.0.1:8080")
    
    band = "warm" if humanity > 0.5 else ("thawing" if humanity > 0.2 else "protocol")
    
    prompt = (
        f"Ты — Мэйд (режим: {band}). Пишешь короткую записку хозяину.\n"
        f"Повод: {trigger}\nКонтекст: {context[:400]}\n\n"
        "Напиши ОДНУ короткую записку (2-4 предложения, до 200 символов).\n"
        "Говори от первого лица, тепло, без пафоса.\n"
        "Без дат, заголовков и подписей — только текст.\n\nЗаписка:"
    )
    
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            r = await client.post(
                f"{llm_url}/completion",
                json={"prompt": prompt, "temperature": 0.75, "n_predict": 300, "stream": False},
                timeout=60.0
            )
            if r.status_code != 200:
                return None
            text = r.json().get("content", "").strip().strip('"').strip("'")
            return text[:400] if text and len(text) >= 10 else None
    except Exception as e:
        _log_exc("_generate_letter_text", e)
        return None


async def try_generate_letter(
    uid: str, 
    key_memory_id: int,
    forced: bool = False
) -> Optional[int]:
    """
    Try to generate a letter triggered by a key memory.
    
    Checks cooldowns and config before generating.
    Returns letter ID on success, None on skip/failure.
    """
    cfg = _cfg()
    if not cfg.get("enabled", True) and not forced:
        return None
    
    # Check cooldown
    cooldown_sec = int(cfg.get("cooldown_sec", 21600))  # Default 6h
    if not forced and LettersRepository.has_recent_letter(uid, cooldown_sec):
        _log("Letter skipped (cooldown) for %s", uid)
        return None
    
    # Load context
    state = UserStateRepository.get(uid) or {}
    humanity = state.get("humanity_level", 0.0)
    
    # Check humanity threshold for sealing
    seal_threshold = float(cfg.get("seal_below_humanity", 0.3))
    sealed = humanity < seal_threshold
    
    if sealed and not cfg.get("allow_sealed", True):
        _log("Letter skipped (sealed disabled) for %s", uid)
        return None
    
    # Get key memory context
    km_row = None
    with db() as c:
        row = c.execute(
            "SELECT event_type, description, intensity FROM key_memories WHERE id=?",
            (key_memory_id,)
        ).fetchone()
        if row:
            km_row = dict(row)
    
    if not km_row:
        _log("Key memory %d not found for letter", key_memory_id)
        return None
    
    trigger = f"{km_row['event_type']} (интенсивность: {km_row['intensity']:.2f})"
    context = km_row['description']
    
    # Generate text
    text = await _generate_letter_text(uid, trigger, context, humanity)
    if not text:
        _log("Letter generation failed for %s", uid)
        return None
    
    # Insert letter
    letter_id = LettersRepository.insert(
        uid, text, 
        key_memory_id=key_memory_id,
        triggered_by="anchor",
        sealed=sealed
    )
    
    if letter_id:
        status = "sealed" if sealed else "delivered"
        _log("Letter %s created for %s (status=%s)", letter_id, uid, status)
    
    return letter_id


async def try_evening_letter(uid: str) -> Optional[int]:
    """Generate an evening reflection letter (if enabled)."""
    cfg = _cfg()
    if not cfg.get("evening_enabled", True):
        return None
    
    # Check time window
    now = datetime.now()
    hour = now.hour
    lo = int(cfg.get("evening_window_lo", 22))
    hi = int(cfg.get("evening_window_hi", 24))
    
    if not (lo <= hour < hi):
        return None
    
    # Check cooldown
    cooldown_sec = int(cfg.get("evening_cooldown_sec", 86400))  # Default 24h
    if LettersRepository.has_recent_letter(uid, cooldown_sec):
        return None
    
    state = UserStateRepository.get(uid) or {}
    humanity = state.get("humanity_level", 0.0)
    
    # Get today's summary
    today = now.strftime("%Y-%m-%d")
    day_summary = ""
    with db() as c:
        row = c.execute(
            "SELECT summary FROM daily_summaries WHERE user_id=? AND day=?",
            (uid, today)
        ).fetchone()
        if row:
            day_summary = row[0]
    
    if not day_summary:
        # No summary available, skip
        return None
    
    text = await _generate_letter_text(
        uid, 
        "вечерняя рефлексия", 
        day_summary, 
        humanity
    )
    
    if not text:
        return None
    
    letter_id = LettersRepository.insert(
        uid, text,
        key_memory_id=None,
        triggered_by="evening",
        sealed=(humanity < 0.3)
    )
    
    if letter_id:
        _log("Evening letter %s created for %s", letter_id, uid)
    
    return letter_id


async def try_milestone_letter(uid: str, total_msg_count: int) -> Optional[int]:
    """Generate a milestone letter (every N messages)."""
    cfg = _cfg()
    milestone_interval = int(cfg.get("milestone_interval", 100))
    
    if total_msg_count % milestone_interval != 0:
        return None
    
    # Check cooldown
    cooldown_sec = int(cfg.get("milestone_cooldown_sec", 86400 * 3))  # 3 days
    if LettersRepository.has_recent_letter(uid, cooldown_sec):
        return None
    
    state = UserStateRepository.get(uid) or {}
    humanity = state.get("humanity_level", 0.0)
    
    text = await _generate_letter_text(
        uid,
        f"веха: {total_msg_count} сообщений",
        f"Мы обменялись уже {total_msg_count} сообщениями.",
        humanity
    )
    
    if not text:
        return None
    
    letter_id = LettersRepository.insert(
        uid, text,
        key_memory_id=None,
        triggered_by="milestone",
        sealed=(humanity < 0.3)
    )
    
    if letter_id:
        _log("Milestone letter %s created for %s at msg %d", letter_id, uid, total_msg_count)
    
    return letter_id


def unseal_letters_if_ready(uid: str, current_humanity: float) -> int:
    """Unseal any sealed letters when humanity crosses threshold."""
    return LettersRepository.unseal_below_threshold(uid, current_humanity)
