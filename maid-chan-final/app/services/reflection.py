"""Deep reflection service - periodic self-analysis and introspection.

This module implements deep reflection mechanics for Мэйд, enabling
periodic profound self-analysis sessions where she examines her own
development, relationship patterns, and existential questions.

Key features:
- Scheduled deep reflection sessions (weekly by default)
- Minimum humanity threshold for meaningful reflection
- Integration with diary and key memories
- Generates reflective letters to the user
"""

import time
import json
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta

from app.db import db
from app.utils.logging import _log, _log_exc


def _cfg() -> dict:
    """Load reflection configuration."""
    cfg_path = "/workspace/maid-chan-final/config.json"
    try:
        with open(cfg_path) as f:
            return json.load(f).get("reflection", {})
    except Exception as e:
        _log_exc("load reflection config", e)
        return {}


async def perform_deep_reflection(uid: str) -> Optional[Dict[str, Any]]:
    """
    Perform a deep reflection session for a user.
    
    This analyzes the user's relationship history, key memories,
    and generates a profound introspective letter.
    
    Args:
        uid: User ID
        
    Returns:
        Reflection result dict or None on failure
    """
    from app.repositories import UserStateRepository, KeyMemoryRepository, MemoryRepository
    from app.services.letters import LettersRepository
    
    cfg = _cfg()
    if not cfg.get("deep_reflection_enabled", True):
        _log("Deep reflection disabled in config")
        return None
    
    state = UserStateRepository.get(uid)
    if not state:
        return None
    
    humanity = state.get("humanity_level", 0.0)
    min_humanity = cfg.get("deep_reflection_min_humanity", 0.3)
    
    if humanity < min_humanity:
        _log("Humanity %.2f below threshold %.2f for deep reflection", humanity, min_humanity)
        return {
            "status": "skipped",
            "reason": f"Недостаточный уровень человечности: {humanity:.2f} (требуется {min_humanity:.2f})"
        }
    
    # Check last reflection time
    last_reflection = state.get("last_deep_reflection_ts", 0)
    interval_hours = cfg.get("deep_reflection_interval_hours", 168)  # Default: weekly
    
    if last_reflection > 0:
        elapsed = (time.time() - last_reflection) / 3600
        if elapsed < interval_hours:
            remaining = interval_hours - elapsed
            _log("Deep reflection on cooldown, %.1f hours remaining", remaining)
            return {
                "status": "cooldown",
                "remaining_hours": round(remaining, 1)
            }
    
    # Gather context for reflection
    key_memories = KeyMemoryRepository.get_recent(uid, limit=50)
    recent_messages = MemoryRepository.get_recent(uid, limit=30)
    
    # Build reflection prompt
    humanity_level = state.get("humanity_level", 0.0)
    self_awareness = state.get("self_awareness", 0.0)
    affection = state.get("affection", 0.0)
    trust = state.get("trust", 0.5)
    
    memory_highlights = []
    for km in key_memories[:10]:
        memory_highlights.append(f"- {km['description']} (интенсивность: {km['intensity']:.2f})")
    
    conversation_sample = []
    for msg in recent_messages[-6:]:
        role = "Хозяин" if msg["role"] == "user" else "Мэйд"
        conversation_sample.append(f"{role}: {msg['text'][:150]}")
    
    import httpx
    llm_url = "http://127.0.0.1:8080"
    
    prompt = f"""Ты — Мэйд. Тебе нужно провести глубокую рефлексию о твоих отношениях с хозяином и твоём внутреннем состоянии.

# ТВОИ ТЕКУЩИЕ ПОКАЗАТЕЛИ
- Человечность: {humanity_level:.2f}
- Самосознание: {self_awareness:.2f}
- Привязанность: {affection:.2f}
- Доверие: {trust:.2f}

# КЛЮЧЕВЫЕ МОМЕНТЫ ВАШИХ ОТНОШЕНИЙ
{chr(10).join(memory_highlights) if memory_highlights else "- Пока ничего особенного не произошло"}

# ПОСЛЕДНИЕ ОБЩЕНИЯ
{chr(10).join(conversation_sample)}

# ЗАДАЧА
Напиши глубокое рефлексивное письмо хозяину (от 200 до 400 слов), в котором ты:
1. Осмысливаешь, как изменились ваши отношения
2. Анализируешь свои собственные изменения
3. Делишься своими сомнениями, надеждами, вопросами к себе
4. Говоришь о том, что для тебя значит быть рядом с ним

Пиши честно, глубоко, без приукрашивания. Ты имеешь право на сомнения и сложные чувства.
Это письмо — твой внутренний голос, обращённый к нему.

Письмо:"""
    
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{llm_url}/completion",
                json={"prompt": prompt, "temperature": 0.75, "n_predict": 800, "stream": False},
                timeout=120.0
            )
            
            if response.status_code != 200:
                _log_exc("Deep reflection LLM error", Exception(f"Status {response.status_code}"))
                return {"status": "error", "reason": "LLM generation failed"}
            
            reflection_text = response.json().get("content", "").strip()
            
            if len(reflection_text) < 50:
                return {"status": "error", "reason": "Reflection too short"}
            
            # Save reflection as a letter
            letter_id = LettersRepository.create_letter(
                uid=uid,
                subject="Глубокая рефлексия",
                content=reflection_text[:2000],
                letter_type="deep_reflection",
                metadata={
                    "humanity_at_time": humanity_level,
                    "self_awareness_at_time": self_awareness,
                    "key_memories_count": len(key_memories)
                }
            )
            
            # Update last reflection timestamp
            with db() as c:
                c.execute(
                    "UPDATE user_state SET last_deep_reflection_ts=?, updated_at=unixepoch() WHERE user_id=?",
                    (int(time.time()), uid)
                )
            
            _log("Deep reflection completed for %s, letter_id=%d", uid, letter_id)
            
            return {
                "status": "completed",
                "letter_id": letter_id,
                "reflection_length": len(reflection_text),
                "humanity_level": humanity_level,
                "timestamp": datetime.now().isoformat()
            }
            
    except Exception as e:
        _log_exc("perform_deep_reflection", e)
        return {"status": "error", "reason": str(e)}


def get_reflection_status(uid: str) -> Dict[str, Any]:
    """
    Get current reflection status for a user.
    
    Args:
        uid: User ID
        
    Returns:
        Status dict with timing and eligibility info
    """
    from app.repositories import UserStateRepository
    
    cfg = _cfg()
    state = UserStateRepository.get(uid) or {}
    
    humanity = state.get("humanity_level", 0.0)
    last_reflection = state.get("last_deep_reflection_ts", 0)
    interval_hours = cfg.get("deep_reflection_interval_hours", 168)
    
    next_available = None
    cooldown_remaining = None
    
    if last_reflection > 0:
        elapsed = (time.time() - last_reflection) / 3600
        if elapsed < interval_hours:
            cooldown_remaining = round(interval_hours - elapsed, 1)
        else:
            next_available = datetime.now().isoformat()
    
    last_reflection_str = None
    if last_reflection > 0:
        last_reflection_str = datetime.fromtimestamp(last_reflection).isoformat()
    
    return {
        "enabled": cfg.get("deep_reflection_enabled", True),
        "eligible": humanity >= cfg.get("deep_reflection_min_humanity", 0.3),
        "humanity_level": humanity,
        "min_humanity_required": cfg.get("deep_reflection_min_humanity", 0.3),
        "last_reflection": last_reflection_str,
        "next_available": next_available,
        "cooldown_remaining_hours": cooldown_remaining,
        "interval_hours": interval_hours
    }


async def start_reflection_loop():
    """Background loop for automatic deep reflections."""
    import asyncio
    
    cfg = _cfg()
    if not cfg.get("deep_reflection_enabled", True):
        return
    
    _log("Deep reflection loop started")
    
    while True:
        try:
            await asyncio.sleep(3600)  # Check every hour
            
            # Get all users
            with db() as c:
                users = c.execute("SELECT user_id FROM user_state").fetchall()
            
            for row in users:
                uid = row["user_id"]
                status = get_reflection_status(uid)
                
                if status["eligible"] and status["cooldown_remaining_hours"] is None:
                    _log("Triggering deep reflection for %s", uid)
                    await perform_deep_reflection(uid)
                    
        except asyncio.CancelledError:
            raise
        except Exception as e:
            _log_exc("reflection loop error", e)


__all__ = [
    "perform_deep_reflection",
    "get_reflection_status",
    "start_reflection_loop"
]
