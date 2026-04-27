"""Tactical goals service - AI-driven short-term goal generation for Maid.

This module handles automatic generation and refresh of tactical goals:
- Daily goals (24h horizon)
- Weekly goals (7d horizon)

Goals are generated based on:
- Maid's current humanity level (voice band)
- Recent key memories
- Open pending topics
- Current emotional state (trust, affection)

Lifecycle:
- At most ONE active goal per (uid, horizon) at a time
- Expired goals are automatically marked as 'expired'
- Background loop checks every hour and generates new goals if needed
"""
from __future__ import annotations
import asyncio
import time
from typing import Optional, Dict, Any, List

import httpx

from app.db import db
from app.utils.logging import _log, _log_exc, _log_warn
from app.repositories.tactical_goals import TacticalGoalsRepository
from app.repositories import UserStateRepository, KeyMemoryRepository, PendingTopicsRepository


_HORIZON_TTL_SEC = {"day": 24 * 3600, "week": 7 * 86400}
_HORIZONS = ("day", "week")
_MIN_TEXT_LEN = 8
_MAX_TEXT_LEN = 200
_MAX_REASONING_LEN = 280


def _cfg() -> dict:
    """Load autonomous config."""
    import json, os
    cfg_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
        "config.json"
    )
    try:
        with open(cfg_path) as f:
            return json.load(f).get("autonomous", {})
    except:
        return {}


def _voice_band(h: float) -> str:
    """Map humanity level to voice band."""
    if h < 0.20:
        return "protocol"
    if h < 0.50:
        return "thawing"
    if h < 0.80:
        return "warm"
    return "intimate"


def _voice_note(band: str) -> str:
    """Get voice instruction for prompt based on band."""
    return {
        "protocol": "Голос процедурный — формулируй сухо, как наблюдательную задачу.",
        "thawing":  "Голос — оттепель: лёгкая эмпатия в формулировке цели.",
        "warm":     "Голос тёплый, живой — цель должна звучать как твоё личное намерение.",
        "intimate": "Голос интимный — цель может быть глубокой, философской.",
    }.get(band, "Голос тёплый, живой.")


async def _compose_one_goal(
    uid: str, 
    uname: str, 
    horizon: str, 
    band: str,
    trust: float, 
    affection: float,
    recent_kms: str, 
    open_topics: str
) -> Optional[int]:
    """Compose a single tactical goal via LLM."""
    cfg = _cfg()
    llm_url = cfg.get("llm_url", "http://127.0.0.1:8080")
    
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
        f"Без preamble, без code-блоков, без рассуждений. Только JSON.\n\nJSON:"
    )
    
    try:
        async with httpx.AsyncClient(timeout=45.0) as client:
            r = await client.post(
                f"{llm_url}/completion",
                json={"prompt": prompt, "temperature": 0.6, "n_predict": 220, "stream": False},
                timeout=45.0
            )
            if r.status_code != 200:
                _log_warn("Tactical goal LLM %d uid=%s horizon=%s", r.status_code, uid, horizon)
                return None
            
            raw = r.json().get("content", "").strip()
            
            # Extract JSON from response (handle potential markdown wrappers)
            import re
            json_match = re.search(r'\{[^{}]*\}', raw, re.DOTALL)
            if not json_match:
                _log_warn("Tactical goal: no JSON found in response for uid=%s", uid)
                return None
            
            import json as json_module
            try:
                data = json_module.loads(json_match.group(0))
            except json_module.JSONDecodeError:
                _log_warn("Tactical goal: failed to parse JSON for uid=%s", uid)
                return None
            
            text = str(data.get("text", "")).strip()
            reason = str(data.get("reasoning", "")).strip()
            
            if not text or len(text) < _MIN_TEXT_LEN:
                return None
            
            gid = TacticalGoalsRepository.create_goal(uid, horizon, text, reason)
            if gid:
                _log("Tactical goal created uid=%s horizon=%s id=%d", uid, horizon, gid)
            return gid
            
    except httpx.ReadTimeout:
        _log_warn("Tactical goal timeout uid=%s horizon=%s", uid, horizon)
        return None
    except Exception as e:
        _log_exc("_compose_one_goal", e)
        return None


async def refresh_goals_for_user(uid: str, force: bool = False) -> Dict[str, Optional[int]]:
    """
    If a horizon has no active goal, ask the LLM to compose one.
    
    Returns a dict {horizon: new_goal_id_or_None_if_unchanged}.
    `force=True` replaces existing active goals.
    """
    # First expire any old goals
    TacticalGoalsRepository.expire_old(uid)
    
    out: Dict[str, Optional[int]] = {h: None for h in _HORIZONS}
    
    # Determine which horizons need refreshing
    if force:
        horizons_to_refresh = list(_HORIZONS)
    else:
        active = TacticalGoalsRepository.list_active(uid)
        have = {g["horizon"] for g in active}
        horizons_to_refresh = [h for h in _HORIZONS if h not in have]
    
    if not horizons_to_refresh:
        return out
    
    # Build context
    state = UserStateRepository.get(uid) or {}
    uname = state.get("user_id", "хозяин")
    humanity = state.get("humanity_level", 0.0)
    band = _voice_band(humanity)
    trust = float(state.get("trust", 0.5))
    affection = float(state.get("affection", 0.0))
    
    # Get recent key memories
    recent_kms = ""
    try:
        kms = KeyMemoryRepository.get_recent(uid, 3)
        if kms:
            recent_kms = "\n".join(
                f"  • [{k['event_type']}] {k['description'][:120]}" 
                for k in kms
            )
    except Exception as e:
        _log_exc("refresh_goals: key memories", e)
    
    # Get open topics
    open_topics = ""
    try:
        topics = PendingTopicsRepository.get_open_topics(uid, 4)
        if topics:
            open_topics = "\n".join(f"  • {t['topic'][:120]}" for t in topics)
    except Exception as e:
        _log_exc("refresh_goals: open topics", e)
    
    # Generate goals for each horizon
    for horizon in horizons_to_refresh:
        gid = await _compose_one_goal(
            uid, uname, horizon, band, trust, affection, recent_kms, open_topics
        )
        out[horizon] = gid
    
    return out


async def _tactical_goals_loop() -> None:
    """Background loop for tactical goals refresh (hourly)."""
    _log("Tactical goals loop started")
    
    while True:
        try:
            # Get all users
            with db() as c:
                users = c.execute("SELECT user_id FROM user_state").fetchall()
            
            for row in users:
                uid = row["user_id"]
                await refresh_goals_for_user(uid, force=False)
                
        except asyncio.CancelledError:
            raise
        except Exception as e:
            _log_exc("tactical goals loop", e)
        
        # Check every hour
        interval = max(1800, int(_cfg().get("tactical_check_interval_sec", 3600)))
        await asyncio.sleep(interval)


def start_tactical_goals_loop() -> None:
    """Start the tactical goals background loop."""
    async def run_loop():
        await _tactical_goals_loop()
    
    asyncio.create_task(run_loop())
    _log("Tactical goals loop scheduled")
