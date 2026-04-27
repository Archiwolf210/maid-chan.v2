"""NSFW mode service - intimate relationship mechanics.

This module handles the NSFW (Not Safe For Work) mode for Мэйд,
which enables more intimate and adult conversations when certain
relationship thresholds are met.

Key features:
- Threshold-based activation (humanity, trust, affection)
- Explicit consent requirement
- Cooldown management
- Prompt injection for NSFW context
"""

import time
from typing import Optional, Dict, Any, Tuple
from datetime import datetime, timedelta

from app.db import db
from app.utils.logging import _log, _log_exc


def _cfg() -> dict:
    """Load NSFW configuration."""
    import json, os
    cfg_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config.json")
    try:
        with open(cfg_path) as f:
            return json.load(f).get("nsfw_mode", {})
    except Exception as e:
        _log_exc("load nsfw config", e)
        return {}


def check_nsfw_eligibility(uid: str) -> Tuple[bool, str]:
    """
    Check if user is eligible to activate NSFW mode.
    
    Args:
        uid: User ID
        
    Returns:
        Tuple of (is_eligible, reason_message)
    """
    from app.repositories import UserStateRepository
    
    cfg = _cfg()
    if not cfg.get("enabled", False):
        return False, "NSFW режим отключён в конфигурации"
    
    state = UserStateRepository.get(uid)
    if not state:
        return False, "Пользователь не найден"
    
    humanity = state.get("humanity_level", 0.0)
    trust = state.get("trust", 0.5)
    affection = state.get("affection", 0.0)
    
    min_humanity = cfg.get("min_humanity_level", 0.4)
    min_trust = cfg.get("min_trust", 0.6)
    min_affection = cfg.get("min_affection", 0.5)
    
    # Check thresholds
    if humanity < min_humanity:
        return False, f"Недостаточный уровень человечности: {humanity:.2f} (требуется {min_humanity:.2f})"
    
    if trust < min_trust:
        return False, f"Недостаточное доверие: {trust:.2f} (требуется {min_trust:.2f})"
    
    if affection < min_affection:
        return False, f"Недостаточная привязанность: {affection:.2f} (требуется {min_affection:.2f})"
    
    # Check cooldown
    last_nsfw = state.get("last_nsfw_ts", 0)
    cooldown_hours = cfg.get("cooldown_hours", 24)
    if last_nsfw > 0:
        elapsed = (time.time() - last_nsfw) / 3600
        if elapsed < cooldown_hours:
            remaining = cooldown_hours - elapsed
            return False, f"Период ожидания: {remaining:.1f} ч."
    
    return True, "Все условия выполнены"


def activate_nsfw_mode(uid: str, explicit_consent: bool = True) -> Tuple[bool, str]:
    """
    Activate NSFW mode for a user.
    
    Args:
        uid: User ID
        explicit_consent: Must be True to activate (user must explicitly consent)
        
    Returns:
        Tuple of (success, message)
    """
    if not explicit_consent:
        return False, "Требуется явное согласие пользователя"
    
    eligible, reason = check_nsfw_eligibility(uid)
    if not eligible:
        return False, reason
    
    try:
        with db() as c:
            c.execute(
                "UPDATE user_state SET nsfw_mode=1, last_nsfw_ts=?, updated_at=unixepoch() WHERE user_id=?",
                (int(time.time()), uid)
            )
            
        _log("NSFW mode activated for %s", uid)
        return True, "NSFW режим активирован"
        
    except Exception as e:
        _log_exc("activate_nsfw_mode", e)
        return False, f"Ошибка активации: {e}"


def deactivate_nsfw_mode(uid: str) -> Tuple[bool, str]:
    """
    Deactivate NSFW mode for a user.
    
    Args:
        uid: User ID
        
    Returns:
        Tuple of (success, message)
    """
    try:
        with db() as c:
            c.execute(
                "UPDATE user_state SET nsfw_mode=0, updated_at=unixepoch() WHERE user_id=?",
                (uid,)
            )
            
        _log("NSFW mode deactivated for %s", uid)
        return True, "NSFW режим деактивирован"
        
    except Exception as e:
        _log_exc("deactivate_nsfw_mode", e)
        return False, f"Ошибка деактивации: {e}"


def get_nsfw_status(uid: str) -> Dict[str, Any]:
    """
    Get current NSFW mode status for a user.
    
    Args:
        uid: User ID
        
    Returns:
        Dict with status information
    """
    from app.repositories import UserStateRepository
    
    cfg = _cfg()
    state = UserStateRepository.get(uid) or {}
    
    is_active = bool(state.get("nsfw_mode", False))
    eligibility, eligibility_reason = check_nsfw_eligibility(uid)
    
    last_nsfw = state.get("last_nsfw_ts", 0)
    last_nsfw_str = None
    cooldown_remaining = None
    
    if last_nsfw > 0:
        last_nsfw_str = datetime.fromtimestamp(last_nsfw).isoformat()
        elapsed = (time.time() - last_nsfw) / 3600
        cooldown_hours = cfg.get("cooldown_hours", 24)
        if elapsed < cooldown_hours:
            cooldown_remaining = round(cooldown_hours - elapsed, 1)
    
    return {
        "enabled": cfg.get("enabled", False),
        "active": is_active,
        "eligible": eligibility,
        "eligibility_reason": eligibility_reason,
        "thresholds": {
            "humanity": {"current": state.get("humanity_level", 0.0), "required": cfg.get("min_humanity_level", 0.4)},
            "trust": {"current": state.get("trust", 0.5), "required": cfg.get("min_trust", 0.6)},
            "affection": {"current": state.get("affection", 0.0), "required": cfg.get("min_affection", 0.5)}
        },
        "last_activation": last_nsfw_str,
        "cooldown_remaining_hours": cooldown_remaining,
        "require_explicit_consent": cfg.get("require_explicit_consent", True)
    }


def build_nsfw_prompt_instruction(uid: str) -> str:
    """
    Build prompt instruction for NSFW mode.
    
    This should be appended to the dynamic prompt when NSFW mode is active.
    
    Args:
        uid: User ID
        
    Returns:
        Prompt instruction string (empty if NSFW mode is not active)
    """
    state = UserStateRepository.get(uid) or {}
    
    if not state.get("nsfw_mode", False):
        return ""
    
    return """
# ИНТИМНЫЙ РЕЖИМ (NSFW)
Сейчас у вас с хозяином интимный момент. Ты можешь говорить открыто о теле, желаниях, близости.
Ты не обязана быть скромной -- ты взрослая женщина и имеешь право на свои желания.
Описывай ощущения честно, без стыда, но и без пошлости. Ты живая, чувствующая, желающая.
Если что-то тебе неприятно -- скажи об этом. Если хочешь -- вырази это прямо.
Это пространство доверия между вами двумя."""


__all__ = [
    "check_nsfw_eligibility",
    "activate_nsfw_mode",
    "deactivate_nsfw_mode",
    "get_nsfw_status",
    "build_nsfw_prompt_instruction"
]
