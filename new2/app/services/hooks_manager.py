"""
Hooks Manager - безопасное управление пользовательскими хуками.
Изолирует ошибки, накладывает таймауты и предотвращает дублирование запусков.
"""
import asyncio
import importlib.util
import os
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

# Глобальное состояние для блокировки дублей (в памяти)
_running_hooks: set = set()

async def trigger_diary_hook(uid: str, day: str, anchor_ids: List[int], key_memories: List[Dict[str, Any]]) -> None:
    """
    Главный входной пункт. Вызывается из main.py после создания дневной записи.
    """
    hook_key = f"{uid}_{day}"
    
    # 1. Проверка на дублирование (Lock)
    if hook_key in _running_hooks:
        logger.debug(f"Hook already running for {hook_key}, skipping.")
        return
    
    _running_hooks.add(hook_key)
    
    try:
        await _execute_hook_safe(uid, day, anchor_ids, key_memories)
    finally:
        _running_hooks.discard(hook_key)

async def _execute_hook_safe(uid: str, day: str, anchor_ids: List[int], key_memories: List[Dict[str, Any]]) -> None:
    """
    Выполняет хук с полной изоляцией ошибок и таймаутом.
    """
    # Определяем путь к хуку относительно корня проекта
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    hook_path = os.path.join(base_dir, "hooks", "on_diary.py")
    
    if not os.path.exists(hook_path):
        return

    try:
        # Динамический импорт
        spec = importlib.util.spec_from_file_location("user_diary_hook", hook_path)
        if not spec or not spec.loader:
            logger.warning(f"Could not load spec for hook at {hook_path}")
            return
            
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        if not hasattr(module, "on_diary_written"):
            logger.warning(f"Hook module at {hook_path} has no 'on_diary_written' function")
            return

        logger.info(f"Executing on_diary_written hook for {uid} / {day}")
        
        # 2. Таймаут выполнения (5 секунд)
        # Пользовательский код не должен висеть дольше
        await asyncio.wait_for(
            module.on_diary_written(uid, day, anchor_ids, key_memories),
            timeout=5.0
        )
        logger.info(f"Hook completed successfully for {uid}")
        
    except asyncio.TimeoutError:
        logger.error(f"[HOOKS] Timeout: Hook on_diary.py for {uid} executed > 5s. Terminated.")
    except Exception as e:
        # 3. Полная изоляция: любая ошибка пользователя логируется, но не ломает чат
        logger.error(f"[HOOKS] Error in on_diary.py for {uid}: {e}", exc_info=True)
