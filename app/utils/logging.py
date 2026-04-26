"""
Centralized logging utility for Digital Human project.
Eliminates duplication of _log() and _log_exc() across modules.
"""

import sys
import traceback
from datetime import datetime
from typing import Any, Optional

# Late import to avoid circular dependency
_config: Optional[dict] = None
_base_dir_cache: Optional[str] = None


def _get_config() -> dict:
    """Lazy load config to avoid circular imports."""
    global _config
    if _config is None:
        from main import load_config
        _config = load_config()
    return _config


def _get_base_dir() -> str:
    """Get base directory from config."""
    global _base_dir_cache
    if _base_dir_cache is None:
        from main import _base_dir
        _base_dir_cache = _base_dir()
    return _base_dir_cache


def _log(msg: str, *a: Any) -> None:
    """
    Thread-safe logging with timestamp and optional formatting.
    
    Args:
        msg: Message string (may contain % format placeholders)
        *a: Arguments for % formatting
    """
    try:
        cfg = _get_config()
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = (msg % a) if a else msg
        text = f"[{ts}] {line}"
        
        if cfg.get("log_to_file"):
            import os
            log_path = os.path.join(_get_base_dir(), "digital_human.log")
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(text + "\n")
        
        if cfg.get("log_to_stdout", True):
            print(text, file=sys.stderr, flush=True)
            
    except Exception as e:
        # Fallback: never crash logging itself
        print(f"[LOG ERROR] {e}: {msg}", file=sys.stderr, flush=True)


def _log_exc(msg: str, exc: Optional[Exception] = None, *a: Any) -> None:
    """
    Log error message with full traceback.
    
    Args:
        msg: Error message (may contain % format placeholders)
        exc: Exception object (if None, uses sys.exc_info())
        *a: Arguments for % formatting
    """
    try:
        # Format the message
        line = (msg % a) if a else msg
        
        # Get traceback
        if exc is None:
            tb = traceback.format_exc()
        else:
            tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        
        # Log both message and traceback
        _log(f"{line}\n{tb}")
        
    except Exception as fallback_e:
        # Fallback: never crash logging itself
        print(f"[LOG ERROR] {fallback_e}: {msg}", file=sys.stderr, flush=True)


# Convenience exports
__all__ = ["_log", "_log_exc"]
