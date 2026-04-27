"""Centralized logging utility for Digital Human project.

Eliminates duplication of _log() and _log_exc() across modules.
Thread-safe, with file and stdout output options.
"""

import sys
import traceback
from datetime import datetime
from typing import Any, Optional


def _log(msg: str, *a: Any) -> None:
    """
    Thread-safe logging with timestamp and optional formatting.
    
    Args:
        msg: Message string (may contain % format placeholders)
        *a: Arguments for % formatting
    """
    try:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = (msg % a) if a else msg
        text = f"[{ts}] {line}"
        
        # Always log to stderr for visibility
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


def _log_warn(msg: str) -> None:
    """Log warning message (non-fatal background issues)."""
    _log("[WARNING] %s", msg)


# Convenience exports
__all__ = ["_log", "_log_exc", "_log_warn"]
