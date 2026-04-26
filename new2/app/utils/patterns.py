"""
Utility patterns for keyword matching and emotion detection.
Isolated from main.py to break circular dependencies and improve architecture.
"""
import re
from typing import List, Dict, Optional

# Global cache for compiled regex patterns
_KW_CACHE: Dict[str, re.Pattern] = {}


def _get_compiled_pattern(pattern_str: str) -> re.Pattern:
    """Get or create a compiled regex pattern from cache."""
    if pattern_str not in _KW_CACHE:
        _KW_CACHE[pattern_str] = re.compile(pattern_str, re.IGNORECASE)
    return _KW_CACHE[pattern_str]


def _kw_count(keywords: List[str], text: str) -> int:
    """
    Count occurrences of keywords in text using word-boundary matching.
    Prevents false positives like 'несчастье' matching 'счастье'.
    """
    count = 0
    for kw in keywords:
        if len(kw) < 2:
            continue
        # Pattern: start of string or non-word char + keyword + end of string or non-word char
        pattern = rf"(?:^|\W){re.escape(kw)}(?:\W|$)"
        regex = _get_compiled_pattern(pattern)
        count += len(regex.findall(text))
    return count


def _kw_any(keywords: List[str], text: str) -> bool:
    """Check if any of the keywords exist in the text."""
    return _kw_count(keywords, text) > 0


def _detect_emotion(text: str) -> Optional[str]:
    """
    Simple emotion detection based on keyword lists.
    Returns 'positive', 'negative', or None.
    """
    positive = ["рад", "счастлив", "люблю", "смешно", "весело", "классно", "здорово"]
    negative = ["груст", "зл", "больн", "страш", "плохо", "ужасно", "тоск"]
    
    if _kw_any(positive, text):
        return "positive"
    if _kw_any(negative, text):
        return "negative"
    return None
