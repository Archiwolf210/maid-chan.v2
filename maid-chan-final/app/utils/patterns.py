"""Pattern matching utilities for text analysis.

Word-boundary keyword matching, emotion detection, and other NLP helpers.
All functions are pure (no side effects) and thread-safe.
"""

import re
from typing import List, Union


def _kw_any(kws: Union[List[str], tuple], text: str) -> bool:
    """
    Check if ANY keyword appears in text with word boundaries.
    
    Uses Unicode-aware word boundaries to avoid false positives
    (e.g., "плохо" should NOT match inside "неплохо").
    
    Args:
        kws: List or tuple of keywords to search for
        text: Text to search in
    
    Returns:
        True if any keyword found with proper word boundaries
    """
    for kw in kws:
        # Word boundary pattern that works with Cyrillic
        pattern = r"(?:^|\W)" + re.escape(kw) + r"(?:\W|$)"
        if re.search(pattern, text, flags=re.IGNORECASE | re.UNICODE):
            return True
    return False


def _kw_count(kws: Union[List[str], tuple], text: str) -> int:
    """
    Count how many keywords appear in text with word boundaries.
    
    Args:
        kws: List or tuple of keywords to search for
        text: Text to search in
    
    Returns:
        Number of distinct keywords found
    """
    count = 0
    for kw in kws:
        pattern = r"(?:^|\W)" + re.escape(kw) + r"(?:\W|$)"
        if re.search(pattern, text, flags=re.IGNORECASE | re.UNICODE):
            count += 1
    return count


# Emotion keyword dictionaries (Russian)
_EMOTION_KEYWORDS = {
    "joy": ["рад", "счастлив", "весел", "ликую", "восхищ", "восторг"],
    "sadness": ["груст", "печал", "тоск", "уныл", "скорб", "рыда"],
    "anger": ["зл", "гнев", "бешен", "ярост", "раздраж", "возмущ"],
    "fear": ["страх", "боюсь", "опаса", "тревож", "паник", "ужас"],
    "surprise": ["удив", "пораж", "неожидан", "внезап", "ошелом"],
    "disgust": ["отвращ", "мерз", "тошн", "гадк", "против"],
    "love": ["люб", "обожа", "дорог", "сердц", "нежн", "ласк"],
    "neutral": ["норм", "обыч", "спокой", "равнодуш", "никак"],
}


def _detect_emotion(text: str) -> str:
    """
    Detect dominant emotion in text based on keyword matching.
    
    Args:
        text: Input text to analyze
    
    Returns:
        Emotion label: 'joy', 'sadness', 'anger', 'fear', 
                      'surprise', 'disgust', 'love', or 'neutral'
    """
    text_lower = text.lower()
    max_count = 0
    detected = "neutral"
    
    for emotion, keywords in _EMOTION_KEYWORDS.items():
        count = _kw_count(keywords, text_lower)
        if count > max_count:
            max_count = count
            detected = emotion
    
    return detected


__all__ = ["_kw_any", "_kw_count", "_detect_emotion"]
