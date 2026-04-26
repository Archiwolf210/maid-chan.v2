"""
Pattern matching utilities for keyword detection and emotion analysis.
Extracted from main.py to eliminate circular dependencies.
"""
import re
from typing import List, Dict, Tuple

# Compiled regex cache for performance
_KW_CACHE: Dict[str, re.Pattern] = {}


def _get_kw_pattern(keyword: str) -> re.Pattern:
    """Get or compile a word-boundary pattern for a keyword."""
    if keyword not in _KW_CACHE:
        # Match keyword only on word boundaries (not as substring)
        # Handles: "счастье" matches, but "несчастье" does not
        pattern = re.compile(rf"(?:^|\W)({re.escape(keyword)})(?:\W|$)", re.IGNORECASE)
        _KW_CACHE[keyword] = pattern
    return _KW_CACHE[keyword]


def _kw_count(keywords: List[str], text: str) -> int:
    """
    Count how many keywords appear in text with proper word boundaries.
    
    Args:
        keywords: List of keywords to search for
        text: Text to search in
        
    Returns:
        Number of unique keywords found (with word boundary matching)
    """
    count = 0
    text_lower = text.lower()
    for kw in keywords:
        if len(kw) < 2:
            continue
        pattern = _get_kw_pattern(kw.lower())
        if pattern.search(text_lower):
            count += 1
    return count


def _kw_any(keywords: List[str], text: str) -> bool:
    """
    Check if any keyword appears in text with proper word boundaries.
    
    Args:
        keywords: List of keywords to search for
        text: Text to search in
        
    Returns:
        True if at least one keyword is found
    """
    return _kw_count(keywords, text) > 0


def _detect_emotion(text: str) -> Tuple[str, float]:
    """
    Detect dominant emotion in text using keyword matching.
    
    Args:
        text: Input text to analyze
        
    Returns:
        Tuple of (emotion_tag, confidence_score)
        emotion_tag: One of 'joy', 'sadness', 'anger', 'fear', 'surprise', 'neutral'
        confidence_score: 0.0 to 1.0
    """
    # Emotion keyword dictionaries (Russian)
    _EMOTIONS = {
        'joy': ['радость', 'счастье', 'весело', 'замечательно', 'прекрасно', 
                'отлично', 'восхищён', 'ликую', 'доволен', 'смех', 'хохот',
                'улыбка', 'смеюсь', 'рада', 'рад'],
        'sadness': ['грусть', 'печаль', 'тоска', 'уныние', 'мрачно', 'тяжело',
                    'больно', 'страдание', 'рыдаю', 'плачу', 'слёзы', 'депрессия',
                    'подавлен', 'разбит', 'устал', 'опустошён'],
        'anger': ['злость', 'гнев', 'ярость', 'бешенство', 'возмущение', 'ненависть',
                  'раздражение', 'злолю', 'бесят', 'достали', 'надоело', 'терпеть не могу'],
        'fear': ['страх', 'ужас', 'паника', 'тревога', 'боязнь', 'испуг', 'нервничаю',
                 'волнуюсь', 'боюсь', 'страшно', 'жутко', 'пугающе'],
        'surprise': ['удивление', 'неожиданно', 'внезапно', 'поразительно', 'изумлён',
                     'шокирован', 'не верю', 'обалдеть', 'ого', 'ничего себе']
    }
    
    text_lower = text.lower()
    scores: Dict[str, int] = {}
    
    for emotion, keywords in _EMOTIONS.items():
        count = _kw_count(keywords, text_lower)
        if count > 0:
            scores[emotion] = count
    
    if not scores:
        return ('neutral', 0.0)
    
    # Find dominant emotion
    dominant = max(scores.items(), key=lambda x: x[1])
    emotion_tag = dominant[0]
    raw_score = dominant[1]
    
    # Normalize score (cap at 1.0, scale by total matches)
    total_matches = sum(scores.values())
    confidence = min(1.0, (raw_score / total_matches) * (1 + min(total_matches, 5) * 0.1))
    
    return (emotion_tag, round(confidence, 2))
