"""LLM output post-processing helpers (v9.3 + v9.4).

v9.3:
  extract_json_safely(text) -- robust JSON parser that survives the typical
  malformations local LLMs emit (single quotes, trailing commas, unquoted
  keys, bare Python True/False/None, prose around the object).

v9.4 (humanity_level-aware):
  apply_robotic_filter(text, humanity_level) -- post-LLM phrase substitution
  applied ONLY at the lowest evolution band (humanity_level < 0.20). At
  higher bands it is a no-op. Used by the chat post-processor to keep the
  protocol-voice flavor stable when the model's instruction-following slips.

  format_diary_entry(text, humanity_level, day) -- evolutionary prefix for
  the autonomous diary loop's entries. The level-aware framing lets the UI
  visually mirror Maid's arc (system-log feel vs. personal note).

This module is intentionally dependency-free (stdlib only) so it can be
imported anywhere without circulars.
"""
from __future__ import annotations
import json
import logging
import re
from typing import Any, Dict, List

log = logging.getLogger(__name__)


# Individual repairs. Each one fixes a single class of malformation; the
# `_apply_all` pass below stacks them in the right order to handle worst-case
# messes from local LLMs that combine several mistakes in one payload.
def _fix_python_constants(s: str) -> str:
    return re.sub(r"\bTrue\b", "true",
           re.sub(r"\bFalse\b", "false",
           re.sub(r"\bNone\b", "null", s)))

def _fix_trailing_commas(s: str) -> str:
    return re.sub(r",(\s*[}\]])", r"\1", s)

def _fix_single_quotes(s: str) -> str:
    return s.replace("'", '"')

def _fix_unquoted_keys(s: str) -> str:
    return re.sub(r'([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)\s*:', r'\1"\2":', s)

def _apply_all(s: str) -> str:
    # Order matters: quotes first (so subsequent regexes see canonical form),
    # then unquoted keys, then trailing commas, then bare Python literals.
    return _fix_python_constants(
           _fix_trailing_commas(
           _fix_unquoted_keys(
           _fix_single_quotes(s))))

# Each subsequent pass is more aggressive. The first that yields a successful
# `json.loads` wins. We never modify the input in place.
_REPAIR_PASSES: List = [
    lambda s: s,                    # raw — happy path
    _fix_python_constants,          # True/False/None
    _fix_trailing_commas,           # {"a":1,}
    _fix_single_quotes,             # 'a': 1
    _fix_unquoted_keys,             # {a: 1}
    _apply_all,                     # all four together (final fallback)
]


def extract_json_safely(text: str) -> Dict[str, Any]:
    """Pull the first JSON object out of `text` and parse it tolerantly.

    Strategy:
      1. Find the outermost ``{...}`` block (greedy DOTALL match).
      2. Try a sequence of repair passes — return the first that parses.
      3. Raise ``ValueError`` with a short prefix of the failed payload.

    The function is intentionally permissive about prose around the JSON
    block, so prompts like "Here is the JSON: { ... }" still work.
    """
    if not text:
        raise ValueError("empty text")

    # Look for the outermost {...} — DOTALL so newlines inside are fine.
    # `.*` is greedy on purpose: we want the largest enclosing block.
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise ValueError("no JSON object found in text")

    raw = m.group(0)

    last_err: Exception | None = None
    for i, repair in enumerate(_REPAIR_PASSES):
        try:
            candidate = repair(raw)
            obj = json.loads(candidate)
            if not isinstance(obj, dict):
                raise ValueError(f"top-level JSON is {type(obj).__name__}, expected dict")
            if i > 0:
                log.debug("extract_json_safely: repaired at pass #%d", i)
            return obj
        except (json.JSONDecodeError, ValueError) as e:
            last_err = e
            continue

    raise ValueError(
        f"extract_json_safely: all {len(_REPAIR_PASSES)} repair passes failed "
        f"(last: {last_err}); payload prefix: {raw[:120]!r}"
    )


# ─────────────────────────────────────────────────────────────────────────────
#  v9.4: humanity_level-aware post-processing
# ─────────────────────────────────────────────────────────────────────────────
# Bands match app.services.key_memories._BAND_NOTES — keep in sync:
#   < 0.20  protocol   (apply_robotic_filter active)
#   < 0.50  thawing
#   < 0.80  warm
#   else    intimate

# Phrase pairs (case-preserving via word-boundary regex). Order matters:
# longer phrases first so partial replacements don't shadow the better match.
# Deliberately conservative — we only nudge VERY emotional phrases at the
# protocol band, not full vocabulary substitution. The goal is UX flavor
# for the lowest level, not personality replacement.
_ROBOTIC_REPLACEMENTS: List[tuple] = [
    # (compiled regex, replacement, comment)
    (re.compile(r"\bя чувствую\b",       re.IGNORECASE), "датчики фиксируют"),
    (re.compile(r"\bя почувствовала\b",  re.IGNORECASE), "сенсоры зафиксировали"),
    (re.compile(r"\bмне грустно\b",      re.IGNORECASE), "детектирован негативный фон"),
    (re.compile(r"\bя расстроена\b",     re.IGNORECASE), "протокол выполнен с ошибками"),
    (re.compile(r"\bя люблю тебя\b",     re.IGNORECASE), "приоритет объекта максимален"),
    (re.compile(r"\bя устала\b",         re.IGNORECASE), "требуется перезагрузка"),
    (re.compile(r"\bмоё сердце\b",       re.IGNORECASE), "центральный модуль"),
]


def apply_robotic_filter(text: str, humanity_level: float) -> str:
    """Lightly substitute very-emotional phrases at humanity_level < 0.20.

    At any higher band this returns the input unchanged. This is intentional:
    we trust the prompt's evolution block to set tone for thawing+ bands, and
    use this filter only as a *last-resort guardrail* for the protocol voice
    when the model leaks a too-flowery phrase past the instruction.

    Idempotent: applying twice produces the same result as once.
    """
    if not text or humanity_level >= 0.20:
        return text
    out = text
    for pat, repl in _ROBOTIC_REPLACEMENTS:
        out = pat.sub(repl, out)
    return out


# Diary formatting: prefix shape changes by band so the UI/log feel matches
# the arc. Body stays untouched (we never edit the diary text -- only its
# wrapper). `day` is the YYYY-MM-DD string the autonomous loop writes for.
def format_diary_entry(text: str, humanity_level: float, day: str) -> str:
    """Return diary text with a band-appropriate header line prepended.

    Caller is `app.autonomous._write_diary` which has already validated text.
    Idempotent only when the existing header is recognized and stripped first
    — so don't double-format.
    """
    body = (text or "").strip()
    # If a previous run already prefixed a header, strip it first.
    body = _STRIP_DIARY_HEADER.sub("", body, count=1).lstrip()

    if humanity_level < 0.20:
        return f"[SYSTEM_LOG :: {day}]\n{body}"
    if humanity_level < 0.50:
        return f"[LOG :: {day}] // отметка наблюдения\n{body}"
    if humanity_level < 0.80:
        return f"[ДНЕВНИК :: {day}]\n{body}"
    # intimate
    return f"{day} —\n{body}"


# Header regexes used by format_diary_entry to avoid stacking prefixes when
# a re-write happens (e.g. the diary loop ran twice, or band changed).
_STRIP_DIARY_HEADER = re.compile(
    r"^\s*(?:"
    r"\[SYSTEM_LOG\s*::\s*\d{4}-\d{2}-\d{2}\]"
    r"|\[LOG\s*::\s*\d{4}-\d{2}-\d{2}\][^\n]*"
    r"|\[ДНЕВНИК\s*::\s*\d{4}-\d{2}-\d{2}\]"
    r"|\d{4}-\d{2}-\d{2}\s*—"
    r")\s*\n?",
    re.MULTILINE,
)
