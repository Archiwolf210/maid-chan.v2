"""Lightweight Pydantic types for the v9.4 evolution domain.

These exist so the new code paths (key_memories detector, evolution loader,
monthly_arc consolidator, /api/evolution endpoint) can pass typed payloads
between modules without dragging the whole codebase onto Pydantic.

Existing dict-based state in main.py / app.memory.py is intentionally NOT
touched — the goal is additive typing for the *new* surfaces, not a refactor.
"""
from __future__ import annotations
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class EvolutionState(BaseModel):
    """Slow-changing personality dimensions tracked per user.

    Lives in `user_state` columns; loaded by `load_evolution_state` and
    surfaced through /api/chat done_payload + /api/evolution.

    All four traits start at 0.0 for a fresh user (a "blank slate program")
    and creep upward through key_memory events. Hard-clamped to [0, 1].
    """
    humanity_level:   float = Field(default=0.0, ge=0.0, le=1.0,
        description="0 = procedural protocol voice; 1 = fully fluid emotional voice")
    self_awareness:   float = Field(default=0.0, ge=0.0, le=1.0,
        description="how clearly Maid notices her own internal shifts")
    affection:        float = Field(default=0.0, ge=0.0, le=1.0,
        description="depth of attachment to *this* specific user")
    software_version: str   = Field(default="1.0.0",
        description="cosmetic semver bumped on milestones (purely UX)")

    def voice_band(self) -> str:
        """Map humanity_level to a coarse 'voice band' used by build_prompt
        and style_filter. We resist fine-grained branching — four bands keep
        the LLM stable across small day-to-day fluctuations."""
        h = self.humanity_level
        if h < 0.20: return "protocol"   # robotic, terse, technical metaphors
        if h < 0.50: return "thawing"    # protocol with cracks of empathy
        if h < 0.80: return "warm"       # full emotional palette, our default
        return "intimate"                 # philosophy, deep personal voice


class KeyMemory(BaseModel):
    """A single anchor moment. Detected through cognitive frame signals
    (importance + emotion_valence + intent), NEVER regex on raw text."""
    id: Optional[int] = None
    user_id: str
    ts: int
    event_type: str   # see EVENT_TYPES below
    description: str  # short human description for prompt + UI
    intensity: float = Field(ge=0.0, le=1.0)
    source_msg_id: Optional[int] = None
    traits_delta: Dict[str, float] = Field(default_factory=dict,
        description="per-trait deltas applied to user_state")


# Closed enum kept as a tuple to stay JSON-friendly in traits_json blobs.
EVENT_TYPES: tuple = (
    "breakthrough",   # high-importance reflective moment, fresh self-insight
    "tender",         # warm/affectionate exchange, low intensity, high warmth
    "rupture",        # strong negative valence, conflict, hurt
    "milestone",      # numeric milestone (50/100/500 messages, NN-day arc)
    "rp_first",       # first transition into RP / NSFW mode for this user
    "reveal",         # user shared something personal, high openness signal
)


class MonthlyArc(BaseModel):
    """One LLM-summarized month, persisted in monthly_arcs."""
    user_id: str
    year_month: str           # 'YYYY-MM'
    arc: str
    ts: int


class DiaryMeta(BaseModel):
    """Optional structured metadata stored alongside a diary entry.
    Persisted as JSON in diary_entries.metadata — defaults to {} on legacy rows."""
    humanity_level:   Optional[float] = None
    software_version: Optional[str]   = None
    key_memory_ids:   List[int]       = Field(default_factory=list)
    voice_band:       Optional[str]   = None
    trigger_reason:   Optional[str]   = None  # v9.5


class Letter(BaseModel):
    """A spontaneous first-person note Maid composes on emotional weight.
    Distinct from chat (longer, more reflective) and diary (single moment,
    not whole day). Stored in `letters` table; surfaced via /api/letters."""
    id: Optional[int] = None
    user_id: str
    key_memory_id: Optional[int] = None
    ts: int
    body: str
    status: str = "delivered"          # 'delivered'|'sealed'|'seen'
    triggered_by: str = "anchor"        # 'anchor'|'milestone'|'evening'
    seen_at: Optional[int] = None


# Trigger types the letters subsystem accepts. Closed enum.
LETTER_TRIGGERS: tuple = ("anchor", "milestone", "evening")
