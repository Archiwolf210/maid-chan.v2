"""API endpoints for chat and user management.

This module contains all HTTP handlers (FastAPI routers), separated from
business logic. All endpoints are async and use SSE for streaming responses.
"""

from fastapi import APIRouter, Request, HTTPException, Depends
from fastapi.responses import StreamingResponse, JSONResponse
import json
import asyncio
from typing import AsyncGenerator, Optional

from app.utils.logging import _log, _log_exc, _log_warn
from app.repositories import (
    UserStateRepository,
    MemoryRepository,
    LongTermMemoryRepository,
    KeyMemoryRepository,
    RPSceneRepository,
    DiaryRepository
)
from app.db import db, reset_user_data


router = APIRouter()


def _jdumps(obj: dict) -> bytes:
    """Fast JSON dumps to bytes."""
    return json.dumps(obj, ensure_ascii=False).encode("utf-8")


@router.post("/api/chat")
async def chat_endpoint(request: Request):
    """
    SSE chat endpoint - main conversation interface.
    
    Expects JSON: {"uid": str, "message": str}
    Streams SSE events: token, done, error, citations, memory_trace
    """
    # Import chat handler from core (lazy to avoid circular deps)
    from main import handle_chat_stream
    
    try:
        payload = await request.json()
        uid = payload.get("uid", "master")
        user_text = payload.get("message", "").strip()
        
        if not user_text:
            return StreamingResponse(
                iter([b"data: " + _jdumps({"type": "error", "message": "empty message"}) + b"\n\n"]),
                media_type="text/event-stream"
            )
        
        async def generate() -> AsyncGenerator[bytes, None]:
            """Stream SSE events from chat pipeline."""
            try:
                async for chunk in handle_chat_stream(uid, user_text):
                    yield chunk
                    # Check if client disconnected
                    if await request.is_disconnected():
                        _log("Client %s disconnected, stopping stream", uid)
                        break
            except asyncio.CancelledError:
                _log("Chat stream cancelled for %s", uid)
                raise
            except Exception as e:
                _log_exc("chat endpoint error", e)
                yield b"data: " + _jdumps({"type": "error", "message": "internal error"}).encode() + b"\n\n"
        
        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )
        
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="invalid JSON")
    except Exception as e:
        _log_exc("chat endpoint unexpected error", e)
        raise HTTPException(status_code=500, detail="internal error")


@router.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "service": "maid-companion", "version": "10.0.0"}


@router.post("/api/users/{uid}/reset")
async def reset_user(uid: str):
    """Reset user state and memory (admin operation)."""
    try:
        reset_user_data(uid)
        _log("User %s reset complete", uid)
        return {"status": "ok", "message": f"User {uid} reset"}
        
    except Exception as e:
        _log_exc(f"reset user {uid} error", e)
        raise HTTPException(status_code=500, detail="reset failed")


@router.get("/api/users/{uid}/state")
async def get_user_state(uid: str):
    """Get current user state (debug/admin endpoint)."""
    state = UserStateRepository.get(uid)
    if not state:
        raise HTTPException(status_code=404, detail="user not found")
    
    # Remove sensitive fields if needed
    safe_state = {k: v for k, v in state.items() if k != 'password_hash'}
    return safe_state


@router.get("/api/users/{uid}/memory")
async def get_user_memory(uid: str, limit: int = 20):
    """Get recent user memory (debug/admin endpoint)."""
    memory = MemoryRepository.get_recent(uid, limit)
    return {"messages": memory, "count": len(memory)}


@router.get("/api/users/{uid}/diary")
async def get_user_diary(uid: str):
    """Get latest diary entry (debug/admin endpoint)."""
    diary = DiaryRepository.get_latest(uid)
    if not diary:
        return {"entry": None, "message": "no diary entries"}
    
    return {"entry": diary}


@router.get("/api/users/{uid}/rp-scene")
async def get_rp_scene(uid: str):
    """Get current RP scene (debug/admin endpoint)."""
    scene = RPSceneRepository.get(uid)
    return scene


@router.get("/api/users/{uid}/key-memories")
async def get_key_memories(uid: str, limit: int = 10):
    """Get recent key memories (debug/admin endpoint)."""
    memories = KeyMemoryRepository.get_recent(uid, limit)
    return {"memories": memories, "count": len(memories)}


@router.get("/api/users/{uid}/diary/entries")
async def get_diary_entries(uid: str, limit: int = 30):
    """Get list of diary entries (newest first)."""
    from app.repositories.diary import DiaryRepository
    entries = DiaryRepository.list_days(uid, limit)
    return {"entries": entries, "count": len(entries)}


@router.get("/api/users/{uid}/diary/{day}")
async def get_diary_entry(uid: str, day: str):
    """Get specific diary entry by day (YYYY-MM-DD)."""
    from app.repositories.diary import DiaryRepository
    entry = DiaryRepository.get_entry(uid, day)
    if not entry:
        raise HTTPException(status_code=404, detail="entry not found")
    return entry


@router.get("/api/users/{uid}/goals")
async def get_tactical_goals(uid: str):
    """Get active tactical goals."""
    from app.repositories.tactical_goals import TacticalGoalsRepository
    goals = TacticalGoalsRepository.list_active(uid)
    return {"goals": goals, "count": len(goals)}


@router.post("/api/users/{uid}/goals")
async def create_tactical_goal(uid: str, request: Request):
    """Create a new tactical goal (admin/debug)."""
    from app.repositories.tactical_goals import TacticalGoalsRepository
    try:
        payload = await request.json()
        horizon = payload.get("horizon", "day")
        text = payload.get("text", "")
        reasoning = payload.get("reasoning", "")
        
        if not text:
            raise HTTPException(status_code=400, detail="text required")
        
        goal_id = TacticalGoalsRepository.create_goal(uid, horizon, text, reasoning)
        if not goal_id:
            raise HTTPException(status_code=400, detail="invalid horizon")
        
        return {"status": "ok", "goal_id": goal_id}
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="invalid JSON")


@router.post("/api/users/{uid}/goals/{goal_id}/complete")
async def complete_goal(uid: str, goal_id: int):
    """Mark a goal as completed."""
    from app.repositories.tactical_goals import TacticalGoalsRepository
    success = TacticalGoalsRepository.mark_done(uid, goal_id)
    if not success:
        raise HTTPException(status_code=404, detail="goal not found or not active")
    return {"status": "ok"}


@router.get("/api/users/{uid}/letters")
async def get_letters(uid: str, limit: int = 20, include_sealed: bool = False):
    """Get recent letters from Maid."""
    from app.repositories.letters import LettersRepository
    letters = LettersRepository.list_recent(uid, limit, include_sealed)
    return {"letters": letters, "count": len(letters)}


@router.get("/api/users/{uid}/letters/{letter_id}")
async def get_letter(uid: str, letter_id: int):
    """Get specific letter."""
    from app.repositories.letters import LettersRepository
    letter = LettersRepository.get_letter(uid, letter_id)
    if not letter:
        raise HTTPException(status_code=404, detail="letter not found")
    
    # Mark as seen if not already
    if letter.get("seen_at") is None:
        LettersRepository.mark_seen(uid, letter_id)
        letter["seen_at"] = int(time.time())
    
    return letter


@router.post("/api/users/{uid}/letters/{letter_id}/seen")
async def mark_letter_seen(uid: str, letter_id: int):
    """Mark letter as seen."""
    from app.repositories.letters import LettersRepository
    success = LettersRepository.mark_seen(uid, letter_id)
    return {"status": "ok", "marked": success}


@router.get("/api/users/{uid}/proactive")
async def get_proactive_messages(uid: str, consume: bool = True):
    """Get queued proactive messages (if any)."""
    from app.services.autonomous import get_proactive_pending
    items = get_proactive_pending(uid, consume)
    return {"messages": items, "count": len(items)}


@router.get("/api/users/{uid}/evolution")
async def get_evolution_timeline(uid: str, limit: int = 200):
    """Get personality evolution timeline for charting."""
    from app.services.key_memories import get_evolution_timeline as get_timeline
    timeline = get_timeline(uid, limit)
    return {"timeline": timeline, "points": len(timeline)}


@router.delete("/api/users/{uid}/pending-messages")
async def cleanup_pending_messages(uid: str):
    """Manually trigger cleanup of stale pending messages."""
    try:
        deleted = MemoryRepository.cleanup_stale_pending(max_age_seconds=60)
        return {"status": "ok", "deleted": deleted}
    except Exception as e:
        _log_exc("cleanup pending messages", e)
        raise HTTPException(status_code=500, detail="cleanup failed")


@router.get("/api/status")
async def server_status():
    """Get server status and configuration summary."""
    from main import load_config, VERSION
    cfg = load_config()
    
    return {
        "status": "running",
        "version": VERSION,
        "config": {
            "llm_url": cfg.get("llm_url", "unknown"),
            "memory_limit": cfg.get("memory", {}).get("short_term_limit", 20),
            "embedding_enabled": cfg.get("memory", {}).get("embedding_enabled", False),
            "nsfw_mode": cfg.get("nsfw_mode", False),
            "remote_mode": cfg.get("server", {}).get("remote_mode", "local_trusted")
        }
    }


# ─────────────────────────────────────────────────────────────────────────────
#  NSFW MODE ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/api/users/{uid}/nsfw/status")
async def get_nsfw_status_endpoint(uid: str):
    """Get NSFW mode status for a user."""
    from app.services.nsfw import get_nsfw_status as get_status
    status = get_status(uid)
    return status


@router.post("/api/users/{uid}/nsfw/activate")
async def activate_nsfw(uid: str, request: Request):
    """Activate NSFW mode (requires explicit consent)."""
    from app.services.nsfw import activate_nsfw_mode
    try:
        payload = await request.json()
    except:
        payload = {}
    
    explicit_consent = payload.get("explicit_consent", True)
    success, message = activate_nsfw_mode(uid, explicit_consent)
    
    if success:
        return {"status": "ok", "message": message}
    else:
        raise HTTPException(status_code=400, detail=message)


@router.post("/api/users/{uid}/nsfw/deactivate")
async def deactivate_nsfw(uid: str):
    """Deactivate NSFW mode."""
    from app.services.nsfw import deactivate_nsfw_mode
    success, message = deactivate_nsfw_mode(uid)
    
    if success:
        return {"status": "ok", "message": message}
    else:
        raise HTTPException(status_code=400, detail=message)


# ─────────────────────────────────────────────────────────────────────────────
#  DEEP REFLECTION ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/api/users/{uid}/reflection/status")
async def get_reflection_status_endpoint(uid: str):
    """Get deep reflection status for a user."""
    from app.services.reflection import get_reflection_status
    status = get_reflection_status(uid)
    return status


@router.post("/api/users/{uid}/reflection/trigger")
async def trigger_deep_reflection(uid: str):
    """Trigger a deep reflection session manually."""
    from app.services.reflection import perform_deep_reflection
    result = await perform_deep_reflection(uid)
    
    if result is None:
        raise HTTPException(status_code=500, detail="Reflection failed")
    
    if result.get("status") == "error":
        raise HTTPException(status_code=400, detail=result.get("reason", "Unknown error"))
    
    return {"status": "ok", **result}
