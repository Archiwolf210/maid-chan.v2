"""API endpoints for chat functionality.

This module contains all HTTP handlers related to chat, separated from business logic.
"""

from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import StreamingResponse
import json
import asyncio
from typing import AsyncGenerator

router = APIRouter()


def _jdumps(obj: dict) -> str:
    """Fast JSON dumps."""
    return json.dumps(obj, ensure_ascii=False, separators=(',', ':'))


@router.post("/api/chat")
async def chat_endpoint(request: Request):
    """SSE chat endpoint - main conversation interface."""
    from app.utils.logging import _log, _log_exc
    from main import _chat_sse, load_config
    
    try:
        payload = await request.json()
        uid = payload.get("uid", "default")
        user_text = payload.get("message", "").strip()
        
        if not user_text:
            return StreamingResponse(
                iter([b"data: " + _jdumps({"type": "error", "message": "empty message"}).encode() + b"\n\n"]),
                media_type="text/event-stream"
            )
        
        # Get config for context window
        config = load_config()
        
        async def generate() -> AsyncGenerator[bytes, None]:
            """Stream SSE events from chat pipeline."""
            try:
                async for chunk in _chat_sse(uid, user_text):
                    yield chunk
                    # Check if client disconnected
                    if await request.is_disconnected():
                        _log(f"Client {uid} disconnected, stopping stream")
                        break
            except asyncio.CancelledError:
                _log(f"Chat stream cancelled for {uid}")
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
    return {"status": "ok", "service": "maid-companion"}


@router.post("/api/users/{uid}/reset")
async def reset_user(uid: str):
    """Reset user state and memory (admin operation)."""
    from app.utils.logging import _log
    from app.repositories import (
        UserStateRepository,
        MemoryRepository,
        LongTermMemoryRepository,
        KeyMemoryRepository,
        RPSceneRepository,
        DiaryRepository
    )
    
    try:
        # Delete from all tables
        with __import__('app.db').db() as c:
            c.execute("DELETE FROM memory WHERE user_id=?", (uid,))
            c.execute("DELETE FROM long_term_memory WHERE user_id=?", (uid,))
            c.execute("DELETE FROM key_memories WHERE user_id=?", (uid,))
            c.execute("DELETE FROM diary_entries WHERE user_id=?", (uid,))
            c.execute("DELETE FROM letters WHERE user_id=?", (uid,))
            c.execute("DELETE FROM tactical_goals WHERE user_id=?", (uid,))
            c.execute("DELETE FROM rp_scene WHERE user_id=?", (uid,))
            c.execute("DELETE FROM user_state WHERE user_id=?", (uid,))
        
        _log(f"User {uid} reset complete")
        return {"status": "ok", "message": f"User {uid} reset"}
        
    except Exception as e:
        _log_exc(f"reset user {uid} error", e)
        raise HTTPException(status_code=500, detail="reset failed")


@router.get("/api/users/{uid}/state")
async def get_user_state(uid: str):
    """Get current user state (debug/admin endpoint)."""
    from app.utils.logging import _log
    from app.repositories import UserStateRepository
    
    state = UserStateRepository.get(uid)
    if not state:
        raise HTTPException(status_code=404, detail="user not found")
    
    # Remove sensitive fields if needed
    safe_state = {k: v for k, v in state.items() if k != 'password_hash'}
    return safe_state


@router.get("/api/users/{uid}/memory")
async def get_user_memory(uid: str, limit: int = 20):
    """Get recent user memory (debug/admin endpoint)."""
    from app.utils.logging import _log
    from app.repositories import MemoryRepository
    
    memory = MemoryRepository.get_recent(uid, limit)
    return {"messages": memory, "count": len(memory)}


@router.get("/api/users/{uid}/diary")
async def get_user_diary(uid: str):
    """Get latest diary entry (debug/admin endpoint)."""
    from app.utils.logging import _log
    from app.repositories import DiaryRepository
    
    diary = DiaryRepository.get_latest(uid)
    if not diary:
        return {"entry": None, "message": "no diary entries"}
    
    return {"entry": diary}


@router.get("/api/users/{uid}/rp-scene")
async def get_rp_scene(uid: str):
    """Get current RP scene (debug/admin endpoint)."""
    from app.utils.logging import _log
    from app.repositories import RPSceneRepository
    
    scene = RPSceneRepository.get(uid)
    return scene
