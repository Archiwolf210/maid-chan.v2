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
