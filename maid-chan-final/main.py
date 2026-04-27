"""Digital Human v10 -- Maid Companion Main Entry Point.

FastAPI + uvicorn + httpx (async SSE) / SQLite WAL / llama.cpp backend
Consolidated architecture with clean module separation:
  - app/core/     : Application context, DI container
  - app/api/      : HTTP endpoints (FastAPI routers)
  - app/repositories/ : Data access layer
  - app/services/ : Business logic services
  - app/utils/    : Pure helper functions

v10.0 fixes:
  - Atomic persist_and_apply with transaction chaining
  - Background cleanup of stale pending messages
  - SQL injection prevention via allowlist validation
  - Thread-safe caches with proper locking
  - Eliminated circular dependencies via AppContext
"""

from __future__ import annotations
import asyncio
import json
import os
import sys
import time
import traceback
from contextlib import asynccontextmanager
from datetime import datetime
from typing import AsyncGenerator, Dict, Any, Optional

# FastAPI imports
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles

# Local imports (no circular dependencies)
from app.utils.logging import _log, _log_exc, _log_warn
from app.utils.patterns import _kw_any, _kw_count, _detect_emotion
from app.db import db, init_db, set_db_path
from app.core.context import get_context, init_context, shutdown_context
from app.personality import _SEED, _HOST_ARCHY

VERSION = "10.0.0"

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "memory.db")
CFG_PATH = os.path.join(BASE_DIR, "config.json")
TOKEN_PATH = os.path.join(BASE_DIR, "app_token.txt")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOGS_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

def load_config() -> Dict[str, Any]:
    """Load configuration from JSON file."""
    try:
        with open(CFG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        _log_warn("config.json not found, using defaults")
        return {}
    except Exception as e:
        _log_exc("load_config failed", e)
        return {}


def load_app_token() -> str:
    """Load API token from file."""
    try:
        with open(TOKEN_PATH, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        _log_warn("app_token.txt not found, auth disabled")
        return ""
    except Exception as e:
        _log_exc("load_app_token failed", e)
        return ""


# ─────────────────────────────────────────────────────────────────────────────
#  LLM HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _llm_url() -> str:
    """Get LLM server URL from config."""
    cfg = load_config()
    return cfg.get("llm_url", "http://127.0.0.1:8080")


async def _stream(messages: list) -> AsyncGenerator[str, None]:
    """Stream tokens from LLM server via SSE."""
    import httpx
    
    url = f"{_llm_url()}/completion"
    cfg = load_config()
    profile = cfg.get("inference", {}).get("active_profile", "daily")
    inf_cfg = cfg.get("inference", {}).get("profiles", {}).get(profile, {})
    
    payload = {
        "prompt": _build_prompt_string(messages),
        "temperature": inf_cfg.get("temperature", 0.7),
        "n_predict": inf_cfg.get("n_predict", 512),
        "stream": True
    }
    
    try:
        async with httpx.AsyncClient(timeout=180.0) as client:
            async with client.stream("POST", url, json=payload) as resp:
                if resp.status_code != 200:
                    yield f"[LLM error: {resp.status_code}]"
                    return
                
                async for line in resp.aiter_lines():
                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data)
                            content = chunk.get("content", "")
                            if content:
                                yield content
                        except json.JSONDecodeError:
                            continue
                            
    except httpx.TimeoutException:
        _log("LLM timeout")
        yield "[Тайм-аут -- модель не ответила за 3 минуты]"
    except Exception as e:
        _log_exc("LLM stream error", e)
        yield f"[Ошибка LLM: {e}]"


def _build_prompt_string(messages: list) -> str:
    """Convert messages list to prompt string for llama.cpp."""
    result = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            result.append(f"### System:\n{content}\n")
        elif role == "user":
            result.append(f"### User:\n{content}\n")
        elif role == "assistant":
            result.append(f"### Assistant:\n{content}\n")
    result.append("### Assistant:\n")
    return "".join(result)


# ─────────────────────────────────────────────────────────────────────────────
#  PROMPT BUILDING
# ─────────────────────────────────────────────────────────────────────────────

def build_prompt(uid: str, cog: Any, ltm_facts: list) -> tuple:
    """
    Build system prompt (static + dynamic parts).
    
    Returns:
        Tuple of (static_system_prompt, dynamic_system_prompt)
    """
    from app.repositories import (
        UserStateRepository, 
        MemoryRepository, 
        LongTermMemoryRepository,
        KeyMemoryRepository,
        RPSceneRepository
    )
    
    # Load state
    state = UserStateRepository.get(uid) or {}
    
    # Static part: personality seed + host profile
    static_parts = [_SEED]
    if uid == "master":
        static_parts.append(_HOST_ARCHY)
    static_prompt = "\n".join(static_parts)
    
    # Dynamic part: current state, recent memories, LTM facts
    dynamic_parts = []
    
    # Current emotional state
    mood = state.get("mood", 0.5)
    trust = state.get("trust", 0.5)
    affection = state.get("affection", 0.0)
    dynamic_parts.append(
        f"# ТЕКУЩЕЕ СОСТОЯНИЕ\nНастроение: {mood:.2f}, Доверие: {trust:.2f}, Привязанность: {affection:.2f}"
    )
    
    # RP scene if active
    rp_scene = RPSceneRepository.get(uid)
    if rp_scene.get("mode") != "normal":
        dynamic_parts.append(
            f"\n# RP СЦЕНА\nРежим: {rp_scene['mode']}\nЛокация: {rp_scene['location']}\nСценарий: {rp_scene['scenario']}"
        )
    
    # Recent key memories (evolution anchors)
    key_mems = KeyMemoryRepository.get_recent(uid, limit=3)
    if key_mems:
        mem_texts = [f"- {m['description']} (интенсивность: {m['intensity']:.2f})" for m in key_mems]
        dynamic_parts.append("\n# НЕДАВНИЕ ВАЖНЫЕ МОМЕНТЫ\n" + "\n".join(mem_texts))
    
    # LTM facts (relevant to current conversation)
    if ltm_facts:
        fact_texts = [f"- {f['fact']}" for f in ltm_facts[:5]]
        dynamic_parts.append("\n# ПАМЯТЬ (факты о пользователе)\n" + "\n".join(fact_texts))
    
    dynamic_prompt = "\n".join(dynamic_parts)
    
    return static_prompt, dynamic_prompt


# ─────────────────────────────────────────────────────────────────────────────
#  MEMORY OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

def get_ltm_relevant(uid: str, query: str, limit: int = 8) -> list:
    """
    Get relevant long-term memories for a query.
    
    v10.0: Simple keyword-based retrieval (embeddings optional).
    For production: integrate fastembed for semantic search.
    """
    from app.repositories import LongTermMemoryRepository
    
    # Get all facts
    facts = LongTermMemoryRepository.get_facts(uid, limit=50)
    
    if not facts:
        return []
    
    # Simple keyword scoring
    query_words = set(query.lower().split())
    scored = []
    
    for fact in facts:
        fact_text = fact.get("fact", "").lower()
        importance = fact.get("importance", 0.5)
        
        # Count keyword overlaps
        overlap = len(query_words & set(fact_text.split()))
        score = overlap * 0.3 + importance * 0.7
        
        if overlap > 0 or importance > 0.7:
            scored.append((score, fact))
    
    # Sort by score and return top results
    scored.sort(reverse=True, key=lambda x: x[0])
    return [f for _, f in scored[:limit]]


# ─────────────────────────────────────────────────────────────────────────────
#  CHAT HANDLER (CORE LOGIC)
# ─────────────────────────────────────────────────────────────────────────────

async def handle_chat_stream(uid: str, user_text: str) -> AsyncGenerator[bytes, None]:
    """
    Main chat handler - processes user message and streams response.
    
    This is the core conversation pipeline:
    1. Load context (state, memories)
    2. Save user message as pending
    3. Build prompt with context
    4. Stream LLM response
    5. Post-process (save assistant message, update state, detect key memories)
    6. Cleanup on errors
    """
    from app.repositories import (
        UserStateRepository,
        MemoryRepository,
        UserStateRepository
    )
    from app.services.key_memories import analyze_for_key_memory, persist_and_apply
    
    loop = asyncio.get_running_loop()
    ctx = get_context()
    cfg = ctx.config or load_config()
    limit = cfg.get("memory", {}).get("short_term_limit", 20)
    
    # Send memory trace event
    yield b"data: " + json.dumps({"type": "memory_trace", "step": "recalling"}).encode() + b"\n\n"
    
    # Load LTM facts (blocking I/O → run in executor if needed)
    ltm_facts = await loop.run_in_executor(
        ctx.executor, get_ltm_relevant, uid, user_text, 8
    )
    
    # Send citations if any
    if ltm_facts:
        citations = [
            {"n": i+1, "fact": f["fact"][:240], "importance": round(f["importance"], 2)}
            for i, f in enumerate(ltm_facts[:6])
        ]
        yield b"data: " + json.dumps({"type": "citations", "items": citations}).encode() + b"\n\n"
        yield b"data: " + json.dumps({"type": "memory_trace", "step": "recalled", "count": len(citations)}).encode() + b"\n\n"
    else:
        yield b"data: " + json.dumps({"type": "memory_trace", "step": "empty"}).encode() + b"\n\n"
    
    # Load recent history
    history = await loop.run_in_executor(
        ctx.executor, MemoryRepository.get_recent, uid, limit
    )
    
    # Convert history to message format
    history_msgs = [{"role": m["role"], "content": m["text"]} for m in history]
    
    # Save user message as PENDING
    user_msg_id = await loop.run_in_executor(
        ctx.executor, MemoryRepository.add, uid, "user", user_text, {}, "user_input", "pending"
    )
    
    # Build prompt
    static_prompt, dynamic_prompt = await loop.run_in_executor(
        ctx.executor, build_prompt, uid, None, ltm_facts
    )
    
    # Construct messages for LLM
    messages = (
        [{"role": "system", "content": static_prompt + "\n\n" + dynamic_prompt}]
        + history_msgs
        + [{"role": "user", "content": user_text}]
    )
    
    # Stream response
    full_response = ""
    try:
        async for token in _stream(messages):
            full_response += token
            yield b"data: " + json.dumps({"type": "token", "text": token}).encode() + b"\n\n"
        
        full_response = full_response.strip()
        
        if not full_response or full_response.startswith("[Ошибка") or full_response.startswith("[Тайм-аут"):
            # Error response - discard pending message
            await loop.run_in_executor(ctx.executor, MemoryRepository.discard_pending, uid, user_msg_id)
            yield b"data: " + json.dumps({"type": "error", "message": "LLM failed"}).encode() + b"\n\n"
            return
        
        # Save assistant message
        asst_msg_id = await loop.run_in_executor(
            ctx.executor, MemoryRepository.add, uid, "assistant", full_response, {}, "response", "done"
        )
        
        # Update user state (increment counters)
        new_state = await loop.run_in_executor(
            ctx.executor, UserStateRepository.increment_msg_count, uid, 1, 1
        )
        
        # Detect key memories (evolution)
        await _process_evolution(uid, user_text, full_response, user_msg_id, asst_msg_id, new_state)
        
        # Send done event
        yield b"data: " + json.dumps({
            "type": "done",
            "message": full_response,
            "state": {"msg_count": new_state.get("msg_count", 0)}
        }).encode() + b"\n\n"
        
    except Exception as e:
        _log_exc("chat stream error", e)
        # Discard pending message on error
        try:
            await loop.run_in_executor(ctx.executor, MemoryRepository.discard_pending, uid, user_msg_id)
        except:
            pass
        yield b"data: " + json.dumps({"type": "error", "message": "internal error"}).encode() + b"\n\n"


async def _process_evolution(
    uid: str, 
    user_text: str, 
    reply: str, 
    user_msg_id: int, 
    asst_msg_id: int,
    state: Dict[str, Any]
) -> None:
    """Process evolution (key memory detection) in background."""
    from app.services.key_memories import analyze_for_key_memory, persist_and_apply
    
    loop = asyncio.get_running_loop()
    ctx = get_context()
    
    total_count = state.get("total_msg_count", 0)
    
    # Create cognitive view objects
    class CogView:
        emotion_valence = 0.0
        emotion_tag = ""
        maid_emotion = ""
    
    anchored_ids = []
    
    # Analyze user message
    try:
        km_user = await loop.run_in_executor(
            ctx.executor,
            analyze_for_key_memory,
            uid, "user", user_text, CogView(),
            0.5, user_msg_id, None, total_count
        )
        if km_user:
            rid = await loop.run_in_executor(ctx.executor, persist_and_apply, km_user)
            if rid and km_user.intensity >= 0.8:
                anchored_ids.append(rid)
    except Exception as e:
        _log_exc("user key memory analysis", e)
    
    # Analyze assistant message
    try:
        km_asst = await loop.run_in_executor(
            ctx.executor,
            analyze_for_key_memory,
            uid, "assistant", reply, CogView(),
            0.5, asst_msg_id, None, total_count
        )
        if km_asst:
            rid = await loop.run_in_executor(ctx.executor, persist_and_apply, km_asst)
            if rid and km_asst.intensity >= 0.8:
                anchored_ids.append(rid)
    except Exception as e:
        _log_exc("assistant key memory analysis", e)
    
    if anchored_ids:
        _log("Evolution: %d anchor(s) detected for %s", len(anchored_ids), uid)


# ─────────────────────────────────────────────────────────────────────────────
#  BACKGROUND TASKS
# ─────────────────────────────────────────────────────────────────────────────

async def cleanup_pending_task():
    """Background task: remove stale pending messages every 5 minutes."""
    from app.repositories import MemoryRepository
    
    await asyncio.sleep(300)  # First run after 5 min
    while True:
        try:
            deleted = MemoryRepository.cleanup_stale_pending(max_age_seconds=3600)
            if deleted > 0:
                _log("Cleaned up %d stale pending messages", deleted)
        except Exception as e:
            _log_exc("cleanup_pending_task error", e)
        await asyncio.sleep(300)


# ─────────────────────────────────────────────────────────────────────────────
#  FASTAPI APP
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    _log("Starting Maid Companion v%s", VERSION)
    
    # Initialize database
    set_db_path(DB_PATH)
    init_db()
    
    # Load config and initialize context
    cfg = load_config()
    init_context(cfg, max_workers=16)
    
    # Start background tasks
    cleanup_task = asyncio.create_task(cleanup_pending_task())

    # Start autonomous loops (proactive + diary + tactical goals)
    try:
        from app.services.autonomous import start_autonomous_loops
        from app.services.tactical_goals import start_tactical_goals_loop
        start_autonomous_loops()
        start_tactical_goals_loop()
    except Exception as e:
        _log_exc("Failed to start autonomous loops", e)
    
    _log("Server ready")
    
    yield
    
    # Shutdown
    _log("Shutting down...")
    cleanup_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass
    shutdown_context()


# Create FastAPI app
app = FastAPI(title="Maid Companion", version=VERSION, lifespan=lifespan)

# CORS middleware
cfg = load_config()
cors_origins = cfg.get("server", {}).get("cors_origins", ["http://127.0.0.1:5000"])
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
from app.api.chat import router as chat_router
app.include_router(chat_router)

# Static files (if web directory exists)
WEB_DIR = os.path.join(BASE_DIR, "web")
if os.path.exists(WEB_DIR):
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint - serves index.html or API info."""
    index_path = os.path.join(WEB_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"service": "maid-companion", "version": VERSION, "status": "running"}


# Main entry point
if __name__ == "__main__":
    import uvicorn
    
    cfg = load_config()
    host = cfg.get("server", {}).get("host", "127.0.0.1")
    port = cfg.get("server", {}).get("port", 5000)
    
    _log("Starting uvicorn on %s:%d", host, port)
    uvicorn.run(app, host=host, port=port, log_level="info")
