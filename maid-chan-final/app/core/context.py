"""Application context and dependency injection container.

This module provides a centralized AppContext class that holds all shared
resources (DB connections, HTTP clients, config, caches) to eliminate
circular dependencies between modules.

Usage:
    from app.core.context import get_context
    ctx = get_context()
    config = ctx.config
"""

import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from concurrent.futures import ThreadPoolExecutor
import threading


@dataclass
class AppContext:
    """Global application context (DI container)."""
    
    # Configuration
    config: Dict[str, Any] = field(default_factory=dict)
    
    # Thread pool for blocking I/O operations
    executor: Optional[ThreadPoolExecutor] = None
    
    # In-memory caches (push-based invalidation)
    key_memories_cache: Dict[str, list] = field(default_factory=dict)
    proactive_queue: Dict[str, Any] = field(default_factory=dict)
    
    # Async locks for thread-safe operations
    _lock_key_memories: threading.Lock = field(default_factory=threading.Lock)
    _lock_proactive: threading.Lock = field(default_factory=threading.Lock)
    
    # HTTP client (httpx.AsyncClient) - lazy initialized
    http_client: Any = None
    
    # Embedder state (fastembed) - lazy initialized
    embedder_model: Any = None
    embedder_lock: threading.Lock = field(default_factory=threading.Lock)
    
    def get_key_memories_cached(self, uid: str) -> list:
        """Thread-safe read of key memories cache."""
        with self._lock_key_memories:
            return list(self.key_memories_cache.get(uid, []))
    
    def set_key_memories_cached(self, uid: str, memories: list) -> None:
        """Thread-safe write of key memories cache."""
        with self._lock_key_memories:
            self.key_memories_cache[uid] = memories
    
    def invalidate_key_memories_cache(self, uid: str) -> None:
        """Thread-safe cache invalidation."""
        with self._lock_key_memories:
            self.key_memories_cache.pop(uid, None)
    
    def prepend_key_memory(self, uid: str, memory: dict) -> None:
        """Thread-safe cache prepend (newest first)."""
        with self._lock_key_memories:
            if uid not in self.key_memories_cache:
                self.key_memories_cache[uid] = []
            self.key_memories_cache[uid].insert(0, memory)
            # Keep cache size bounded
            if len(self.key_memories_cache[uid]) > 20:
                self.key_memories_cache[uid] = self.key_memories_cache[uid][:20]
    
    def get_proactive_queue(self, uid: str) -> Any:
        """Thread-safe read of proactive queue."""
        with self._lock_proactive:
            return self.proactive_queue.get(uid)
    
    def set_proactive_queue(self, uid: str, queue: Any) -> None:
        """Thread-safe write of proactive queue."""
        with self._lock_proactive:
            self.proactive_queue[uid] = queue
    
    def clear_proactive_queue(self, uid: str) -> None:
        """Thread-safe queue removal."""
        with self._lock_proactive:
            self.proactive_queue.pop(uid, None)


# Global singleton instance
_context: Optional[AppContext] = None
_context_lock = threading.Lock()


def get_context() -> AppContext:
    """Get or create the global application context."""
    global _context
    if _context is None:
        with _context_lock:
            if _context is None:
                _context = AppContext()
    return _context


def init_context(config: Dict[str, Any], max_workers: int = 16) -> AppContext:
    """Initialize the global context with configuration."""
    global _context
    with _context_lock:
        if _context is None:
            _context = AppContext(
                config=config,
                executor=ThreadPoolExecutor(max_workers=max_workers)
            )
        else:
            _context.config = config
            if _context.executor is None:
                _context.executor = ThreadPoolExecutor(max_workers=max_workers)
    return _context


def shutdown_context() -> None:
    """Shutdown the context and release resources."""
    global _context
    if _context is not None:
        if _context.executor is not None:
            _context.executor.shutdown(wait=False)
        if _context.http_client is not None:
            try:
                asyncio.get_running_loop().create_task(_context.http_client.aclose())
            except RuntimeError:
                pass  # No event loop running
        _context = None
