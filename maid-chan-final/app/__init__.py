"""Digital Human v10 -- Maid Companion.

Final consolidated package structure (v10.0):
    core/        -- Application context, DI container, global state
    api/         -- FastAPI routers and HTTP handlers
    repositories -- Data access layer (Repository pattern)
    services/    -- Business logic services (key_memories, letters, etc.)
    utils/       -- Pure helper functions (logging, patterns, style_filter)

This structure eliminates circular dependencies and provides clear separation
of concerns between layers.
"""

__version__ = "10.0.0"
