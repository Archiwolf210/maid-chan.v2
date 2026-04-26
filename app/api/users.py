"""API endpoints for user management."""

from fastapi import APIRouter, HTTPException
from typing import List

router = APIRouter()


@router.get("/api/users")
async def list_users():
    """List all users (admin endpoint)."""
    from app.utils.logging import _log
    from app.repositories import UserStateRepository
    from app.db import db
    
    try:
        with db() as c:
            rows = c.execute("SELECT user_id, created_at, updated_at FROM user_state ORDER BY updated_at DESC").fetchall()
        
        return {
            "users": [
                {"uid": row[0], "created_at": row[1], "updated_at": row[2]}
                for row in rows
            ],
            "count": len(rows)
        }
    except Exception as e:
        _log_exc("list users error", e)
        raise HTTPException(status_code=500, detail="failed to list users")


@router.delete("/api/users/{uid}")
async def delete_user(uid: str):
    """Delete user and all associated data (admin endpoint)."""
    from app.utils.logging import _log, _log_exc
    from app.db import db
    
    try:
        with db() as c:
            # Delete from all tables in order (respecting foreign keys)
            tables = [
                'tactical_goals', 'letters', 'diary_entries', 
                'key_memories', 'long_term_memory', 'memory',
                'rp_scene', 'user_state'
            ]
            for table in tables:
                c.execute(f"DELETE FROM {table} WHERE user_id=?", (uid,))
        
        _log(f"User {uid} deleted")
        return {"status": "ok", "message": f"User {uid} deleted"}
        
    except Exception as e:
        _log_exc(f"delete user {uid} error", e)
        raise HTTPException(status_code=500, detail="delete failed")
