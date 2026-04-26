"""API router aggregator.

This module collects all API routers and provides a single include point for main.py.
"""

from fastapi import APIRouter
from app.api.chat import router as chat_router
from app.api.users import router as users_router

api_router = APIRouter()

# Include all routers with their prefixes
api_router.include_router(chat_router)
api_router.include_router(users_router, prefix="/api/users", tags=["users"])
