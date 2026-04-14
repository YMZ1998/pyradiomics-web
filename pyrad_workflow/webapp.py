"""Compatibility module for the refactored FastAPI application."""

from .app import app, create_app

__all__ = ["app", "create_app"]
