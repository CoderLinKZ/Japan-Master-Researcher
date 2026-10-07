"""FastAPI boundary for the durable JMR workflow."""

from .app import create_app

__all__ = ["create_app"]
