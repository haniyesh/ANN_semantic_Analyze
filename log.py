"""Centralised logging setup.

Call `setup_logging()` once at process startup (main.py / server.py entrypoints).
Use `get_logger(__name__)` in every module instead of bare print().

Level hierarchy used in this project:
  DEBUG    — per-item pipeline trace (disabled in production)
  INFO     — normal operational events (news scored, cache written…)
  WARNING  — degraded but serviceable (DB unavailable, model missing…)
  ERROR    — item lost or endpoint failed (exception caught and handled)
  CRITICAL — fatal startup failure
"""
import logging
import os
import sys
from pathlib import Path


def setup_logging(
    level: str | None = None,
    log_file: str | None = None,
) -> None:
    """Configure root logger.  Call once before any module imports log."""
    level = level or os.getenv("LOG_LEVEL", "INFO")
    fmt   = "%(asctime)s %(name)-28s %(levelname)-8s %(message)s"
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=fmt,
        datefmt="%Y-%m-%dT%H:%M:%S",
        handlers=handlers,
        force=True,
    )
    # Suppress chatty third-party loggers
    for noisy in (
        "transformers", "httpx", "httpcore", "asyncio",
        "aiohttp", "telethon", "uvicorn.access",
        "hf_xet", "huggingface_hub",
    ):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
