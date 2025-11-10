"""Environment configuration utilities."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

logger = logging.getLogger(__name__)


def load_environment(dotenv_path: Optional[Path] = None) -> None:
    """Load environment variables from a .env file if present."""
    default_path = Path(__file__).resolve().parent.parent / ".env"
    target = dotenv_path or default_path
    loaded = load_dotenv(dotenv_path=target, override=False)
    if loaded:
        logger.debug("Loaded environment variables from %s", target)
    else:
        logger.debug("No .env file found at %s (skipping)", target)

    if not os.getenv("OPENAI_API_KEY"):
        logger.warning(
            "OPENAI_API_KEY is not set. OpenAI Whisper transcription will fail until it is configured."
        )

