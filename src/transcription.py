"""Speech-to-text adapters (local Whisper only)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Protocol

import numpy as np

logger = logging.getLogger(__name__)


class SpeechToTextError(RuntimeError):
    """Raised when transcription fails."""


@dataclass
class TranscriptionResult:
    """Container for transcription outputs."""

    text: str
    language: Optional[str] = None
    raw: Optional[dict] = None


class SpeechToTextService(Protocol):
    """Interface for speech-to-text providers."""

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> TranscriptionResult:
        """Return the transcription for the given audio clip."""


class WhisperLocalTranscriber:
    """Runs the open-source Whisper model locally."""

    def __init__(self, model_name: str = "base", device: Optional[str] = None) -> None:
        try:
            import whisper  # type: ignore
        except ImportError as exc:  # pragma: no cover - dependency hint
            raise ImportError(
                "OpenAI's whisper package is required for WhisperLocalTranscriber"
            ) from exc

        self._whisper = whisper
        self._model = whisper.load_model(model_name, device=device)

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> TranscriptionResult:
        # Whisper expects float32 numpy array at 16000 Hz; resample handled internally.
        audio = audio.squeeze()
        result = self._model.transcribe(audio, fp16=False)
        text = result.get("text", "").strip()
        if not text:
            raise SpeechToTextError("Whisper returned empty transcription")
        language = result.get("language")
        return TranscriptionResult(text=text, language=language, raw=result)

