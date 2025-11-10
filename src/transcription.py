"""Speech-to-text adapters."""

from __future__ import annotations

import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol

import numpy as np
import soundfile as sf

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


class OpenAIWhisperTranscriber:
    """Calls OpenAI Whisper API to transcribe audio."""

    def __init__(self, model: Optional[str] = None) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - dependency hint
            raise ImportError(
                "openai package is required for OpenAIWhisperTranscriber"
            ) from exc

        self._client = OpenAI()
        self._model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini-transcribe")

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> TranscriptionResult: #audio is from output of MicrophoneRecorder.record
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = Path(tmp.name)
            sf.write(tmp_path, audio, sample_rate) #writes audio to temporary file
        try:
            with open(tmp_path, "rb") as file_handle:
                response = self._client.audio.transcriptions.create( #creates transcription request to OpenAI API, returns AudioTranscription object
                    model=self._model,
                    file=file_handle,
                )
        except Exception as exc:  # pragma: no cover - passthrough
            logger.exception("OpenAI transcription failed")
            raise SpeechToTextError("OpenAI transcription failed") from exc
        finally:
            tmp_path.unlink(missing_ok=True)

        text = getattr(response, "text", None)
        if not text:
            raise SpeechToTextError("No transcription text returned")
        language = getattr(response, "language", None)
        return TranscriptionResult(text=text, language=language, raw=_maybe_to_dict(response))


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
        original_dtype = audio.dtype
        if audio.ndim > 1: #If X > 1
            audio = audio.mean(axis=1).astype(original_dtype, copy=False) #average across channels X
        else:
            audio = audio.squeeze() #Collapse dimensions of audio array (N, 1) to 1D (N,) - 1 is the number of channels
        result = self._model.transcribe(audio, fp16=False)
        text = result.get("text", "").strip()
        if not text:
            raise SpeechToTextError("Whisper returned empty transcription")
        language = result.get("language")
        return TranscriptionResult(text=text, language=language, raw=result)


def _maybe_to_dict(response: object) -> Optional[dict]: #convert OpenAI's AudioTranscription object to dict
    """Best-effort conversion of OpenAI response objects to dicts."""
    if response is None:
        return None
    if isinstance(response, dict):
        return response
    if hasattr(response, "model_dump"):
        return response.model_dump()
    if hasattr(response, "__dict__"):
        return dict(response.__dict__)
    return None

