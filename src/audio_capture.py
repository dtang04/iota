"""Microphone capture utilities."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
import logging
import threading
from pathlib import Path
from typing import Optional, List

import numpy as np
import sounddevice as sd
import soundfile as sf


@dataclass
class AudioCaptureConfig:
    """Configuration options for microphone capture."""

    sample_rate: int = 16_000
    channels: int = 1
    dtype: str = "float32"


class MicrophoneRecorder:
    """Records audio from the default input device."""

    def __init__(self, config: Optional[AudioCaptureConfig] = None) -> None:
        self.config = config or AudioCaptureConfig()

    def record(self, duration: float) -> np.ndarray:
        """Record audio for a fixed duration in seconds."""
        frames = int(duration * self.config.sample_rate)
        if frames <= 0:
            raise ValueError("duration must be positive")

        with _input_stream(self.config) as stream: #creates context manager as StreamWrapper instance
            stream.start()
            audio = stream.record(frames) #returns an array of floating-point amplitudes - 1/1600o of a second
        return audio.reshape(-1, self.config.channels)

    def save_wav(self, audio: np.ndarray, output_path: str | Path) -> None:
        """Persist recorded audio to a WAV file."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(
            file=path,
            data=audio,
            samplerate=self.config.sample_rate,
            format="WAV",
        )


@contextlib.contextmanager
def _input_stream(config: AudioCaptureConfig):
    """Context manager that yields an input stream."""
    stream = sd.InputStream( #instantiates new stream instance with params
        samplerate=config.sample_rate,
        channels=config.channels,
        dtype=config.dtype,
    )
    try:
        yield _StreamWrapper(stream)
    finally:
        stream.close()


class _StreamWrapper:
    """Thin wrapper to expose a consistent record API."""

    def __init__(self, stream: sd.InputStream) -> None:
        self._stream = stream

    def start(self) -> None:
        """Start the stream, guarding against repeated starts."""
        if not self._stream.active:
            self._stream.start()

    def record(self, frames: int) -> np.ndarray:
        """Record a fixed number of frames."""
        data, _overflowed = self._stream.read(frames)
        return data

    def close(self) -> None:
        self._stream.close()


logger = logging.getLogger(__name__)


class StreamingMicrophoneRecorder:
    """Capture audio until stop() is invoked, buffering samples incrementally."""

    def __init__(self, config: Optional[AudioCaptureConfig] = None) -> None:
        self.config = config or AudioCaptureConfig()
        self._stream: Optional[sd.InputStream] = None
        self._buffer: List[np.ndarray] = []
        self._lock = threading.Lock()

    def start(self) -> None:
        """Begin capturing audio using a callback-based stream."""
        if self._stream is not None:
            raise RuntimeError("Recorder already running")

        self._buffer = []
        self._stream = sd.InputStream(
            samplerate=self.config.sample_rate,
            channels=self.config.channels,
            dtype=self.config.dtype,
            callback=self._callback,
        )
        self._stream.start()

    def stop(self) -> np.ndarray:
        """Stop capturing and return the buffered audio."""
        if self._stream is None:
            raise RuntimeError("Recorder not running")

        self._stream.stop()
        self._stream.close()
        self._stream = None

        with self._lock:
            if not self._buffer:
                return np.empty((0, self.config.channels), dtype=self.config.dtype)
            audio = np.concatenate(self._buffer, axis=0)
            self._buffer = []
        return audio

    def is_running(self) -> bool:
        return self._stream is not None

    def _callback(self, indata: np.ndarray, frames: int, time, status) -> None:  # type: ignore[override]
        if status:
            logger.warning("Input stream status: %s", status)
        with self._lock:
            self._buffer.append(indata.copy())

