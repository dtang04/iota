from __future__ import annotations

import io
import subprocess
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.config import load_environment
from src.summarization import SummarizationError, summarize_with_ollama
from src.tokenization import LLMTokenizer
from src.transcription import (
    OpenAIWhisperTranscriber,
    SpeechToTextError,
    WhisperLocalTranscriber,
)

load_environment()

app = FastAPI(title="Iota Mic-to-Token")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

static_dir = Path(__file__).parent / "web"
app.mount("/static", StaticFiles(directory=static_dir), name="static")


class TranscriptionResponse(BaseModel):
    transcript: str
    language: Optional[str]
    tokens: list[int]
    token_count: int
    encoding_name: str
    summary: Optional[str] = None
    answer: Optional[str] = None


def _load_audio_bytes(data: bytes) -> tuple[np.ndarray, int]:
    try:
        audio, sample_rate = sf.read(io.BytesIO(data), dtype="float32")
    except Exception:
        converted = _convert_with_ffmpeg(data)
        try:
            audio, sample_rate = sf.read(io.BytesIO(converted), dtype="float32")
        except Exception as exc:  # pragma: no cover - invalid uploads
            raise HTTPException(status_code=400, detail=f"Unable to read audio file: {exc}") from exc

    if audio.ndim == 1:
        audio = audio[:, np.newaxis]
    return audio, int(sample_rate)


def _convert_with_ffmpeg(data: bytes) -> bytes:
    try:
        process = subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                "pipe:0",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-f",
                "wav",
                "pipe:1",
            ],
            input=data,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:  # pragma: no cover - system config
        raise HTTPException(status_code=500, detail="ffmpeg conversion failed; ensure ffmpeg is installed") from exc
    return process.stdout


def _normalize_bool(value: Optional[str]) -> bool:
    if value is None:
        return False
    return value.lower() in {"true", "1", "yes", "on"}


def _select_transcriber(provider: str, openai_model: Optional[str], whisper_model: Optional[str]):
    if provider == "openai":
        return OpenAIWhisperTranscriber(model=openai_model or None)
    if provider == "whisper-local":
        return WhisperLocalTranscriber(model_name=whisper_model or "base")
    raise HTTPException(status_code=400, detail=f"Unsupported provider: {provider}")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(static_dir / "index.html")


@app.post("/transcribe", response_model=TranscriptionResponse) #transcribe endpoint
async def transcribe_audio(
    file: UploadFile = File(...),
    provider: str = Form("openai"),
    openai_model: Optional[str] = Form(None),
    whisper_model: Optional[str] = Form("base"),
    tokenizer_model: Optional[str] = Form(None),
    encoding_name: Optional[str] = Form(None),
    summarize: Optional[str] = Form("false"),
    ollama_model: Optional[str] = Form("llama3"),
    ollama_url: Optional[str] = Form("http://localhost:11434/api/generate"),
) -> TranscriptionResponse:
    audio_bytes = await file.read()
    audio, sample_rate = _load_audio_bytes(audio_bytes)

    try:
        transcriber = _select_transcriber(provider, openai_model, whisper_model)
        transcription = transcriber.transcribe(audio, sample_rate)
    except SpeechToTextError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    tokenizer = LLMTokenizer(model=tokenizer_model or None, encoding_name=encoding_name or None)
    token_result = tokenizer.encode(transcription.text)

    summary_text: Optional[str] = None
    answer_text: Optional[str] = None
    if _normalize_bool(summarize):
        try:
            summary = summarize_with_ollama(
                transcription.text,
                model=ollama_model or None,
                url=ollama_url or None,
            )
        except SummarizationError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        else:
            summary_text = summary.summary
            answer_text = summary.answer

    return TranscriptionResponse(
        transcript=transcription.text,
        language=transcription.language,
        tokens=token_result.tokens,
        token_count=token_result.count(),
        encoding_name=token_result.encoding_name,
        summary=summary_text,
        answer=answer_text,
    )

