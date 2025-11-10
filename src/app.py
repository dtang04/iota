"""CLI entrypoint: microphone -> tokens pipeline."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from audio_capture import AudioCaptureConfig, MicrophoneRecorder
from config import load_environment
from tokenization import LLMTokenizer
from transcription import (
    OpenAIWhisperTranscriber,
    SpeechToTextError,
    SpeechToTextService,
    TranscriptionResult,
    WhisperLocalTranscriber,
)
from summarization import SummarizationError, summarize_with_ollama

load_environment()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("mic-to-tokens")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record audio and convert to LLM tokens.")
    parser.add_argument("--duration", type=float, default=5.0, help="Recording duration in seconds")
    parser.add_argument("--sample-rate", type=int, default=16_000, help="Audio sample rate")
    parser.add_argument("--channels", type=int, default=1, help="Number of audio channels")
    parser.add_argument("--provider", choices=("openai", "whisper-local"), default="openai", help="Transcription backend")
    parser.add_argument("--openai-model", type=str, default=None, help="Override OpenAI Whisper model id")
    parser.add_argument("--whisper-model", type=str, default="base", help="Local Whisper model name")
    parser.add_argument("--tokenizer-model", type=str, default=None, help="LLM model name for tokenization")
    parser.add_argument("--encoding", type=str, default=None, help="Explicit tiktoken encoding name")
    parser.add_argument("--output-wav", type=str, default=None, help="Optional path to store raw audio")
    parser.add_argument("--print-tokens", action="store_true", help="Print token ids instead of just counts")
    parser.add_argument("--summarize", action="store_true", help="Summarize transcript with local LLM (Ollama).")
    parser.add_argument("--ollama-model", type=str, default=None, help="Ollama model name (default: llama3)")
    parser.add_argument("--ollama-url", type=str, default=None, help="Ollama HTTP endpoint (default: http://localhost:11434/api/generate)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = AudioCaptureConfig(
        sample_rate=args.sample_rate,
        channels=args.channels,
    )
    recorder = MicrophoneRecorder(config) #instansiates Micropohone Recorder

    logger.info("Recording audio for %.2f seconds ...", args.duration)
    audio = recorder.record(args.duration)
    logger.info("Captured audio with shape %s", audio.shape)

    if args.output_wav:
        recorder.save_wav(audio, args.output_wav)
        logger.info("Saved audio to %s", Path(args.output_wav).resolve())

    transcriber = build_transcriber(args, config.sample_rate)
    try:
        transcription = transcriber.transcribe(audio, config.sample_rate)
    except SpeechToTextError as exc:
        logger.error("Transcription failed: %s", exc)
        return 1

    logger.info("Transcript: %s", transcription.text)
    tokenizer = LLMTokenizer(model=args.tokenizer_model, encoding_name=args.encoding)
    token_result = tokenizer.encode(transcription.text)
    logger.info(
        "Tokenized using encoding '%s': %s tokens",
        token_result.encoding_name,
        token_result.count(),
    )
    if args.print_tokens:
        logger.info("Token ids: %s", token_result.tokens)

    if args.summarize:
        try:
            summary = summarize_with_ollama(
                transcription.text,
                model=args.ollama_model,
                url=args.ollama_url,
            )
        except SummarizationError as exc:
            logger.error("Summarization failed: %s", exc)
        else:
            logger.info("Summary (%s):\n%s", summary.model, summary.summary)
            logger.info("Answer: %s", summary.answer)
    return 0


def build_transcriber(args: argparse.Namespace, sample_rate: int) -> SpeechToTextService:
    if args.provider == "openai":
        return OpenAIWhisperTranscriber(model=args.openai_model)
    if args.provider == "whisper-local":
        return WhisperLocalTranscriber(model_name=args.whisper_model)
    raise ValueError(f"Unsupported provider: {args.provider}")


if __name__ == "__main__":
    sys.exit(main())

