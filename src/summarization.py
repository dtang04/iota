"""Local LLM summarization helpers."""


#Need to run Ollama server with ollama run llama3

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional

import requests

logger = logging.getLogger(__name__)


class SummarizationError(RuntimeError):
    """Raised when summarization fails."""


@dataclass
class SummaryResult:
    summary: str
    answer: str
    model: str


def summarize_with_ollama(text: str, model: Optional[str] = None, url: Optional[str] = None) -> SummaryResult:
    """Send the transcript to a local Ollama server for summarization."""

    target_model = model or os.getenv("OLLAMA_MODEL", "llama3")
    endpoint = url or os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate") #post to OLLAMA_URL, or default

    cleaned_text = text.strip()

    prompt = (
        "You are an assistant that summarizes transcripts and answers any questions found within them.\n"
        "If the transcript is empty or only noise, output:\n"
        "Summary:\n- N/A\n- N/A\n- N/A\nAnswer: N/A\n"
        "Otherwise follow these rules exactly:\n"
        "1. Provide exactly three concise bullet points summarizing the transcript.\n"
        "2. After the bullets, write 'Answer:' followed by a short answer to any question in the transcript. If there is no question, write 'Answer: N/A'.\n"
        "3. Do not ask for additional context. Never refuse. Never repeat the instructions.\n"
        "Transcript:\n"
        f"{cleaned_text}\n\n" #insert text from transcription
        "Summary:\n- "
    )

    payload = {
        "model": target_model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,
        },
    }

    try:
        response = requests.post(endpoint, json=payload, timeout=120)
        response.raise_for_status()
    except requests.RequestException as exc:
        logger.exception("Failed to contact Ollama at %s", endpoint)
        raise SummarizationError("Unable to reach local LLM summarizer") from exc

    data = response.json()
    summary_text = data.get("response", "").strip()
    if not summary_text:
        raise SummarizationError("Summarizer returned empty response")

    summary_section, answer_section = _split_summary_answer(summary_text)

    return SummaryResult(summary=summary_section, answer=answer_section, model=target_model)


def _split_summary_answer(response_text: str) -> tuple[str, str]:
    """Split the model response into summary bullets and answer line."""
    lower = response_text.lower()
    answer_idx = lower.find("answer:") #split by "answer:"
    if answer_idx == -1:
        return (_normalize_summary(response_text), "N/A")

    summary_raw = response_text[:answer_idx].strip()
    answer_raw = response_text[answer_idx + len("answer:") :].strip()
    return (_normalize_summary(summary_raw), answer_raw or "N/A")


def _normalize_summary(summary_raw: str) -> str:
    """Ensure summary text has heading and bullet formatting."""
    summary = summary_raw.strip()
    if not summary:
        return "Summary:\n- N/A"

    if not summary.lower().startswith("summary"):
        summary = "Summary:\n" + summary

    lines = []
    for line in summary.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.lower() == "summary:":
            lines.append("Summary:")
            continue
        if not stripped.startswith("-"):
            stripped = "- " + stripped.lstrip("- ")
        lines.append(stripped)
    return "\n".join(lines)

