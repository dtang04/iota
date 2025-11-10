"""Simple Tkinter front end to control microphone recording."""

from __future__ import annotations

import threading
import tkinter as tk
from tkinter import messagebox

from audio_capture import AudioCaptureConfig, StreamingMicrophoneRecorder
from config import load_environment
from tokenization import LLMTokenizer
from summarization import SummarizationError, summarize_with_ollama
from transcription import (
    OpenAIWhisperTranscriber,
    SpeechToTextError,
    SpeechToTextService,
    WhisperLocalTranscriber,
)

load_environment()


class AudioGuiApp:
    """Tk-based interface with record/stop controls."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Mic to Tokens")

        self.recorder = StreamingMicrophoneRecorder(AudioCaptureConfig())

        self.provider_var = tk.StringVar(value="openai")
        self.openai_model_var = tk.StringVar(value="")
        self.whisper_model_var = tk.StringVar(value="base")
        self.tokenizer_model_var = tk.StringVar(value="")
        self.encoding_var = tk.StringVar(value="")
        self.status_var = tk.StringVar(value="Idle")
        self.summarize_var = tk.BooleanVar(value=False)
        self.ollama_model_var = tk.StringVar(value="llama3")
        self.ollama_url_var = tk.StringVar(value="http://localhost:11434/api/generate")

        self._build_ui()

    def _build_ui(self) -> None:
        padding = {"padx": 10, "pady": 5}

        provider_frame = tk.LabelFrame(self.root, text="Transcription Provider")
        provider_frame.pack(fill="x", **padding)

        tk.Radiobutton(
            provider_frame,
            text="OpenAI Whisper",
            variable=self.provider_var,
            value="openai",
        ).pack(anchor="w", padx=5, pady=2)
        tk.Radiobutton(
            provider_frame,
            text="Local Whisper",
            variable=self.provider_var,
            value="whisper-local",
        ).pack(anchor="w", padx=5, pady=2)

        model_frame = tk.LabelFrame(self.root, text="Model Settings")
        model_frame.pack(fill="x", **padding)

        tk.Label(model_frame, text="OpenAI model:").grid(row=0, column=0, sticky="w")
        tk.Entry(model_frame, textvariable=self.openai_model_var).grid(
            row=0, column=1, sticky="ew", padx=5
        )
        tk.Label(model_frame, text="Whisper model:").grid(row=1, column=0, sticky="w")
        tk.Entry(model_frame, textvariable=self.whisper_model_var).grid(
            row=1, column=1, sticky="ew", padx=5
        )
        tk.Label(model_frame, text="Tokenizer model:").grid(row=2, column=0, sticky="w")
        tk.Entry(model_frame, textvariable=self.tokenizer_model_var).grid(
            row=2, column=1, sticky="ew", padx=5
        )
        tk.Label(model_frame, text="Encoding override:").grid(row=3, column=0, sticky="w")
        tk.Entry(model_frame, textvariable=self.encoding_var).grid(
            row=3, column=1, sticky="ew", padx=5
        )
        model_frame.columnconfigure(1, weight=1)

        control_frame = tk.Frame(self.root)
        control_frame.pack(fill="x", **padding)

        self.record_button = tk.Button(
            control_frame, text="Record Audio", command=self.start_recording
        )
        self.record_button.pack(side="left", expand=True, fill="x", padx=5)

        self.stop_button = tk.Button(
            control_frame,
            text="Stop Recording",
            command=self.stop_recording,
            state="disabled",
        )
        self.stop_button.pack(side="left", expand=True, fill="x", padx=5)

        self.exit_button = tk.Button(
            control_frame,
            text="Exit",
            command=self._exit_app,
        )
        self.exit_button.pack(side="left", expand=True, fill="x", padx=5)

        status_frame = tk.Frame(self.root)
        status_frame.pack(fill="x", **padding)
        tk.Label(status_frame, text="Status:").pack(side="left")
        self.status_label = tk.Label(status_frame, textvariable=self.status_var)
        self.status_label.pack(side="left", padx=5)

        summarize_frame = tk.LabelFrame(self.root, text="Local Summarization (Ollama)")
        summarize_frame.pack(fill="x", **padding)
        tk.Checkbutton(
            summarize_frame,
            text="Generate summary after transcription",
            variable=self.summarize_var,
        ).grid(row=0, column=0, columnspan=2, sticky="w", padx=5, pady=2)
        tk.Label(summarize_frame, text="Ollama model:").grid(row=1, column=0, sticky="w")
        tk.Entry(summarize_frame, textvariable=self.ollama_model_var).grid(
            row=1, column=1, sticky="ew", padx=5
        )
        tk.Label(summarize_frame, text="Endpoint URL:").grid(row=2, column=0, sticky="w")
        tk.Entry(summarize_frame, textvariable=self.ollama_url_var).grid(
            row=2, column=1, sticky="ew", padx=5
        )
        summarize_frame.columnconfigure(1, weight=1)

    def start_recording(self) -> None:
        if self.recorder.is_running():
            return
        try:
            self.recorder.start()
        except Exception as exc:  # pragma: no cover - GUI feedback
            messagebox.showerror("Recording Error", str(exc))
            return

        self.status_var.set("Recording... press stop when finished.")
        self.record_button.config(state="disabled")
        self.stop_button.config(state="normal")
        self._set_exit_enabled(False)

    def stop_recording(self) -> None:
        if not self.recorder.is_running():
            return
        try:
            audio = self.recorder.stop()
        except Exception as exc:  # pragma: no cover - GUI feedback
            messagebox.showerror("Recording Error", str(exc))
            return

        self.record_button.config(state="normal")
        self.stop_button.config(state="disabled")
        if audio.size == 0:
            self.status_var.set("No audio captured.")
            print("No audio captured.")
            self._set_exit_enabled(True)
            return

        self.status_var.set("Processing transcription...")
        worker = threading.Thread(
            target=self._process_audio, args=(audio,), daemon=True
        )
        worker.start()

    def _process_audio(self, audio) -> None:
        try:
            transcriber = self._build_transcriber()
            transcription = transcriber.transcribe(audio, self.recorder.config.sample_rate)
        except SpeechToTextError as exc:
            self._update_status("Transcription failed.")
            print(f"Transcription failed: {exc}")
            self._set_exit_enabled(True)
            return
        except Exception as exc:  # pragma: no cover - unexpected errors
            self._update_status("Transcription error.")
            print(f"Unexpected transcription error: {exc}")
            self._set_exit_enabled(True)
            return

        tokenizer = LLMTokenizer(
            model=self.tokenizer_model_var.get().strip() or None,
            encoding_name=self.encoding_var.get().strip() or None,
        )
        token_result = tokenizer.encode(transcription.text)

        print("=== Transcription Result ===")
        print(f"Text: {transcription.text}")
        if transcription.language:
            print(f"Language: {transcription.language}")
        print(f"Token count ({token_result.encoding_name}): {token_result.count()}")
        #print(f"Tokens: {token_result.tokens}")

        if self.summarize_var.get():
            try:
                summary = summarize_with_ollama( #take raw transcription text and summarize with local LLM (Ollama)
                    transcription.text,
                    model=self.ollama_model_var.get().strip() or None,
                    url=self.ollama_url_var.get().strip() or None,
                )
            except SummarizationError as exc:
                self._update_status("Summary failed.")
                print(f"Summarization failed: {exc}")
            else:
                print(f"Summary ({summary.model}):\n{summary.summary}")
                print(f"Answer: {summary.answer}")

        self._update_status("Done.")
        self._set_exit_enabled(True)

    def _build_transcriber(self) -> SpeechToTextService:
        provider = self.provider_var.get()
        if provider == "openai":
            model = self.openai_model_var.get().strip() or None
            return OpenAIWhisperTranscriber(model=model)
        if provider == "whisper-local":
            model_name = self.whisper_model_var.get().strip() or "base"
            return WhisperLocalTranscriber(model_name=model_name)
        raise ValueError(f"Unsupported provider: {provider}")

    def _update_status(self, message: str) -> None:
        def setter() -> None:
            self.status_var.set(message)

        self.root.after(0, setter)

    def _set_exit_enabled(self, enabled: bool) -> None:
        def setter() -> None:
            state = "normal" if enabled else "disabled"
            self.exit_button.config(state=state)

        self.root.after(0, setter)

    def _exit_app(self) -> None:
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    AudioGuiApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

