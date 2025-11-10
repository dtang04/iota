## Mic-to-Token Pipeline Prototype

This project provides the initial scaffolding for an application that records audio from a microphone, transcribes the speech to text, and converts that text into tokens ready for use with a Large Language Model (LLM).

### 1. Environment Setup

- Install system dependencies:
  - macOS: `brew install portaudio ffmpeg`
  - Linux: `sudo apt-get install portaudio19-dev ffmpeg`
- Create a virtual environment and install Python requirements:
  ```bash
  cd /Users/dylantang/Desktop/Project
  python3 -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt
  ```

### 2. Configuration

- Create a `.env` file in the project root with at least:
  ```
  OPENAI_API_KEY=sk-...
  OPENAI_MODEL=gpt-4o-mini-transcribe  # optional override
  ```
  The runner automatically loads this file via `python-dotenv`.

### 3. Usage

The orchestrator script `src/app.py` records a short audio clip, transcribes it, and prints token ids.

```bash
python src/app.py --duration 5 --output-wav mic_sample.wav
```

For an interactive experience with record/stop buttons, launch the Tkinter GUI:

```bash
python src/gui.py
```

Speak while “Recording...” is displayed, then press “Stop Recording.” The transcript and tokens will print in the terminal.

To generate local summaries (and have questions auto-answered) via [Ollama](https://ollama.ai), install and run an Ollama model (e.g. `ollama run llama3`), then either:

- CLI: `python src/app.py --duration 5 --summarize --ollama-model llama3`
- GUI: enable “Generate summary after transcription” and specify the model/endpoint if different from defaults.

### 4. Project Structure

```
src/
  __init__.py
  app.py
  audio_capture.py
  config.py
  transcription.py
  tokenization.py
  summarization.py
  gui.py
```

### 5. Next Steps

- Implement streaming audio capture and partial transcription.
- Add UI (CLI prompts or web interface) to visualize transcripts and tokens.
- Integrate with downstream LLM request pipeline. 

