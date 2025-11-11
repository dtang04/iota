## Mic-to-Token Pipeline Prototype

This project provides the initial scaffolding for an application that records audio from a microphone, transcribes the speech to text, and converts that text into tokens ready for use with a Large Language Model (LLM).

Current Models:
- Transcription: Local Whisper (via openai-whisper)
- LLM - Ollama 3

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

- Create a `.env` file in the project root if you need other environment values (no OpenAI key required).

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

Optional GUI features:

- Enable “Generate summary after transcription” to send the transcript to a local Ollama model for summarization and question answering.
- Enable “Save transcript and terminal output…” to write each run to a timestamped text file in your chosen directory (default `outputs/`).

To generate local summaries (and have questions auto-answered) via [Ollama](https://ollama.ai), install and run an Ollama model (e.g. `ollama run llama3`), then enable summarization in the UI or CLI options.

### 4. Web App

Serve a FastAPI backend and static web client to control the pipeline from a browser:

```bash
uvicorn web_server:app --reload
```

Open [http://localhost:8000/](http://localhost:8000/) to:

- Record locally with Whisper
- Configure models/encodings and optional Ollama summarization
- Download transcripts and summaries as `.txt`

When deploying to Render, no OpenAI API key is needed because transcription runs locally.

### 5. Project Structure

```
src/
  __init__.py
  app.py
  audio_capture.py
  config.py
  web_server.py
  transcription.py
  tokenization.py
  summarization.py
  gui.py
web/
  index.html
  styles.css
  app.js
```

### 6. Next Steps

