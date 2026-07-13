# Voice Cloner

A local voice-cloning app built on [kyutai-labs/pocket-tts](https://github.com/kyutai-labs/pocket-tts) — a lightweight, CPU-only TTS model. Clone a voice from an uploaded sample or a browser mic recording, then generate speech from any text using that voice.

## Setup

```powershell
pip install -r requirements.txt
```

## Run

```powershell
uvicorn backend.app:app --host 0.0.0.0 --port 8000
```

Then open http://localhost:8000 in your browser.

The first request downloads the Pocket TTS model (and, for built-in voices, the reference sample) from Hugging Face, so it may take a minute the first time.

## Usage

- **Built-in voices** are listed automatically (alba, cosette, marius, ...).
- **Clone a voice**: click "+ Clone" in the sidebar, name it, then either upload a clean 5-30s `.wav`/`.mp3`/`.flac`/`.ogg` sample of a single speaker, or record yourself directly in the browser. The cloned voice is saved to `voices/*.safetensors` and appears in the voice list for reuse across sessions.
- **Generate speech**: pick a voice, type text, click Generate. Play it back or download the `.wav`.

## Configuration

- `POCKET_TTS_LANGUAGE` env var selects the model language (e.g. `french_24l`, `german_24l`, `italian_24l`, `spanish_24l`, `portuguese_24l`). Defaults to English.

## Project layout

```
backend/app.py     FastAPI server: voice cloning + generation endpoints, serves the frontend
frontend/          Static single-page UI (HTML/CSS/vanilla JS)
voices/            Cloned voice embeddings (.safetensors) + metadata.json
```
