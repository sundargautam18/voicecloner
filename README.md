# Voice Cloner

A local voice-cloning app built on [kyutai-labs/pocket-tts](https://github.com/kyutai-labs/pocket-tts) — a lightweight, CPU-only TTS model. Clone a voice from an uploaded sample or a browser mic recording, then generate speech from any text using that voice. Everything runs locally: no API keys, no GPU required.

## Requirements

- Python 3.10-3.14
- ~2 GB free disk space (PyTorch + model weights)
- Internet access on first run only, to download the Pocket TTS model and built-in voice samples from Hugging Face

## Enabling cloning from your own audio (gated model)

The 26 built-in voices work out of the box. Cloning a voice from your own uploaded/recorded audio uses Kyutai's gated voice-cloning weights, which require a one-time Hugging Face login. First, regardless of which option below you use:

1. Create a free account at https://huggingface.co if you don't have one.
2. Visit https://huggingface.co/kyutai/pocket-tts and accept the model's terms of use.
3. Create an access token at https://huggingface.co/settings/tokens (the default "read" permission is enough).

Then authenticate with **one** of these three equivalent options:

- **In-app (easiest)**: open the app, click "⚙ Settings" in the sidebar, paste the token, and click "Save token". The app validates it against Hugging Face and persists it to the standard Hugging Face credential cache — no restart needed.
- **CLI, inside the activated virtual environment**:
  ```bash
  hf auth login
  ```
- **Environment variable**, set before starting the server (useful for containers/CI):
  - Windows: `$env:HF_TOKEN = "hf_..."`
  - Linux/macOS: `export HF_TOKEN=hf_...`

Without one of these, the "Clone a new voice" feature returns a 400 error explaining that voice-cloning weights couldn't be downloaded; the built-in voice catalog still works fine either way.

## Generating a matching photo sequence (Gemini, optional)

After generating speech, you can click "🖼 Generate matching photos (.zip)" to have Google's Gemini turn the narration into a sequence of AI-generated photos (one per sentence, or per group of sentences if the text is long), packaged as a `.zip` — handy as b-roll to drop straight into a video editor alongside the generated audio.

This requires a Gemini API key:

1. Create a free key at https://aistudio.google.com/apikey.
2. Either paste it into "⚙ Settings" in the sidebar and click "Save settings", or set the `GEMINI_API_KEY` environment variable before starting the server (Windows: `$env:GEMINI_API_KEY = "AIza..."`, Linux/macOS: `export GEMINI_API_KEY=AIza...`).

Without a key, the button returns a 400 error explaining the key isn't configured yet; everything else in the app works fine without it.

## Setup

Use a virtual environment so the app's dependencies stay isolated from the rest of your system and nothing conflicts with other Python projects.

### Windows (PowerShell)

```powershell
cd voiceclone
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If activation is blocked by execution policy, run once as your user (not admin):
```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

### Windows (Git Bash / WSL-style shell)

```bash
cd voiceclone
python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt
```

### Linux / macOS (bash/zsh)

```bash
cd voiceclone
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

With the virtual environment activated, run the server with **`python -m uvicorn`** rather than the bare `uvicorn` command — on Windows, pip-installed scripts often land in a directory that isn't on `PATH` (that's the cause of a `uvicorn: command not found` / `'uvicorn' is not recognized` error), while `python -m uvicorn` always works because it just asks the active Python interpreter to run the module:

```bash
python -m uvicorn backend.app:app --host 0.0.0.0 --port 8000
```

Then open http://localhost:8000 in your browser.

The first request downloads the Pocket TTS model (~440 MB) and, for built-in voices, the reference sample, from Hugging Face — this can take a minute or two. Subsequent runs are instant since everything is cached locally (default Hugging Face cache dir: `~/.cache/huggingface` on Linux/macOS, `%USERPROFILE%\.cache\huggingface` on Windows).

To stop the server, press `Ctrl+C`. To leave the virtual environment afterwards, run `deactivate`.

### Re-running later

You don't need to reinstall dependencies each time — just re-activate the existing `.venv` and start the server:

**Windows (PowerShell):**
```powershell
cd voiceclone
.\.venv\Scripts\Activate.ps1
python -m uvicorn backend.app:app --host 0.0.0.0 --port 8000
```

**Windows (Git Bash) / Linux / macOS:**
```bash
cd voiceclone
source .venv/Scripts/activate   # Linux/macOS: .venv/bin/activate
python -m uvicorn backend.app:app --host 0.0.0.0 --port 8000
```

## Usage

- **Built-in voices** are listed automatically (alba, cosette, marius, ...).
- **Clone a voice**: click "+ Clone" in the sidebar, name it, then either upload a clean 5-30s `.wav`/`.mp3`/`.flac`/`.ogg` sample of a single speaker, or record yourself directly in the browser (grant microphone permission when prompted). The cloned voice is saved to `voices/*.safetensors` and appears in the voice list for reuse across sessions.
- **Generate speech**: pick a voice, type text, click Generate. Play it back or download the `.wav`.
- **Delete a cloned voice**: hover it in the sidebar and click the trash icon. Built-in voices can't be deleted.

## Configuration

- `POCKET_TTS_LANGUAGE` env var selects the model language: `english` (default), `french_24l`, `german_24l`, `italian_24l`, `spanish_24l`, `portuguese_24l`. Set it before starting the server, e.g.:
  - Windows: `$env:POCKET_TTS_LANGUAGE = "french_24l"`
  - Linux/macOS: `export POCKET_TTS_LANGUAGE=french_24l`

## Project layout

```
backend/app.py     FastAPI server: voice cloning + generation endpoints, serves the frontend
frontend/          Static single-page UI (HTML/CSS/vanilla JS)
voices/            Cloned voice embeddings (.safetensors) + metadata.json
requirements.txt   Python dependencies
```

## Troubleshooting

- **"Model is still loading, try again shortly" (503)**: the server just started and is still loading weights; wait a few seconds and retry.
- **Slow first generation**: expected — the model and default voice sample are being downloaded and cached.
- **Microphone recording doesn't work**: browsers only allow mic access on `localhost` or HTTPS. Access the app via `http://localhost:8000`, not a raw IP address.
- **`ModuleNotFoundError` on startup**: the virtual environment isn't activated, or `pip install -r requirements.txt` wasn't run inside it. Re-check the Setup steps above.
- **`uvicorn: command not found` / `'uvicorn' is not recognized`**: use `python -m uvicorn ...` instead of the bare `uvicorn` command (see Run section above).
- **Port already in use**: change the port, e.g. `python -m uvicorn backend.app:app --port 8001`.
- **Cloning fails with "could not download the weights for the model with voice cloning"**: you haven't accepted the gated model's terms and/or logged in via `hf auth login` yet — see "Enabling cloning from your own audio" above.
- **"Gemini API key not configured"**: add a key in Settings or set `GEMINI_API_KEY` — see "Generating a matching photo sequence" above.
- **Photo sequence generation is slow or fails partway through**: each photo is a separate Gemini API call; longer text produces more images (capped at 12) and takes longer. A failure on one segment (e.g. rate limiting) aborts the whole request — wait a moment and retry.
