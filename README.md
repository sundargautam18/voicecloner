# Voice Cloner

A local voice-cloning app built on [kyutai-labs/pocket-tts](https://github.com/kyutai-labs/pocket-tts) — a lightweight, CPU-only TTS model. Clone a voice from an uploaded sample or a browser mic recording, then generate speech from any text using that voice. Everything runs locally: no API keys, no GPU required.

## Requirements

- Python 3.10-3.14
- ~2 GB free disk space (PyTorch + model weights)
- Internet access on first run only, to download the Pocket TTS model and built-in voice samples from Hugging Face

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
