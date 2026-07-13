"""FastAPI backend for the Voice Cloner app, built on kyutai-labs/pocket-tts."""

import io
import json
import logging
import math
import os
import re
import uuid
import wave
import zipfile
from contextlib import asynccontextmanager
from pathlib import Path

import huggingface_hub
import numpy as np
from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from pocket_tts import TTSModel
from pocket_tts.utils.utils import _ORIGINS_OF_PREDEFINED_VOICES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("voiceclone")

ROOT_DIR = Path(__file__).resolve().parent.parent
VOICES_DIR = ROOT_DIR / "voices"
FRONTEND_DIR = ROOT_DIR / "frontend"
METADATA_PATH = VOICES_DIR / "metadata.json"
APP_SETTINGS_PATH = ROOT_DIR / "settings.json"
VOICES_DIR.mkdir(exist_ok=True)

BUILTIN_VOICES = sorted(_ORIGINS_OF_PREDEFINED_VOICES.keys())
ALLOWED_UPLOAD_SUFFIXES = {".wav", ".mp3", ".flac", ".ogg"}
MAX_STORYBOARD_IMAGES = 12
GEMINI_IMAGE_MODEL = "gemini-2.5-flash-image"

state = {"model": None, "voice_states": {}}


def load_metadata() -> dict:
    if METADATA_PATH.exists():
        return json.loads(METADATA_PATH.read_text())
    return {}


def save_metadata(metadata: dict) -> None:
    METADATA_PATH.write_text(json.dumps(metadata, indent=2))


def load_app_settings() -> dict:
    if APP_SETTINGS_PATH.exists():
        return json.loads(APP_SETTINGS_PATH.read_text())
    return {}


def save_app_settings(data: dict) -> None:
    APP_SETTINGS_PATH.write_text(json.dumps(data, indent=2))


def get_gemini_api_key() -> str | None:
    return load_app_settings().get("gemini_api_key") or os.environ.get("GEMINI_API_KEY") or None


def split_into_segments(text: str, max_segments: int = MAX_STORYBOARD_IMAGES) -> list[str]:
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]
    if not sentences:
        return []
    if len(sentences) <= max_segments:
        return sentences
    bucket_size = math.ceil(len(sentences) / max_segments)
    return [" ".join(sentences[i : i + bucket_size]) for i in range(0, len(sentences), bucket_size)]


@asynccontextmanager
async def lifespan(app: FastAPI):
    language = os.environ.get("POCKET_TTS_LANGUAGE") or None
    logger.info("Loading pocket-tts model (language=%s)...", language or "default")
    state["model"] = TTSModel.load_model(language=language)
    logger.info("Model loaded on device %s", state["model"].device)
    yield
    state["model"] = None


app = FastAPI(title="Voice Cloner", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_model() -> TTSModel:
    model = state["model"]
    if model is None:
        raise HTTPException(status_code=503, detail="Model is still loading, try again shortly.")
    return model


def resolve_voice_source(voice_id: str, metadata: dict) -> str:
    if voice_id in BUILTIN_VOICES:
        return voice_id
    if voice_id in metadata:
        return str(VOICES_DIR / metadata[voice_id]["file"])
    raise HTTPException(status_code=404, detail=f"Unknown voice '{voice_id}'")


def get_voice_state(voice_id: str, metadata: dict):
    cached = state["voice_states"].get(voice_id)
    if cached is not None:
        return cached
    model = get_model()
    source = resolve_voice_source(voice_id, metadata)
    voice_state = model.get_state_for_audio_prompt(source, truncate=True)
    state["voice_states"][voice_id] = voice_state
    return voice_state


def tensor_to_wav_bytes(audio, sample_rate: int) -> bytes:
    samples = audio.detach().cpu().numpy()
    if samples.ndim > 1:
        samples = samples.reshape(-1)
    pcm16 = (np.clip(samples, -1.0, 1.0) * 32767.0).astype(np.int16)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm16.tobytes())
    return buffer.getvalue()


@app.get("/api/voices")
def list_voices():
    metadata = load_metadata()
    builtin = [{"id": name, "name": name, "type": "builtin"} for name in BUILTIN_VOICES]
    cloned = [
        {"id": voice_id, "name": info["name"], "type": "cloned"}
        for voice_id, info in metadata.items()
    ]
    return {"voices": builtin + cloned}


@app.post("/api/voices")
async def clone_voice(name: str = Form(...), file: UploadFile = File(...)):
    name = name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Voice name cannot be empty.")

    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_UPLOAD_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Use one of {sorted(ALLOWED_UPLOAD_SUFFIXES)}.",
        )

    voice_id = uuid.uuid4().hex
    raw_path = VOICES_DIR / f"{voice_id}_raw{suffix}"
    raw_path.write_bytes(await file.read())

    model = get_model()
    try:
        voice_state = model.get_state_for_audio_prompt(raw_path, truncate=True)
    except Exception as exc:
        raw_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"Could not process audio: {exc}") from exc
    finally:
        raw_path.unlink(missing_ok=True)

    from pocket_tts import export_model_state

    safetensors_path = VOICES_DIR / f"{voice_id}.safetensors"
    export_model_state(voice_state, safetensors_path)
    state["voice_states"][voice_id] = voice_state

    metadata = load_metadata()
    metadata[voice_id] = {"name": name, "file": safetensors_path.name}
    save_metadata(metadata)

    return {"id": voice_id, "name": name, "type": "cloned"}


@app.delete("/api/voices/{voice_id}")
def delete_voice(voice_id: str):
    metadata = load_metadata()
    if voice_id not in metadata:
        raise HTTPException(status_code=404, detail="Cloned voice not found.")
    safetensors_path = VOICES_DIR / metadata[voice_id]["file"]
    safetensors_path.unlink(missing_ok=True)
    del metadata[voice_id]
    save_metadata(metadata)
    state["voice_states"].pop(voice_id, None)
    return {"ok": True}


@app.post("/api/generate")
def generate_speech(text: str = Form(...), voice_id: str = Form(...)):
    text = text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    model = get_model()
    metadata = load_metadata()
    voice_state = get_voice_state(voice_id, metadata)

    audio = model.generate_audio(voice_state, text, copy_state=True)
    wav_bytes = tensor_to_wav_bytes(audio, model.sample_rate)
    return Response(content=wav_bytes, media_type="audio/wav")


@app.get("/api/settings")
def get_settings():
    token = huggingface_hub.get_token()
    username = None
    if token:
        try:
            username = huggingface_hub.whoami(token=token).get("name")
        except Exception:
            username = None
    return {
        "hf_token_configured": bool(token),
        "hf_username": username,
        "gemini_api_key_configured": bool(get_gemini_api_key()),
    }


@app.post("/api/settings")
def set_settings(payload: dict = Body(...)):
    hf_token = (payload.get("hf_token") or "").strip()
    gemini_api_key = (payload.get("gemini_api_key") or "").strip()
    if not hf_token and not gemini_api_key:
        raise HTTPException(status_code=400, detail="Provide a Hugging Face token and/or a Gemini API key.")

    if hf_token:
        try:
            huggingface_hub.login(token=hf_token, add_to_git_credential=False, skip_if_logged_in=False)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Could not validate Hugging Face token: {exc}") from exc

    if gemini_api_key:
        app_settings = load_app_settings()
        app_settings["gemini_api_key"] = gemini_api_key
        save_app_settings(app_settings)

    current_token = huggingface_hub.get_token()
    username = None
    if current_token:
        try:
            username = huggingface_hub.whoami(token=current_token).get("name")
        except Exception:
            username = None

    return {
        "hf_token_configured": bool(current_token),
        "hf_username": username,
        "gemini_api_key_configured": bool(get_gemini_api_key()),
    }


@app.post("/api/storyboard")
def generate_storyboard(text: str = Form(...)):
    text = text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    api_key = get_gemini_api_key()
    if not api_key:
        raise HTTPException(
            status_code=400,
            detail="Gemini API key not configured. Add it in Settings first.",
        )

    try:
        from google import genai
    except ImportError as exc:
        raise HTTPException(
            status_code=500, detail="google-genai package is not installed. Run pip install -r requirements.txt."
        ) from exc

    segments = split_into_segments(text)
    if not segments:
        raise HTTPException(status_code=400, detail="No text to generate images from.")

    client = genai.Client(api_key=api_key)
    zip_buffer = io.BytesIO()
    manifest = []

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for i, segment in enumerate(segments, start=1):
            prompt = (
                "Create a photorealistic, cinematic still that visually depicts this moment "
                f"from a narration, suitable as a b-roll frame for video editing: {segment}"
            )
            try:
                response = client.models.generate_content(model=GEMINI_IMAGE_MODEL, contents=[prompt])
            except Exception as exc:
                raise HTTPException(
                    status_code=502, detail=f"Gemini image generation failed on segment {i}: {exc}"
                ) from exc

            image_bytes = None
            for part in response.candidates[0].content.parts:
                inline_data = getattr(part, "inline_data", None)
                if inline_data and inline_data.data:
                    image_bytes = inline_data.data
                    break

            if image_bytes is None:
                raise HTTPException(status_code=502, detail=f"Gemini did not return an image for segment {i}.")

            filename = f"{i:02d}.png"
            zf.writestr(filename, image_bytes)
            manifest.append({"file": filename, "text": segment})

        zf.writestr("manifest.json", json.dumps(manifest, indent=2))

    zip_buffer.seek(0)
    return Response(
        content=zip_buffer.getvalue(),
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=storyboard.zip"},
    )


app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
