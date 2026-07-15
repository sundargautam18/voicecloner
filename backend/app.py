"""FastAPI backend for the Voice Cloner app, built on kyutai-labs/pocket-tts."""

import io
import json
import logging
import os
import subprocess
import tempfile
import textwrap
import uuid
import wave
from contextlib import asynccontextmanager
from pathlib import Path

import huggingface_hub
import imageio_ffmpeg
import numpy as np
from dotenv import load_dotenv
from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from google import genai
from google.genai import errors as genai_errors
from pocket_tts import TTSModel
from pocket_tts.utils.utils import _ORIGINS_OF_PREDEFINED_VOICES

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("voiceclone")

ROOT_DIR = Path(__file__).resolve().parent.parent
VOICES_DIR = ROOT_DIR / "voices"
FRONTEND_DIR = ROOT_DIR / "frontend"
METADATA_PATH = VOICES_DIR / "metadata.json"
VOICES_DIR.mkdir(exist_ok=True)

BUILTIN_VOICES = sorted(_ORIGINS_OF_PREDEFINED_VOICES.keys())
ALLOWED_UPLOAD_SUFFIXES = {".wav", ".mp3", ".flac", ".ogg"}
GEMINI_MODEL = "gemini-flash-latest"
GEMINI_IMAGE_MODEL = "gemini-2.5-flash-image"
FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()
FONT_CANDIDATES = [
    Path("C:/Windows/Fonts/arial.ttf"),
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
]

state = {"model": None, "voice_states": {}}


def load_metadata() -> dict:
    if METADATA_PATH.exists():
        return json.loads(METADATA_PATH.read_text())
    return {}


def save_metadata(metadata: dict) -> None:
    METADATA_PATH.write_text(json.dumps(metadata, indent=2))


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


def audio_to_samples(audio) -> np.ndarray:
    samples = audio.detach().cpu().numpy()
    if samples.ndim > 1:
        samples = samples.reshape(-1)
    return samples


def tensor_to_wav_bytes(audio, sample_rate: int) -> bytes:
    samples = audio_to_samples(audio)
    pcm16 = (np.clip(samples, -1.0, 1.0) * 32767.0).astype(np.int16)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm16.tobytes())
    return buffer.getvalue()


def find_caption_font() -> Path | None:
    for candidate in FONT_CANDIDATES:
        if candidate.exists():
            return candidate
    return None


def escape_ffmpeg_path(path: str) -> str:
    return path.replace("\\", "/").replace(":", "\\:")


def generate_background_image(client: "genai.Client", text: str) -> bytes:
    prompt = (
        "Create a visually fitting background image for a short narrated video. "
        f'The narration is about: "{text[:500]}". '
        "Style: cinematic, atmospheric, high quality, 16:9 widescreen. "
        "Do not include any text, letters, words, logos, watermarks, or human faces in the image."
    )
    try:
        response = client.models.generate_content(model=GEMINI_IMAGE_MODEL, contents=prompt)
    except genai_errors.APIError as exc:
        raise HTTPException(status_code=502, detail=f"Gemini image request failed: {exc}") from exc

    for part in response.candidates[0].content.parts:
        if part.inline_data and part.inline_data.data:
            return part.inline_data.data
    raise HTTPException(status_code=502, detail="Gemini did not return an image.")


def build_video_bytes(image_bytes: bytes, wav_bytes: bytes, duration_seconds: float, caption: str) -> bytes:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        image_path = tmp_path / "background.png"
        audio_path = tmp_path / "audio.wav"
        caption_path = tmp_path / "caption.txt"
        output_path = tmp_path / "output.mp4"

        image_path.write_bytes(image_bytes)
        audio_path.write_bytes(wav_bytes)

        wrapped_lines = textwrap.wrap(caption, width=42) or [""]
        fontsize = 32 if len(wrapped_lines) <= 6 else max(20, int(32 * 6 / len(wrapped_lines)))
        caption_path.write_text("\n".join(wrapped_lines), encoding="utf-8")

        fps = 25
        frames = max(1, round(duration_seconds * fps))

        video_filter = (
            "scale=1280:720:force_original_aspect_ratio=increase,"
            "crop=1280:720,"
            f"zoompan=z='min(zoom+0.0012,1.2)':d={frames}:s=1280x720:fps={fps},"
            "setsar=1"
        )

        drawtext_opts = [
            f"textfile='{escape_ffmpeg_path(str(caption_path))}'",
            f"fontsize={fontsize}",
            "fontcolor=white",
            "borderw=2",
            "bordercolor=black@0.7",
            "line_spacing=6",
            "x=(w-text_w)/2",
            "y=h-text_h-50",
        ]
        font = find_caption_font()
        if font:
            drawtext_opts.insert(0, f"fontfile='{escape_ffmpeg_path(str(font))}'")
        drawtext_filter = "drawtext=" + ":".join(drawtext_opts)

        cmd = [
            FFMPEG_EXE, "-y",
            "-loop", "1", "-i", str(image_path),
            "-i", str(audio_path),
            "-filter:v", f"{video_filter},{drawtext_filter}",
            "-map", "0:v", "-map", "1:a",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k",
            "-shortest", "-movflags", "+faststart",
            str(output_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise HTTPException(status_code=500, detail=f"Video rendering failed: {result.stderr[-800:]}")
        return output_path.read_bytes()


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


@app.post("/api/generate-video")
def generate_video(text: str = Form(...), voice_id: str = Form(...)):
    text = text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=400,
            detail="Gemini isn't configured. Set GEMINI_API_KEY in your .env and restart the server.",
        )

    model = get_model()
    metadata = load_metadata()
    voice_state = get_voice_state(voice_id, metadata)

    audio = model.generate_audio(voice_state, text, copy_state=True)
    samples = audio_to_samples(audio)
    duration_seconds = len(samples) / model.sample_rate
    wav_bytes = tensor_to_wav_bytes(audio, model.sample_rate)

    client = genai.Client(api_key=api_key)
    image_bytes = generate_background_image(client, text)
    video_bytes = build_video_bytes(image_bytes, wav_bytes, duration_seconds, text)

    return Response(content=video_bytes, media_type="video/mp4")


@app.post("/api/script")
def generate_script(prompt: str = Form(...)):
    prompt = prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt cannot be empty.")

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=400,
            detail="Gemini isn't configured. Set GEMINI_API_KEY in your .env and restart the server.",
        )

    client = genai.Client(api_key=api_key)
    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=(
                "Write a short script meant to be read aloud by a text-to-speech voice. "
                "Return only the spoken text itself, with no titles, labels, stage directions, "
                "or markdown formatting.\n\n"
                f"Topic: {prompt}"
            ),
        )
    except genai_errors.APIError as exc:
        raise HTTPException(status_code=502, detail=f"Gemini request failed: {exc}") from exc

    text = (response.text or "").strip()
    if not text:
        raise HTTPException(status_code=502, detail="Gemini returned an empty response.")
    return {"text": text}


@app.get("/api/settings")
def get_settings():
    token = huggingface_hub.get_token()
    username = None
    if token:
        try:
            username = huggingface_hub.whoami(token=token).get("name")
        except Exception:
            username = None
    return {"hf_token_configured": bool(token), "hf_username": username, "gemini_configured": bool(os.environ.get("GEMINI_API_KEY"))}


@app.post("/api/settings")
def set_settings(payload: dict = Body(...)):
    token = (payload.get("hf_token") or "").strip()
    if not token:
        raise HTTPException(status_code=400, detail="Token cannot be empty.")
    try:
        huggingface_hub.login(token=token, add_to_git_credential=False, skip_if_logged_in=False)
        username = huggingface_hub.whoami(token=token).get("name")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not validate token: {exc}") from exc
    return {"hf_token_configured": True, "hf_username": username}


app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
